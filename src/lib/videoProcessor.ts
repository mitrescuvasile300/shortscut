/**
 * Browser-based video processing with ffmpeg.wasm
 * Handles: download → face detection (scene-aware, dual-mode capable)
 *          → crop 9:16 → burn ASS subtitles → encode
 *
 * Improvements over v1:
 *  - Pre-warms MediaPipe in parallel with ffmpeg load + video download
 *  - Per-clip CropPlan with multiple segments (true scene tracking)
 *  - Dual-mode split-screen for wide two-person shots
 *  - veryfast/CRF 23 encoding (was ultrafast/28 — visibly soft output)
 *  - Frame-accurate seek (-ss AFTER -i) for clip start
 */

import { FFmpeg } from "@ffmpeg/ffmpeg";
import { toBlobURL } from "@ffmpeg/util";
import {
  type CropPlan,
  type CropSegment,
  disposeFaceDetector,
  getVideoDimensions,
  planClipCropFromFrames,
  prewarmFaceDetector,
  type TrackingKeyframe,
} from "./faceDetection";
import {
  downloadClipSegment,
  getFileSize,
  supportsRangeRequests,
} from "./segmentDownloader";

let ffmpegInstance: FFmpeg | null = null;
let ffmpegLoaded = false;

export interface ProcessingProgress {
  clipIndex: number;
  totalClips: number;
  stage:
    | "loading"
    | "downloading"
    | "detecting"
    | "processing"
    | "encoding"
    | "done"
    | "error";
  percent: number;
  message: string;
}

export type ProgressCallback = (progress: ProcessingProgress) => void;

// ── ffmpeg.wasm singleton ──────────────────────────────────────────────
async function getFFmpeg(_onProgress?: ProgressCallback): Promise<FFmpeg> {
  if (ffmpegInstance && ffmpegLoaded) return ffmpegInstance;
  const ffmpeg = new FFmpeg();
  ffmpeg.on("log", ({ message }) => {
    console.log("[ffmpeg]", message);
  });
  const baseURL = "https://unpkg.com/@ffmpeg/core@0.12.10/dist/esm";
  await ffmpeg.load({
    coreURL: await toBlobURL(`${baseURL}/ffmpeg-core.js`, "text/javascript"),
    wasmURL: await toBlobURL(`${baseURL}/ffmpeg-core.wasm`, "application/wasm"),
  });
  ffmpegInstance = ffmpeg;
  ffmpegLoaded = true;
  return ffmpeg;
}

// ── Chunked download (unchanged from v1) ───────────────────────────────
const CHUNK_SIZE = 8 * 1024 * 1024;

async function downloadVideo(
  url: string,
  onProgress?: (percent: number) => void,
): Promise<Uint8Array> {
  const probeResp = await fetch(url, { headers: { Range: "bytes=0-0" } });
  let totalSize = 0;
  if (probeResp.status === 206) {
    const cr = probeResp.headers.get("Content-Range");
    if (cr) {
      const match = cr.match(/\/(\d+)/);
      if (match) totalSize = parseInt(match[1], 10);
    }
  }
  if (!totalSize) {
    const fullResp = await fetch(url);
    if (!fullResp.ok) throw new Error(`Download failed: ${fullResp.status}`);
    totalSize = Number(fullResp.headers.get("content-length") || 0);
    if (totalSize > 0 && totalSize <= CHUNK_SIZE) {
      const reader = fullResp.body?.getReader();
      if (!reader) throw new Error("No response body");
      const chunks: Uint8Array[] = [];
      let received = 0;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        received += value.length;
        if (onProgress) onProgress(Math.round((received / totalSize) * 100));
      }
      const result = new Uint8Array(received);
      let offset = 0;
      for (const chunk of chunks) {
        result.set(chunk, offset);
        offset += chunk.length;
      }
      return result;
    }
    if (!totalSize) {
      const reader = fullResp.body?.getReader();
      if (!reader) throw new Error("No response body");
      const chunks: Uint8Array[] = [];
      let received = 0;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        received += value.length;
      }
      const result = new Uint8Array(received);
      let offset = 0;
      for (const chunk of chunks) {
        result.set(chunk, offset);
        offset += chunk.length;
      }
      return result;
    }
  }
  console.log(
    `[download] Total size: ${(totalSize / 1024 / 1024).toFixed(1)}MB, chunk size: ${(CHUNK_SIZE / 1024 / 1024).toFixed(0)}MB`,
  );
  const result = new Uint8Array(totalSize);
  let downloaded = 0;
  while (downloaded < totalSize) {
    const start = downloaded;
    const end = Math.min(downloaded + CHUNK_SIZE - 1, totalSize - 1);
    const chunkResp = await fetch(url, {
      headers: { Range: `bytes=${start}-${end}` },
    });
    if (!chunkResp.ok && chunkResp.status !== 206) {
      throw new Error(
        `Chunk download failed: ${chunkResp.status} (bytes ${start}-${end})`,
      );
    }
    const reader = chunkResp.body?.getReader();
    if (!reader) throw new Error("No chunk body");
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      result.set(value, downloaded);
      downloaded += value.length;
      if (onProgress) onProgress(Math.round((downloaded / totalSize) * 100));
    }
  }
  return result;
}

// ── Clip config ────────────────────────────────────────────────────────
export interface ClipConfig {
  index: number;
  title: string;
  startTime: number;
  endTime: number;
  assSubtitles?: string;
  /** Legacy single-X crop (still supported). New code should set cropPlan. */
  cropX?: number;
  /** Full crop plan from planClipCrop — preferred over cropX */
  cropPlan?: CropPlan;
}

function formatTime(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = (seconds % 60).toFixed(3);
  return `${h.toString().padStart(2, "0")}:${m.toString().padStart(2, "0")}:${s.padStart(6, "0")}`;
}

// ── Filter construction ────────────────────────────────────────────────

/**
 * Build a piecewise-linear ffmpeg `crop` X expression from keyframes.
 *
 * Keyframes are in segment-relative time (t=0 at segment start).
 * Caller passes `tOffset` = segment-start in clip time, because the
 * `crop` filter receives clip-time `t` (the outer ffmpeg already trims
 * the input to [clip.startTime, clip.endTime]).
 *
 * Output (for keyframes (s0,x0)..(sN,xN), with absolute times Ti = sI+tOffset):
 *   if(lt(t,T1), x0 + (t-T0)/(T1-T0)*(x1-x0),
 *     if(lt(t,T2), x1 + (t-T1)/(T2-T1)*(x2-x1),
 *       ...
 *         xN))
 *
 * For t < T0 the expression naturally evaluates to the first branch,
 * which extrapolates linearly toward x0 — but since the outer trim
 * filter discards those frames anyway, it doesn't matter.
 *
 * Clamp the produced expression to [0, srcW-cropW] via `clip(...)` so
 * we never produce an out-of-bounds crop X if floats drift.
 *
 * Caller must wrap the returned expression in single quotes when using it
 * in either -vf or filter_complex (commas inside if(...) would otherwise
 * be parsed as filter-chain separators).
 */
function buildTrackingExpr(
  keyframes: TrackingKeyframe[],
  tOffset: number,
  srcW: number,
  cropW: number,
): string {
  const maxCropX = srcW - cropW;
  if (keyframes.length === 0) {
    return String(Math.round(maxCropX / 2));
  }
  if (keyframes.length === 1) {
    const x = Math.max(0, Math.min(Math.round(keyframes[0].x), maxCropX));
    return String(x);
  }

  // Absolute times (clip-relative)
  const T = keyframes.map(k => k.t + tOffset);
  const X = keyframes.map(k => Math.round(k.x));

  // Build nested if() from the right: start with the last x, then wrap.
  let expr = String(X[X.length - 1]);
  for (let i = T.length - 2; i >= 0; i--) {
    const t0 = T[i].toFixed(4);
    const t1 = T[i + 1].toFixed(4);
    const x0 = X[i];
    const x1 = X[i + 1];
    const dx = x1 - x0;
    const dt = T[i + 1] - T[i];
    if (dt <= 0 || dx === 0) {
      // Degenerate segment: hold x0 until next
      expr = `if(lt(t,${t1}),${x0},${expr})`;
    } else {
      // x0 + (t - t0) * (dx/dt)
      const slope = (dx / dt).toFixed(6);
      expr = `if(lt(t,${t1}),${x0}+(t-${t0})*${slope},${expr})`;
    }
  }
  // Final clip to bounds (commas inside clip() are fine — we single-quote
  // the whole expression at the call site)
  return `clip(${expr},0,${maxCropX})`;
}

/**
 * Build a -filter_complex string for a clip with a multi-segment plan.
 *
 * Strategy:
 *   - For each segment, build a separate filter chain (crop+scale, or
 *     split/crop/vstack for dual, or crop=x='expr' for tracking)
 *   - Use `enable='between(t,a,b)'` so each only renders during its segment
 *   - Overlay them onto a black canvas to avoid frame ordering issues
 *
 * For simple plans (one single-mode segment) we still emit a plain -vf.
 */
function buildFilters(
  plan: CropPlan,
  outW: number,
  outH: number,
  hasSubs: boolean,
): { vf?: string; filterComplex?: string; outLabel?: string } {
  const { videoWidth: srcW, videoHeight: srcH, segments } = plan;
  const cropW = Math.round((srcH * 9) / 16);
  const cropH = srcH;

  // Fast path: one segment, single/center → plain -vf with static crop
  if (
    segments.length === 1 &&
    (segments[0].mode === "single" || segments[0].mode === "center")
  ) {
    const seg = segments[0];
    const cropX =
      seg.cropX !== undefined
        ? Math.max(0, Math.min(seg.cropX, srcW - cropW))
        : Math.round((srcW - cropW) / 2);
    const base = `crop=${cropW}:${cropH}:${cropX}:0,scale=${outW}:${outH}`;
    const vf = hasSubs ? `${base},ass=subs.ass` : base;
    return { vf };
  }

  // Fast path: one segment, tracking → plain -vf with crop=x='expr'
  if (segments.length === 1 && segments[0].mode === "tracking") {
    const seg = segments[0];
    const expr = buildTrackingExpr(seg.keyframes ?? [], 0, srcW, cropW);
    // Quote the expression so ffmpeg parses it as a single argument
    // (escape colons and commas inside it via \, already done in clip(...))
    const base = `crop=${cropW}:${cropH}:'${expr}':0,scale=${outW}:${outH}`;
    const vf = hasSubs ? `${base},ass=subs.ass` : base;
    return { vf };
  }

  // Fast path: one segment, dual mode → filter_complex with split+vstack
  if (segments.length === 1 && segments[0].mode === "dual") {
    const seg = segments[0];
    const halfH = Math.floor(outH / 2);

    // FIX: Each half-screen has aspect ratio 9:8, not 9:16!
    // Using 9:16 cropW here would squish faces ~180% when scaling to halfH.
    const dualCropW = Math.min(srcW, Math.round((srcH * 9) / 8));

    const cx1 = clampCrop(
      Math.round((seg.face1X ?? 0) - dualCropW / 2),
      srcW,
      dualCropW,
    );
    const cx2 = clampCrop(
      Math.round((seg.face2X ?? 0) - dualCropW / 2),
      srcW,
      dualCropW,
    );
    let fc =
      `[0:v]split=2[top][bot];` +
      `[top]crop=${dualCropW}:${cropH}:${cx1}:0,scale=${outW}:${halfH}[t];` +
      `[bot]crop=${dualCropW}:${cropH}:${cx2}:0,scale=${outW}:${halfH}[b];` +
      `[t][b]vstack=inputs=2[stacked]`;
    if (hasSubs) {
      fc += `;[stacked]ass=subs.ass[v]`;
      return { filterComplex: fc, outLabel: "[v]" };
    }
    return { filterComplex: fc, outLabel: "[stacked]" };
  }

  // Multi-segment: build N branches, each enabled for its time window,
  // overlay onto a base canvas. Each branch crops from the SAME source
  // (we duplicate with split).
  const n = segments.length;
  const splitOuts = Array.from({ length: n }, (_, i) => `[s${i}]`).join("");
  let fc = `[0:v]split=${n}${splitOuts}`;

  const overlayInputs: string[] = [];
  for (let i = 0; i < n; i++) {
    const seg = segments[i];
    const t0 = seg.startTime;
    const t1 = seg.endTime;
    if (seg.mode === "dual") {
      const halfH = Math.floor(outH / 2);
      // FIX: Each half-screen has aspect ratio 9:8, not 9:16
      const dualCropW = Math.min(srcW, Math.round((srcH * 9) / 8));
      const cx1 = clampCrop(
        Math.round((seg.face1X ?? 0) - dualCropW / 2),
        srcW,
        dualCropW,
      );
      const cx2 = clampCrop(
        Math.round((seg.face2X ?? 0) - dualCropW / 2),
        srcW,
        dualCropW,
      );
      // Each split output needs its own split→crop→vstack
      fc +=
        `;[s${i}]split=2[s${i}a][s${i}b]` +
        `;[s${i}a]crop=${dualCropW}:${cropH}:${cx1}:0,scale=${outW}:${halfH}[s${i}ac]` +
        `;[s${i}b]crop=${dualCropW}:${cropH}:${cx2}:0,scale=${outW}:${halfH}[s${i}bc]` +
        `;[s${i}ac][s${i}bc]vstack=inputs=2,setpts=PTS-STARTPTS[seg${i}]`;
    } else if (seg.mode === "tracking") {
      // Tracking: piecewise-linear crop=x='expr'. The expression sees clip-
      // relative time `t` because the outer ffmpeg command already trims
      // the source to [clip.startTime, clip.endTime]. We pass tOffset=t0
      // so segment-relative keyframes are translated into clip-relative.
      const expr = buildTrackingExpr(seg.keyframes ?? [], t0, srcW, cropW);
      fc += `;[s${i}]crop=${cropW}:${cropH}:'${expr}':0,scale=${outW}:${outH},setpts=PTS-STARTPTS[seg${i}]`;
    } else {
      const cropX =
        seg.cropX !== undefined
          ? clampCrop(seg.cropX, srcW, cropW)
          : Math.round((srcW - cropW) / 2);
      fc += `;[s${i}]crop=${cropW}:${cropH}:${cropX}:0,scale=${outW}:${outH},setpts=PTS-STARTPTS[seg${i}]`;
    }
    overlayInputs.push(`[seg${i}]`);
    // Time-gated: only show this branch within [t0, t1)
    // We use trim+setpts to limit the temporal extent, then overlay sequentially.
    fc += `;${overlayInputs[i]}trim=start=${t0.toFixed(3)}:end=${t1.toFixed(3)},setpts=PTS-STARTPTS[clip${i}]`;
  }

  // Concat all clipN streams in order
  const concatIns = Array.from({ length: n }, (_, i) => `[clip${i}]`).join("");
  fc += `;${concatIns}concat=n=${n}:v=1:a=0[concat]`;

  if (hasSubs) {
    fc += `;[concat]ass=subs.ass[v]`;
    return { filterComplex: fc, outLabel: "[v]" };
  }
  return { filterComplex: fc, outLabel: "[concat]" };
}

function clampCrop(x: number, srcW: number, cropW: number): number {
  return Math.max(0, Math.min(x, srcW - cropW));
}

// ── Silence removal ────────────────────────────────────────────────────
// Detects dead-time (silent pauses ≥ 0.8 s) and removes them so the
// output feels snappier.  Short padding (~120 ms) is kept at cut
// boundaries for natural-sounding transitions.

interface SilenceInterval {
  start: number; // clip-relative seconds
  end: number;
}

interface SpeakingSegment {
  srcStart: number; // clip-relative
  srcEnd: number;
  outStart: number; // output-timeline start
}

/**
 * Run ffmpeg `silencedetect` on a clip and return silence intervals.
 * Times are clip-relative (0 = clip start).
 */
async function detectSilenceInClip(
  ffmpeg: FFmpeg,
  clipStart: number,
  clipDuration: number,
  thresholdDb = -30,
  minDuration = 0.8,
): Promise<SilenceInterval[]> {
  const logLines: string[] = [];
  const handler = ({ message }: { message: string }) => {
    logLines.push(message);
  };
  ffmpeg.on("log", handler);

  const fastSeek = Math.max(0, clipStart - 0.5);
  const innerSeek = clipStart - fastSeek;

  try {
    await ffmpeg.exec([
      "-ss",
      formatTime(fastSeek),
      "-i",
      "source.mp4",
      "-ss",
      formatTime(innerSeek),
      "-t",
      clipDuration.toFixed(3),
      "-af",
      `silencedetect=noise=${thresholdDb}dB:d=${minDuration}`,
      "-vn",
      "-f",
      "null",
      "-",
    ]);
  } catch {
    return [];
  }

  const silences: SilenceInterval[] = [];
  let curStart: number | null = null;
  for (const line of logLines) {
    const sm = line.match(/silence_start:\s*([\d.]+)/);
    if (sm) {
      curStart = Number.parseFloat(sm[1]);
      continue;
    }
    const em = line.match(/silence_end:\s*([\d.]+)/);
    if (em && curStart !== null) {
      silences.push({
        start: curStart,
        end: Number.parseFloat(em[1]),
      });
      curStart = null;
    }
  }
  if (curStart !== null) {
    silences.push({ start: curStart, end: clipDuration });
  }
  return silences;
}

/**
 * Convert silence intervals into speaking segments with output offsets.
 * Keeps `padding` seconds at each cut boundary for natural transitions.
 */
function buildSpeakingSegments(
  silences: SilenceInterval[],
  clipDuration: number,
  padding = 0.12,
): SpeakingSegment[] {
  if (!silences.length)
    return [{ srcStart: 0, srcEnd: clipDuration, outStart: 0 }];

  // Merge silences closer than 0.25 s
  const merged: SilenceInterval[] = [];
  for (const s of silences) {
    if (merged.length && s.start - merged[merged.length - 1].end < 0.25) {
      merged[merged.length - 1].end = s.end;
    } else {
      merged.push({ start: s.start, end: s.end });
    }
  }

  const segments: SpeakingSegment[] = [];
  let out = 0;
  let lastEnd = 0;

  for (const silence of merged) {
    const segEnd = Math.min(silence.start + padding, clipDuration);
    if (segEnd - lastEnd > 0.05) {
      segments.push({ srcStart: lastEnd, srcEnd: segEnd, outStart: out });
      out += segEnd - lastEnd;
    }
    lastEnd = Math.max(silence.end - padding, lastEnd);
  }
  if (clipDuration - lastEnd > 0.05) {
    segments.push({ srcStart: lastEnd, srcEnd: clipDuration, outStart: out });
  }
  return segments;
}

/** Map a source time to its position in the silence-removed output. */
function remapTimeToOutput(t: number, segs: SpeakingSegment[]): number | null {
  for (const s of segs) {
    if (t >= s.srcStart && t <= s.srcEnd) return s.outStart + (t - s.srcStart);
  }
  return null;
}

/** Rewrite ASS subtitle timings to match the silence-removed timeline. */
function adjustAssForSilenceRemoval(
  ass: string,
  segs: SpeakingSegment[],
): string {
  const parseT = (s: string) => {
    const m = s.match(/(\d+):(\d{2}):(\d{2})\.(\d{2})/);
    return m ? +m[1] * 3600 + +m[2] * 60 + +m[3] + +m[4] / 100 : 0;
  };
  const fmtT = (t: number) => {
    const h = Math.floor(t / 3600);
    const m = Math.floor((t % 3600) / 60);
    const s = Math.floor(t % 60);
    const cs = Math.round((t % 1) * 100);
    return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${String(cs).padStart(2, "0")}`;
  };

  return ass
    .split("\n")
    .map(line => {
      if (!line.startsWith("Dialogue:")) return line;
      const parts = line.split(",");
      if (parts.length < 3) return line;
      const st = parseT(parts[1]);
      const et = parseT(parts[2]);
      const mid = (st + et) / 2;
      const nm = remapTimeToOutput(mid, segs);
      if (nm === null) return null; // subtitle falls in silence → drop
      const ns = remapTimeToOutput(st, segs) ?? Math.max(0, nm - (et - st) / 2);
      const ne = remapTimeToOutput(et, segs) ?? ns + (et - st);
      parts[1] = fmtT(ns);
      parts[2] = fmtT(ne);
      return parts.join(",");
    })
    .filter(l => l !== null)
    .join("\n");
}

/** Look up the crop X from a CropPlan at a given clip-relative time. */
function getCropXAtTime(plan: CropPlan, t: number): number {
  const cropW = Math.round((plan.videoHeight * 9) / 16);
  const maxX = plan.videoWidth - cropW;
  for (const seg of plan.segments) {
    if (t < seg.startTime || t > seg.endTime) continue;
    if (seg.mode === "center" || seg.mode === "single")
      return seg.cropX ?? Math.round(maxX / 2);
    if (seg.mode === "dual") {
      const mid = ((seg.face1X ?? 0) + (seg.face2X ?? 0)) / 2;
      return clampCrop(Math.round(mid - cropW / 2), plan.videoWidth, cropW);
    }
    if (seg.mode === "tracking" && seg.keyframes?.length) {
      const kfs = seg.keyframes;
      const rt = t - seg.startTime;
      if (rt <= kfs[0].t) return Math.round(kfs[0].x);
      if (rt >= kfs[kfs.length - 1].t) return Math.round(kfs[kfs.length - 1].x);
      for (let i = 0; i < kfs.length - 1; i++) {
        if (rt >= kfs[i].t && rt <= kfs[i + 1].t) {
          const f = (rt - kfs[i].t) / (kfs[i + 1].t - kfs[i].t);
          return Math.round(kfs[i].x + f * (kfs[i + 1].x - kfs[i].x));
        }
      }
    }
  }
  return Math.round(maxX / 2);
}

/**
 * Build a filter_complex that removes silence AND applies crop + scale
 * in a single pass.  Each speaking segment gets its own trim/crop branch,
 * then all branches are concatenated.
 */
function buildSilenceRemovalFilters(
  segs: SpeakingSegment[],
  plan: CropPlan,
  outW: number,
  outH: number,
  hasSubs: boolean,
): { filterComplex: string; outVLabel: string; outALabel: string } {
  const { videoWidth: srcW, videoHeight: srcH } = plan;
  const cropW = Math.round((srcH * 9) / 16);
  const cropH = srcH;
  const n = segs.length;

  const dualSeg = plan.segments.find(s => s.mode === "dual");

  let fc = "";
  const vs = Array.from({ length: n }, (_, i) => `[v${i}]`).join("");
  const as = Array.from({ length: n }, (_, i) => `[a${i}]`).join("");
  fc += `[0:v]split=${n}${vs};[0:a]asplit=${n}${as}`;

  for (let i = 0; i < n; i++) {
    const s = segs[i];
    const midT = (s.srcStart + s.srcEnd) / 2;

    if (dualSeg) {
      const halfH = Math.floor(outH / 2);
      const dualCropW = Math.min(srcW, Math.round((srcH * 9) / 8));
      const cx1 = clampCrop(
        Math.round((dualSeg.face1X ?? 0) - dualCropW / 2),
        srcW,
        dualCropW,
      );
      const cx2 = clampCrop(
        Math.round((dualSeg.face2X ?? 0) - dualCropW / 2),
        srcW,
        dualCropW,
      );
      fc += `;[v${i}]trim=start=${s.srcStart.toFixed(3)}:end=${s.srcEnd.toFixed(3)},setpts=PTS-STARTPTS,split=2[v${i}t][v${i}b]`;
      fc += `;[v${i}t]crop=${dualCropW}:${cropH}:${cx1}:0,scale=${outW}:${halfH}[v${i}tc]`;
      fc += `;[v${i}b]crop=${dualCropW}:${cropH}:${cx2}:0,scale=${outW}:${halfH}[v${i}bc]`;
      fc += `;[v${i}tc][v${i}bc]vstack=inputs=2[vo${i}]`;
    } else {
      const cx = getCropXAtTime(plan, midT);
      fc += `;[v${i}]trim=start=${s.srcStart.toFixed(3)}:end=${s.srcEnd.toFixed(3)},setpts=PTS-STARTPTS,crop=${cropW}:${cropH}:${cx}:0,scale=${outW}:${outH}[vo${i}]`;
    }
    fc += `;[a${i}]atrim=start=${s.srcStart.toFixed(3)}:end=${s.srcEnd.toFixed(3)},asetpts=PTS-STARTPTS[ao${i}]`;
  }

  const ci = Array.from({ length: n }, (_, i) => `[vo${i}][ao${i}]`).join("");
  fc += `;${ci}concat=n=${n}:v=1:a=1[cv][ca]`;

  if (hasSubs) {
    fc += `;[cv]ass=subs.ass[fv]`;
    return { filterComplex: fc, outVLabel: "[fv]", outALabel: "[ca]" };
  }
  return { filterComplex: fc, outVLabel: "[cv]", outALabel: "[ca]" };
}

// ── Frame-accurate sampling via ffmpeg.wasm ───────────────────────────
/**
 * Extract `fps` frames per second of clip duration as decoded image
 * frames, using ffmpeg's `fps=N` filter (frame-accurate, deterministic).
 *
 * Why this instead of `<video>.currentTime`:
 *   - Firefox: currentTime has ±2ms precision baseline, ±41ms with
 *     fingerprinting protection enabled
 *   - Chrome: known "drift" at start-of-video and on rapid seeks
 *   - Safari: longstanding seek-accuracy bugs (WebKit #52697)
 * The previous pipeline could produce staircase trajectories where two
 * consecutive samples landed on the same source frame, then jumped two
 * frames, which the smoothing pipeline dutifully followed → visible
 * camera weirdness. ffmpeg's `fps` filter samples at exact intervals.
 *
 * Frames are downscaled to half-resolution for face detection: BlazeFace
 * needs roughly 128-px faces in the input, so 960×540 of a 1080p source
 * still contains plenty of detail for the model, and decode is 4× faster.
 *
 * RETURNS a lazy provider — caller asks for frame `i` one at a time, and
 * each canvas is garbage-collectable after the call. This keeps peak
 * memory around one canvas (~2 MB at 960×540) instead of N (480 MB for
 * a 60s clip at 4 fps).
 *
 * The underlying JPEGs are pre-extracted into ffmpeg's MEMFS (small —
 * ~80 kB each at q=5) and deleted after the caller signals done().
 */
async function extractFramesViaFfmpeg(
  ffmpeg: FFmpeg,
  startTime: number,
  duration: number,
  fps: number,
  srcW: number,
  srcH: number,
): Promise<{
  numFrames: number;
  getFrame: (i: number) => Promise<HTMLCanvasElement | null>;
  cleanup: () => Promise<void>;
}> {
  // Downscale only when the source is large (>1280 wide). For 720p/1080p
  // we keep source resolution — BlazeFace works best on faces ≥40 px, and
  // halving 1080p source would put marginal-distance faces below that.
  // For 4K (3840) we downscale to 1280 (3× reduction, faces still ≥50 px
  // for typical compositions) → keeps decode time bounded.
  const TARGET_MAX_W = 1280;
  let scaleW = srcW;
  let scaleH = srcH;
  if (srcW > TARGET_MAX_W) {
    scaleW = TARGET_MAX_W;
    scaleH = Math.round((TARGET_MAX_W / srcW) * srcH);
    // Make sure height is even (some codecs require it; ffmpeg auto-rounds
    // anyway but being explicit is cleaner)
    if (scaleH % 2 === 1) scaleH -= 1;
  }

  // Sample directory in MEMFS
  const dir = "samples";
  try {
    await ffmpeg.createDir(dir);
  } catch {
    /* already exists */
  }
  // Wipe any leftovers from a previous clip
  try {
    const existing = await ffmpeg.listDir(dir);
    for (const node of existing) {
      if (!node.isDir) {
        await ffmpeg.deleteFile(`${dir}/${node.name}`).catch(() => {});
      }
    }
  } catch {
    /* dir didn't exist or was empty */
  }

  // -ss BEFORE -i = fast keyframe seek; we then -t for duration.
  // The `fps` filter produces exactly N frames/sec, regardless of how
  // the input GOP is structured.
  await ffmpeg.exec([
    "-ss",
    startTime.toFixed(3),
    "-i",
    "source.mp4",
    "-t",
    duration.toFixed(3),
    "-vf",
    `fps=${fps},scale=${scaleW}:${scaleH}`,
    "-q:v",
    "5", // JPEG quality (1=best, 31=worst). 5 = high quality, small files.
    "-an",
    "-y",
    `${dir}/f%04d.jpg`,
  ]);

  // List produced files (sorted alphabetically = sorted by frame number)
  const nodes = await ffmpeg.listDir(dir);
  const names = nodes
    .filter(n => !n.isDir && n.name.startsWith("f") && n.name.endsWith(".jpg"))
    .map(n => n.name)
    .sort();

  const getFrame = async (i: number): Promise<HTMLCanvasElement | null> => {
    if (i < 0 || i >= names.length) return null;
    const path = `${dir}/${names[i]}`;
    try {
      const data = await ffmpeg.readFile(path);
      if (!(data instanceof Uint8Array)) return null;
      const blob = new Blob([data.buffer as ArrayBuffer], {
        type: "image/jpeg",
      });
      const bitmap = await createImageBitmap(blob);
      const canvas = document.createElement("canvas");
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      if (!ctx) {
        bitmap.close();
        return null;
      }
      ctx.drawImage(bitmap, 0, 0);
      bitmap.close();
      // Eagerly delete the source file once decoded — keeps MEMFS small
      await ffmpeg.deleteFile(path).catch(() => {});
      return canvas;
    } catch {
      return null;
    }
  };

  const cleanup = async () => {
    try {
      const remaining = await ffmpeg.listDir(dir);
      for (const node of remaining) {
        if (!node.isDir) {
          await ffmpeg.deleteFile(`${dir}/${node.name}`).catch(() => {});
        }
      }
    } catch {
      /* dir gone */
    }
  };

  return { numFrames: names.length, getFrame, cleanup };
}

// ── Process a single clip ──────────────────────────────────────────────
async function processClip(
  ffmpeg: FFmpeg,
  clip: ClipConfig,
  videoWidth: number,
  videoHeight: number,
): Promise<Uint8Array> {
  const duration = clip.endTime - clip.startTime;
  const outputFile = `output_${clip.index}.mp4`;
  const outW = 1080;
  const outH = 1920;
  const hasSubs = !!clip.assSubtitles;
  let assContent = clip.assSubtitles || "";

  // Build a CropPlan from whatever we have
  let plan: CropPlan;
  if (clip.cropPlan && clip.cropPlan.segments.length > 0) {
    plan = clip.cropPlan;
  } else {
    const cropW = Math.round((videoHeight * 9) / 16);
    const cropX =
      clip.cropX !== undefined && clip.cropX >= 0 && cropW < videoWidth
        ? Math.min(clip.cropX, videoWidth - cropW)
        : Math.round((videoWidth - cropW) / 2);
    plan = {
      videoWidth,
      videoHeight,
      segments: [
        {
          startTime: 0,
          endTime: duration,
          mode: clip.cropX !== undefined ? "single" : "center",
          cropX,
        },
      ],
    };
  }

  // ── Silence detection & removal ──────────────────────────────────
  // Detect dead-time pauses ≥ 0.8 s and remove them for a snappier result.
  let speakingSegs: SpeakingSegment[] | null = null;
  try {
    const silences = await detectSilenceInClip(
      ffmpeg,
      clip.startTime,
      duration,
    );
    const totalSilence = silences.reduce((s, si) => s + (si.end - si.start), 0);
    if (totalSilence > 0.8 && silences.length > 0) {
      const segs = buildSpeakingSegments(silences, duration);
      if (segs.length > 1) {
        speakingSegs = segs;
        const outDur = segs.reduce(
          (s, seg) => s + (seg.srcEnd - seg.srcStart),
          0,
        );
        console.log(
          `[silenceRemoval] Clip ${clip.index}: removed ${totalSilence.toFixed(1)}s silence ` +
            `(${duration.toFixed(1)}s → ${outDur.toFixed(1)}s, ${segs.length} segments)`,
        );
        // Adjust subtitle timings to the new timeline
        if (hasSubs && assContent) {
          assContent = adjustAssForSilenceRemoval(assContent, segs);
        }
      }
    }
  } catch (e) {
    console.warn(`[silenceRemoval] Clip ${clip.index} detection failed:`, e);
  }

  if (hasSubs && assContent) {
    const encoder = new TextEncoder();
    await ffmpeg.writeFile("subs.ass", encoder.encode(assContent));
  }

  // -ss BEFORE -i is fast-seek (keyframe-aligned, can be off by 1-2s).
  // For frame-accurate start we want -ss AFTER -i, at the cost of decoding
  // from the previous keyframe. We compromise: fast-seek close (-ss before
  // -i to 0.5s before target), then precise-seek inside (-ss after -i).
  const fastSeek = Math.max(0, clip.startTime - 0.5);
  const innerSeek = clip.startTime - fastSeek;

  const args: string[] = [
    "-ss",
    formatTime(fastSeek),
    "-i",
    "source.mp4",
    "-ss",
    formatTime(innerSeek),
    "-t",
    String(duration.toFixed(3)),
  ];

  if (speakingSegs) {
    // Combined silence-removal + crop + scale filter chain
    const { filterComplex, outVLabel, outALabel } = buildSilenceRemovalFilters(
      speakingSegs,
      plan,
      outW,
      outH,
      hasSubs,
    );
    args.push(
      "-filter_complex",
      filterComplex,
      "-map",
      outVLabel,
      "-map",
      outALabel,
    );
  } else {
    // Original filter chain (no silence to remove)
    const { vf, filterComplex, outLabel } = buildFilters(
      plan,
      outW,
      outH,
      hasSubs,
    );
    if (filterComplex && outLabel) {
      args.push(
        "-filter_complex",
        filterComplex,
        "-map",
        outLabel,
        "-map",
        "0:a?",
      );
    } else if (vf) {
      args.push("-vf", vf);
    }
  }

  args.push(
    "-c:v",
    "libx264",
    "-preset",
    "veryfast",
    "-crf",
    "23",
    "-c:a",
    "aac",
    "-b:a",
    "128k",
    "-movflags",
    "+faststart",
    "-y",
    outputFile,
  );

  await ffmpeg.exec(args);

  const data = await ffmpeg.readFile(outputFile);
  if (!(data instanceof Uint8Array)) {
    throw new Error("Unexpected output type from ffmpeg");
  }

  await ffmpeg.deleteFile(outputFile).catch(() => {});
  if (hasSubs) {
    await ffmpeg.deleteFile("subs.ass").catch(() => {});
  }
  return data;
}

// ── Segmented pipeline (per-clip download) ─────────────────────────────
/**
 * Process clips by downloading only each clip's segment of the video.
 * Uses mp4box.js to parse the MP4 header and produce valid fMP4 segments.
 * This avoids downloading the full video into browser memory.
 */
async function processAllClipsSegmented(
  videoUrl: string,
  clips: ClipConfig[],
  onProgress: ProgressCallback,
  audioUrl: string | null | undefined,
  ffmpegPromise: Promise<FFmpeg>,
  prewarmPromise: Promise<void>,
): Promise<Map<number, Blob>> {
  const results = new Map<number, Blob>();

  // Download full audio if separate (it's small, typically 2–10 MB)
  let audioData: Uint8Array | null = null;
  if (audioUrl) {
    onProgress({
      clipIndex: 0,
      totalClips: clips.length,
      stage: "downloading",
      percent: 0,
      message: "Se descarcă audio-ul...",
    });
    audioData = await downloadVideo(audioUrl, (p) =>
      onProgress({
        clipIndex: 0,
        totalClips: clips.length,
        stage: "downloading",
        percent: Math.round(p * 0.3),
        message: `Se descarcă audio... ${p}%`,
      }),
    );
    console.log(
      `[segmented] Audio: ${(audioData.length / 1e6).toFixed(1)} MB`,
    );
  }

  const ffmpeg = await ffmpegPromise;
  await prewarmPromise;

  if (audioData) {
    await ffmpeg.writeFile("audio.m4a", audioData);
  }

  for (let i = 0; i < clips.length; i++) {
    const clip = clips[i];
    const n = i + 1;

    try {
      // ── 1. Download clip segment ─────────────────────────────────
      onProgress({
        clipIndex: n,
        totalClips: clips.length,
        stage: "downloading",
        percent: 0,
        message: `Se descarcă segmentul ${n}/${clips.length}...`,
      });

      const seg = await downloadClipSegment(
        videoUrl,
        clip.startTime,
        clip.endTime,
        (p) =>
          onProgress({
            clipIndex: n,
            totalClips: clips.length,
            stage: "downloading",
            percent: p,
            message: `Se descarcă segmentul ${n}/${clips.length}... ${p}%`,
          }),
      );

      // ── 2. Write to ffmpeg FS ────────────────────────────────────
      await ffmpeg.writeFile("source.mp4", seg.data);

      // ── 3. Mux with audio if separate ────────────────────────────
      if (audioData) {
        onProgress({
          clipIndex: n,
          totalClips: clips.length,
          stage: "loading",
          percent: 0,
          message: "Se combină video + audio...",
        });
        const audioSeek = Math.max(0, clip.startTime - 15);
        const muxDur = clip.endTime - clip.startTime + 30;
        await ffmpeg.exec([
          "-i",
          "source.mp4",
          "-ss",
          audioSeek.toFixed(3),
          "-i",
          "audio.m4a",
          "-t",
          muxDur.toFixed(3),
          "-map",
          "0:v",
          "-map",
          "1:a",
          "-c",
          "copy",
          "-y",
          "muxed.mp4",
        ]);
        await ffmpeg.deleteFile("source.mp4").catch(() => {});
        const muxData = (await ffmpeg.readFile(
          "muxed.mp4",
        )) as Uint8Array;
        await ffmpeg.writeFile("source.mp4", muxData);
        await ffmpeg.deleteFile("muxed.mp4").catch(() => {});
      }

      // ── 4. Face detection ────────────────────────────────────────
      const vW = seg.videoWidth;
      const vH = seg.videoHeight;

      if (vW > 0 && vH > 0) {
        onProgress({
          clipIndex: n,
          totalClips: clips.length,
          stage: "detecting",
          percent: 0,
          message: `🎯 Face detection ${n}/${clips.length}: ${clip.title}`,
        });
        try {
          const dur = clip.endTime - clip.startTime;
          const fps = dur <= 60 ? 4 : dur <= 120 ? 2 : 1;
          const { numFrames, getFrame, cleanup } =
            await extractFramesViaFfmpeg(
              ffmpeg,
              clip.startTime,
              dur,
              fps,
              vW,
              vH,
            );
          try {
            const plan = await planClipCropFromFrames({
              frameProvider: getFrame,
              numFrames,
              fps,
              clipStartTime: clip.startTime,
              clipEndTime: clip.endTime,
              videoWidth: vW,
              videoHeight: vH,
            });
            clip.cropPlan = plan;
            const desc = plan.segments
              .map(
                (s: CropSegment) =>
                  `${s.mode}@${s.startTime.toFixed(1)}-${s.endTime.toFixed(1)}s`,
              )
              .join(" | ");
            console.log(
              `[segmented] Clip ${n}: ${numFrames} frames @ ${fps}fps → ${desc}`,
            );
          } finally {
            await cleanup();
          }
        } catch (err) {
          console.warn(
            `[segmented] Face detection clip ${n} failed:`,
            err,
          );
        }
      }

      // ── 5. Process clip ──────────────────────────────────────────
      onProgress({
        clipIndex: n,
        totalClips: clips.length,
        stage: "processing",
        percent: Math.round((i / clips.length) * 100),
        message: `Se procesează Short ${n}/${clips.length}: ${clip.title}`,
      });

      const clipDuration = clip.endTime - clip.startTime;
      ffmpeg.on("progress", ({ time }) => {
        const cp = Math.min(
          100,
          Math.round((time / 1_000_000 / clipDuration) * 100),
        );
        onProgress({
          clipIndex: n,
          totalClips: clips.length,
          stage: "encoding",
          percent: Math.round(
            ((i + cp / 100) / clips.length) * 100,
          ),
          message: `Se encodează Short ${n}/${clips.length}... ${cp}%`,
        });
      });

      const outputData = await processClip(ffmpeg, clip, vW, vH);
      results.set(
        clip.index,
        new Blob([outputData.buffer as ArrayBuffer], {
          type: "video/mp4",
        }),
      );

      // ── 6. Clean up segment from ffmpeg FS ───────────────────────
      await ffmpeg.deleteFile("source.mp4").catch(() => {});
    } catch (err) {
      console.error(`[segmented] Error clip ${n}:`, err);
      onProgress({
        clipIndex: n,
        totalClips: clips.length,
        stage: "error",
        percent: 0,
        message: `Eroare la Short ${n}: ${
          err instanceof Error ? err.message : "Eroare necunoscută"
        }`,
      });
    }
  }

  // Final cleanup
  if (audioData) await ffmpeg.deleteFile("audio.m4a").catch(() => {});
  disposeFaceDetector();

  onProgress({
    clipIndex: clips.length,
    totalClips: clips.length,
    stage: "done",
    percent: 100,
    message: `✅ ${results.size}/${clips.length} shorts generate!`,
  });

  return results;
}

// ── Main pipeline ──────────────────────────────────────────────────────
export async function processAllClips(
  videoUrl: string,
  clips: ClipConfig[],
  onProgress: ProgressCallback,
  audioUrl?: string | null,
): Promise<Map<number, Blob>> {
  const results = new Map<number, Blob>();

  // Kick off face detector pre-warm in parallel with everything else
  const prewarm = prewarmFaceDetector();

  onProgress({
    clipIndex: 0,
    totalClips: clips.length,
    stage: "loading",
    percent: 0,
    message: "Se încarcă procesorul video...",
  });

  // ffmpeg load can run in parallel with the prewarm
  const ffmpegPromise = getFFmpeg(onProgress);

  // ── Decide download strategy ─────────────────────────────────────
  // For large files (>50 MB), use per-clip segment download to avoid
  // allocating the full video in browser memory (fixes "Array buffer
  // allocation failed" with 4K / long videos).
  let useSegmented = false;
  try {
    const [rangeOk, totalSize] = await Promise.all([
      supportsRangeRequests(videoUrl),
      getFileSize(videoUrl),
    ]);
    if (rangeOk && totalSize > 50 * 1024 * 1024) {
      console.log(
        `[processAllClips] ${(totalSize / 1e6).toFixed(0)} MB > 50 MB threshold → segment download`,
      );
      useSegmented = true;
    } else if (rangeOk && totalSize > 0) {
      console.log(
        `[processAllClips] ${(totalSize / 1e6).toFixed(0)} MB ≤ 50 MB → full download`,
      );
    }
  } catch (e) {
    console.log("[processAllClips] Could not probe file, using full download:", e);
  }

  if (useSegmented) {
    try {
      return await processAllClipsSegmented(
        videoUrl,
        clips,
        onProgress,
        audioUrl,
        ffmpegPromise,
        prewarm,
      );
    } catch (e) {
      console.warn(
        "[processAllClips] Segment download failed, falling back to full download:",
        e,
      );
      // Fall through to original full-download approach
    }
  }

  // ── Full download (original approach) ────────────────────────────
  onProgress({
    clipIndex: 0,
    totalClips: clips.length,
    stage: "downloading",
    percent: 0,
    message: "Se descarcă video-ul...",
  });

  const videoData = await downloadVideo(videoUrl, percent => {
    onProgress({
      clipIndex: 0,
      totalClips: clips.length,
      stage: "downloading",
      percent: audioUrl ? Math.round(percent * 0.8) : percent,
      message: `Se descarcă video-ul... ${audioUrl ? Math.round(percent * 0.8) : percent}%`,
    });
  });

  let audioData: Uint8Array | null = null;
  if (audioUrl) {
    onProgress({
      clipIndex: 0,
      totalClips: clips.length,
      stage: "downloading",
      percent: 80,
      message: "Se descarcă audio-ul...",
    });
    audioData = await downloadVideo(audioUrl, percent => {
      onProgress({
        clipIndex: 0,
        totalClips: clips.length,
        stage: "downloading",
        percent: 80 + Math.round(percent * 0.2),
        message: `Se descarcă audio-ul... ${percent}%`,
      });
    });
    console.log(
      `[download] Audio: ${(audioData.length / 1024 / 1024).toFixed(1)}MB`,
    );
  }

  // Wait for ffmpeg now (likely already done)
  const ffmpeg = await ffmpegPromise;

  // Mux video+audio if separate streams
  let muxedData: Uint8Array;
  if (audioData) {
    onProgress({
      clipIndex: 0,
      totalClips: clips.length,
      stage: "loading",
      percent: 0,
      message: "Se combină video + audio...",
    });
    await ffmpeg.writeFile("video_only.mp4", videoData);
    await ffmpeg.writeFile("audio_only.m4a", audioData);
    await ffmpeg.exec([
      "-i",
      "video_only.mp4",
      "-i",
      "audio_only.m4a",
      "-c",
      "copy",
      "-movflags",
      "+faststart",
      "-y",
      "muxed.mp4",
    ]);
    muxedData = (await ffmpeg.readFile("muxed.mp4")) as Uint8Array;
    await ffmpeg.deleteFile("video_only.mp4").catch(() => {});
    await ffmpeg.deleteFile("audio_only.m4a").catch(() => {});
    await ffmpeg.deleteFile("muxed.mp4").catch(() => {});
    console.log(
      `[mux] Muxed: ${(muxedData.length / 1024 / 1024).toFixed(1)}MB`,
    );
  } else {
    muxedData = videoData;
  }

  // Blob URL for face detection
  const videoBlob = new Blob([muxedData.buffer as ArrayBuffer], {
    type: "video/mp4",
  });
  const videoBlobUrl = URL.createObjectURL(videoBlob);

  // Read dimensions once (used by all clips)
  let videoWidth = 0;
  let videoHeight = 0;
  try {
    const dims = await getVideoDimensions(videoBlobUrl);
    videoWidth = dims.width;
    videoHeight = dims.height;
    console.log(`[faceDetection] Video: ${videoWidth}x${videoHeight}`);
  } catch (e) {
    console.warn(
      "[faceDetection] Could not read dimensions, will skip face detection:",
      e,
    );
  }

  // Wait for pre-warm to be done so the first clip doesn't pay the load cost
  await prewarm;

  // We need source.mp4 in ffmpeg's FS for both face-detection (frame extract)
  // and the encode pass. Write once, before the detection loop.
  await ffmpeg.writeFile("source.mp4", muxedData);

  // Face detection per clip: extract frames via ffmpeg (frame-accurate)
  // and feed them into planClipCropFromFrames. This replaces the previous
  // <video>.currentTime seek-and-grab approach which has known accuracy
  // issues across browsers (Firefox ±2ms, Chrome drift, Safari bugs).
  if (videoWidth > 0 && videoHeight > 0) {
    onProgress({
      clipIndex: 0,
      totalClips: clips.length,
      stage: "detecting",
      percent: 0,
      message: "🎯 Se detectează fețele pentru crop inteligent...",
    });

    for (let i = 0; i < clips.length; i++) {
      const clip = clips[i];
      onProgress({
        clipIndex: i + 1,
        totalClips: clips.length,
        stage: "detecting",
        percent: Math.round(((i + 1) / clips.length) * 100),
        message: `🎯 Face detection ${i + 1}/${clips.length}: ${clip.title}`,
      });
      try {
        const duration = clip.endTime - clip.startTime;
        // Adaptive sampling: 4 fps for short clips, 2 for medium, 1 for long
        const fps = duration <= 60 ? 4 : duration <= 120 ? 2 : 1;
        const { numFrames, getFrame, cleanup } = await extractFramesViaFfmpeg(
          ffmpeg,
          clip.startTime,
          duration,
          fps,
          videoWidth,
          videoHeight,
        );
        try {
          const plan = await planClipCropFromFrames({
            frameProvider: getFrame,
            numFrames,
            fps,
            clipStartTime: clip.startTime,
            clipEndTime: clip.endTime,
            videoWidth,
            videoHeight,
          });
          clip.cropPlan = plan;
          const desc = plan.segments
            .map(
              (s: CropSegment) =>
                `${s.mode}@${s.startTime.toFixed(1)}-${s.endTime.toFixed(1)}s${
                  s.mode === "single" ? `(x=${s.cropX})` : ""
                }${
                  s.mode === "dual"
                    ? `(f1=${Math.round(s.face1X ?? 0)},f2=${Math.round(s.face2X ?? 0)})`
                    : ""
                }${
                  s.mode === "tracking"
                    ? `(${(s.keyframes ?? []).length}kf)`
                    : ""
                }`,
            )
            .join(" | ");
          console.log(
            `[faceDetection] Clip ${i + 1}: ${numFrames} frames @ ${fps}fps → ${desc}`,
          );
        } finally {
          await cleanup();
        }
      } catch (err) {
        console.warn(
          `[faceDetection] Clip ${i + 1} failed, will use center crop:`,
          err,
        );
      }
    }
  }
  disposeFaceDetector();

  // Process each clip
  for (let i = 0; i < clips.length; i++) {
    const clip = clips[i];
    onProgress({
      clipIndex: i + 1,
      totalClips: clips.length,
      stage: "processing",
      percent: Math.round((i / clips.length) * 100),
      message: `Se procesează Short ${i + 1}/${clips.length}: ${clip.title}`,
    });

    try {
      const clipDuration = clip.endTime - clip.startTime;
      ffmpeg.on("progress", ({ time }) => {
        const clipPercent = Math.min(
          100,
          Math.round((time / 1_000_000 / clipDuration) * 100),
        );
        onProgress({
          clipIndex: i + 1,
          totalClips: clips.length,
          stage: "encoding",
          percent: Math.round(((i + clipPercent / 100) / clips.length) * 100),
          message: `Se encodează Short ${i + 1}/${clips.length}... ${clipPercent}%`,
        });
      });

      const outputData = await processClip(
        ffmpeg,
        clip,
        videoWidth,
        videoHeight,
      );
      const blob = new Blob([outputData.buffer as ArrayBuffer], {
        type: "video/mp4",
      });
      results.set(clip.index, blob);
    } catch (error) {
      console.error(`Error processing clip ${i + 1}:`, error);
      onProgress({
        clipIndex: i + 1,
        totalClips: clips.length,
        stage: "error",
        percent: 0,
        message: `Eroare la Short ${i + 1}: ${
          error instanceof Error ? error.message : "Eroare necunoscută"
        }`,
      });
    }
  }

  await ffmpeg.deleteFile("source.mp4").catch(() => {});
  URL.revokeObjectURL(videoBlobUrl);

  onProgress({
    clipIndex: clips.length,
    totalClips: clips.length,
    stage: "done",
    percent: 100,
    message: `✅ ${results.size}/${clips.length} shorts generate!`,
  });

  return results;
}
