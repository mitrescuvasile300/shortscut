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

  if (hasSubs && clip.assSubtitles) {
    const encoder = new TextEncoder();
    await ffmpeg.writeFile("subs.ass", encoder.encode(clip.assSubtitles));
  }

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

  const { vf, filterComplex, outLabel } = buildFilters(
    plan,
    outW,
    outH,
    hasSubs,
  );

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

  args.push(
    "-c:v",
    "libx264",
    "-preset",
    "veryfast", // was "ultrafast"
    "-crf",
    "23", // was "28"
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
