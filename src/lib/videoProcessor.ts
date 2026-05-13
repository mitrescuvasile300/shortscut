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
  planClipCrop,
  prewarmFaceDetector,
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
 * Build a -filter_complex string for a clip with a multi-segment plan.
 *
 * Strategy:
 *   - For each segment, build a separate filter chain (crop+scale, or
 *     split/crop/vstack for dual)
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

  // Fast path: one segment, single mode (or center) → plain -vf
  if (segments.length === 1 && segments[0].mode !== "dual") {
    const seg = segments[0];
    const cropX =
      seg.cropX !== undefined
        ? Math.max(0, Math.min(seg.cropX, srcW - cropW))
        : Math.round((srcW - cropW) / 2);
    const base = `crop=${cropW}:${cropH}:${cropX}:0,scale=${outW}:${outH}`;
    const vf = hasSubs ? `${base},ass=subs.ass` : base;
    return { vf };
  }

  // Fast path: one segment, dual mode → filter_complex with split+vstack
  if (segments.length === 1 && segments[0].mode === "dual") {
    const seg = segments[0];
    const halfH = Math.floor(outH / 2);
    const cx1 = clampCrop(
      Math.round((seg.face1X ?? 0) - cropW / 2),
      srcW,
      cropW,
    );
    const cx2 = clampCrop(
      Math.round((seg.face2X ?? 0) - cropW / 2),
      srcW,
      cropW,
    );
    let fc =
      `[0:v]split=2[top][bot];` +
      `[top]crop=${cropW}:${cropH}:${cx1}:0,scale=${outW}:${halfH}[t];` +
      `[bot]crop=${cropW}:${cropH}:${cx2}:0,scale=${outW}:${halfH}[b];` +
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
      const cx1 = clampCrop(
        Math.round((seg.face1X ?? 0) - cropW / 2),
        srcW,
        cropW,
      );
      const cx2 = clampCrop(
        Math.round((seg.face2X ?? 0) - cropW / 2),
        srcW,
        cropW,
      );
      // Each split output needs its own split→crop→vstack
      fc +=
        `;[s${i}]split=2[s${i}a][s${i}b]` +
        `;[s${i}a]crop=${cropW}:${cropH}:${cx1}:0,scale=${outW}:${halfH}[s${i}ac]` +
        `;[s${i}b]crop=${cropW}:${cropH}:${cx2}:0,scale=${outW}:${halfH}[s${i}bc]` +
        `;[s${i}ac][s${i}bc]vstack=inputs=2,setpts=PTS-STARTPTS[seg${i}]`;
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

  // Face detection per clip (now produces a CropPlan)
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
        const plan = await planClipCrop(
          videoBlobUrl,
          clip.startTime,
          clip.endTime,
          videoWidth,
          videoHeight,
        );
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
              }`,
          )
          .join(" | ");
        console.log(`[faceDetection] Clip ${i + 1} plan: ${desc}`);
      } catch (err) {
        console.warn(
          `[faceDetection] Clip ${i + 1} failed, will use center crop:`,
          err,
        );
      }
    }
  }
  disposeFaceDetector();

  // Write source for processing
  await ffmpeg.writeFile("source.mp4", muxedData);

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
