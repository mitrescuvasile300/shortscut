/**
 * Browser-based video processing with ffmpeg.wasm
 * Handles: download → face detection → crop 9:16 → burn ASS subtitles → encode
 */

import { FFmpeg } from "@ffmpeg/ffmpeg";
import { toBlobURL } from "@ffmpeg/util";
import {
  detectFaceCropPosition,
  disposeFaceDetector,
  getVideoDimensions,
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
  percent: number; // 0-100
  message: string;
}

export type ProgressCallback = (progress: ProcessingProgress) => void;

/**
 * Load ffmpeg.wasm (lazy, singleton)
 */
async function getFFmpeg(_onProgress?: ProgressCallback): Promise<FFmpeg> {
  if (ffmpegInstance && ffmpegLoaded) return ffmpegInstance;

  const ffmpeg = new FFmpeg();

  ffmpeg.on("log", ({ message }) => {
    console.log("[ffmpeg]", message);
  });

  // Load the core from CDN
  const baseURL = "https://unpkg.com/@ffmpeg/core@0.12.10/dist/esm";
  await ffmpeg.load({
    coreURL: await toBlobURL(`${baseURL}/ffmpeg-core.js`, "text/javascript"),
    wasmURL: await toBlobURL(
      `${baseURL}/ffmpeg-core.wasm`,
      "application/wasm"
    ),
  });

  ffmpegInstance = ffmpeg;
  ffmpegLoaded = true;
  return ffmpeg;
}

/**
 * Download video from URL with progress, using chunked Range requests
 * to avoid proxy timeout/size limits (Convex HTTP actions ~20MB limit).
 */
const CHUNK_SIZE = 8 * 1024 * 1024; // 8MB per request (safe under Convex limits)

async function downloadVideo(
  url: string,
  onProgress?: (percent: number) => void
): Promise<Uint8Array> {
  // First, get the total size with a HEAD-like Range request
  const probeResp = await fetch(url, {
    headers: { Range: "bytes=0-0" },
  });

  let totalSize = 0;
  if (probeResp.status === 206) {
    // Parse Content-Range: bytes 0-0/TOTAL
    const cr = probeResp.headers.get("Content-Range");
    if (cr) {
      const match = cr.match(/\/(\d+)/);
      if (match) totalSize = parseInt(match[1], 10);
    }
  }

  // If we couldn't get size via Range, try Content-Length from full request
  if (!totalSize) {
    const fullResp = await fetch(url);
    if (!fullResp.ok) throw new Error(`Download failed: ${fullResp.status}`);
    totalSize = Number(fullResp.headers.get("content-length") || 0);

    // If small enough, just use this response directly
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
      for (const chunk of chunks) { result.set(chunk, offset); offset += chunk.length; }
      return result;
    }

    // For large files without Range support, stream the full response
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
      for (const chunk of chunks) { result.set(chunk, offset); offset += chunk.length; }
      return result;
    }
  }

  console.log(`[download] Total size: ${(totalSize / 1024 / 1024).toFixed(1)}MB, chunk size: ${(CHUNK_SIZE / 1024 / 1024).toFixed(0)}MB`);

  // Download in chunks using Range requests
  const result = new Uint8Array(totalSize);
  let downloaded = 0;

  while (downloaded < totalSize) {
    const start = downloaded;
    const end = Math.min(downloaded + CHUNK_SIZE - 1, totalSize - 1);

    const chunkResp = await fetch(url, {
      headers: { Range: `bytes=${start}-${end}` },
    });

    if (!chunkResp.ok && chunkResp.status !== 206) {
      throw new Error(`Chunk download failed: ${chunkResp.status} (bytes ${start}-${end})`);
    }

    const reader = chunkResp.body?.getReader();
    if (!reader) throw new Error("No chunk body");

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      result.set(value, downloaded);
      downloaded += value.length;
      if (onProgress) {
        onProgress(Math.round((downloaded / totalSize) * 100));
      }
    }
  }

  return result;
}

export interface ClipConfig {
  index: number;
  title: string;
  startTime: number;
  endTime: number;
  assSubtitles?: string; // ASS file content for this clip
  cropX?: number; // X offset for 9:16 crop (from face detection)
}

function formatTime(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  return `${h.toString().padStart(2, "0")}:${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
}

/**
 * Process a single clip: cut, crop 9:16, burn subtitles
 */
async function processClip(
  ffmpeg: FFmpeg,
  clip: ClipConfig,
  videoWidth: number,
  videoHeight: number
): Promise<Uint8Array> {
  const duration = clip.endTime - clip.startTime;
  const outputFile = `output_${clip.index}.mp4`;

  // Write ASS subtitles if provided
  if (clip.assSubtitles) {
    const encoder = new TextEncoder();
    await ffmpeg.writeFile("subs.ass", encoder.encode(clip.assSubtitles));
  }

  // Calculate crop dimensions
  const cropW = Math.round((videoHeight * 9) / 16);
  const cropH = videoHeight;

  // Use face detection crop position or center
  let cropX: number;
  if (
    clip.cropX !== undefined &&
    clip.cropX >= 0 &&
    cropW < videoWidth
  ) {
    cropX = Math.min(clip.cropX, videoWidth - cropW);
  } else {
    cropX = Math.round((videoWidth - cropW) / 2);
  }

  // Build crop filter string with exact position
  const cropFilter = `crop=${cropW}:${cropH}:${cropX}:0,scale=1080:1920`;
  const vf = clip.assSubtitles
    ? `${cropFilter},ass=subs.ass`
    : cropFilter;

  // Build ffmpeg command
  const args = [
    "-ss",
    formatTime(clip.startTime),
    "-i",
    "source.mp4",
    "-t",
    String(Math.ceil(duration)),
    "-vf",
    vf,
    "-c:v",
    "libx264",
    "-preset",
    "ultrafast",
    "-crf",
    "28",
    "-c:a",
    "aac",
    "-b:a",
    "128k",
    "-movflags",
    "+faststart",
    "-y",
    outputFile,
  ];

  await ffmpeg.exec(args);

  // Read output
  const data = await ffmpeg.readFile(outputFile);
  if (!(data instanceof Uint8Array)) {
    throw new Error("Unexpected output type from ffmpeg");
  }

  // Cleanup
  await ffmpeg.deleteFile(outputFile).catch(() => {});
  if (clip.assSubtitles) {
    await ffmpeg.deleteFile("subs.ass").catch(() => {});
  }

  return data;
}

/**
 * Main pipeline: process all clips from a video
 * Now with face detection for intelligent 9:16 cropping
 */
export async function processAllClips(
  videoUrl: string,
  clips: ClipConfig[],
  onProgress: ProgressCallback,
  audioUrl?: string | null
): Promise<Map<number, Blob>> {
  const results = new Map<number, Blob>();

  // Step 1: Load ffmpeg
  onProgress({
    clipIndex: 0,
    totalClips: clips.length,
    stage: "loading",
    percent: 0,
    message: "Se încarcă procesorul video...",
  });

  const ffmpeg = await getFFmpeg(onProgress);

  // Step 2: Download video
  onProgress({
    clipIndex: 0,
    totalClips: clips.length,
    stage: "downloading",
    percent: 0,
    message: "Se descarcă video-ul...",
  });

  const videoData = await downloadVideo(videoUrl, (percent) => {
    onProgress({
      clipIndex: 0,
      totalClips: clips.length,
      stage: "downloading",
      percent: audioUrl ? Math.round(percent * 0.8) : percent, // Leave room for audio download
      message: `Se descarcă video-ul... ${audioUrl ? Math.round(percent * 0.8) : percent}%`,
    });
  });

  // Download audio separately if provided (adaptive streams = video-only + audio-only)
  let audioData: Uint8Array | null = null;
  if (audioUrl) {
    onProgress({
      clipIndex: 0,
      totalClips: clips.length,
      stage: "downloading",
      percent: 80,
      message: "Se descarcă audio-ul...",
    });
    audioData = await downloadVideo(audioUrl, (percent) => {
      onProgress({
        clipIndex: 0,
        totalClips: clips.length,
        stage: "downloading",
        percent: 80 + Math.round(percent * 0.2),
        message: `Se descarcă audio-ul... ${percent}%`,
      });
    });
    console.log(`[download] Audio: ${(audioData.length / 1024 / 1024).toFixed(1)}MB`);
  }

  // If we have separate audio, mux them first with ffmpeg
  let muxedData: Uint8Array;
  if (audioData) {
    onProgress({
      clipIndex: 0,
      totalClips: clips.length,
      stage: "loading",
      percent: 0,
      message: "Se combină video + audio...",
    });

    const ffmpegForMux = await getFFmpeg(onProgress);
    await ffmpegForMux.writeFile("video_only.mp4", videoData);
    await ffmpegForMux.writeFile("audio_only.m4a", audioData);

    await ffmpegForMux.exec([
      "-i", "video_only.mp4",
      "-i", "audio_only.m4a",
      "-c", "copy",
      "-movflags", "+faststart",
      "-y", "muxed.mp4",
    ]);

    muxedData = await ffmpegForMux.readFile("muxed.mp4") as Uint8Array;
    await ffmpegForMux.deleteFile("video_only.mp4").catch(() => {});
    await ffmpegForMux.deleteFile("audio_only.m4a").catch(() => {});
    await ffmpegForMux.deleteFile("muxed.mp4").catch(() => {});
    console.log(`[mux] Muxed: ${(muxedData.length / 1024 / 1024).toFixed(1)}MB`);
  } else {
    muxedData = videoData;
  }

  // Create a blob URL for face detection (needs <video> element)
  const videoBlob = new Blob([muxedData.buffer as ArrayBuffer], { type: "video/mp4" });
  const videoBlobUrl = URL.createObjectURL(videoBlob);

  // Step 3: Face detection for each clip
  onProgress({
    clipIndex: 0,
    totalClips: clips.length,
    stage: "detecting",
    percent: 0,
    message: "🎯 Se detectează fețele pentru crop inteligent...",
  });

  let videoWidth = 0;
  let videoHeight = 0;

  try {
    const dims = await getVideoDimensions(videoBlobUrl);
    videoWidth = dims.width;
    videoHeight = dims.height;
    console.log(`[faceDetection] Video: ${videoWidth}x${videoHeight}`);

    for (let i = 0; i < clips.length; i++) {
      const clip = clips[i];
      onProgress({
        clipIndex: i + 1,
        totalClips: clips.length,
        stage: "detecting",
        percent: Math.round(((i + 1) / clips.length) * 100),
        message: `🎯 Face detection ${i + 1}/${clips.length}: ${clip.title}`,
      });

      const result = await detectFaceCropPosition(
        videoBlobUrl,
        clip.startTime,
        clip.endTime
      );
      clip.cropX = result.cropX >= 0 ? result.cropX : undefined;
      console.log(
        `[faceDetection] Clip ${i + 1}: ${result.method} → cropX=${clip.cropX ?? "center"}`
      );
    }
  } catch (err) {
    console.warn("[faceDetection] Failed, will use center crop:", err);
  } finally {
    disposeFaceDetector();
  }

  // Write video to ffmpeg filesystem (use muxed video+audio if available)
  await ffmpeg.writeFile("source.mp4", muxedData);

  // Step 4: Process each clip with face-aware crop
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
      // Track ffmpeg progress for this clip
      const clipDuration = clip.endTime - clip.startTime;
      ffmpeg.on("progress", ({ time }) => {
        const clipPercent = Math.min(
          100,
          Math.round((time / 1000000 / clipDuration) * 100)
        );
        onProgress({
          clipIndex: i + 1,
          totalClips: clips.length,
          stage: "encoding",
          percent: Math.round(
            ((i + clipPercent / 100) / clips.length) * 100
          ),
          message: `Se encodează Short ${i + 1}/${clips.length}... ${clipPercent}%`,
        });
      });

      const outputData = await processClip(
        ffmpeg,
        clip,
        videoWidth,
        videoHeight
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
        message: `Eroare la Short ${i + 1}: ${error instanceof Error ? error.message : "Eroare necunoscută"}`,
      });
    }
  }

  // Cleanup
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
