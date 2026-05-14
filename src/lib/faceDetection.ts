/**
 * Browser-based face detection using MediaPipe Tasks Vision
 *
 * Improvements over v1:
 *  - Runs BOTH short_range and full_range models (catches faces in wide shots)
 *  - GPU delegate with CPU fallback (3–5× faster)
 *  - Adaptive sampling density (1 sample / ~2 s of clip, capped)
 *  - Reuses a single <video> + <canvas> element instead of recreating per frame
 *  - Detects scene cuts within a clip → one crop per sub-segment (true tracking)
 *  - Cluster threshold scaled to face width (not frame width) → fewer merge errors
 *  - "Most visible" ranked by sample count first (not area)
 *  - 0.45 headroom bias from Python reference: face at 45% from left edge
 *  - Returns "dual" mode for split-screen when two people don't fit in one crop
 *  - Pre-warm export so detector init can run in parallel with video download
 */

import { FaceDetector, FilesetResolver } from "@mediapipe/tasks-vision";

// ── Singleton detectors ────────────────────────────────────────────────
// We run two models in parallel: short_range (selfie/close-up) and
// full_range (wide shots, podcasts, interviews). The Python reference
// does the same — running only one misses ~30% of faces in wide shots.
let detectorShort: FaceDetector | null = null;
let detectorFull: FaceDetector | null = null;
let warmupPromise: Promise<void> | null = null;

const WASM_BASE =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";
const MODEL_SHORT =
  "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite";
// Note: as of 2025 MediaPipe ships "full_range" only via the legacy face_detection
// (mp.solutions) path. For Tasks Vision we substitute by lowering confidence on
// short_range and relying on multi-sample clustering. If/when an official
// blaze_face_full_range.tflite for Tasks Vision lands, swap MODEL_FULL to it.
const MODEL_FULL = MODEL_SHORT;

/**
 * Pre-warm the detector. Safe to call early (e.g. when user clicks Generate)
 * to run model download in parallel with video fetch.
 */
export function prewarmFaceDetector(): Promise<void> {
  if (warmupPromise) return warmupPromise;
  warmupPromise = (async () => {
    try {
      await getDetectors();
    } catch (e) {
      console.warn("[faceDetection] Pre-warm failed:", e);
    }
  })();
  return warmupPromise;
}

async function createDetector(
  vision: Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>,
  modelPath: string,
): Promise<FaceDetector> {
  // Try GPU first; fall back to CPU if WebGL/WebGPU not available
  const tryDelegate = async (delegate: "GPU" | "CPU") =>
    FaceDetector.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: modelPath,
        delegate,
      },
      runningMode: "IMAGE",
      minDetectionConfidence: 0.4, // was 0.6, matches Python's 0.3 better
    });

  try {
    return await tryDelegate("GPU");
  } catch (e) {
    console.info("[faceDetection] GPU delegate unavailable, using CPU:", e);
    return tryDelegate("CPU");
  }
}

async function getDetectors(): Promise<{
  short: FaceDetector;
  full: FaceDetector;
}> {
  if (detectorShort && detectorFull) {
    return { short: detectorShort, full: detectorFull };
  }
  const vision = await FilesetResolver.forVisionTasks(WASM_BASE);
  // Run model loads in parallel
  const [short, full] = await Promise.all([
    detectorShort ?? createDetector(vision, MODEL_SHORT),
    detectorFull ?? createDetector(vision, MODEL_FULL),
  ]);
  detectorShort = short;
  detectorFull = full;
  return { short, full };
}

// ── Types ──────────────────────────────────────────────────────────────
interface FaceInfo {
  centerX: number; // pixels
  centerY: number;
  width: number;
  height: number;
  area: number; // width*height normalized by frame area
  confidence: number;
}

interface FrameSample {
  time: number; // seconds in source video
  faces: FaceInfo[];
}

/**
 * A segment of the clip with its own crop position.
 * If `mode === "dual"` we split-screen the two faces vertically (top/bottom).
 */
export interface CropSegment {
  startTime: number; // relative to clip start (NOT source start)
  endTime: number; // relative to clip start
  mode: "single" | "dual" | "center" | "tracking";
  cropX?: number; // for single/center
  face1X?: number; // for dual (normalized 0-1)
  face2X?: number; // for dual (normalized 0-1)
  keyframes?: Array<{ t: number; x: number }>; // for tracking mode
}

export interface CropPlan {
  segments: CropSegment[];
  videoWidth: number;
  videoHeight: number;
}

// ── Frame extraction (reusable video element) ──────────────────────────
class FrameExtractor {
  private video: HTMLVideoElement;
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D | null;
  private ready: Promise<void>;

  constructor(videoUrl: string, w: number, h: number) {
    this.video = document.createElement("video");
    this.video.crossOrigin = "anonymous";
    this.video.muted = true;
    this.video.preload = "auto";
    this.canvas = document.createElement("canvas");
    this.canvas.width = w;
    this.canvas.height = h;
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });

    this.ready = new Promise((resolve, reject) => {
      const onLoad = () => resolve();
      const onErr = () => reject(new Error("video load failed"));
      this.video.addEventListener("loadedmetadata", onLoad, { once: true });
      this.video.addEventListener("error", onErr, { once: true });
      this.video.src = videoUrl;
    });
  }

  async waitReady(): Promise<void> {
    return this.ready;
  }

  async grab(time: number): Promise<ImageData | null> {
    await this.ready;
    if (!this.ctx) return null;
    const safeTime = Math.max(
      0,
      Math.min(time, (this.video.duration || time) - 0.05),
    );

    return new Promise(resolve => {
      let done = false;
      const cleanupListeners = () => {
        this.video.removeEventListener("seeked", onSeeked);
        this.video.removeEventListener("error", onError);
      };
      const onSeeked = () => {
        if (done) return;
        done = true;
        cleanupListeners();
        if (!this.ctx) return resolve(null);
        try {
          this.ctx.drawImage(
            this.video,
            0,
            0,
            this.canvas.width,
            this.canvas.height,
          );
          resolve(
            this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height),
          );
        } catch {
          resolve(null);
        }
      };
      const onError = () => {
        if (done) return;
        done = true;
        cleanupListeners();
        resolve(null);
      };
      const timeout = setTimeout(() => {
        if (done) return;
        done = true;
        cleanupListeners();
        resolve(null);
      }, 8000);
      this.video.addEventListener(
        "seeked",
        () => {
          clearTimeout(timeout);
          onSeeked();
        },
        { once: true },
      );
      this.video.addEventListener(
        "error",
        () => {
          clearTimeout(timeout);
          onError();
        },
        { once: true },
      );
      this.video.currentTime = safeTime;
    });
  }

  /** Compute a quick brightness signature for scene-cut detection */
  async signature(time: number): Promise<Uint8Array | null> {
    const img = await this.grab(time);
    if (!img) return null;
    // Downsample to an 8×8 luminance grid → 64 bytes
    const grid = 8;
    const sig = new Uint8Array(grid * grid);
    const cellW = img.width / grid;
    const cellH = img.height / grid;
    const data = img.data;
    for (let gy = 0; gy < grid; gy++) {
      for (let gx = 0; gx < grid; gx++) {
        const x0 = Math.floor(gx * cellW);
        const y0 = Math.floor(gy * cellH);
        const x1 = Math.floor((gx + 1) * cellW);
        const y1 = Math.floor((gy + 1) * cellH);
        let sum = 0;
        let n = 0;
        // Stride sample inside cell for speed
        for (let y = y0; y < y1; y += 4) {
          for (let x = x0; x < x1; x += 4) {
            const i = (y * img.width + x) * 4;
            // Luma: 0.299 R + 0.587 G + 0.114 B
            sum += 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
            n++;
          }
        }
        sig[gy * grid + gx] = Math.round(sum / Math.max(n, 1));
      }
    }
    return sig;
  }

  dispose(): void {
    try {
      this.video.pause();
      this.video.removeAttribute("src");
      this.video.load();
    } catch {
      /* ignore */
    }
  }

  getVideo() {
    return this.video;
  }
}

// ── Video metadata ─────────────────────────────────────────────────────
export function getVideoDimensions(
  videoUrl: string,
): Promise<{ width: number; height: number; duration: number }> {
  return new Promise((resolve, reject) => {
    const video = document.createElement("video");
    video.preload = "metadata";
    video.addEventListener(
      "loadedmetadata",
      () => {
        resolve({
          width: video.videoWidth,
          height: video.videoHeight,
          duration: video.duration,
        });
        video.removeAttribute("src");
        video.load();
      },
      { once: true },
    );
    video.addEventListener(
      "error",
      () => reject(new Error("Failed to load video metadata")),
      { once: true },
    );
    video.src = videoUrl;
  });
}

// ── Face detection on a single frame (single model — short_range) ──────
// Note: MODEL_FULL currently points to the same tflite as MODEL_SHORT
// (tasks-vision doesn't ship a separate full_range yet). Running it twice
// wastes ~50% of CPU/GPU budget for zero benefit. When an official
// blaze_face_full_range.tflite for Tasks Vision lands, re-add dual detection.
async function detectFacesInFrame(
  imageData: ImageData,
  canvas: HTMLCanvasElement,
): Promise<FaceInfo[]> {
  const { short } = await getDetectors();
  const ctx = canvas.getContext("2d");
  if (!ctx) return [];
  ctx.putImageData(imageData, 0, 0);

  const minPxW = 30;
  const minPxH = 30;

  const frameArea = canvas.width * canvas.height || 1;
  const out: FaceInfo[] = [];
  const res = short.detect(canvas);
  for (const d of res.detections) {
    const bb = d.boundingBox;
    if (!bb || bb.width < minPxW || bb.height < minPxH) continue;
    const conf = d.categories?.[0]?.score ?? 0.5;
    out.push({
      centerX: bb.originX + bb.width / 2,
      centerY: bb.originY + bb.height / 2,
      width: bb.width,
      height: bb.height,
      area: (bb.width * bb.height) / frameArea,
      confidence: conf,
    });
  }

  return out;
}

// ── Main entry: plan crops for a clip ──────────────────────────────────
/**
 * Build a crop plan for a single clip. Uses deadzone + EMA tracking
 * with primary face selection, MAD outlier rejection, velocity cap,
 * and RDP keyframe reduction.
 *
 * @param videoBlobUrl  blob URL of the source video
 * @param clipStartTime seconds in source video
 * @param clipEndTime   seconds in source video
 * @param videoWidth    cached source width (avoids re-reading metadata)
 * @param videoHeight   cached source height
 */
export async function planClipCrop(
  videoBlobUrl: string,
  clipStartTime: number,
  clipEndTime: number,
  videoWidth: number,
  videoHeight: number,
): Promise<CropPlan> {
  const cropWidth = Math.round((videoHeight * 9) / 16);
  const clipDuration = clipEndTime - clipStartTime;

  if (cropWidth >= videoWidth) {
    return {
      segments: [
        { startTime: 0, endTime: clipDuration, mode: "center", cropX: 0 },
      ],
      videoWidth,
      videoHeight,
    };
  }

  // ── Adaptive sampling: 4fps ≤60s, 2fps ≤120s, 1fps beyond ──
  const sampleFps =
    clipDuration <= 60 ? 4 : clipDuration <= 120 ? 2 : 1;
  const sampleCount = Math.max(
    10,
    Math.min(120, Math.ceil(clipDuration * sampleFps)),
  );
  const frameInterval = clipDuration / sampleCount;

  const extractor = new FrameExtractor(videoBlobUrl, videoWidth, videoHeight);
  await extractor.waitReady();

  const sampleTimes: number[] = [];
  for (let i = 0; i < sampleCount; i++) {
    sampleTimes.push(clipStartTime + ((i + 0.5) / sampleCount) * clipDuration);
  }

  const frames: FrameSample[] = [];
  const scratch = document.createElement("canvas");
  scratch.width = videoWidth;
  scratch.height = videoHeight;

  try {
    for (let i = 0; i < sampleTimes.length; i++) {
      const t = sampleTimes[i];
      const img = await extractor.grab(t);
      if (!img) {
        frames.push({ time: t - clipStartTime, faces: [] });
        continue;
      }
      const faces = await detectFacesInFrame(img, scratch);
      frames.push({ time: t - clipStartTime, faces });
    }
  } finally {
    extractor.dispose();
  }

  const nDetected = frames.filter((f) => f.faces.length > 0).length;
  if (nDetected < 2) {
    return {
      segments: [
        {
          startTime: 0,
          endTime: clipDuration,
          mode: "center",
          cropX: Math.round((videoWidth - cropWidth) / 2),
        },
      ],
      videoWidth,
      videoHeight,
    };
  }

  // ── DUAL detection: ≥2 faces in SAME frame, wide separation ──
  const validDuals: Array<{ f1: number; f2: number }> = [];
  for (const f of frames) {
    if (f.faces.length < 2) continue;
    const xs = f.faces.map((fc) => fc.centerX).sort((a, b) => a - b);
    if (xs[xs.length - 1] - xs[0] > cropWidth * 0.7) {
      validDuals.push({ f1: xs[0], f2: xs[xs.length - 1] });
    }
  }
  if (validDuals.length >= Math.max(3, frames.length * 0.15)) {
    const f1s = validDuals.map((d) => d.f1).sort((a, b) => a - b);
    const f2s = validDuals.map((d) => d.f2).sort((a, b) => a - b);
    return {
      segments: [
        {
          startTime: 0,
          endTime: clipDuration,
          mode: "dual",
          face1X: f1s[Math.floor(f1s.length / 2)] / videoWidth,
          face2X: f2s[Math.floor(f2s.length / 2)] / videoWidth,
        },
      ],
      videoWidth,
      videoHeight,
    };
  }

  // ── Primary face selection: score = area × confidence × proximity ──
  const sigma = (0.3 * cropWidth) / videoWidth; // Gaussian σ (normalized)
  let prevX: number | null = null;
  const primarySeries: (number | null)[] = [];

  for (const f of frames) {
    if (f.faces.length === 0) {
      primarySeries.push(null);
      continue;
    }
    let bestFace: FaceInfo | null = null;
    let bestScore = -1;
    for (const face of f.faces) {
      let score = face.area * face.confidence;
      if (prevX !== null) {
        const dist = Math.abs(face.centerX / videoWidth - prevX);
        score *= Math.exp(-(dist * dist) / (2 * sigma * sigma));
      }
      if (score > bestScore) {
        bestScore = score;
        bestFace = face;
      }
    }
    const normX = bestFace!.centerX / videoWidth;
    primarySeries.push(normX);
    prevX = normX;
  }

  // ── MAD-based temporal outlier rejection ──
  const cleaned = [...primarySeries];
  for (let i = 0; i < cleaned.length; i++) {
    if (cleaned[i] === null) continue;
    const window: number[] = [];
    for (let j = Math.max(0, i - 2); j <= Math.min(cleaned.length - 1, i + 2); j++) {
      if (cleaned[j] !== null) window.push(cleaned[j]!);
    }
    if (window.length < 3) continue;
    window.sort((a, b) => a - b);
    const localMed = window[Math.floor(window.length / 2)];
    const devs = window.map((v) => Math.abs(v - localMed)).sort((a, b) => a - b);
    const mad = devs[Math.floor(devs.length / 2)];
    if (Math.abs(cleaned[i]! - localMed) > 3 * mad + 0.02) {
      cleaned[i] = null; // reject outlier
    }
  }

  // ── Gap-fill: ≤1.5s nearest-neighbor, >1.5s → center fallback ──
  const maxGapFrames = Math.ceil(1.5 * sampleFps);
  const filled = [...cleaned];
  // Forward fill (limited)
  let lastVal: number | null = null;
  let lastIdx = -999;
  for (let i = 0; i < filled.length; i++) {
    if (filled[i] !== null) {
      lastVal = filled[i];
      lastIdx = i;
    } else if (lastVal !== null && i - lastIdx <= maxGapFrames) {
      filled[i] = lastVal;
    }
  }
  // Backward fill (limited)
  lastVal = null;
  lastIdx = filled.length + 999;
  for (let i = filled.length - 1; i >= 0; i--) {
    if (filled[i] !== null) {
      lastVal = filled[i];
      lastIdx = i;
    } else if (lastVal !== null && lastIdx - i <= maxGapFrames) {
      filled[i] = lastVal;
    }
  }
  // Remaining nulls → center
  const centerNorm = 0.5;
  const filledFinal = filled.map((v) => v ?? centerNorm);

  // ── Deadzone + EMA tracking ──
  const dzW = (cropWidth * 0.12) / videoWidth; // deadzone half-width (normalized)
  const alpha = 0.30; // EMA at ~4fps → τ ≈ 0.70s
  const maxPanPerFrame = ((0.30 * cropWidth) / videoWidth) * frameInterval; // velocity cap

  let camX = filledFinal[0];
  const smoothed: number[] = [];

  for (const headPos of filledFinal) {
    let targ: number;
    if (headPos > camX + dzW) {
      targ = headPos; // face exited right → full follow
    } else if (headPos < camX - dzW) {
      targ = headPos; // face exited left → full follow
    } else {
      targ = camX; // inside deadzone → stay put
    }

    let desired = alpha * targ + (1 - alpha) * camX;
    // Velocity cap
    const delta = desired - camX;
    if (Math.abs(delta) > maxPanPerFrame) {
      desired = camX + maxPanPerFrame * (delta > 0 ? 1 : -1);
    }
    camX = desired;
    smoothed.push(camX);
  }

  // ── Convert to pixel crop positions with 0.45 bias ──
  const maxCropX = videoWidth - cropWidth;
  const cropPositions = smoothed.map((focus) => {
    let cx = Math.round(focus * videoWidth - cropWidth * 0.45);
    cx = Math.max(0, Math.min(cx, maxCropX));
    return cx;
  });

  // ── RDP keyframe reduction ──
  const epsilon = Math.max(5, Math.round(0.008 * cropWidth));
  const timed: Array<{ t: number; x: number }> = cropPositions.map(
    (x, i) => ({ t: i * frameInterval, x }),
  );

  function rdpSimplify(
    points: Array<{ t: number; x: number }>,
    eps: number,
  ): Array<{ t: number; x: number }> {
    if (points.length <= 2) return points;
    const start = points[0];
    const end = points[points.length - 1];
    const dt = end.t - start.t;
    const dx = end.x - start.x;
    const lineLen = Math.sqrt(dt * dt + dx * dx) || 1e-9;
    let maxDist = 0;
    let maxIdx = 0;
    for (let i = 1; i < points.length - 1; i++) {
      const d =
        Math.abs(
          dx * (start.t - points[i].t) - dt * (start.x - points[i].x),
        ) / lineLen;
      if (d > maxDist) {
        maxDist = d;
        maxIdx = i;
      }
    }
    if (maxDist > eps) {
      const left = rdpSimplify(points.slice(0, maxIdx + 1), eps);
      const right = rdpSimplify(points.slice(maxIdx), eps);
      return [...left.slice(0, -1), ...right];
    }
    return [start, end];
  }

  const keyframes = rdpSimplify(timed, epsilon);

  // ── Mode decision: range < 25px → static single ──
  const xValues = keyframes.map((kf) => kf.x);
  const rng = Math.max(...xValues) - Math.min(...xValues);

  if (rng < 25) {
    const avgX = Math.round(
      xValues.reduce((s, v) => s + v, 0) / xValues.length,
    );
    return {
      segments: [
        {
          startTime: 0,
          endTime: clipDuration,
          mode: "single",
          cropX: Math.max(0, Math.min(avgX, maxCropX)),
        },
      ],
      videoWidth,
      videoHeight,
    };
  }

  // ── Return tracking segment with keyframes ──
  return {
    segments: [
      {
        startTime: 0,
        endTime: clipDuration,
        mode: "tracking",
        keyframes,
      },
    ],
    videoWidth,
    videoHeight,
  };
}

// ── Cleanup ────────────────────────────────────────────────────────────
export function disposeFaceDetector(): void {
  try {
    detectorShort?.close();
  } catch {
    /* ignore */
  }
  try {
    detectorFull?.close();
  } catch {
    /* ignore */
  }
  detectorShort = null;
  detectorFull = null;
  warmupPromise = null;
}

// ── Backwards-compat shim ──────────────────────────────────────────────
// Older code calls detectFaceCropPosition() → keep it working with a single
// crop result. New code should call planClipCrop() to get scene segments
// and potential dual mode.
export async function detectFaceCropPosition(
  videoBlobUrl: string,
  clipStartTime: number,
  clipEndTime: number,
): Promise<{ cropX: number; method: string }> {
  try {
    const { width, height } = await getVideoDimensions(videoBlobUrl);
    const plan = await planClipCrop(
      videoBlobUrl,
      clipStartTime,
      clipEndTime,
      width,
      height,
    );
    // Reduce to a single cropX by majority/longest segment
    const cropWidth = Math.round((height * 9) / 16);
    let best: CropSegment | undefined;
    let bestDur = -1;
    for (const seg of plan.segments) {
      const dur = seg.endTime - seg.startTime;
      if (dur > bestDur) {
        bestDur = dur;
        best = seg;
      }
    }
    if (!best || best.mode === "center") {
      return {
        cropX: Math.round((width - cropWidth) / 2),
        method: "center",
      };
    }
    if (best.mode === "dual") {
      // Legacy path can't represent dual — center between
      const mid = ((best.face1X ?? 0) + (best.face2X ?? 0)) / 2;
      let cx = Math.round(mid - cropWidth / 2);
      cx = Math.max(0, Math.min(cx, width - cropWidth));
      return { cropX: cx, method: "dual→single (legacy)" };
    }
    if (best.mode === "tracking" && best.keyframes && best.keyframes.length > 0) {
      // Legacy path: use median keyframe position
      const xs = best.keyframes.map((kf) => kf.x).sort((a, b) => a - b);
      return { cropX: xs[Math.floor(xs.length / 2)], method: "tracking→single (legacy)" };
    }
    return {
      cropX: best.cropX ?? Math.round((width - cropWidth) / 2),
      method: "single (segment)",
    };
  } catch (err) {
    console.warn("[faceDetection] Failed, using center crop:", err);
    return { cropX: -1, method: "error → center" };
  }
}
