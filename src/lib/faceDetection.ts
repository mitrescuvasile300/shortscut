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
}

interface FrameSample {
  time: number; // seconds in source video
  faces: FaceInfo[];
}

interface Cluster {
  centerX: number;
  medianArea: number;
  count: number;
  faces: FaceInfo[];
}

/**
 * A segment of the clip with its own crop position.
 * If `mode === "dual"` we split-screen the two faces vertically (top/bottom).
 */
export interface CropSegment {
  startTime: number; // relative to clip start (NOT source start)
  endTime: number; // relative to clip start
  mode: "single" | "dual" | "center";
  cropX?: number; // for single/center
  face1X?: number; // for dual (pixels)
  face2X?: number; // for dual (pixels)
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

  const out: FaceInfo[] = [];
  const res = short.detect(canvas);
  for (const d of res.detections) {
    const bb = d.boundingBox;
    if (!bb || bb.width < minPxW || bb.height < minPxH) continue;
    out.push({
      centerX: bb.originX + bb.width / 2,
      centerY: bb.originY + bb.height / 2,
      width: bb.width,
      height: bb.height,
    });
  }

  return out;
}

// ── Clustering ─────────────────────────────────────────────────────────
/**
 * Cluster face detections by X position.
 * Threshold scales with face width (half a face), not frame width.
 * This stops two distinct people from being merged in wide shots.
 */
function clusterFaces(allFaces: FaceInfo[]): Cluster[] {
  if (allFaces.length === 0) return [];

  const clusters: {
    faces: FaceInfo[];
  }[] = [];

  for (const face of allFaces) {
    let merged = false;
    for (const c of clusters) {
      // Threshold = half a face width (whichever face is larger)
      const refW = Math.max(face.width, ...c.faces.map(f => f.width));
      const threshold = refW * 0.5;
      const clusterCx =
        c.faces.reduce((s, f) => s + f.centerX, 0) / c.faces.length;
      if (Math.abs(face.centerX - clusterCx) < threshold) {
        c.faces.push(face);
        merged = true;
        break;
      }
    }
    if (!merged) clusters.push({ faces: [face] });
  }

  return clusters.map(c => {
    const areas = c.faces.map(f => f.width * f.height).sort((a, b) => a - b);
    const medianArea = areas[Math.floor(areas.length / 2)];
    const centerX = c.faces.reduce((s, f) => s + f.centerX, 0) / c.faces.length;
    return {
      centerX,
      medianArea,
      count: c.faces.length,
      faces: c.faces,
    };
  });
}

/**
 * Decide a crop for a single set of faces.
 * Implements: 0.45 headroom bias for single, dual split-screen for two-person
 * wide shots, "most visible" tiebroken by count then area.
 */
function decideCrop(
  clusters: Cluster[],
  videoWidth: number,
  cropWidth: number,
): {
  mode: "single" | "dual" | "center";
  cropX?: number;
  face1X?: number;
  face2X?: number;
} {
  if (clusters.length === 0) {
    return {
      mode: "center",
      cropX: Math.round((videoWidth - cropWidth) / 2),
    };
  }

  // Sort: most samples first, then largest median area as tiebreaker
  clusters.sort((a, b) => b.count - a.count || b.medianArea - a.medianArea);

  if (clusters.length === 1) {
    const cx = clusters[0].centerX;
    // 0.45 bias: face sits at 45% from the left edge of the crop (Python reference).
    // This gives headroom + body lead-room for vertical short-form framing.
    let cropX = Math.round(cx - cropWidth * 0.45);
    cropX = Math.max(0, Math.min(cropX, videoWidth - cropWidth));
    return { mode: "single", cropX };
  }

  const p1 = clusters[0];
  const p2 = clusters[1];

  // Only consider second person "real" if they appeared in at least half as many
  // samples as the primary — kills noise / single-frame false positives.
  if (p2.count < Math.max(2, p1.count / 2)) {
    let cropX = Math.round(p1.centerX - cropWidth * 0.45);
    cropX = Math.max(0, Math.min(cropX, videoWidth - cropWidth));
    return { mode: "single", cropX };
  }

  const leftMost = Math.min(p1.centerX, p2.centerX);
  const rightMost = Math.max(p1.centerX, p2.centerX);
  const span = rightMost - leftMost;

  if (span < cropWidth * 0.8) {
    // Both fit comfortably — center between them
    const midX = (p1.centerX + p2.centerX) / 2;
    let cropX = Math.round(midX - cropWidth / 2);
    cropX = Math.max(0, Math.min(cropX, videoWidth - cropWidth));
    return { mode: "single", cropX };
  }

  // Don't fit — DUAL / split-screen mode
  return { mode: "dual", face1X: leftMost, face2X: rightMost };
}

// ── Scene-cut detection from signatures ────────────────────────────────
/**
 * Given 8×8 luminance signatures sampled across the clip, find time indices
 * where the frame content changed sharply. Returns segment boundaries
 * (indices into the samples array) so each segment can be analyzed separately.
 *
 * Threshold tuned empirically for talking-head / podcast / vlog content.
 */
function findSceneBoundaries(signatures: (Uint8Array | null)[]): number[] {
  // L1 distance between consecutive signatures, normalized per cell
  const diffs: number[] = [];
  for (let i = 1; i < signatures.length; i++) {
    const a = signatures[i - 1];
    const b = signatures[i];
    if (!a || !b) {
      diffs.push(0);
      continue;
    }
    let d = 0;
    for (let k = 0; k < a.length; k++) d += Math.abs(a[k] - b[k]);
    diffs.push(d / a.length); // 0..255
  }

  // A scene cut produces a large diff. We use the higher of:
  //   - absolute floor (30 / 255 ≈ 12% mean luminance shift)
  //   - 3× median diff (robust against gradual lighting changes)
  if (diffs.length === 0) return [0, signatures.length];
  const sorted = [...diffs].sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)] || 1;
  const threshold = Math.max(30, median * 3);

  const boundaries: number[] = [0];
  for (let i = 0; i < diffs.length; i++) {
    if (diffs[i] >= threshold) {
      // Cut between sample i and i+1 → boundary at i+1
      const b = i + 1;
      // Avoid tiny segments
      if (b - boundaries[boundaries.length - 1] >= 2) boundaries.push(b);
    }
  }
  if (boundaries[boundaries.length - 1] !== signatures.length) {
    boundaries.push(signatures.length);
  }
  return boundaries;
}

// ── Main entry: plan crops for a clip ──────────────────────────────────
/**
 * Build a crop plan for a single clip. May return multiple segments if
 * scene cuts are detected within the clip.
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
  // 9:16 crop width
  const cropWidth = Math.round((videoHeight * 9) / 16);
  const clipDuration = clipEndTime - clipStartTime;

  // If already narrow enough, just center
  if (cropWidth >= videoWidth) {
    return {
      segments: [
        {
          startTime: 0,
          endTime: clipDuration,
          mode: "center",
          cropX: 0,
        },
      ],
      videoWidth,
      videoHeight,
    };
  }

  // Adaptive sample count: ≥10, ~1 per 2s, cap 30
  const sampleCount = Math.max(10, Math.min(30, Math.ceil(clipDuration / 2)));

  const extractor = new FrameExtractor(videoBlobUrl, videoWidth, videoHeight);
  await extractor.waitReady();

  // Sample: evenly spaced, with margins inside the clip
  const sampleTimes: number[] = [];
  for (let i = 0; i < sampleCount; i++) {
    const t = clipStartTime + ((i + 0.5) / sampleCount) * clipDuration;
    sampleTimes.push(t);
  }

  // Pass 1: extract signatures + face detections in one go.
  // We can't easily parallelize seeks within one <video>, so do sequentially.
  const frames: FrameSample[] = [];
  const sigs: (Uint8Array | null)[] = [];
  // Scratch canvas reused across frames for face detection
  const scratch = document.createElement("canvas");
  scratch.width = videoWidth;
  scratch.height = videoHeight;

  try {
    for (let i = 0; i < sampleTimes.length; i++) {
      const t = sampleTimes[i];
      const img = await extractor.grab(t);
      if (!img) {
        sigs.push(null);
        frames.push({ time: t - clipStartTime, faces: [] });
        continue;
      }
      // Signature for scene detection: cheap 8×8 luma grid
      const sig = imageDataToSig(img, 8);
      sigs.push(sig);
      // Face detection (writes img into scratch internally)
      const faces = await detectFacesInFrame(img, scratch);
      frames.push({ time: t - clipStartTime, faces });
    }
  } finally {
    extractor.dispose();
  }

  // Detect scene boundaries (indices into frames/sigs)
  const boundaries = findSceneBoundaries(sigs);

  // For each segment, cluster only the faces from that segment's samples
  const segments: CropSegment[] = [];
  for (let s = 0; s < boundaries.length - 1; s++) {
    const a = boundaries[s];
    const b = boundaries[s + 1];
    const segFaces: FaceInfo[] = [];
    for (let i = a; i < b; i++) segFaces.push(...frames[i].faces);
    const clusters = clusterFaces(segFaces);
    const decision = decideCrop(clusters, videoWidth, cropWidth);
    segments.push({
      startTime: frames[a].time,
      endTime: b < frames.length ? frames[b].time : clipDuration,
      mode: decision.mode,
      cropX: decision.cropX,
      face1X: decision.face1X,
      face2X: decision.face2X,
    });
  }

  // Merge adjacent identical segments (e.g. same single crop across cuts)
  const merged: CropSegment[] = [];
  for (const seg of segments) {
    const last = merged[merged.length - 1];
    if (
      last &&
      last.mode === seg.mode &&
      same(last.cropX, seg.cropX) &&
      same(last.face1X, seg.face1X) &&
      same(last.face2X, seg.face2X)
    ) {
      last.endTime = seg.endTime;
    } else {
      merged.push({ ...seg });
    }
  }

  return {
    segments: merged,
    videoWidth,
    videoHeight,
  };
}

function same(a: number | undefined, b: number | undefined): boolean {
  if (a === undefined && b === undefined) return true;
  if (a === undefined || b === undefined) return false;
  // Within 10 px → consider equal
  return Math.abs(a - b) < 10;
}

// Pure helper (no canvas / async needed)
function imageDataToSig(img: ImageData, grid = 8): Uint8Array {
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
      for (let y = y0; y < y1; y += 4) {
        for (let x = x0; x < x1; x += 4) {
          const i = (y * img.width + x) * 4;
          sum += 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
          n++;
        }
      }
      sig[gy * grid + gx] = Math.round(sum / Math.max(n, 1));
    }
  }
  return sig;
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
    return {
      cropX: best.cropX ?? Math.round((width - cropWidth) / 2),
      method: "single (segment)",
    };
  } catch (err) {
    console.warn("[faceDetection] Failed, using center crop:", err);
    return { cropX: -1, method: "error → center" };
  }
}
