/**
 * Browser-based face detection + dynamic camera tracking.
 *
 * What changed in this version (and why):
 *   • Replaces "scene-cut segments with one static crop each" — the previous
 *     pipeline's luminance-diff scene detector triggered on stage lighting,
 *     gestures, and source camera motion, producing visible crop jumps.
 *   • Adds proper face tracking: dense sampling, primary-face selection by
 *     (area × confidence × proximity), temporal outlier rejection (MAD),
 *     deadzone + EMA smoothing, velocity cap, RDP keyframe reduction.
 *   • Adds a "tracking" CropSegment mode → videoProcessor emits a
 *     piecewise-linear ffmpeg crop=x='expr' filter.
 *   • Same-frame dual detection kept (split-screen still works).
 *   • Headroom bias restored to 0.45 (face at 45% from left of crop).
 *
 * Numerical design (all values are deliberate, not heuristics):
 *   - Sampling: 4 fps ≤60s clips, 2 fps ≤120s, 1 fps beyond. At 4 fps a
 *     face moving 200 px/s advances 50 px between samples — well above
 *     detector noise (~5 px) and below crop width (~600 px).
 *   - Outlier MAD threshold: |x - local_median| > 3·MAD + 0.02 (norm units).
 *     The +0.02 = ~38 px on 1920 source; stops rejection in tight clusters.
 *   - Gap fill: ≤1.5 s (≤6 frames at 4 fps) linear interpolation; longer
 *     gaps are left as null and produce a "center" fallback for that span —
 *     making up data over multi-second face-loss causes more harm than good.
 *   - Deadzone: dz = 0.12 × cropWidth in pixels (gates WHEN to move).
 *   - EMA α: 0.30 at 4 fps → time constant τ = -dt/ln(1-α) = 0.70 s
 *     (the previous α=0.1 at 2 fps gave τ=4.7 s — visibly laggy).
 *   - Velocity cap: 0.30 × cropWidth per second (≈182 px/s on a 608-px crop;
 *     cinematic pan is ~25%/s in commercial film, 30% accommodates stand-up).
 *   - Headroom: face placed at 0.45 × cropWidth from left edge of crop.
 *   - RDP tolerance: max(5 px, 0.008 × cropWidth). Below visible jitter (~3 px).
 *   - Static threshold: range < 25 px → emit `single` mode (no expr cost).
 */

import { FaceDetector, FilesetResolver } from "@mediapipe/tasks-vision";

// ── Detector lifecycle ─────────────────────────────────────────────────
let detectorPrimary: FaceDetector | null = null;
let detectorPermissive: FaceDetector | null = null;
let warmupPromise: Promise<void> | null = null;

const WASM_BASE =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";
const MODEL_PATH =
  "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite";

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
  minConfidence: number,
): Promise<FaceDetector> {
  const tryDelegate = async (delegate: "GPU" | "CPU") =>
    FaceDetector.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_PATH, delegate },
      runningMode: "IMAGE",
      minDetectionConfidence: minConfidence,
    });
  try {
    return await tryDelegate("GPU");
  } catch (e) {
    console.info("[faceDetection] GPU delegate unavailable, using CPU:", e);
    return tryDelegate("CPU");
  }
}

async function getDetectors(): Promise<{
  primary: FaceDetector;
  permissive: FaceDetector;
}> {
  if (detectorPrimary && detectorPermissive) {
    return { primary: detectorPrimary, permissive: detectorPermissive };
  }
  const vision = await FilesetResolver.forVisionTasks(WASM_BASE);
  const [p, q] = await Promise.all([
    detectorPrimary ?? createDetector(vision, 0.5),
    detectorPermissive ?? createDetector(vision, 0.25),
  ]);
  detectorPrimary = p;
  detectorPermissive = q;
  return { primary: p, permissive: q };
}

export function disposeFaceDetector(): void {
  try {
    detectorPrimary?.close();
  } catch {
    /* ignore */
  }
  try {
    detectorPermissive?.close();
  } catch {
    /* ignore */
  }
  detectorPrimary = null;
  detectorPermissive = null;
  warmupPromise = null;
}

// ── Types ──────────────────────────────────────────────────────────────
interface FaceInfo {
  centerX: number; // pixels
  centerY: number;
  width: number;
  height: number;
  score: number; // confidence 0..1
}

export interface TrackingKeyframe {
  t: number; // seconds relative to SEGMENT start
  x: number; // crop X in source pixels (top-left of crop window)
}

/**
 * One time span of the clip with its own crop behavior.
 *
 *   center   — no faces / fallback: static centered crop
 *   single   — one face span: static crop at `cropX`
 *   dual     — two faces visible same-frame, split-screen
 *   tracking — face moves enough to need dynamic crop: piecewise-linear
 *              keyframes; videoProcessor renders as crop=x='expr'
 */
export interface CropSegment {
  startTime: number; // relative to clip start
  endTime: number;
  mode: "single" | "dual" | "center" | "tracking";
  cropX?: number; // single/center
  face1X?: number; // dual (pixels)
  face2X?: number; // dual (pixels)
  keyframes?: TrackingKeyframe[]; // tracking
}

export interface CropPlan {
  segments: CropSegment[];
  videoWidth: number;
  videoHeight: number;
}

// ── Frame extraction ───────────────────────────────────────────────────
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
      this.video.addEventListener("loadedmetadata", () => resolve(), {
        once: true,
      });
      this.video.addEventListener(
        "error",
        () => reject(new Error("video load failed")),
        { once: true },
      );
      this.video.src = videoUrl;
    });
  }

  waitReady(): Promise<void> {
    return this.ready;
  }

  async grab(time: number): Promise<HTMLCanvasElement | null> {
    await this.ready;
    if (!this.ctx) return null;
    const safeTime = Math.max(
      0,
      Math.min(time, (this.video.duration || time) - 0.05),
    );
    return new Promise(resolve => {
      let done = false;
      const cleanup = () => {
        this.video.removeEventListener("seeked", onSeek);
        this.video.removeEventListener("error", onError);
      };
      const onSeek = () => {
        if (done) return;
        done = true;
        cleanup();
        if (!this.ctx) return resolve(null);
        try {
          this.ctx.drawImage(
            this.video,
            0,
            0,
            this.canvas.width,
            this.canvas.height,
          );
          resolve(this.canvas);
        } catch {
          resolve(null);
        }
      };
      const onError = () => {
        if (done) return;
        done = true;
        cleanup();
        resolve(null);
      };
      const timeout = setTimeout(() => {
        if (done) return;
        done = true;
        cleanup();
        resolve(null);
      }, 8000);
      this.video.addEventListener(
        "seeked",
        () => {
          clearTimeout(timeout);
          onSeek();
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

  dispose(): void {
    try {
      this.video.pause();
      this.video.removeAttribute("src");
      this.video.load();
    } catch {
      /* ignore */
    }
  }
}

// ── Per-frame face detection ───────────────────────────────────────────
async function detectFacesInCanvas(
  canvas: HTMLCanvasElement,
): Promise<FaceInfo[]> {
  const { primary, permissive } = await getDetectors();
  const MIN_PX = 30;

  const collect = (det: FaceDetector): FaceInfo[] => {
    const out: FaceInfo[] = [];
    const res = det.detect(canvas);
    const cw = canvas.width;
    const ch = canvas.height;
    for (const d of res.detections) {
      const bb = d.boundingBox;
      if (!bb) continue;
      if (bb.width < MIN_PX || bb.height < MIN_PX) continue;
      const score = d.categories?.[0]?.score ?? 0.5;
      // BlazeFace keypoint order: 0=left eye, 1=right eye, 2=nose tip,
      // 3=mouth, 4=left ear tragion, 5=right ear tragion.
      // The nose is the most stable visual anchor across head turns
      // (bbox center can shift toward the ear in profile views).
      // Keypoints are in NORMALIZED 0..1 coords — multiply by canvas dims.
      const nose = d.keypoints?.[2];
      const centerX = nose ? nose.x * cw : bb.originX + bb.width / 2;
      const centerY = nose ? nose.y * ch : bb.originY + bb.height / 2;
      out.push({
        centerX,
        centerY,
        width: bb.width,
        height: bb.height,
        score,
      });
    }
    return out;
  };

  const a = collect(primary);
  if (a.length >= 2) return a;
  const b = collect(permissive);
  const merged = [...a];
  for (const f of b) {
    const dup = merged.some(g => iou(f, g) > 0.4);
    if (!dup) merged.push(f);
  }
  return merged;
}

function iou(a: FaceInfo, b: FaceInfo): number {
  const ax0 = a.centerX - a.width / 2;
  const ay0 = a.centerY - a.height / 2;
  const ax1 = ax0 + a.width;
  const ay1 = ay0 + a.height;
  const bx0 = b.centerX - b.width / 2;
  const by0 = b.centerY - b.height / 2;
  const bx1 = bx0 + b.width;
  const by1 = by0 + b.height;
  const ix0 = Math.max(ax0, bx0);
  const iy0 = Math.max(ay0, by0);
  const ix1 = Math.min(ax1, bx1);
  const iy1 = Math.min(ay1, by1);
  if (ix1 <= ix0 || iy1 <= iy0) return 0;
  const inter = (ix1 - ix0) * (iy1 - iy0);
  const union = a.width * a.height + b.width * b.height - inter;
  return inter / union;
}

// ── Math helpers ───────────────────────────────────────────────────────
function median(arr: number[]): number {
  if (arr.length === 0) return 0;
  const s = [...arr].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(v, hi));
}

// ── Public entry point ─────────────────────────────────────────────────
export function getVideoDimensions(
  videoUrl: string,
): Promise<{ width: number; height: number; duration: number }> {
  return new Promise((resolve, reject) => {
    const video = document.createElement("video");
    video.preload = "metadata";
    video.muted = true;
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

/**
 * Plan a crop for one clip. Returns one or more CropSegments; each segment
 * covers a contiguous time range and has its own crop behavior.
 *
 * Two ways to call:
 *   a) Pass `videoBlobUrl` (legacy) → uses <video>.currentTime to seek
 *      for each sample. Cross-browser seek accuracy is poor (±41ms in
 *      Firefox, drift in Chrome, broken in Safari), causing staircase
 *      trajectories that the camera then dutifully follows.
 *   b) Pass `frameProvider` + `numFrames` (preferred) → a function that
 *      returns the i-th sample frame lazily. We decode one frame at a
 *      time so we never hold more than a single canvas in memory.
 *
 * Either way, `fps` is the sampling rate (caller-chosen). Without a
 * provider, planClipCrop picks the rate adaptively and seeks accordingly.
 */
export interface PlanClipCropOptions {
  /** Legacy path: blob URL of source video; uses <video>.currentTime */
  videoBlobUrl?: string;
  /** Preferred path: lazy frame provider (one canvas per call, in order) */
  frameProvider?: (i: number) => Promise<HTMLCanvasElement | null>;
  /** Number of frames the provider can supply (required with frameProvider) */
  numFrames?: number;
  /** Sampling rate of the provider's frames (required with frameProvider) */
  fps?: number;
  clipStartTime: number;
  clipEndTime: number;
  videoWidth: number;
  videoHeight: number;
}

export async function planClipCropFromFrames(
  opts: PlanClipCropOptions,
): Promise<CropPlan> {
  const { clipStartTime, clipEndTime, videoWidth, videoHeight } = opts;
  const cropWidth = Math.round((videoHeight * 9) / 16);
  const duration = clipEndTime - clipStartTime;
  const maxCropX = videoWidth - cropWidth;

  // Trivial early return
  if (cropWidth >= videoWidth) {
    return {
      segments: [{ startTime: 0, endTime: duration, mode: "center", cropX: 0 }],
      videoWidth,
      videoHeight,
    };
  }

  // Resolve sampling rate
  let fps: number;
  let nSamples: number;
  let useExtractor = false;
  if (opts.frameProvider && opts.fps && opts.numFrames !== undefined) {
    fps = opts.fps;
    nSamples = opts.numFrames;
  } else if (opts.videoBlobUrl) {
    fps = duration <= 60 ? 4 : duration <= 120 ? 2 : 1;
    nSamples = Math.max(4, Math.floor(duration * fps));
    useExtractor = true;
  } else {
    throw new Error(
      "planClipCrop: need frameProvider+numFrames or videoBlobUrl",
    );
  }
  const dt = 1 / fps;

  // ── Pass 1: per-frame detection + primary-face selection ───────────
  // primary score = area × confidence × proximity_to_prev_primary
  // proximity = exp(-dist² / (2·sigma²)). All distances in NORMALIZED
  // 0..1 fractions of the canvas — works regardless of whether canvases
  // are at source resolution or downscaled.
  //   sigma = 0.30 × cropWidth / videoWidth  (canvas-width-relative)
  const sigmaNorm = (0.3 * cropWidth) / videoWidth;
  const twoSigma2Norm = 2 * sigmaNorm * sigmaNorm;
  const allFacesPerFrame: FaceInfo[][] = new Array(nSamples);
  const rawPrimaryX: (number | null)[] = new Array(nSamples).fill(null);
  /** Primary face area as fraction of canvas area (for cut detection) */
  const primaryArea: (number | null)[] = new Array(nSamples).fill(null);
  let prevPrimaryNorm: { x: number; y: number } | null = null;

  let extractor: FrameExtractor | null = null;
  if (useExtractor && opts.videoBlobUrl) {
    extractor = new FrameExtractor(opts.videoBlobUrl, videoWidth, videoHeight);
    await extractor.waitReady();
  }

  try {
    for (let i = 0; i < nSamples; i++) {
      let canvas: HTMLCanvasElement | null = null;
      if (opts.frameProvider) {
        canvas = await opts.frameProvider(i);
      } else if (extractor) {
        const t = clipStartTime + (i + 0.5) * dt;
        canvas = await extractor.grab(t);
      }
      if (!canvas) {
        allFacesPerFrame[i] = [];
        continue;
      }
      // Canvas may be at scaled-down resolution (e.g. half-res for speed).
      // Normalize face positions by the canvas's own pixel width so that
      // the value is a 0..1 fraction of the visible frame — that fraction
      // applies equally at source resolution.
      const canvasW = canvas.width || videoWidth;
      const canvasH = canvas.height || videoHeight;
      const canvasArea = canvasW * canvasH;
      const faces = await detectFacesInCanvas(canvas);
      allFacesPerFrame[i] = faces;
      if (faces.length === 0) continue;
      let best = faces[0];
      let bestScore = -1;
      for (const f of faces) {
        const fNormX = f.centerX / canvasW;
        const fNormY = f.centerY / canvasH;
        const fNormArea = (f.width * f.height) / canvasArea;
        let proximity = 1.0;
        if (prevPrimaryNorm) {
          const dx = fNormX - prevPrimaryNorm.x;
          const dy = fNormY - prevPrimaryNorm.y;
          proximity = Math.exp(-(dx * dx + dy * dy) / twoSigma2Norm);
        }
        const s = fNormArea * f.score * proximity;
        if (s > bestScore) {
          bestScore = s;
          best = f;
        }
      }
      prevPrimaryNorm = {
        x: best.centerX / canvasW,
        y: best.centerY / canvasH,
      };
      rawPrimaryX[i] = best.centerX / canvasW;
      // Store NORMALIZED area for scale-invariant cut detection
      primaryArea[i] = (best.width * best.height) / canvasArea;
    }
  } finally {
    extractor?.dispose();
  }

  const nDetected = rawPrimaryX.filter(v => v !== null).length;
  if (nDetected < Math.max(2, Math.ceil(nSamples * 0.15))) {
    console.log(
      `[faceDetection] Only ${nDetected}/${nSamples} detections → center`,
    );
    return {
      segments: [
        {
          startTime: 0,
          endTime: duration,
          mode: "center",
          cropX: Math.round(maxCropX / 2),
        },
      ],
      videoWidth,
      videoHeight,
    };
  }

  // ── Pass 2: dual-face decision ─────────────────────────────────────
  const dual = decideDual(allFacesPerFrame, videoWidth, cropWidth);
  if (dual) {
    console.log(
      `[faceDetection] Dual: f1=${Math.round(dual.face1X)}, f2=${Math.round(dual.face2X)}`,
    );
    return {
      segments: [
        {
          startTime: 0,
          endTime: duration,
          mode: "dual",
          face1X: dual.face1X,
          face2X: dual.face2X,
        },
      ],
      videoWidth,
      videoHeight,
    };
  }

  // ── Pass 3: temporal outlier rejection ─────────────────────────────
  const cleaned = rejectOutliers(rawPrimaryX);

  // ── Pass 3.5: DEFAULT-STATIC-FIRST CHECK ───────────────────────────
  // User spec: "default to static centered crop; only move the camera
  // when the face goes more than half off-screen."
  //
  //   1. Compute the optimal static crop position: median of all valid
  //      face X positions, biased by 0.45 × cropWidth (upper-third).
  //   2. Count what fraction of valid samples have the face center
  //      OUTSIDE the resulting crop window [cropX, cropX+cropWidth].
  //      Face center outside window ⟺ more than half of the face is
  //      off-screen.
  //   3. If ≤ 15% of samples exit → emit a SINGLE static crop. The few
  //      exit frames will clip the face partially — that's the user's
  //      accepted tradeoff.
  //   4. Else → fall through to tracking.
  const validNorms: number[] = [];
  for (const v of cleaned) if (v !== null) validNorms.push(v);
  if (validNorms.length >= 2) {
    const medianFaceNorm = median(validNorms);
    const medianFacePx = medianFaceNorm * videoWidth;
    let staticCropX = Math.round(medianFacePx - cropWidth * 0.45);
    staticCropX = clamp(staticCropX, 0, maxCropX);
    let exits = 0;
    for (const v of validNorms) {
      const facePx = v * videoWidth;
      if (facePx < staticCropX || facePx > staticCropX + cropWidth) exits++;
    }
    const exitFraction = exits / validNorms.length;
    const STATIC_THRESHOLD = 0.15;
    if (exitFraction <= STATIC_THRESHOLD) {
      console.log(
        `[faceDetection] Static single (x=${staticCropX}, exit=${(exitFraction * 100).toFixed(0)}% ≤ 15%)`,
      );
      return {
        segments: [
          {
            startTime: 0,
            endTime: duration,
            mode: "single",
            cropX: staticCropX,
          },
        ],
        videoWidth,
        videoHeight,
      };
    }
    console.log(
      `[faceDetection] Engaging tracking (${(exitFraction * 100).toFixed(0)}% of frames have face outside static crop)`,
    );
  }

  // ── Pass 4: gap fill (short gaps only, linear interpolation) ───────
  // HARD-RESET MARKERS: gaps longer than 0.75s mark a "discontinuity" —
  // the camera state should reset rather than EMA-into-a-stale-position
  // when the face re-acquires.
  const maxFillFrames = Math.ceil(1.5 * fps);
  const hardResetThresholdFrames = Math.ceil(0.75 * fps);
  const filled = fillShortGaps(cleaned, maxFillFrames);
  const hardResetAt = findGapStarts(cleaned, hardResetThresholdFrames);

  // ── Pass 5: deadzone + EMA + velocity cap + CUT DETECTION ──────────
  //   - Cut detection: if face position jumps >0.35 normalized between
  //     consecutive valid samples AND the face area changes >50%,
  //     it's a real source-camera cut → snap camera instantly to
  //     the new position (skip velocity cap for that frame).
  //   - Hard reset: at gap boundaries longer than 0.75s, reset camX.
  //   - Re-anchoring: every 3 seconds, if camera has been static
  //     (within deadzone) for ≥3 s and a valid face exists, snap
  //     camX = head to eliminate accumulated drift.
  const dzNorm = (0.12 * cropWidth) / videoWidth;
  const alpha = 0.3;
  const maxPanNormPerStep = ((0.3 * cropWidth) / videoWidth) * dt;
  const cutPositionDelta = 0.25; // normalized — face shifted >25% of frame
  const cutAreaRatio = 1.5; // either grew >1.5x or shrunk to <1/1.5
  const reanchorEverySec = 3.0;
  const reanchorEveryFrames = Math.round(reanchorEverySec * fps);

  let camX: number | null = null;
  /** Index where camera last actively moved (left deadzone) */
  let lastMovedIdx = -1;
  /** Last valid sample index processed */
  let lastValidIdx = -1;
  const cam: (number | null)[] = new Array(nSamples).fill(null);

  for (let i = 0; i < nSamples; i++) {
    const head = filled[i];

    // Hard reset on long gap: when a long null gap starts, reset camX so
    // the next valid detection re-anchors freshly
    if (hardResetAt.has(i)) {
      camX = null;
      lastMovedIdx = -1;
    }

    if (head === null) {
      cam[i] = null;
      continue;
    }

    // CUT DETECTION (between this and the previous valid sample)
    let isCut = false;
    if (lastValidIdx >= 0 && camX !== null) {
      const prevHead = filled[lastValidIdx];
      const prevArea = primaryArea[lastValidIdx];
      const curArea = primaryArea[i];
      if (prevHead !== null && Math.abs(head - prevHead) > cutPositionDelta) {
        // Position changed dramatically. Check if face identity also changed
        // (proxy: face area changed substantially).
        if (
          prevArea !== null &&
          curArea !== null &&
          (curArea / prevArea > cutAreaRatio ||
            prevArea / curArea > cutAreaRatio)
        ) {
          isCut = true;
        } else if (prevArea === null || curArea === null) {
          // No area info on one side → treat as cut to be safe
          isCut = true;
        }
      }
    }

    if (camX === null || isCut) {
      // Cold start OR detected cut → snap immediately
      camX = head;
      cam[i] = camX;
      lastValidIdx = i;
      lastMovedIdx = i;
      continue;
    }

    // Capture into a definitely-number local for TS narrowing through reassign
    const cx: number = camX;
    let targ: number;
    let movedThisStep = false;
    if (head > cx + dzNorm || head < cx - dzNorm) {
      targ = head; // face exited deadzone — follow it
      movedThisStep = true;
    } else {
      targ = cx;
    }
    let nextX: number = alpha * targ + (1 - alpha) * cx;
    const dv = nextX - cx;
    if (dv > maxPanNormPerStep) nextX = cx + maxPanNormPerStep;
    else if (dv < -maxPanNormPerStep) nextX = cx - maxPanNormPerStep;

    // RE-ANCHORING: if camera has been static for ≥3s and the face is
    // within the deadzone, snap to face. This eliminates drift that
    // accumulates from EMA half-steps inside the deadzone.
    if (
      !movedThisStep &&
      lastMovedIdx >= 0 &&
      i - lastMovedIdx >= reanchorEveryFrames
    ) {
      nextX = head;
      lastMovedIdx = i; // reset the timer
    } else if (movedThisStep) {
      lastMovedIdx = i;
    }

    camX = nextX;
    cam[i] = camX;
    lastValidIdx = i;
  }

  // ── Pass 6: convert to pixel crop X with 0.45 headroom bias ────────
  const cropPositions: (number | null)[] = cam.map(v =>
    v === null ? null : clamp(v * videoWidth - cropWidth * 0.45, 0, maxCropX),
  );

  // ── Pass 7: build CropSegments ─────────────────────────────────────
  const segments: CropSegment[] = [];
  let runStart = 0;
  let runIsValid = cropPositions[0] !== null;

  const flushRun = (endIdx: number) => {
    const t0 = runStart === 0 ? 0 : runStart * dt;
    const t1 =
      endIdx === nSamples - 1
        ? duration
        : Math.min(duration, (endIdx + 1) * dt);

    if (!runIsValid) {
      segments.push({
        startTime: t0,
        endTime: t1,
        mode: "center",
        cropX: Math.round(maxCropX / 2),
      });
      return;
    }

    const xs: number[] = [];
    for (let i = runStart; i <= endIdx; i++) {
      const v = cropPositions[i];
      if (v !== null) xs.push(v);
    }
    if (xs.length === 0) {
      segments.push({
        startTime: t0,
        endTime: t1,
        mode: "center",
        cropX: Math.round(maxCropX / 2),
      });
      return;
    }
    const range = Math.max(...xs) - Math.min(...xs);

    // Static → single mode
    if (range < 25) {
      const avg = xs.reduce((s, v) => s + v, 0) / xs.length;
      segments.push({
        startTime: t0,
        endTime: t1,
        mode: "single",
        cropX: clamp(Math.round(avg), 0, maxCropX),
      });
      return;
    }

    // Tracking → build segment-relative keyframes, simplify with RDP
    const kfs: TrackingKeyframe[] = [];
    for (let i = runStart; i <= endIdx; i++) {
      const v = cropPositions[i];
      if (v !== null) {
        const sampleAbsT = (i + 0.5) * dt;
        const segRelT = Math.max(0, sampleAbsT - t0);
        kfs.push({ t: segRelT, x: v });
      }
    }
    const segLen = t1 - t0;
    if (kfs.length > 0) {
      if (kfs[0].t > 0) kfs.unshift({ t: 0, x: kfs[0].x });
      if (kfs[kfs.length - 1].t < segLen) {
        kfs.push({ t: segLen, x: kfs[kfs.length - 1].x });
      }
    }
    const eps = Math.max(5, cropWidth * 0.008);
    const reduced = rdp(kfs, eps);

    segments.push({
      startTime: t0,
      endTime: t1,
      mode: "tracking",
      keyframes: reduced,
    });
  };

  for (let i = 1; i < nSamples; i++) {
    const valid = cropPositions[i] !== null;
    if (valid !== runIsValid) {
      flushRun(i - 1);
      runStart = i;
      runIsValid = valid;
    }
  }
  flushRun(nSamples - 1);

  const summary = segments
    .map(s => {
      if (s.mode === "tracking") {
        const kfs = s.keyframes ?? [];
        const r =
          kfs.length > 0
            ? Math.max(...kfs.map(k => k.x)) - Math.min(...kfs.map(k => k.x))
            : 0;
        return `${s.mode}[${s.startTime.toFixed(1)}-${s.endTime.toFixed(1)}s, ${kfs.length}kf, range=${Math.round(r)}px]`;
      }
      return `${s.mode}[${s.startTime.toFixed(1)}-${s.endTime.toFixed(1)}s${
        s.cropX !== undefined ? `, x=${s.cropX}` : ""
      }]`;
    })
    .join(" ");
  console.log(`[faceDetection] plan: ${summary}`);

  return { segments, videoWidth, videoHeight };
}

/**
 * Backwards-compatible wrapper for the legacy signature.
 * Equivalent to planClipCropFromFrames({ videoBlobUrl, ... }).
 */
export async function planClipCrop(
  videoBlobUrl: string,
  clipStartTime: number,
  clipEndTime: number,
  videoWidth: number,
  videoHeight: number,
): Promise<CropPlan> {
  return planClipCropFromFrames({
    videoBlobUrl,
    clipStartTime,
    clipEndTime,
    videoWidth,
    videoHeight,
  });
}

// ── Helpers ────────────────────────────────────────────────────────────

/**
 * Return the set of sample indices where a null gap of >= `minGapLen`
 * frames begins. Used to mark hard-reset boundaries in the smoothing pass.
 */
function findGapStarts(
  track: (number | null)[],
  minGapLen: number,
): Set<number> {
  const out = new Set<number>();
  let i = 0;
  while (i < track.length) {
    if (track[i] !== null) {
      i++;
      continue;
    }
    let j = i;
    while (j < track.length && track[j] === null) j++;
    if (j - i >= minGapLen) {
      // Mark the index AFTER the gap (where camera should reset on re-acquire)
      if (j < track.length) out.add(j);
    }
    i = j;
  }
  return out;
}

function rejectOutliers(track: (number | null)[]): (number | null)[] {
  const w = 2;
  const out = track.slice();
  for (let i = 0; i < track.length; i++) {
    const cur = track[i];
    if (cur === null) continue;
    const lo = Math.max(0, i - w);
    const hi = Math.min(track.length, i + w + 1);
    const ctx: number[] = [];
    for (let j = lo; j < hi; j++) {
      const v = track[j];
      if (v !== null && j !== i) ctx.push(v);
    }
    if (ctx.length < 2) continue;
    const med = median(ctx);
    const mad = median(ctx.map(v => Math.abs(v - med)));
    if (Math.abs(cur - med) > 3 * mad + 0.02) {
      out[i] = null;
    }
  }
  return out;
}

function fillShortGaps(
  track: (number | null)[],
  maxGapFrames: number,
): (number | null)[] {
  const out = track.slice();
  let i = 0;
  while (i < out.length) {
    if (out[i] !== null) {
      i++;
      continue;
    }
    let j = i;
    while (j < out.length && out[j] === null) j++;
    const gapLen = j - i;
    const prev = i > 0 ? out[i - 1] : null;
    const next = j < out.length ? out[j] : null;
    if (gapLen <= maxGapFrames && prev !== null && next !== null) {
      for (let k = i; k < j; k++) {
        const f = (k - i + 1) / (j - i + 1);
        out[k] = prev + (next - prev) * f;
      }
    } else if (gapLen <= maxGapFrames && prev !== null) {
      for (let k = i; k < j; k++) out[k] = prev;
    } else if (gapLen <= maxGapFrames && next !== null) {
      for (let k = i; k < j; k++) out[k] = next;
    }
    i = j;
  }
  return out;
}

/**
 * Decide if the clip is a real two-person wide shot.
 *
 * Constraints (ALL must hold simultaneously, in ≥85% of detection frames):
 *   - Two faces visible in the SAME frame
 *   - Separated horizontally by ≥ 0.8 × cropWidth (else they'd fit in one crop)
 *   - At SIMILAR Y position (within 25% of frame height). Kills the common
 *     comedian-on-stage + audience-in-front-rows false positive.
 *   - SIMILAR SIZE (max/min area ratio ≤ 2.0). Same camera distance.
 *
 * Returns face1X/face2X in SOURCE pixels (what videoProcessor renders).
 */
function decideDual(
  framesFaces: FaceInfo[][],
  videoWidth: number,
  cropW: number,
): { face1X: number; face2X: number } | null {
  const minSepNorm = (cropW * 0.8) / videoWidth;
  const Y_TOLERANCE = 0.25; // normalized canvas-height fraction
  const SIZE_RATIO_MAX = 2.0;
  const CONSISTENCY_THRESHOLD = 0.85;

  // Estimate canvas dimensions from the max face bbox extents across the clip
  // (face boxes approach but don't exceed canvas borders). All FaceInfo
  // coordinates use the same canvas, so a single estimate suffices.
  let estCanvasW = 0;
  let estCanvasH = 0;
  for (const faces of framesFaces) {
    for (const f of faces) {
      const r = f.centerX + f.width / 2;
      const b = f.centerY + f.height / 2;
      if (r > estCanvasW) estCanvasW = r;
      if (b > estCanvasH) estCanvasH = b;
    }
  }
  // Add 5% safety margin — faces don't reach the very edge of frame
  estCanvasW = Math.max(estCanvasW * 1.05, 1);
  estCanvasH = Math.max(estCanvasH * 1.05, 1);

  let framesWithAny = 0;
  let framesWithDual = 0;
  const validLeftsNorm: number[] = [];
  const validRightsNorm: number[] = [];

  for (const faces of framesFaces) {
    if (faces.length === 0) continue;
    framesWithAny++;
    if (faces.length < 2) continue;
    // Filter tiny detections (canvas-relative width < 3%)
    const big = faces.filter(
      f => f.width / estCanvasW > 0.03 && f.height / estCanvasH > 0.03,
    );
    if (big.length < 2) continue;
    const sorted = [...big].sort((a, b) => a.centerX - b.centerX);
    const left = sorted[0];
    const right = sorted[sorted.length - 1];
    // Horizontal separation constraint
    const sepNorm = (right.centerX - left.centerX) / estCanvasW;
    if (sepNorm < minSepNorm) continue;
    // Y position constraint (kills comedian+audience)
    const dyNorm = Math.abs(left.centerY - right.centerY) / estCanvasH;
    if (dyNorm > Y_TOLERANCE) continue;
    // Size constraint
    const aL = left.width * left.height;
    const aR = right.width * right.height;
    if (aL <= 0 || aR <= 0) continue;
    const sizeRatio = Math.max(aL, aR) / Math.min(aL, aR);
    if (sizeRatio > SIZE_RATIO_MAX) continue;
    framesWithDual++;
    validLeftsNorm.push(left.centerX / estCanvasW);
    validRightsNorm.push(right.centerX / estCanvasW);
  }

  if (framesWithAny === 0) return null;
  if (framesWithDual / framesWithAny < CONSISTENCY_THRESHOLD) return null;
  if (validLeftsNorm.length < 3) return null;

  return {
    face1X: median(validLeftsNorm) * videoWidth,
    face2X: median(validRightsNorm) * videoWidth,
  };
}

function rdp(points: TrackingKeyframe[], epsilon: number): TrackingKeyframe[] {
  if (points.length <= 2) return points.slice();
  const keep = new Uint8Array(points.length);
  keep[0] = 1;
  keep[points.length - 1] = 1;
  const stack: [number, number][] = [[0, points.length - 1]];
  while (stack.length > 0) {
    const top = stack.pop();
    if (!top) break;
    const [a, b] = top;
    if (b - a <= 1) continue;
    const pa = points[a];
    const pb = points[b];
    const dt2d = pb.t - pa.t;
    const dx2d = pb.x - pa.x;
    const len2 = dt2d * dt2d + dx2d * dx2d;
    let maxD = 0;
    let maxI = a;
    for (let i = a + 1; i < b; i++) {
      const pi = points[i];
      let d: number;
      if (len2 === 0) {
        const ddt = pi.t - pa.t;
        const ddx = pi.x - pa.x;
        d = Math.sqrt(ddt * ddt + ddx * ddx);
      } else {
        const cross = Math.abs(dx2d * (pi.t - pa.t) - dt2d * (pi.x - pa.x));
        d = cross / Math.sqrt(len2);
      }
      if (d > maxD) {
        maxD = d;
        maxI = i;
      }
    }
    if (maxD > epsilon) {
      keep[maxI] = 1;
      stack.push([a, maxI]);
      stack.push([maxI, b]);
    }
  }
  const out: TrackingKeyframe[] = [];
  for (let i = 0; i < keep.length; i++) if (keep[i]) out.push(points[i]);
  return out;
}

// ── Backwards-compat shim ──────────────────────────────────────────────
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
    const cropW = Math.round((height * 9) / 16);
    let best = plan.segments[0];
    let bestDur = 0;
    for (const s of plan.segments) {
      const d = s.endTime - s.startTime;
      if (d > bestDur) {
        bestDur = d;
        best = s;
      }
    }
    if (best.mode === "single" || best.mode === "center") {
      return { cropX: best.cropX ?? -1, method: best.mode };
    }
    if (best.mode === "dual") {
      const mid = ((best.face1X ?? 0) + (best.face2X ?? 0)) / 2;
      return {
        cropX: clamp(Math.round(mid - cropW / 2), 0, width - cropW),
        method: "dual→single (legacy)",
      };
    }
    const kfs = best.keyframes ?? [];
    if (kfs.length === 0) {
      return {
        cropX: Math.round((width - cropW) / 2),
        method: "tracking-empty",
      };
    }
    const avg = kfs.reduce((s, k) => s + k.x, 0) / kfs.length;
    return {
      cropX: clamp(Math.round(avg), 0, width - cropW),
      method: "tracking→single (legacy)",
    };
  } catch (err) {
    console.warn("[faceDetection] Failed, using center crop:", err);
    return { cropX: -1, method: "error → center" };
  }
}
