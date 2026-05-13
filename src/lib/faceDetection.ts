/**
 * Browser-based face detection using MediaPipe Tasks Vision
 * Matches the logic from the local Python script:
 * - Samples multiple frames per clip
 * - Detects ALL faces per frame
 * - Clusters faces by X position
 * - Multi-person logic: fit both or focus on most visible
 */

import { FaceDetector, FilesetResolver } from "@mediapipe/tasks-vision";

let faceDetector: FaceDetector | null = null;

/**
 * Initialize the face detector (lazy, singleton)
 */
async function getDetector(): Promise<FaceDetector> {
  if (faceDetector) return faceDetector;

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm"
  );

  faceDetector = await FaceDetector.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
      delegate: "CPU",
    },
    runningMode: "IMAGE",
    minDetectionConfidence: 0.6,
  });

  return faceDetector;
}

interface FaceInfo {
  centerX: number; // center X in pixels
  centerY: number;
  width: number;   // face bounding box width
  height: number;
}

interface CropResult {
  cropX: number;    // X offset for crop
  method: string;   // description of what happened
}

/**
 * Extract a frame from a video blob at a given time
 */
function extractFrame(
  videoUrl: string,
  timeSeconds: number,
  videoWidth: number,
  videoHeight: number
): Promise<ImageData | null> {
  return new Promise((resolve) => {
    const video = document.createElement("video");
    video.crossOrigin = "anonymous";
    video.muted = true;
    video.preload = "auto";

    const canvas = document.createElement("canvas");
    canvas.width = videoWidth;
    canvas.height = videoHeight;
    const ctx = canvas.getContext("2d");

    let resolved = false;
    const cleanup = () => {
      if (!resolved) {
        resolved = true;
        video.pause();
        video.removeAttribute("src");
        video.load();
      }
    };

    const timeout = setTimeout(() => {
      cleanup();
      resolve(null);
    }, 10000);

    video.addEventListener("seeked", () => {
      if (resolved) return;
      if (!ctx) { cleanup(); resolve(null); return; }
      ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
      const imageData = ctx.getImageData(0, 0, videoWidth, videoHeight);
      clearTimeout(timeout);
      cleanup();
      resolve(imageData);
    }, { once: true });

    video.addEventListener("error", () => {
      clearTimeout(timeout);
      cleanup();
      resolve(null);
    }, { once: true });

    video.addEventListener("loadedmetadata", () => {
      video.currentTime = Math.min(timeSeconds, video.duration - 0.1);
    }, { once: true });

    video.src = videoUrl;
  });
}

/**
 * Get video dimensions from a blob URL
 */
export function getVideoDimensions(
  videoUrl: string
): Promise<{ width: number; height: number; duration: number }> {
  return new Promise((resolve, reject) => {
    const video = document.createElement("video");
    video.preload = "metadata";

    video.addEventListener("loadedmetadata", () => {
      resolve({
        width: video.videoWidth,
        height: video.videoHeight,
        duration: video.duration,
      });
      video.removeAttribute("src");
      video.load();
    }, { once: true });

    video.addEventListener("error", () => {
      reject(new Error("Failed to load video metadata"));
    }, { once: true });

    video.src = videoUrl;
  });
}

/**
 * Detect faces in a single frame using MediaPipe
 */
async function detectFacesInFrame(
  imageData: ImageData,
  frameWidth: number,
  frameHeight: number
): Promise<FaceInfo[]> {
  const detector = await getDetector();

  // Create a canvas with the image data for MediaPipe
  const canvas = document.createElement("canvas");
  canvas.width = frameWidth;
  canvas.height = frameHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) return [];
  ctx.putImageData(imageData, 0, 0);

  const result = detector.detect(canvas);
  const faces: FaceInfo[] = [];

  for (const detection of result.detections) {
    const bb = detection.boundingBox;
    if (!bb) continue;

    const faceWidth = bb.width;
    const faceHeight = bb.height;

    // Filter out tiny faces (< 0.5% of frame) — likely false positives
    const faceArea = faceWidth * faceHeight;
    const frameArea = frameWidth * frameHeight;
    if (faceArea / frameArea < 0.005) continue;

    faces.push({
      centerX: bb.originX + faceWidth / 2,
      centerY: bb.originY + faceHeight / 2,
      width: faceWidth,
      height: faceHeight,
    });
  }

  return faces;
}

/**
 * Cluster faces by X position (faces within 15% of frame width = same person)
 */
function clusterFaces(
  allFaces: FaceInfo[],
  frameWidth: number
): Array<{ centerX: number; totalArea: number; count: number }> {
  const threshold = frameWidth * 0.15;
  const clusters: Array<{
    sumX: number;
    totalArea: number;
    count: number;
    faces: FaceInfo[];
  }> = [];

  for (const face of allFaces) {
    let merged = false;
    for (const cluster of clusters) {
      const clusterCenterX = cluster.sumX / cluster.count;
      if (Math.abs(face.centerX - clusterCenterX) < threshold) {
        cluster.sumX += face.centerX;
        cluster.totalArea += face.width * face.height;
        cluster.count += 1;
        cluster.faces.push(face);
        merged = true;
        break;
      }
    }
    if (!merged) {
      clusters.push({
        sumX: face.centerX,
        totalArea: face.width * face.height,
        count: 1,
        faces: [face],
      });
    }
  }

  return clusters.map((c) => ({
    centerX: c.sumX / c.count,
    totalArea: c.totalArea,
    count: c.count,
  }));
}

/**
 * Calculate optimal crop X position for 9:16 from face detection results
 * Matches the local script's v2 logic:
 * - Multi-person: try to fit both, else focus on most visible
 * - Single person: center on face
 * - No face: center crop
 */
export async function detectFaceCropPosition(
  videoBlobUrl: string,
  clipStartTime: number,
  clipEndTime: number
): Promise<CropResult> {
  try {
    const { width: videoWidth, height: videoHeight } =
      await getVideoDimensions(videoBlobUrl);

    // 9:16 crop dimensions
    const cropWidth = Math.round((videoHeight * 9) / 16);

    // If video is already narrower than or equal to 9:16, no face detection needed
    if (cropWidth >= videoWidth) {
      return { cropX: 0, method: "video already narrow enough" };
    }

    const clipDuration = clipEndTime - clipStartTime;

    // Sample 7 frames at 10%, 25%, 40%, 50%, 60%, 75%, 90% of clip duration
    const samplePoints = [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9];
    const allFaces: FaceInfo[] = [];

    for (const pct of samplePoints) {
      const sampleTime = clipStartTime + clipDuration * pct;
      const frameData = await extractFrame(
        videoBlobUrl,
        sampleTime,
        videoWidth,
        videoHeight
      );
      if (!frameData) continue;

      const faces = await detectFacesInFrame(frameData, videoWidth, videoHeight);
      allFaces.push(...faces);
    }

    if (allFaces.length === 0) {
      // No faces detected — center crop (fallback)
      const centerX = Math.round((videoWidth - cropWidth) / 2);
      return { cropX: centerX, method: "no face → center" };
    }

    // Cluster faces by X position
    const clusters = clusterFaces(allFaces, videoWidth);

    if (clusters.length === 0) {
      const centerX = Math.round((videoWidth - cropWidth) / 2);
      return { cropX: centerX, method: "no clusters → center" };
    }

    let targetCenterX: number;
    let method: string;

    if (clusters.length === 1) {
      // Single person — center on them
      targetCenterX = clusters[0].centerX;
      method = `1 person → crop X centered on face`;
    } else {
      // Multiple persons — sort by visibility (total area)
      clusters.sort((a, b) => b.totalArea - a.totalArea);
      const person1 = clusters[0];
      const person2 = clusters[1];

      // Check if both fit in 9:16 crop
      const leftMost = Math.min(person1.centerX, person2.centerX);
      const rightMost = Math.max(person1.centerX, person2.centerX);
      const span = rightMost - leftMost;

      if (span < cropWidth * 0.8) {
        // Both fit — center between them
        targetCenterX = (person1.centerX + person2.centerX) / 2;
        method = `2 persons (both fit) → centered between`;
      } else {
        // Don't fit — focus on the most visible person
        targetCenterX = person1.centerX;
        method = `2 persons (focus on most visible)`;
      }
    }

    // Clamp cropX to valid range
    let cropX = Math.round(targetCenterX - cropWidth / 2);
    cropX = Math.max(0, Math.min(cropX, videoWidth - cropWidth));

    return { cropX, method };
  } catch (error) {
    console.warn("Face detection failed, using center crop:", error);
    return { cropX: -1, method: "error → center" };
  }
}

/**
 * Cleanup face detector resources
 */
export function disposeFaceDetector(): void {
  if (faceDetector) {
    faceDetector.close();
    faceDetector = null;
  }
}
