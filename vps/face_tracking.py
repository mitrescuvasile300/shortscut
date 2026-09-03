"""
Face detection + tracking for the ShortsCut VPS pipeline.

This is a verbatim port of ``detect_faces_for_clip()`` from the local
``src/lib/shortscut_pipeline.py`` script (the one the user runs on their own
machine to get high-quality shorts). Keeping the detection/tracking logic
byte-for-byte identical means the VPS produces the same crop decisions
(static single / dynamic tracking / dual split-screen / center fallback) as
the local script instead of always falling back to a plain center crop.

Requires ``opencv-python-headless`` and ``mediapipe`` to be installed in the
VPS's Python environment (see vps/requirements.txt). If they are missing,
detection silently degrades to a center crop (matching the local script's
own fallback behaviour).

Output shapes of ``detect_faces_for_clip`` (unchanged from the local script):
  {"mode": "center", "crop_x": int}
  {"mode": "single", "crop_x": int}
  {"mode": "dual", "face1_x": float, "face2_x": float}
  {"mode": "tracking", "keyframes": [(t, x), ...]}

``build_face_crop_plan()`` below adapts these into the crop_plan schema that
server.py's ``_build_filters`` / ``get_crop_x_at_time`` already understand
(the same schema the browser's client-side face tracker used to be the only
producer of).
"""
import json
import math
import shutil
import statistics
import subprocess
import sys
import textwrap
from pathlib import Path


def detect_faces_for_clip(python_path: str, video_path: Path,
                          start_time: float, duration: float,
                          src_w: int, src_h: int, crop_w: int) -> dict:
    """Detect and TRACK face positions throughout the clip for dynamic crop.
    Uses ffmpeg fps filter for frame extraction + MediaPipe/Haar cascade.
    Returns:
      - {"mode": "tracking", "keyframes": [(t, crop_x), ...]} for dynamic face-following
      - {"mode": "single", "crop_x": int} for static crop
      - {"mode": "dual", "face1_x": float, "face2_x": float} for split-screen
      - {"mode": "center", "crop_x": int} for fallback
    """
    # Extract frames at 2fps using ffmpeg (single fast command, reliable on all codecs)
    tmp_frames_dir = Path(video_path).parent / ".face_frames"
    if tmp_frames_dir.exists():
        shutil.rmtree(tmp_frames_dir)
    tmp_frames_dir.mkdir(exist_ok=True)

    # Use fps filter to extract exactly 2 frames per second
    extract_fps = 2 if duration <= 90 else 1  # lower rate for very long clips
    subprocess.run([
        "ffmpeg", "-y", "-ss", str(start_time), "-t", str(duration),
        "-i", str(video_path),
        "-vf", f"fps={extract_fps}", "-q:v", "3",
        str(tmp_frames_dir / "frame_%04d.jpg")
    ], capture_output=True, timeout=180)

    frame_paths = sorted(tmp_frames_dir.glob("frame_*.jpg"))
    frame_interval = 1.0 / extract_fps  # seconds between frames

    if not frame_paths:
        print(f"    ⚠️  Could not extract any frames")
        shutil.rmtree(tmp_frames_dir, ignore_errors=True)
        return {"mode": "center", "crop_x": (src_w - crop_w) // 2}

    frame_paths_json = json.dumps([str(p) for p in frame_paths])

    # Run face detection on ALL frames in subprocess
    face_script = textwrap.dedent(f"""\
import json, sys, os, statistics, math, urllib.request

frame_paths = {frame_paths_json}
src_w = {src_w}
src_h = {src_h}

def log(msg):
    print(f"DIAG: {{msg}}", file=sys.stderr)

# Per-frame results: list of [{{x, y, w, h, score}}, ...] OR None.
# We keep ALL detected faces per frame (not just one primary), because the
# planning pass needs the full list to distinguish "single speaker" from
# "two people simultaneously visible" (real dual-mode).
results = [None] * len(frame_paths)

# ── METHOD 1: MediaPipe Tasks API ──
# The legacy `mp.solutions.face_detection` was removed in recent mediapipe
# builds (the user's environment shows: "module 'mediapipe' has no attribute
# 'solutions'"). The current API is `mediapipe.tasks.python.vision`, which
# requires a TFLite model file passed explicitly.
mp_count = 0
try:
    import cv2
    import numpy as np
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision

    # Download blaze_face_short_range model on first use; cache in /tmp.
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
    model_path = "/tmp/blaze_face_short_range.tflite"
    if not os.path.exists(model_path):
        log("Downloading BlazeFace model...")
        urllib.request.urlretrieve(MODEL_URL, model_path)

    base = mp_tasks.BaseOptions(model_asset_path=model_path)
    opts = mp_vision.FaceDetectorOptions(
        base_options=base,
        min_detection_confidence=0.3,
    )
    detector = mp_vision.FaceDetector.create_from_options(opts)
    log(f"MediaPipe Tasks {{mp.__version__}} + OpenCV {{cv2.__version__}}, {{len(frame_paths)}} frames")

    for idx, fp in enumerate(frame_paths):
        img_bgr = cv2.imread(fp)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h_img, w_img = img_rgb.shape[:2]
        # MediaPipe Tasks expects a uint8 RGB Image
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        res = detector.detect(mp_img)
        if not res.detections:
            continue
        frame_faces = []
        # In Tasks API, BoundingBox uses pixel coords (origin_x, origin_y,
        # width, height). Keypoints are NormalizedKeypoint in [0,1].
        # Keypoint order for BlazeFace: 0=left eye, 1=right eye, 2=nose tip,
        # 3=mouth, 4=left ear tragion, 5=right ear tragion.
        min_face_px = max(40, int(w_img * 0.04))
        for d in res.detections:
            bb = d.bounding_box
            if bb.width < min_face_px or bb.height < min_face_px:
                continue
            score = d.categories[0].score if d.categories else 0.5
            # Prefer nose keypoint as anchor (more stable than bbox center
            # during head turns; bbox grows toward the back of the head).
            cx_px = bb.origin_x + bb.width / 2.0
            cy_px = bb.origin_y + bb.height / 2.0
            if d.keypoints and len(d.keypoints) > 2:
                kp = d.keypoints[2]
                cx_px = kp.x * w_img
                cy_px = kp.y * h_img
            frame_faces.append({{
                "x": cx_px / w_img,          # 0..1, normalized X
                "y": cy_px / h_img,          # 0..1, normalized Y
                "w": bb.width / w_img,       # 0..1, normalized width
                "h": bb.height / h_img,      # 0..1, normalized height
                "score": float(score),
            }})
        if frame_faces:
            results[idx] = frame_faces
            mp_count += 1
    detector.close()
    log(f"MediaPipe: {{mp_count}}/{{len(frame_paths)}} frames")
except Exception as e:
    import traceback
    log(f"MediaPipe error: {{e}}")
    log(traceback.format_exc().replace(chr(10), ' | '))

# ── METHOD 2: Haar cascade fallback for frames where MP failed ──
# Tightened minSize from 3% to 8% of frame width — at 3%, Haar caught
# audience members in stand-up footage, leading to spurious dual detection.
# 8% of 1920 = 154 px; that's about a face at ~3m, which is what we want.
haar_count = 0
try:
    import cv2
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if os.path.exists(cascade_path):
        face_cascade = cv2.CascadeClassifier(cascade_path)
        for idx, fp in enumerate(frame_paths):
            if results[idx] is not None:
                continue
            img = cv2.imread(fp)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            h_img, w_img = gray.shape
            min_face = max(80, int(w_img * 0.08))
            faces = face_cascade.detectMultiScale(
                gray, 1.1, 4, minSize=(min_face, min_face)
            )
            if len(faces) == 0:
                continue
            frame_faces = []
            for (x, y, fw, fh) in faces:
                frame_faces.append({{
                    "x": (x + fw / 2.0) / w_img,
                    "y": (y + fh / 2.0) / h_img,
                    "w": fw / w_img,
                    "h": fh / h_img,
                    "score": 0.5,  # Haar has no confidence score
                }})
            results[idx] = frame_faces
            haar_count += 1
        if haar_count:
            log(f"Haar: filled {{haar_count}} more frames")
except Exception as e:
    log(f"Haar error: {{e}}")

# Edge-analysis fallback removed: it guessed a face position even when none
# existed, which is worse than letting downstream emit an honest center crop.

detected = sum(1 for r in results if r is not None)
log(f"Total: {{detected}}/{{len(results)}} frames with faces")
print(json.dumps({{"results": results, "mp": mp_count, "haar": haar_count}}))
""")

    r = subprocess.run(
        [python_path, "-c", face_script],
        capture_output=True, text=True, timeout=300,
    )

    # Print diagnostics
    if r.stderr:
        for line in r.stderr.strip().split("\n"):
            if line.startswith("DIAG:"):
                print(f"    {line[5:].strip()}")

    try:
        data = json.loads(r.stdout.strip().split("\n")[-1])
    except Exception:
        print(f"    ⚠️  Face detect parse error. stderr={r.stderr[:300]}")
        shutil.rmtree(tmp_frames_dir, ignore_errors=True)
        return {"mode": "center", "crop_x": (src_w - crop_w) // 2}

    shutil.rmtree(tmp_frames_dir, ignore_errors=True)

    raw = data.get("results", [])
    n = len(raw)
    n_detected = sum(1 for r in raw if r is not None and len(r) > 0)

    if n_detected < 2:
        print(f"    ⚠️  Only {n_detected} face detections — using center")
        return {"mode": "center", "crop_x": (src_w - crop_w) // 2}

    fps_sample = extract_fps
    dt = 1.0 / fps_sample
    max_crop_x = src_w - crop_w

    # ──────────────────────────────────────────────────────────────────
    #  Step 1: DUAL DETECTION (strict — two real speakers, side by side)
    # ──────────────────────────────────────────────────────────────────
    # Both faces must:
    #   - Be visible in the SAME frame (not just somewhere across the clip)
    #   - Be separated horizontally by ≥ 0.8 × cropWidth (so the split makes
    #     visual sense — they wouldn't fit in a single 9:16 crop)
    #   - Be at SIMILAR Y position (within ±25% of frame height). This is
    #     the constraint that kills the "comedian on stage + audience in
    #     front" false positive: comedian face Y ≈ 0.30, audience Y ≈ 0.85.
    #   - Be SIMILAR in size (max/min area ratio ≤ 2.0). Same camera,
    #     same distance → similar bounding-box sizes.
    # And the dual configuration must hold in ≥ 85% of frames-with-faces.
    min_sep_norm = (crop_w * 0.8) / src_w   # 0.253 on 1920×1080
    y_tolerance = 0.25
    size_ratio_max = 2.0
    dual_consistency = 0.85

    frames_with_any = 0
    frames_with_valid_dual = 0
    valid_lefts = []
    valid_rights = []

    for frame_faces in raw:
        if not frame_faces:
            continue
        frames_with_any += 1
        if len(frame_faces) < 2:
            continue
        # Filter out any tiny detections (defensive — detection script also filters)
        big = [f for f in frame_faces if f["w"] > 0.03 and f["h"] > 0.03]
        if len(big) < 2:
            continue
        sorted_faces = sorted(big, key=lambda f: f["x"])
        left = sorted_faces[0]
        right = sorted_faces[-1]
        if right["x"] - left["x"] < min_sep_norm:
            continue
        if abs(left["y"] - right["y"]) > y_tolerance:
            continue
        a_left = left["w"] * left["h"]
        a_right = right["w"] * right["h"]
        if min(a_left, a_right) < 1e-4:
            continue
        ratio = max(a_left, a_right) / min(a_left, a_right)
        if ratio > size_ratio_max:
            continue
        frames_with_valid_dual += 1
        valid_lefts.append(left["x"])
        valid_rights.append(right["x"])

    if (
        frames_with_any > 0
        and frames_with_valid_dual / frames_with_any >= dual_consistency
        and len(valid_lefts) >= 3
    ):
        f1 = statistics.median(valid_lefts)
        f2 = statistics.median(valid_rights)
        print(f"    👥 Dual mode: left={f1:.2f}, right={f2:.2f} (in {frames_with_valid_dual}/{frames_with_any} frames)")
        return {"mode": "dual", "face1_x": f1, "face2_x": f2}

    # ──────────────────────────────────────────────────────────────────
    #  Step 2: SELECT PRIMARY FACE per frame (largest × confidence ×
    #          proximity_to_previous_primary). This tracks ONE speaker
    #          across frames instead of averaging all detected faces.
    # ──────────────────────────────────────────────────────────────────
    sigma = 0.30  # normalized
    two_sigma_sq = 2 * sigma * sigma

    primary_x = [None] * n
    primary_area = [None] * n
    prev_primary = None

    for idx, frame_faces in enumerate(raw):
        if not frame_faces:
            continue
        best = None
        best_score = -1
        for f in frame_faces:
            area = f["w"] * f["h"]
            score = f["score"]
            proximity = 1.0
            if prev_primary is not None:
                dx = f["x"] - prev_primary["x"]
                dy = f["y"] - prev_primary["y"]
                proximity = math.exp(-(dx * dx + dy * dy) / two_sigma_sq)
            s = area * score * proximity
            if s > best_score:
                best_score = s
                best = f
        if best is not None:
            prev_primary = {"x": best["x"], "y": best["y"]}
            primary_x[idx] = best["x"]
            primary_area[idx] = best["w"] * best["h"]

    # ──────────────────────────────────────────────────────────────────
    #  Step 3: OUTLIER REJECTION (local MAD over ±2 neighbors).
    #          Kills single-frame spikes from poster/mirror false positives.
    # ──────────────────────────────────────────────────────────────────
    cleaned = list(primary_x)
    for i in range(n):
        if cleaned[i] is None:
            continue
        ctx = [primary_x[j] for j in range(max(0, i - 2), min(n, i + 3))
               if j != i and primary_x[j] is not None]
        if len(ctx) < 2:
            continue
        med = statistics.median(ctx)
        mad = statistics.median([abs(v - med) for v in ctx])
        if abs(cleaned[i] - med) > 3 * mad + 0.02:
            cleaned[i] = None

    valid = [v for v in cleaned if v is not None]
    if len(valid) < max(2, int(n * 0.15)):
        print(f"    ⚠️  Only {len(valid)} valid detections after outlier rejection → center")
        return {"mode": "center", "crop_x": max_crop_x // 2}

    # ──────────────────────────────────────────────────────────────────
    #  Step 3.5: BIMODALITY-BASED DUAL FALLBACK.
    #
    #  Step 1 (same-frame dual detection) requires both speakers to be
    #  visible in the same frame ≥85% of the time. In real interview /
    #  podcast footage this often misses dual mode because:
    #    - Strict profile views drop face detector confidence; the
    #      primary track ping-pongs between speakers instead of detecting
    #      both at once
    #    - One speaker frequently looks down at notes / laptop / mug
    #
    #  This pass catches that case: if the cleaned primary-X track is
    #  CLEANLY BIMODAL (two tight clusters separated by ≥0.30 of frame
    #  width, each cluster getting ≥25% of frames, each σ < 0.05), AND
    #  the median face at each cluster passes the Y/size constraints
    #  already used in Step 1, then it's a real dual scene.
    # ──────────────────────────────────────────────────────────────────
    sorted_xs = sorted(valid)
    total = len(sorted_xs)
    min_side = max(2, math.ceil(total * 0.25))
    bimodal_result = None
    if total >= 8:
        # Find split that minimizes total within-cluster SSE
        # Use prefix sums for O(1) variance per split
        ps = [0.0] * (total + 1)
        pss = [0.0] * (total + 1)
        for i in range(total):
            ps[i + 1] = ps[i] + sorted_xs[i]
            pss[i + 1] = pss[i] + sorted_xs[i] * sorted_xs[i]
        def sse(lo, hi):
            nn = hi - lo
            if nn <= 0:
                return 0.0
            s = ps[hi] - ps[lo]
            sq = pss[hi] - pss[lo]
            return sq - (s * s) / nn
        best_split = -1
        best_sse = float("inf")
        for split in range(min_side, total - min_side + 1):
            tot_sse = sse(0, split) + sse(split, total)
            if tot_sse < best_sse:
                best_sse = tot_sse
                best_split = split
        if best_split > 0:
            n1 = best_split
            n2 = total - n1
            c1 = (ps[n1] - ps[0]) / n1
            c2 = (ps[total] - ps[n1]) / n2
            s1 = (sse(0, n1) / n1) ** 0.5
            s2 = (sse(n1, total) / n2) ** 0.5
            MIN_SEP = 0.30
            MAX_SIGMA = 0.05
            if c2 - c1 >= MIN_SEP and s1 < MAX_SIGMA and s2 < MAX_SIGMA:
                # Collect ALL faces in original data near each cluster center
                # for Y/size verification
                tol = 0.05
                c1_faces = []
                c2_faces = []
                for frame_faces in raw:
                    if not frame_faces:
                        continue
                    for f in frame_faces:
                        if abs(f["x"] - c1) < tol:
                            c1_faces.append(f)
                        elif abs(f["x"] - c2) < tol:
                            c2_faces.append(f)
                if len(c1_faces) >= 2 and len(c2_faces) >= 2:
                    med_y1 = statistics.median([f["y"] for f in c1_faces])
                    med_y2 = statistics.median([f["y"] for f in c2_faces])
                    med_a1 = statistics.median([f["w"] * f["h"] for f in c1_faces])
                    med_a2 = statistics.median([f["w"] * f["h"] for f in c2_faces])
                    if (
                        abs(med_y1 - med_y2) <= 0.25
                        and med_a1 > 0 and med_a2 > 0
                        and max(med_a1, med_a2) / min(med_a1, med_a2) <= 2.0
                    ):
                        bimodal_result = {
                            "mode": "dual",
                            "face1_x": c1,
                            "face2_x": c2,
                        }
    if bimodal_result is not None:
        print(f"    👥 Dual mode (bimodal): left={bimodal_result['face1_x']:.2f}, "
              f"right={bimodal_result['face2_x']:.2f}")
        return bimodal_result

    # ──────────────────────────────────────────────────────────────────
    #  Step 4: TRY STATIC SINGLE CROP FIRST.
    #          User spec: "default to static, only move camera when the
    #          face goes more than half off-screen."
    #          1. Find optimal static cropX (median face X, 0.45 headroom).
    #          2. Count how many frames have face center OUTSIDE the
    #             [cropX, cropX+cropW] window.
    #          3. If ≤ 15% exit → use static; the rare exits will clip
    #             the face partially, which the user accepts.
    # ──────────────────────────────────────────────────────────────────
    median_face_x_norm = statistics.median(valid)
    median_face_x_px = median_face_x_norm * src_w
    static_crop_x = int(median_face_x_px - crop_w * 0.45)
    static_crop_x = max(0, min(static_crop_x, max_crop_x))

    exit_count = 0
    for v in valid:
        face_px = v * src_w
        if face_px < static_crop_x or face_px > static_crop_x + crop_w:
            exit_count += 1
    exit_fraction = exit_count / len(valid)

    static_threshold = 0.15
    if exit_fraction <= static_threshold:
        print(
            f"    📷 Static single (x={static_crop_x}, "
            f"face exits crop in {exit_fraction * 100:.0f}% of frames ≤ {int(static_threshold * 100)}%)"
        )
        return {"mode": "single", "crop_x": static_crop_x}

    print(
        f"    🎥 Engaging tracking ({exit_fraction * 100:.0f}% of frames "
        f"have face outside static crop)"
    )

    # ──────────────────────────────────────────────────────────────────
    #  Step 5: TRACKING MODE — smooth follow.
    #
    #  Find hard-reset markers (gaps ≥ 0.75 s) where camera state should
    #  reset rather than EMA-into-stale-position.
    # ──────────────────────────────────────────────────────────────────
    hard_reset = set()
    i = 0
    while i < n:
        if cleaned[i] is not None:
            i += 1
            continue
        j = i
        while j < n and cleaned[j] is None:
            j += 1
        if j - i >= max(1, int(0.75 * fps_sample)) and j < n:
            hard_reset.add(j)
        i = j

    # Gap fill: linear interpolation across short gaps (≤ 1.5 s)
    filled = list(cleaned)
    max_gap = max(1, int(1.5 * fps_sample))
    i = 0
    while i < n:
        if filled[i] is not None:
            i += 1
            continue
        j = i
        while j < n and filled[j] is None:
            j += 1
        gap = j - i
        prev_v = filled[i - 1] if i > 0 else None
        next_v = filled[j] if j < n else None
        if gap <= max_gap and prev_v is not None and next_v is not None:
            for k in range(i, j):
                f = (k - i + 1) / (j - i + 1)
                filled[k] = prev_v + (next_v - prev_v) * f
        elif gap <= max_gap and prev_v is not None:
            for k in range(i, j):
                filled[k] = prev_v
        elif gap <= max_gap and next_v is not None:
            for k in range(i, j):
                filled[k] = next_v
        i = j

    # Smoothing: deadzone + EMA + velocity cap + cut detection + re-anchor
    dz_norm = (crop_w * 0.12) / src_w
    alpha = 0.30                                    # τ = 0.70 s at 4 fps
    max_pan_per_step = ((0.30 * crop_w) / src_w) * dt
    cut_position_delta = 0.25
    cut_area_ratio = 1.5
    reanchor_frames = max(1, int(3.0 * fps_sample))

    cam = [None] * n
    cam_x = None
    last_moved = -1
    last_valid = -1

    for i in range(n):
        head = filled[i]
        if i in hard_reset:
            cam_x = None
            last_moved = -1
        if head is None:
            cam[i] = None
            continue
        is_cut = False
        if last_valid >= 0 and cam_x is not None:
            prev_head = filled[last_valid]
            prev_a = primary_area[last_valid]
            cur_a = primary_area[i]
            if prev_head is not None and abs(head - prev_head) > cut_position_delta:
                if prev_a is not None and cur_a is not None and (
                    cur_a / prev_a > cut_area_ratio or prev_a / cur_a > cut_area_ratio
                ):
                    is_cut = True
                elif prev_a is None or cur_a is None:
                    is_cut = True
        if cam_x is None or is_cut:
            cam_x = head
            cam[i] = cam_x
            last_valid = i
            last_moved = i
            continue
        moved = False
        if head > cam_x + dz_norm or head < cam_x - dz_norm:
            targ = head
            moved = True
        else:
            targ = cam_x
        nxt = alpha * targ + (1 - alpha) * cam_x
        dv = nxt - cam_x
        if dv > max_pan_per_step:
            nxt = cam_x + max_pan_per_step
        elif dv < -max_pan_per_step:
            nxt = cam_x - max_pan_per_step
        if not moved and last_moved >= 0 and i - last_moved >= reanchor_frames:
            nxt = head
            last_moved = i
        elif moved:
            last_moved = i
        cam_x = nxt
        cam[i] = cam_x
        last_valid = i

    # Convert to pixel crop X with 0.45 headroom bias
    crop_positions = []
    for v in cam:
        if v is None:
            crop_positions.append(None)
        else:
            cx = int((v * src_w) - (crop_w * 0.45))
            cx = max(0, min(cx, max_crop_x))
            crop_positions.append(cx)

    # Keyframe reduction (greedy, ε = max(5px, 0.8% × cropWidth))
    valid_idx = [i for i, v in enumerate(crop_positions) if v is not None]
    if not valid_idx:
        return {"mode": "center", "crop_x": max_crop_x // 2}

    keyframes = []
    epsilon = max(5, int(crop_w * 0.008))
    for idx in valid_idx:
        t = idx * dt
        x = crop_positions[idx]
        if not keyframes:
            keyframes.append((t, x))
            continue
        if abs(x - keyframes[-1][1]) > epsilon or idx == valid_idx[-1]:
            keyframes.append((t, x))

    xs_only = [x for (_, x) in keyframes]
    rng_val = max(xs_only) - min(xs_only)
    # If after smoothing the range is small, fall back to single static
    if rng_val < 25:
        avg_x = max(0, min(int(statistics.mean(xs_only)), max_crop_x))
        print(f"    📷 Smoothed range only {rng_val}px — falling back to single (x={avg_x})")
        return {"mode": "single", "crop_x": avg_x}

    print(f"    📹 Tracking: {len(keyframes)} keyframes, range {rng_val}px")
    return {"mode": "tracking", "keyframes": keyframes}




def build_face_crop_plan(python_path: str, video_path, start_time: float,
                          duration: float, src_w: int, src_h: int) -> dict:
    """Run detect_faces_for_clip() and adapt its result into the crop_plan
    schema consumed by server.py's _build_filters / get_crop_x_at_time.
    """
    crop_w = round(src_h * 9 / 16)
    try:
        result = detect_faces_for_clip(
            python_path, Path(video_path), start_time, duration, src_w, src_h, crop_w
        )
    except Exception as e:
        print(f"    ⚠️  Face detection crashed ({e}) — using center crop")
        result = {"mode": "center", "crop_x": (src_w - crop_w) // 2}

    mode = result.get("mode", "center")
    if mode == "tracking":
        segment = {
            "startTime": 0,
            "endTime": duration,
            "mode": "tracking",
            "keyframes": [{"t": t, "x": x} for (t, x) in result["keyframes"]],
        }
    elif mode == "dual":
        segment = {
            "startTime": 0,
            "endTime": duration,
            "mode": "dual",
            "face1X": result["face1_x"] * src_w,
            "face2X": result["face2_x"] * src_w,
        }
    elif mode == "single":
        segment = {
            "startTime": 0,
            "endTime": duration,
            "mode": "single",
            "cropX": result["crop_x"],
        }
    else:
        segment = {
            "startTime": 0,
            "endTime": duration,
            "mode": "center",
            "cropX": result["crop_x"],
        }

    return {
        "videoWidth": src_w,
        "videoHeight": src_h,
        "segments": [segment],
    }
