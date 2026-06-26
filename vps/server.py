#!/usr/bin/env python3
"""
Shortscut VPS Processing Server
Processes video clips with native ffmpeg: crop 9:16, silence removal, subtitle burn, libx264.
"""

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from functools import wraps
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────
API_KEY = os.environ.get("SHORTSCUT_API_KEY", "shortcut-vps-2026")
PORT = int(os.environ.get("PORT", "3458"))
WORK_DIR = Path("/tmp/shortscut-processing")
WORK_DIR.mkdir(parents=True, exist_ok=True)
MAX_CONCURRENT = 2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("shortscut")

# Track processed files for cleanup
processed_files: dict[str, dict] = {}  # file_id -> {path, created, size}
processing_lock = threading.Semaphore(MAX_CONCURRENT)


def cleanup_old_files(max_age_secs=1800):
    """Remove processed files older than max_age_secs."""
    now = time.time()
    to_delete = []
    for fid, info in list(processed_files.items()):
        if now - info["created"] > max_age_secs:
            to_delete.append(fid)
    for fid in to_delete:
        info = processed_files.pop(fid, None)
        if info and os.path.exists(info["path"]):
            os.unlink(info["path"])
            log.info(f"Cleaned up old file: {fid}")


def run_cmd(args, timeout=600, cwd=None):
    """Run a command and return (stdout, stderr, returncode)."""
    log.info(f"Running: {' '.join(args[:10])}...")
    result = subprocess.run(
        args, capture_output=True, text=True, timeout=timeout, cwd=cwd
    )
    return result.stdout, result.stderr, result.returncode


def detect_silence(input_path, start_time, duration, threshold_db=-30, min_dur=0.8):
    """Detect silence intervals in a clip using ffmpeg silencedetect."""
    fast_seek = max(0, start_time - 0.5)
    inner_seek = start_time - fast_seek

    args = [
        "ffmpeg", "-hide_banner",
        "-ss", f"{fast_seek:.3f}",
        "-i", input_path,
        "-ss", f"{inner_seek:.3f}",
        "-t", f"{duration:.3f}",
        "-af", f"silencedetect=noise={threshold_db}dB:d={min_dur}",
        "-vn", "-f", "null", "-"
    ]
    _, stderr, _ = run_cmd(args, timeout=120)

    silences = []
    cur_start = None
    for line in stderr.split("\n"):
        m = re.search(r"silence_start:\s*([\d.]+)", line)
        if m:
            cur_start = float(m.group(1))
            continue
        m = re.search(r"silence_end:\s*([\d.]+)", line)
        if m and cur_start is not None:
            silences.append({"start": cur_start, "end": float(m.group(1))})
            cur_start = None
    if cur_start is not None:
        silences.append({"start": cur_start, "end": duration})

    return silences


def build_speaking_segments(silences, clip_duration, padding=0.12):
    """Convert silence intervals into speaking segments."""
    if not silences:
        return [{"srcStart": 0, "srcEnd": clip_duration, "outStart": 0}]

    # Merge silences closer than 0.25s
    merged = []
    for s in silences:
        if merged and s["start"] - merged[-1]["end"] < 0.25:
            merged[-1]["end"] = s["end"]
        else:
            merged.append({"start": s["start"], "end": s["end"]})

    segments = []
    out = 0
    last_end = 0

    for silence in merged:
        seg_end = min(silence["start"] + padding, clip_duration)
        if seg_end - last_end > 0.05:
            segments.append({"srcStart": last_end, "srcEnd": seg_end, "outStart": out})
            out += seg_end - last_end
        last_end = max(silence["end"] - padding, last_end)

    if clip_duration - last_end > 0.05:
        segments.append({"srcStart": last_end, "srcEnd": clip_duration, "outStart": out})

    return segments


def remap_time(t, segs):
    """Map a source time to its position in silence-removed output."""
    for s in segs:
        if s["srcStart"] <= t <= s["srcEnd"]:
            return s["outStart"] + (t - s["srcStart"])
    return None


def adjust_ass_for_silence_removal(ass_content, segs):
    """Rewrite ASS subtitle timings for silence-removed timeline."""
    def parse_t(s):
        m = re.match(r"(\d+):(\d{2}):(\d{2})\.(\d{2})", s.strip())
        if not m:
            return 0
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) + int(m.group(4)) / 100

    def fmt_t(t):
        t = max(0, t)
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        cs = round((t % 1) * 100)
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    lines = ass_content.split("\n")
    result = []
    for line in lines:
        if not line.startswith("Dialogue:"):
            result.append(line)
            continue
        parts = line.split(",")
        if len(parts) < 3:
            result.append(line)
            continue
        st = parse_t(parts[1])
        et = parse_t(parts[2])
        mid = (st + et) / 2
        nm = remap_time(mid, segs)
        if nm is None:
            continue  # subtitle falls in silence → drop
        ns = remap_time(st, segs)
        if ns is None:
            ns = max(0, nm - (et - st) / 2)
        ne = remap_time(et, segs)
        if ne is None:
            ne = ns + (et - st)
        parts[1] = fmt_t(ns)
        parts[2] = fmt_t(ne)
        result.append(",".join(parts))
    return "\n".join(result)


def clamp_crop(x, src_w, crop_w):
    return max(0, min(x, src_w - crop_w))


def get_crop_x_at_time(plan, t):
    """Look up crop X from a CropPlan at a given clip-relative time."""
    crop_w = round(plan["videoHeight"] * 9 / 16)
    max_x = plan["videoWidth"] - crop_w

    for seg in plan.get("segments", []):
        if t < seg.get("startTime", 0) or t > seg.get("endTime", 999999):
            continue
        mode = seg.get("mode", "center")
        if mode in ("center", "single"):
            cx = seg.get("cropX")
            return clamp_crop(cx if cx is not None else round(max_x / 2), plan["videoWidth"], crop_w)
        if mode == "tracking" and seg.get("keyframes"):
            kfs = seg["keyframes"]
            rt = t - seg.get("startTime", 0)
            if rt <= kfs[0]["t"]:
                return round(kfs[0]["x"])
            if rt >= kfs[-1]["t"]:
                return round(kfs[-1]["x"])
            for i in range(len(kfs) - 1):
                if kfs[i]["t"] <= rt <= kfs[i + 1]["t"]:
                    f = (rt - kfs[i]["t"]) / (kfs[i + 1]["t"] - kfs[i]["t"])
                    return round(kfs[i]["x"] + f * (kfs[i + 1]["x"] - kfs[i]["x"]))
        if mode == "dual":
            mid = ((seg.get("face1X", 0) + seg.get("face2X", 0)) / 2)
            return clamp_crop(round(mid - crop_w / 2), plan["videoWidth"], crop_w)

    return round(max_x / 2)


def build_tracking_expr(keyframes, t_offset, src_w, crop_w):
    """Build an ffmpeg expression for smooth face-tracking crop."""
    if not keyframes:
        return str(round((src_w - crop_w) / 2))
    
    max_x = src_w - crop_w
    expr_parts = []
    for i, kf in enumerate(keyframes):
        t = kf["t"] + t_offset
        x = clamp_crop(round(kf["x"]), src_w, crop_w)
        if i == 0:
            expr_parts.append(f"if(lt(t\\,{t:.3f})\\,{x}")
        else:
            prev = keyframes[i - 1]
            prev_t = prev["t"] + t_offset
            prev_x = clamp_crop(round(prev["x"]), src_w, crop_w)
            # Linear interpolation between prev and current
            expr_parts.append(
                f"if(lt(t\\,{t:.3f})\\,"
                f"{prev_x}+(t-{prev_t:.3f})*{(x - prev_x) / max(0.001, t - prev_t):.1f}"
            )
    
    # Final value (after last keyframe)
    last_x = clamp_crop(round(keyframes[-1]["x"]), src_w, crop_w)
    result = f"{last_x}"
    for part in reversed(expr_parts):
        result = f"{part}\\,{result})"
    
    return result


def process_clip(input_path, clip_config, video_width, video_height, work_dir):
    """Process a single clip with ffmpeg. Returns path to output MP4."""
    start_time = clip_config["start_time"]
    end_time = clip_config["end_time"]
    duration = end_time - start_time
    index = clip_config.get("index", 0)
    out_w, out_h = 1080, 1920
    has_subs = bool(clip_config.get("ass_subtitles"))
    ass_content = clip_config.get("ass_subtitles", "")
    remove_silence = clip_config.get("remove_silence", True)
    crop_plan = clip_config.get("crop_plan")

    output_path = os.path.join(work_dir, f"clip_{index}.mp4")

    # Build default crop plan if none provided
    if not crop_plan or not crop_plan.get("segments"):
        crop_w = round(video_height * 9 / 16)
        crop_x = clip_config.get("crop_x")
        if crop_x is None:
            crop_x = round((video_width - crop_w) / 2)
        else:
            crop_x = min(crop_x, video_width - crop_w)
        crop_plan = {
            "videoWidth": video_width,
            "videoHeight": video_height,
            "segments": [{
                "startTime": 0,
                "endTime": duration,
                "mode": "center" if clip_config.get("crop_x") is None else "single",
                "cropX": crop_x
            }]
        }

    # ── Silence detection & removal ──
    speaking_segs = None
    if remove_silence:
        try:
            silences = detect_silence(input_path, start_time, duration)
            total_silence = sum(s["end"] - s["start"] for s in silences)
            if total_silence > 0.8 and silences:
                segs = build_speaking_segments(silences, duration)
                if len(segs) > 1:
                    speaking_segs = segs
                    out_dur = sum(s["srcEnd"] - s["srcStart"] for s in segs)
                    log.info(
                        f"Clip {index}: removed {total_silence:.1f}s silence "
                        f"({duration:.1f}s → {out_dur:.1f}s, {len(segs)} segments)"
                    )
                    if has_subs and ass_content:
                        ass_content = adjust_ass_for_silence_removal(ass_content, segs)
        except Exception as e:
            log.warning(f"Clip {index} silence detection failed: {e}")

    # Write ASS subtitles if present
    subs_path = os.path.join(work_dir, f"subs_{index}.ass")
    if has_subs and ass_content:
        with open(subs_path, "w", encoding="utf-8") as f:
            f.write(ass_content)

    # Build ffmpeg command
    fast_seek = max(0, start_time - 0.5)
    inner_seek = start_time - fast_seek

    args = [
        "ffmpeg", "-hide_banner", "-y",
        "-ss", f"{fast_seek:.3f}",
        "-i", input_path,
        "-ss", f"{inner_seek:.3f}",
        "-t", f"{duration:.3f}",
    ]

    src_w = crop_plan["videoWidth"]
    src_h = crop_plan["videoHeight"]
    crop_w = round(src_h * 9 / 16)
    crop_h = src_h

    if speaking_segs:
        # Combined silence-removal + crop + scale filter chain
        fc = _build_silence_removal_filters(
            speaking_segs, crop_plan, out_w, out_h, has_subs, subs_path
        )
        args += ["-filter_complex", fc["filter_complex"], "-map", fc["out_v"], "-map", fc["out_a"]]
    else:
        # Standard filter chain
        fc = _build_filters(crop_plan, out_w, out_h, has_subs, subs_path)
        if fc.get("filter_complex"):
            args += ["-filter_complex", fc["filter_complex"], "-map", fc["out_label"]]
            if not speaking_segs:
                args += ["-map", "0:a?"]
        elif fc.get("vf"):
            args += ["-vf", fc["vf"]]

    args += [
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        output_path,
    ]

    _, stderr, rc = run_cmd(args, timeout=300, cwd=work_dir)
    if rc != 0:
        log.error(f"Clip {index} ffmpeg failed:\n{stderr[-2000:]}")
        raise RuntimeError(f"ffmpeg failed for clip {index}: {stderr[-500:]}")

    # Cleanup subs
    if os.path.exists(subs_path):
        os.unlink(subs_path)

    if not os.path.exists(output_path):
        raise RuntimeError(f"Output file not created for clip {index}")

    size = os.path.getsize(output_path)
    log.info(f"Clip {index} processed: {size / 1e6:.1f} MB")
    return output_path


def _build_filters(plan, out_w, out_h, has_subs, subs_path):
    """Build ffmpeg filter chain (no silence removal)."""
    src_w = plan["videoWidth"]
    src_h = plan["videoHeight"]
    crop_w = round(src_h * 9 / 16)
    crop_h = src_h
    segments = plan.get("segments", [])

    # Fast path: one segment, single/center
    if len(segments) == 1 and segments[0].get("mode") in ("single", "center"):
        seg = segments[0]
        cx = seg.get("cropX")
        if cx is not None:
            cx = max(0, min(cx, src_w - crop_w))
        else:
            cx = round((src_w - crop_w) / 2)
        base = f"crop={crop_w}:{crop_h}:{cx}:0,scale={out_w}:{out_h}"
        vf = f"{base},ass='{subs_path}'" if has_subs else base
        return {"vf": vf}

    # Fast path: one segment, tracking
    if len(segments) == 1 and segments[0].get("mode") == "tracking":
        seg = segments[0]
        expr = build_tracking_expr(seg.get("keyframes", []), 0, src_w, crop_w)
        base = f"crop={crop_w}:{crop_h}:'{expr}':0,scale={out_w}:{out_h}"
        vf = f"{base},ass='{subs_path}'" if has_subs else base
        return {"vf": vf}

    # One segment, dual mode
    if len(segments) == 1 and segments[0].get("mode") == "dual":
        seg = segments[0]
        half_h = out_h // 2
        dual_crop_w = min(src_w, round(src_h * 9 / 8))
        cx1 = clamp_crop(round((seg.get("face1X", 0)) - dual_crop_w / 2), src_w, dual_crop_w)
        cx2 = clamp_crop(round((seg.get("face2X", 0)) - dual_crop_w / 2), src_w, dual_crop_w)
        fc = (
            f"[0:v]split=2[top][bot];"
            f"[top]crop={dual_crop_w}:{crop_h}:{cx1}:0,scale={out_w}:{half_h}[t];"
            f"[bot]crop={dual_crop_w}:{crop_h}:{cx2}:0,scale={out_w}:{half_h}[b];"
            f"[t][b]vstack=inputs=2[stacked]"
        )
        if has_subs:
            fc += f";[stacked]ass='{subs_path}'[v]"
            return {"filter_complex": fc, "out_label": "[v]"}
        return {"filter_complex": fc, "out_label": "[stacked]"}

    # Multi-segment: center crop as fallback
    cx = round((src_w - crop_w) / 2)
    base = f"crop={crop_w}:{crop_h}:{cx}:0,scale={out_w}:{out_h}"
    vf = f"{base},ass='{subs_path}'" if has_subs else base
    return {"vf": vf}


def _build_silence_removal_filters(segs, plan, out_w, out_h, has_subs, subs_path):
    """Build filter_complex that removes silence AND applies crop+scale."""
    src_w = plan["videoWidth"]
    src_h = plan["videoHeight"]
    crop_w = round(src_h * 9 / 16)
    crop_h = src_h
    n = len(segs)

    dual_seg = None
    for s in plan.get("segments", []):
        if s.get("mode") == "dual":
            dual_seg = s
            break

    vs = "".join(f"[v{i}]" for i in range(n))
    als = "".join(f"[a{i}]" for i in range(n))
    fc = f"[0:v]split={n}{vs};[0:a]asplit={n}{als}"

    for i, s in enumerate(segs):
        mid_t = (s["srcStart"] + s["srcEnd"]) / 2

        if dual_seg:
            half_h = out_h // 2
            dual_crop_w = min(src_w, round(src_h * 9 / 8))
            cx1 = clamp_crop(round(dual_seg.get("face1X", 0) - dual_crop_w / 2), src_w, dual_crop_w)
            cx2 = clamp_crop(round(dual_seg.get("face2X", 0) - dual_crop_w / 2), src_w, dual_crop_w)
            fc += (
                f";[v{i}]trim=start={s['srcStart']:.3f}:end={s['srcEnd']:.3f},setpts=PTS-STARTPTS,"
                f"split=2[v{i}t][v{i}b]"
                f";[v{i}t]crop={dual_crop_w}:{crop_h}:{cx1}:0,scale={out_w}:{half_h}[v{i}tc]"
                f";[v{i}b]crop={dual_crop_w}:{crop_h}:{cx2}:0,scale={out_w}:{half_h}[v{i}bc]"
                f";[v{i}tc][v{i}bc]vstack=inputs=2[vo{i}]"
            )
        else:
            cx = get_crop_x_at_time(plan, mid_t)
            fc += (
                f";[v{i}]trim=start={s['srcStart']:.3f}:end={s['srcEnd']:.3f},setpts=PTS-STARTPTS,"
                f"crop={crop_w}:{crop_h}:{cx}:0,scale={out_w}:{out_h}[vo{i}]"
            )
        fc += (
            f";[a{i}]atrim=start={s['srcStart']:.3f}:end={s['srcEnd']:.3f},"
            f"asetpts=PTS-STARTPTS[ao{i}]"
        )

    ci = "".join(f"[vo{i}][ao{i}]" for i in range(n))
    fc += f";{ci}concat=n={n}:v=1:a=1[cv][ca]"

    if has_subs:
        fc += f";[cv]ass='{subs_path}'[fv]"
        return {"filter_complex": fc, "out_v": "[fv]", "out_a": "[ca]"}
    return {"filter_complex": fc, "out_v": "[cv]", "out_a": "[ca]"}


def download_video(video_url, audio_url, work_dir):
    """Download video (and optionally separate audio) to work_dir. Returns input path and video dimensions."""
    video_path = os.path.join(work_dir, "source.mp4")

    # Try yt-dlp first for YouTube URLs
    if "youtube.com" in video_url or "youtu.be" in video_url:
        log.info("Downloading with yt-dlp...")
        args = [
            "yt-dlp",
            "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]",
            "--merge-output-format", "mp4",
            "-o", video_path,
            "--no-playlist",
            video_url,
        ]
        _, stderr, rc = run_cmd(args, timeout=300, cwd=work_dir)
        if rc == 0 and os.path.exists(video_path):
            log.info(f"Downloaded with yt-dlp: {os.path.getsize(video_path) / 1e6:.1f} MB")
        else:
            log.warning(f"yt-dlp failed: {stderr[-500:]}")
            video_path = None

    # Direct download fallback (for Piped/cobalt URLs)
    if not os.path.exists(video_path) if video_path else True:
        video_path = os.path.join(work_dir, "source.mp4")
        log.info(f"Direct downloading video...")
        args = ["curl", "-L", "-o", video_path, "-m", "300", "--retry", "2", video_url]
        _, stderr, rc = run_cmd(args, timeout=360)
        if rc != 0 or not os.path.exists(video_path):
            raise RuntimeError(f"Video download failed: {stderr[-500:]}")
        log.info(f"Downloaded video: {os.path.getsize(video_path) / 1e6:.1f} MB")

    # Download separate audio if provided
    if audio_url:
        audio_path = os.path.join(work_dir, "audio.m4a")
        log.info("Downloading separate audio...")
        args = ["curl", "-L", "-o", audio_path, "-m", "300", "--retry", "2", audio_url]
        _, stderr, rc = run_cmd(args, timeout=360)
        if rc == 0 and os.path.exists(audio_path):
            # Merge video + audio
            merged_path = os.path.join(work_dir, "merged.mp4")
            args = [
                "ffmpeg", "-hide_banner", "-y",
                "-i", video_path, "-i", audio_path,
                "-c", "copy", "-movflags", "+faststart", merged_path
            ]
            _, stderr, rc = run_cmd(args, timeout=120)
            if rc == 0 and os.path.exists(merged_path):
                os.unlink(video_path)
                os.rename(merged_path, video_path)
                log.info("Merged video + audio")
            if os.path.exists(audio_path):
                os.unlink(audio_path)

    # Get video dimensions
    args = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json", video_path
    ]
    stdout, _, rc = run_cmd(args, timeout=30)
    width, height = 1920, 1080  # defaults
    if rc == 0:
        try:
            info = json.loads(stdout)
            streams = info.get("streams", [])
            if streams:
                width = streams[0].get("width", 1920)
                height = streams[0].get("height", 1080)
        except json.JSONDecodeError:
            pass

    log.info(f"Video: {width}x{height}")
    return video_path, width, height


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        log.info(f"{self.address_string()} - {format % args}")

    def _check_auth(self):
        auth = self.headers.get("X-API-Key", "")
        if auth != API_KEY:
            self._json_response(401, {"error": "Unauthorized"})
            return False
        return True

    def _json_response(self, status, data):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-API-Key")
        self.end_headers()

    def do_GET(self):
        # Health check
        if self.path == "/health":
            self._json_response(200, {"status": "ok", "service": "shortscut-vps"})
            return

        # Download processed file
        if self.path.startswith("/download/"):
            file_id = self.path.split("/download/")[1]
            info = processed_files.get(file_id)
            if not info or not os.path.exists(info["path"]):
                self._json_response(404, {"error": "File not found"})
                return

            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(info["size"]))
            self.send_header("Content-Disposition", f'attachment; filename="clip_{file_id[:8]}.mp4"')
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            with open(info["path"], "rb") as f:
                shutil.copyfileobj(f, self.wfile)
            return

        self._json_response(404, {"error": "Not found"})

    def do_POST(self):
        if not self._check_auth():
            return

        # Process a full job (download + all clips)
        if self.path == "/process":
            try:
                body = json.loads(self._read_body())
            except (json.JSONDecodeError, ValueError) as e:
                self._json_response(400, {"error": f"Invalid JSON: {e}"})
                return

            video_url = body.get("video_url")
            audio_url = body.get("audio_url")
            youtube_url = body.get("youtube_url")
            clips = body.get("clips", [])
            provided_width = body.get("video_width")
            provided_height = body.get("video_height")

            if not (video_url or youtube_url):
                self._json_response(400, {"error": "video_url or youtube_url required"})
                return
            if not clips:
                self._json_response(400, {"error": "clips array required"})
                return

            # Acquire processing slot
            if not processing_lock.acquire(timeout=5):
                self._json_response(503, {"error": "Server busy, try again later"})
                return

            work_dir = tempfile.mkdtemp(dir=WORK_DIR)
            try:
                cleanup_old_files()

                # Download video
                dl_url = youtube_url or video_url
                input_path, det_width, det_height = download_video(dl_url, audio_url, work_dir)
                video_width = provided_width or det_width
                video_height = provided_height or det_height

                # Process each clip
                results = []
                for clip_config in clips:
                    try:
                        output_path = process_clip(
                            input_path, clip_config, video_width, video_height, work_dir
                        )
                        # Register for download
                        file_id = str(uuid.uuid4())[:12]
                        # Move to persistent location
                        final_path = os.path.join(str(WORK_DIR), f"{file_id}.mp4")
                        shutil.move(output_path, final_path)
                        size = os.path.getsize(final_path)
                        processed_files[file_id] = {
                            "path": final_path,
                            "created": time.time(),
                            "size": size,
                        }
                        results.append({
                            "index": clip_config.get("index", 0),
                            "download_url": f"/download/{file_id}",
                            "file_id": file_id,
                            "size": size,
                            "success": True,
                        })
                    except Exception as e:
                        log.error(f"Clip {clip_config.get('index', '?')} failed: {e}")
                        results.append({
                            "index": clip_config.get("index", 0),
                            "error": str(e),
                            "success": False,
                        })

                self._json_response(200, {
                    "success": True,
                    "video_width": video_width,
                    "video_height": video_height,
                    "clips": results,
                })

            except Exception as e:
                log.error(f"Processing failed: {e}")
                self._json_response(500, {"error": str(e)})
            finally:
                # Cleanup work directory (source video etc)
                shutil.rmtree(work_dir, ignore_errors=True)
                processing_lock.release()
            return

        # ── Transcript extraction via yt-dlp ──────────────────────────────
        if self.path == "/transcript":
            try:
                body = json.loads(self._read_body())
            except (json.JSONDecodeError, ValueError) as e:
                self._json_response(400, {"error": f"Invalid JSON: {e}"})
                return

            video_url = body.get("video_url") or body.get("youtube_url")
            lang = body.get("lang", "en")
            if not video_url:
                self._json_response(400, {"error": "video_url required"})
                return

            work_dir = tempfile.mkdtemp(dir=WORK_DIR)
            try:
                log.info(f"[transcript] Extracting subtitles for {video_url} (lang={lang})")

                # Use yt-dlp to extract subtitles without downloading video
                out_template = os.path.join(work_dir, "subs")
                cmd = [
                    "yt-dlp",
                    "--skip-download",
                    "--write-auto-sub",
                    "--write-sub",
                    "--sub-lang", lang,
                    "--sub-format", "json3",
                    "--output", out_template,
                    video_url,
                ]
                stdout, stderr, rc = run_cmd(cmd, timeout=120, cwd=work_dir)

                if rc != 0:
                    log.error(f"[transcript] yt-dlp failed: {stderr}")
                    self._json_response(500, {
                        "error": f"yt-dlp subtitle extraction failed (rc={rc})",
                        "details": stderr[:500],
                    })
                    return

                # Find the subtitle file (json3 format)
                sub_files = list(Path(work_dir).glob("subs*.json3"))
                if not sub_files:
                    # Try VTT fallback
                    cmd2 = [
                        "yt-dlp",
                        "--skip-download",
                        "--write-auto-sub",
                        "--write-sub",
                        "--sub-lang", lang,
                        "--sub-format", "vtt",
                        "--output", out_template,
                        video_url,
                    ]
                    stdout2, stderr2, rc2 = run_cmd(cmd2, timeout=120, cwd=work_dir)
                    sub_files = list(Path(work_dir).glob("subs*.vtt"))

                if not sub_files:
                    log.warning(f"[transcript] No subtitle files found")
                    self._json_response(404, {
                        "error": "No subtitles found for this video",
                        "details": stderr[:500],
                    })
                    return

                sub_file = sub_files[0]
                sub_content = sub_file.read_text(encoding="utf-8")
                sub_format = sub_file.suffix.lstrip(".")

                # Parse json3 format into segments
                segments = []
                if sub_format == "json3":
                    try:
                        json3_data = json.loads(sub_content)
                        events = json3_data.get("events", [])
                        for event in events:
                            if "segs" not in event:
                                continue
                            start_ms = event.get("tStartMs", 0)
                            dur_ms = event.get("dDurationMs", 0)
                            text = "".join(
                                seg.get("utf8", "") for seg in event["segs"]
                            ).strip()
                            if text and text != "\n":
                                segments.append({
                                    "start": start_ms / 1000.0,
                                    "end": (start_ms + dur_ms) / 1000.0,
                                    "text": text,
                                })
                    except json.JSONDecodeError:
                        pass

                # Parse VTT format as fallback
                if sub_format == "vtt" and not segments:
                    import re as re_mod
                    blocks = sub_content.split("\n\n")
                    for block in blocks:
                        lines = block.strip().split("\n")
                        time_match = None
                        for line in lines:
                            m = re_mod.match(
                                r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})",
                                line,
                            )
                            if m:
                                time_match = m
                                break
                        if time_match:
                            g = time_match.groups()
                            start = int(g[0]) * 3600 + int(g[1]) * 60 + int(g[2]) + int(g[3]) / 1000
                            end = int(g[4]) * 3600 + int(g[5]) * 60 + int(g[6]) + int(g[7]) / 1000
                            text_lines = []
                            found_time = False
                            for line in lines:
                                if "-->" in line:
                                    found_time = True
                                    continue
                                if found_time and line.strip():
                                    # Strip VTT tags
                                    clean = re_mod.sub(r"<[^>]+>", "", line).strip()
                                    if clean:
                                        text_lines.append(clean)
                            text = " ".join(text_lines).strip()
                            if text:
                                segments.append({
                                    "start": start,
                                    "end": end,
                                    "text": text,
                                })

                log.info(f"[transcript] Extracted {len(segments)} segments ({sub_format})")

                self._json_response(200, {
                    "success": True,
                    "format": sub_format,
                    "language": lang,
                    "segments": segments,
                    "segment_count": len(segments),
                    "raw": sub_content if len(sub_content) < 500000 else None,
                })

            except Exception as e:
                log.error(f"[transcript] Error: {e}")
                self._json_response(500, {"error": str(e)})
            finally:
                shutil.rmtree(work_dir, ignore_errors=True)
            return

        self._json_response(404, {"error": "Not found"})


class ThreadedHTTPServer(HTTPServer):
    """Handle requests in threads for concurrent /download during /process."""
    def process_request(self, request, client_address):
        t = threading.Thread(target=self._handle, args=(request, client_address))
        t.daemon = True
        t.start()

    def _handle(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)


def main():
    server = ThreadedHTTPServer(("0.0.0.0", PORT), Handler)
    log.info(f"Shortscut VPS Processing Server starting on port {PORT}")
    log.info(f"API Key: {API_KEY[:8]}...")
    log.info(f"Work dir: {WORK_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
