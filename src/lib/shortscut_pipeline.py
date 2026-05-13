#!/usr/bin/env python3
"""
ShortsCut — Podcast → YouTube Shorts (local script)
=====================================================
Downloads a YouTube video (or uses a local file), transcribes with Whisper,
uses AI to find the most viral moments, crops 9:16 with face detection,
and burns subtitles.

Usage:
    python3 shortscut.py <youtube_url_or_file> [options]
    python3 shortscut.py https://youtube.com/watch?v=... --api-key sk-...
    python3 shortscut.py /path/to/video.mp4 --api-key sk-...
    OPENAI_API_KEY=sk-... python3 shortscut.py https://youtube.com/watch?v=...

Requirements: python3 >= 3.9, ffmpeg (with libass), yt-dlp (for YouTube URLs)
Everything else is installed automatically in a venv.
"""

import argparse
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

# ─────────────────────────── constants ───────────────────────────
VENV_DIR = Path.home() / ".shortscut_venv"
REQUIRED_PACKAGES = ["openai", "opencv-python-headless", "mediapipe", "static-ffmpeg"]
MIN_DURATION = 30
MAX_DURATION = 300  # effectively no upper limit — AI decides natural end
NUM_SHORTS = 5
CANDIDATES_PER_CHUNK = 6
CHUNK_CHARS = 12_000  # ~20 min of transcript per chunk


# ─────────────────────────── venv bootstrap ──────────────────────
def ensure_venv():
    """Create venv and install deps if needed. Returns path to venv Python."""
    pip = VENV_DIR / "bin" / "pip"
    python = VENV_DIR / "bin" / "python"

    if not python.exists():
        print("📦 Creating virtual environment...")
        subprocess.check_call(
            [sys.executable, "-m", "venv", str(VENV_DIR)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        subprocess.check_call(
            [str(pip), "install", "--upgrade", "pip", "-q"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    # Check which packages need installing
    missing = []
    import_map = {
        "openai": "openai",
        "opencv-python-headless": "cv2",
        "mediapipe": "mediapipe",
        "static-ffmpeg": "static_ffmpeg",
    }
    for pkg in REQUIRED_PACKAGES:
        mod = import_map.get(pkg, pkg)
        check = subprocess.run(
            [str(python), "-c", f"import {mod}"],
            capture_output=True,
        )
        if check.returncode != 0:
            missing.append(pkg)

    if missing:
        print(f"📦 Installing: {', '.join(missing)}...")
        subprocess.check_call(
            [str(pip), "install", "-q"] + missing,
            stdout=subprocess.DEVNULL,
        )

    return str(python)


def setup_ffmpeg(python_path: str):
    """Set up ffmpeg from static-ffmpeg package (bundled in venv).
    This gives us a known-good ffmpeg with libass for subtitle burning."""
    setup_script = textwrap.dedent("""\
        import static_ffmpeg, json, os
        ffmpeg, ffprobe = static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()
        print(json.dumps({"ffmpeg": ffmpeg, "ffprobe": ffprobe, "dir": os.path.dirname(ffmpeg)}))
    """)
    r = subprocess.run(
        [python_path, "-c", setup_script],
        capture_output=True, text=True, timeout=120,
    )
    if r.returncode != 0:
        print(f"⚠️  static-ffmpeg setup failed: {r.stderr[:200]}")
        print("  Falling back to system ffmpeg...")
        return

    try:
        data = json.loads(r.stdout.strip().split("\n")[-1])
        ffmpeg_dir = data["dir"]
        # Prepend to PATH so our bundled ffmpeg is used everywhere
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
        ffmpeg_path = data["ffmpeg"]
        # Verify it has libass
        r2 = subprocess.run([ffmpeg_path, "-filters"], capture_output=True, text=True, timeout=10)
        has_ass = "libass" in r2.stdout or " ass " in r2.stdout
        print(f"✅ ffmpeg bundled (static-ffmpeg) — libass: {'✅' if has_ass else '❌'}")
        if not has_ass:
            print("  ⚠️  Subtitles may not burn correctly without libass")
    except Exception as e:
        print(f"⚠️  Could not configure static-ffmpeg: {e}")
        print("  Falling back to system ffmpeg...")


def ensure_tools(need_ytdlp: bool):
    """Check ffmpeg and yt-dlp (if downloading)."""
    if not shutil.which("ffmpeg"):
        print("❌ 'ffmpeg' not found — and static-ffmpeg didn't set it up.")
        print("   Install manually: brew install ffmpeg (macOS) / apt install ffmpeg (Linux)")
        sys.exit(1)

    # Quick libass check
    try:
        r = subprocess.run(["ffmpeg", "-filters"], capture_output=True, text=True, timeout=10)
        if "libass" not in r.stdout and " ass " not in r.stdout:
            print("⚠️  ffmpeg may not have libass. Subtitles might not burn correctly.")
            print("  macOS: brew install ffmpeg | Linux: sudo apt install ffmpeg libass-dev")
    except Exception:
        pass

    if need_ytdlp and not shutil.which("yt-dlp"):
        print("❌ 'yt-dlp' not found. Install it first.")
        print("   brew install yt-dlp   (macOS)")
        print("   pip install yt-dlp    (Linux)")
        sys.exit(1)


def is_url(s: str) -> bool:
    """Check if argument is a URL (YouTube or other) vs a local file."""
    return s.startswith("http://") or s.startswith("https://")


# ─────────────────────────── step 1: download ────────────────────
def download_video(url: str, output_dir: Path, cookies_file: str | None = None) -> Path:
    """Download video with yt-dlp."""
    print("\n🎬 Step 1/5: Downloading video...")
    output_path = output_dir / "source.%(ext)s"
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", str(output_path),
        "--no-playlist",
    ]
    if cookies_file:
        cmd.extend(["--cookies", cookies_file])
    cmd.append(url)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr
        if "Sign in to confirm" in stderr or "bot" in stderr.lower():
            print("❌ YouTube bot detection triggered!")
            print("   This happens on datacenter/VPN IPs.")
            print("   Solutions:")
            print("   1. Run this script on your local machine (home internet)")
            print("   2. Provide cookies: --cookies /path/to/cookies.txt")
            print('      Export from browser with "Get cookies.txt LOCALLY" extension')
        else:
            print(f"❌ Download failed: {stderr[:500]}")
        sys.exit(1)

    # Find the downloaded file
    for f in output_dir.iterdir():
        if f.name.startswith("source") and f.suffix == ".mp4":
            print(f"✅ Downloaded: {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")
            return f
    raise FileNotFoundError("Download failed — no source.mp4 found")


def use_local_file(file_path: str, output_dir: Path) -> Path:
    """Copy or link a local video file to the output directory."""
    print("\n🎬 Step 1/5: Using local video file...")
    src = Path(file_path)
    if not src.exists():
        print(f"❌ File not found: {file_path}")
        sys.exit(1)

    dst = output_dir / f"source{src.suffix}"
    if src.suffix.lower() != ".mp4":
        # Convert to mp4
        print(f"   Converting {src.suffix} → .mp4...")
        dst = output_dir / "source.mp4"
        subprocess.check_call(
            ["ffmpeg", "-y", "-i", str(src), "-c", "copy", str(dst)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    else:
        # Copy (don't symlink — safer)
        shutil.copy2(src, dst)

    print(f"✅ Using: {src.name} ({dst.stat().st_size / 1024 / 1024:.1f} MB)")
    return dst


# ─────────────────────────── step 2: transcribe ──────────────────
def extract_audio(video_path: Path, output_dir: Path) -> Path:
    """Extract mono 16kHz audio from video for Whisper."""
    audio_path = output_dir / "audio.mp3"
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k",
        str(audio_path),
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path


def split_audio(audio_path: Path, max_size_mb: float = 24.0) -> list[Path]:
    """Split audio into chunks if over Whisper's 25 MB limit."""
    size_mb = audio_path.stat().st_size / (1024 * 1024)
    if size_mb <= max_size_mb:
        return [audio_path]

    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
        capture_output=True, text=True,
    )
    duration = float(result.stdout.strip())
    num_chunks = math.ceil(size_mb / max_size_mb)
    chunk_duration = duration / num_chunks

    print(f"  Audio {size_mb:.0f} MB → splitting into {num_chunks} chunks")

    chunks = []
    for i in range(num_chunks):
        start = i * chunk_duration
        chunk_path = audio_path.parent / f"audio_chunk_{i:03d}.mp3"
        cmd = [
            "ffmpeg", "-y", "-i", str(audio_path),
            "-ss", str(start), "-t", str(chunk_duration),
            "-ac", "1", "-ar", "16000", "-b:a", "64k",
            str(chunk_path),
        ]
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        chunks.append(chunk_path)

    return chunks


def transcribe_with_whisper(audio_path: Path, api_key: str, output_dir: Path) -> dict:
    """Transcribe audio using OpenAI Whisper API with word-level timestamps."""
    print("\n🎙️ Step 2/5: Transcribing with Whisper...")

    # Import openai from venv
    venv_site = list((VENV_DIR / "lib").glob("python*/site-packages"))
    if venv_site:
        sys.path.insert(0, str(venv_site[0]))
    import openai

    chunks = split_audio(audio_path)
    all_segments = []
    all_words = []
    offset = 0.0

    for i, chunk in enumerate(chunks):
        if len(chunks) > 1:
            print(f"  Transcribing chunk {i + 1}/{len(chunks)}...")
        else:
            print("  Sending to Whisper API...")

        client = openai.OpenAI(api_key=api_key)

        with open(chunk, "rb") as f:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
            )

        resp_dict = response.model_dump() if hasattr(response, "model_dump") else response

        for seg in resp_dict.get("segments", []):
            all_segments.append({
                "start": seg["start"] + offset,
                "end": seg["end"] + offset,
                "text": seg["text"].strip(),
            })

        for w in resp_dict.get("words", []):
            all_words.append({
                "start": w["start"] + offset,
                "end": w["end"] + offset,
                "word": w["word"],
            })

        if len(chunks) > 1:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(chunk)],
                capture_output=True, text=True,
            )
            offset += float(result.stdout.strip())

    # Build formatted transcript text with timestamps
    lines = []
    for seg in all_segments:
        h = int(seg["start"] // 3600)
        m = int((seg["start"] % 3600) // 60)
        s = int(seg["start"] % 60)
        ts = f"[{h}:{m:02d}:{s:02d}]" if h > 0 else f"[{m:02d}:{s:02d}]"
        lines.append(f"{ts} {seg['text']}")

    transcript_text = "\n".join(lines)
    result = {
        "segments": all_segments,
        "words": all_words,
        "text": transcript_text,
    }

    # Save transcript
    with open(output_dir / "transcript.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    dur_min = all_segments[-1]["end"] / 60 if all_segments else 0
    print(f"✅ Transcription complete: {len(all_segments)} segments, {dur_min:.0f} min")
    return result


# ─────────────────────────── step 3: AI analysis ─────────────────
def split_transcript_into_chunks(text: str) -> list[str]:
    """Split transcript into chunks for AI processing."""
    lines = text.split("\n")
    chunks = []
    current = []
    current_len = 0

    for line in lines:
        current.append(line)
        current_len += len(line) + 1
        if current_len >= CHUNK_CHARS:
            chunks.append("\n".join(current))
            current = []
            current_len = 0

    if current:
        chunks.append("\n".join(current))

    return chunks


def scan_chunk_for_candidates(client, chunk: str, chunk_idx: int,
                               total_chunks: int, video_title: str,
                               language: str) -> list[dict]:
    """Scan a transcript chunk for viral clip candidates."""
    n = CANDIDATES_PER_CHUNK
    prompt = f"""You are scanning SECTION {chunk_idx + 1} of {total_chunks} from the video "{video_title}".

Find up to {n} potential viral short clips (MINIMUM {MIN_DURATION}s, NO UPPER LIMIT) in THIS section.

# DURATION RULES
- Minimum {MIN_DURATION} seconds — never shorter.
- NO maximum limit! Let the clip run as long as the moment deserves.
- End where it NATURALLY ends: after a punchline lands, after a story concludes, after a reaction settles.
- If a great moment runs 90s or even 2+ minutes, INCLUDE IT. Don't cut it short.
- Typical good clips: 30s–90s. But longer is fine if the content is compelling throughout.

# WHAT MAKES A CLIP VIRAL
1. STRONG HOOK — bold claim, surprising statement, question, or pattern interrupt in first 3 seconds
2. SELF-CONTAINED — understandable without the rest of the podcast
3. EMOTIONAL PAYLOAD — surprise, controversy, humor, vulnerability, insight
4. CLEAR PAYOFF — delivers on the hook's promise
5. QUOTABLE LINE — something people would repeat or screenshot

# FUNNY SIGNALS TO SCAN FOR
## 🎤 Punchlines & Reactions — words like "what", "wait", "no way", laughter, "haha", swearing = engagement magnet
## 🔄 Reversal Moments — setup question → unexpected answer, twist endings
## 😬 Awkward Pauses — long gaps followed by a comeback, filler words ("uh", "um") before a punchline
## 💎 Self-Roast / Quotable One-Liners — short declarative sentences that stand alone
## ⚡ Audio Peaks — rapid back-and-forth (alternating short segments), passionate disagreements
## 😂 Laughter & Comedy — [laughter], comedic timing, self-deprecation

# WHAT TO SKIP
- Needs backstory ("as I was saying earlier..."), generic advice, slow build-ups
- Mid-sentence cuts, unresolved thoughts, pure agreement/small talk

# SCORING: Most clips are 5-7. Only 1 per podcast can be 9-10.
- 9-10: Strong hook + surprising insight + quotable line + emotional charge
- 7-8: Solid hook + clear value, but missing one viral element
- 5-6: Interesting but niche, or good content with weak opening
- Below 5: Don't include it

# TIMESTAMP RULES
Use [MM:SS] or [HH:MM:SS] timestamps from the transcript → convert to total seconds.
hookLine = exact sentence copied VERBATIM from the transcript.
endSeconds = where the clip NATURALLY ends (after the payoff/reaction lands).

# LANGUAGE: titles, reasoning in {language}. hookLine verbatim from transcript.

TRANSCRIPT SECTION:
{chunk}"""

    # Build structured output schema
    properties = {}
    for i in range(1, n + 1):
        properties[f"c{i}_title"] = {"type": "string", "description": f"5-8 word viral title in {language}"}
        properties[f"c{i}_hookLine"] = {"type": "string", "description": "Exact opening sentence from transcript"}
        properties[f"c{i}_startSeconds"] = {"type": "integer", "description": "Start time in total seconds"}
        properties[f"c{i}_endSeconds"] = {"type": "integer", "description": "End time in total seconds"}
        properties[f"c{i}_viralScore"] = {"type": "integer", "description": "Viral potential 1-10"}
        properties[f"c{i}_reason"] = {"type": "string", "description": f"Why this would go viral, in {language}"}
    properties["noGoodMoments"] = {
        "type": "string",
        "description": "Set to 'true' if no viral moments found (all scores < 5). Otherwise empty string.",
    }

    try:
        response = client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "clip_candidates",
                    "strict": False,
                    "schema": {
                        "type": "object",
                        "properties": properties,
                    },
                },
            },
            temperature=0.7,
        )
        data = json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"  ⚠️  AI scan failed for chunk {chunk_idx + 1}: {e}")
        return []

    if str(data.get("noGoodMoments", "")).lower() == "true":
        return []

    candidates = []
    for i in range(1, n + 1):
        title = data.get(f"c{i}_title", "")
        start = int(data.get(f"c{i}_startSeconds", 0) or 0)
        end = int(data.get(f"c{i}_endSeconds", 0) or 0)
        score = int(data.get(f"c{i}_viralScore", 0) or 0)
        if not title or start == 0 or score < 5:
            continue
        if end <= start:
            end = start + 60  # default 60s if AI didn't set end
        dur = end - start
        if dur < MIN_DURATION:
            end = start + MIN_DURATION
        candidates.append({
            "title": title,
            "hookLine": data.get(f"c{i}_hookLine", ""),
            "startTime": start,
            "endTime": end,
            "viralScore": min(10, max(1, score)),
            "reason": data.get(f"c{i}_reason", ""),
        })

    return candidates


def correct_timestamps(transcript_text: str, clips: list[dict]) -> list[dict]:
    """Correct AI timestamps by searching for hookLine in actual transcript."""
    ts_regex = re.compile(r"\[(\d{1,2}:\d{2}(?::\d{2})?)\]")
    timestamp_index = []

    for match in ts_regex.finditer(transcript_text):
        parts = match.group(1).split(":")
        if len(parts) == 3:
            secs = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        else:
            secs = int(parts[0]) * 60 + int(parts[1])
        timestamp_index.append({"pos": match.start(), "seconds": secs})

    if not timestamp_index:
        return clips

    corrected = []
    for clip in clips:
        hook = clip.get("hookLine", "")
        if not hook or len(hook) < 10:
            corrected.append(clip)
            continue

        # Try progressively shorter prefixes of the hook line
        hook_pos = -1
        for length in [len(hook), 80, 50, 30]:
            variant = hook[:length]
            if len(variant) < 10:
                continue
            idx = transcript_text.lower().find(variant.lower())
            if idx >= 0:
                hook_pos = idx
                break

        if hook_pos < 0:
            # Try fuzzy: search for any 15-char substring
            for start_i in range(0, min(len(hook) - 15, 50), 10):
                sub = hook[start_i:start_i + 15].lower()
                idx = transcript_text.lower().find(sub)
                if idx >= 0:
                    hook_pos = idx
                    break

        if hook_pos < 0:
            corrected.append(clip)
            continue

        # Find nearest timestamp before this position
        nearest = None
        for ts in reversed(timestamp_index):
            if ts["pos"] <= hook_pos:
                nearest = ts
                break

        if not nearest:
            corrected.append(clip)
            continue

        dur = clip["endTime"] - clip["startTime"]
        corrected.append({
            **clip,
            "startTime": nearest["seconds"],
            "endTime": nearest["seconds"] + dur,
        })

    return corrected


def deduplicate_candidates(candidates: list[dict]) -> list[dict]:
    """Remove overlapping candidates, keeping highest score."""
    sorted_c = sorted(candidates, key=lambda c: c["startTime"])
    result = []

    for c in sorted_c:
        overlap = next(
            (i for i, e in enumerate(result) if abs(e["startTime"] - c["startTime"]) < 15),
            None,
        )
        if overlap is not None:
            if c["viralScore"] > result[overlap]["viralScore"]:
                result[overlap] = c
        else:
            result.append(c)

    return result


def select_best_clips(client, candidates: list[dict], video_title: str,
                      language: str, num_shorts: int) -> list[dict]:
    """Use AI to select the best clips from all candidates."""
    summary = "\n".join(
        f'[{i+1}] "{c["title"]}" | {c["startTime"]}s–{c["endTime"]}s | '
        f'Score: {c["viralScore"]}/10 | Hook: "{c["hookLine"][:80]}" | '
        f'Why: {c["reason"][:100]}'
        for i, c in enumerate(candidates)
    )

    properties = {}
    for i in range(1, num_shorts + 1):
        properties[f"pick{i}_number"] = {"type": "integer", "description": f"Candidate number [N] for pick {i}"}
        properties[f"pick{i}_title"] = {"type": "string", "description": f"Polished title in {language}"}
        properties[f"pick{i}_score"] = {"type": "integer", "description": "Final viral score 1-10"}

    try:
        response = client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[{"role": "user", "content": f"""Select the BEST {num_shorts} clips from {len(candidates)} candidates for "{video_title}".
Consider: VARIETY of topics, QUALITY of hooks, NO time OVERLAP, HUMOR, REVERSALS, CONTROVERSY.
TITLE: 5-8 words, curiosity gap, power words, no spoilers. All text in {language}.
SCORING: Be honest. Most 6-8, max 1 at 9-10.

Candidates:
{summary}"""}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "best_clips",
                    "strict": False,
                    "schema": {"type": "object", "properties": properties},
                },
            },
            temperature=0.5,
        )
        data = json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"  ⚠️  Selection failed: {e} — using top scores instead")
        return sorted(candidates, key=lambda c: -c["viralScore"])[:num_shorts]

    picks = []
    used = set()
    for i in range(1, num_shorts + 1):
        num = int(data.get(f"pick{i}_number", 0) or 0)
        if num < 1 or num > len(candidates) or num in used:
            continue
        used.add(num)
        orig = candidates[num - 1]
        picks.append({
            **orig,
            "title": data.get(f"pick{i}_title", orig["title"]),
            "viralScore": min(10, max(1, int(data.get(f"pick{i}_score", orig["viralScore"]) or orig["viralScore"]))),
        })

    # If AI didn't return enough, fill from top scores
    if len(picks) < num_shorts:
        remaining = [c for c in candidates if c not in picks]
        remaining.sort(key=lambda c: -c["viralScore"])
        for c in remaining:
            if len(picks) >= num_shorts:
                break
            if not any(abs(c["startTime"] - p["startTime"]) < 15 for p in picks):
                picks.append(c)

    return picks


def analyze_transcript(api_key: str, transcript: dict, video_title: str,
                       language: str, num_shorts: int) -> list[dict]:
    """Full two-pass AI analysis: scan chunks → dedupe → select best."""
    print("\n🧠 Step 3/5: AI analysis — finding viral moments...")

    venv_site = list((VENV_DIR / "lib").glob("python*/site-packages"))
    if venv_site:
        for sp in venv_site:
            if str(sp) not in sys.path:
                sys.path.insert(0, str(sp))
    import openai
    client = openai.OpenAI(api_key=api_key)

    chunks = split_transcript_into_chunks(transcript["text"])
    print(f"  Scanning {len(chunks)} transcript sections...")

    all_candidates = []
    for i, chunk in enumerate(chunks):
        print(f"  📡 Section {i + 1}/{len(chunks)}...", end="", flush=True)
        candidates = scan_chunk_for_candidates(client, chunk, i, len(chunks), video_title, language)
        print(f" → {len(candidates)} candidates")
        all_candidates.extend(candidates)

    if not all_candidates:
        print("❌ No viral moments found in any section!")
        return []

    # Correct timestamps using actual transcript positions
    all_candidates = correct_timestamps(transcript["text"], all_candidates)

    # Deduplicate overlapping candidates
    deduped = deduplicate_candidates(all_candidates)
    print(f"  📊 {len(deduped)} unique candidates after dedup (from {len(all_candidates)} total)")

    # Select the best clips
    print(f"  🏆 Selecting top {num_shorts}...")
    selected = select_best_clips(client, deduped, video_title, language, num_shorts)

    # Final timestamp correction pass
    selected = correct_timestamps(transcript["text"], selected)

    print(f"\n✅ Selected {len(selected)} clips:")
    for i, clip in enumerate(selected):
        m, s = divmod(clip["startTime"], 60)
        h, m = divmod(int(m), 60)
        ts = f"{h}:{m:02d}:{int(s):02d}" if h else f"{m:02d}:{int(s):02d}"
        dur = clip["endTime"] - clip["startTime"]
        print(f"  {i+1}. [{ts} +{dur}s] {clip['title']} (Score: {clip['viralScore']}/10)")

    return selected


# ─────────────────────────── step 4: face detection ──────────────
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
import json, sys, os, statistics

frame_paths = {frame_paths_json}
src_w = {src_w}
src_h = {src_h}

def log(msg):
    print(f"DIAG: {{msg}}", file=sys.stderr)

# Per-frame results: list of face_x_normalized or None
results = [None] * len(frame_paths)

# ── METHOD 1: MediaPipe face detection ──
mp_count = 0
try:
    import cv2
    import mediapipe as mp
    log(f"MediaPipe {{mp.__version__}} + OpenCV {{cv2.__version__}}, {{len(frame_paths)}} frames")

    detectors = [
        mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.3),
        mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.3),
    ]

    for idx, fp in enumerate(frame_paths):
        img = cv2.imread(fp)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_xs = []
        for fd in detectors:
            res = fd.process(rgb)
            if res.detections:
                for d in res.detections:
                    b = d.location_data.relative_bounding_box
                    if b.width > 0.02 and b.height > 0.02:
                        frame_xs.append(b.xmin + b.width / 2)
        if frame_xs:
            results[idx] = statistics.median(frame_xs)
            mp_count += 1

    for fd in detectors:
        fd.close()
    log(f"MediaPipe: {{mp_count}}/{{len(frame_paths)}} frames")
except Exception as e:
    log(f"MediaPipe error: {{e}}")

# ── METHOD 2: Haar cascade fallback for frames where MP failed ──
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
            faces = face_cascade.detectMultiScale(gray, 1.05, 3,
                minSize=(int(src_w * 0.03), int(src_h * 0.03)))
            if len(faces) > 0:
                areas = [w * h for (x, y, w, h) in faces]
                best = faces[areas.index(max(areas))]
                results[idx] = (best[0] + best[2] / 2) / src_w
                haar_count += 1
        if haar_count:
            log(f"Haar: filled {{haar_count}} more frames")
except Exception as e:
    log(f"Haar error: {{e}}")

# ── METHOD 3: Edge analysis for remaining gaps ──
edge_count = 0
try:
    import cv2
    for idx, fp in enumerate(frame_paths):
        if results[idx] is not None:
            continue
        img = cv2.imread(fp)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h_img, w_img = gray.shape
        strip_w = w_img // 10
        scores = []
        for s in range(10):
            strip = gray[:, s * strip_w:(s + 1) * strip_w]
            lap = cv2.Laplacian(strip, cv2.CV_64F)
            scores.append(lap.var())
        best = max(range(10), key=lambda i: scores[i])
        results[idx] = (best + 0.5) / 10
        edge_count += 1
    if edge_count:
        log(f"Edge analysis: filled {{edge_count}} more frames")
except Exception as e:
    log(f"Edge error: {{e}}")

detected = sum(1 for r in results if r is not None)
log(f"Total: {{detected}}/{{len(results)}} frames with position")
print(json.dumps({{"results": results, "mp": mp_count, "haar": haar_count, "edge": edge_count}}))
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
    n_detected = sum(1 for r in raw if r is not None)

    if n_detected < 2:
        print(f"    ⚠️  Only {n_detected} face detections — using center")
        return {"mode": "center", "crop_x": (src_w - crop_w) // 2}

    # ── Fill gaps by nearest-neighbor interpolation ──
    filled = list(raw)
    last_good = None
    for i in range(len(filled)):
        if filled[i] is not None:
            last_good = filled[i]
        elif last_good is not None:
            filled[i] = last_good
    last_good = None
    for i in range(len(filled) - 1, -1, -1):
        if filled[i] is not None:
            last_good = filled[i]
        elif last_good is not None:
            filled[i] = last_good

    if any(v is None for v in filled):
        return {"mode": "center", "crop_x": (src_w - crop_w) // 2}

    # ── Check for dual-face scenario ──
    vals_sorted = sorted(filled)
    clusters, cur = [], [vals_sorted[0]]
    for v in vals_sorted[1:]:
        if abs(v - cur[-1]) < 0.15:
            cur.append(v)
        else:
            clusters.append(cur)
            cur = [v]
    clusters.append(cur)
    if len(clusters) >= 2:
        c1 = statistics.median(clusters[0])
        c2 = statistics.median(clusters[1])
        if abs(c1 - c2) > (crop_w / src_w) * 0.8:
            return {"mode": "dual", "face1_x": min(c1, c2), "face2_x": max(c1, c2)}

    # ══════════════════════════════════════════════════════════════════
    # VIRTUAL CAMERA MODEL — damped spring with deadzone + velocity cap
    # Mimics a human camera operator: smooth, no jitter, natural inertia
    # ══════════════════════════════════════════════════════════════════

    max_crop_x = src_w - crop_w

    # ── 1. Pre-smooth raw detections with EMA (low-pass filter) ──
    # Alpha 0.10 ≈ heavy smoothing; kills detection noise/jitter
    ema_alpha = 0.10
    ema = [filled[0]]
    for i in range(1, len(filled)):
        ema.append(ema_alpha * filled[i] + (1 - ema_alpha) * ema[-1])
    # Forward-backward EMA to remove phase delay
    ema_back = [0.0] * len(ema)
    ema_back[-1] = ema[-1]
    for i in range(len(ema) - 2, -1, -1):
        ema_back[i] = ema_alpha * ema[i] + (1 - ema_alpha) * ema_back[i + 1]
    smoothed = [(ema[i] + ema_back[i]) / 2 for i in range(len(ema))]

    # ── 2. Convert to raw target pixel positions ──
    raw_targets = []
    for x_norm in smoothed:
        cx = int(x_norm * src_w - crop_w * 0.45)
        cx = max(0, min(cx, max_crop_x))
        raw_targets.append(cx)

    # ── 3. Virtual camera simulation ──
    # Deadzone: don't move unless target is > deadzone_px away from current position
    # Velocity cap: max pixels per second the crop can move (smooth panning)
    # Damping: exponential approach to target (never overshoot)
    deadzone_px = max(20, int(crop_w * 0.04))  # ~4% of crop width (~43px for 1080-from-1920)
    max_velocity = crop_w * 0.5  # max ~half crop width per second (smooth pan speed)
    approach_rate = 3.0  # damping factor (higher = faster approach once outside deadzone)

    cam_pos = float(raw_targets[0])  # current virtual camera position
    cam_positions = [cam_pos]

    for i in range(1, len(raw_targets)):
        target = float(raw_targets[i])
        delta = target - cam_pos
        dt = frame_interval

        # Deadzone: if target is within deadzone, camera stays put
        if abs(delta) <= deadzone_px:
            cam_positions.append(cam_pos)
            continue

        # Outside deadzone: move towards target with damping
        # Effective delta excludes the deadzone band (so movement starts gently)
        effective_delta = delta - (deadzone_px if delta > 0 else -deadzone_px)
        desired_velocity = effective_delta * approach_rate  # pixels/sec

        # Clamp velocity
        if abs(desired_velocity) > max_velocity:
            desired_velocity = max_velocity if desired_velocity > 0 else -max_velocity

        # Apply movement
        movement = desired_velocity * dt
        cam_pos += movement
        cam_pos = max(0, min(cam_pos, max_crop_x))
        cam_positions.append(cam_pos)

    crop_positions = [int(round(p)) for p in cam_positions]

    # ── 4. Reduce to keyframes (only meaningful movements > 25px) ──
    keyframes = [(0.0, crop_positions[0])]
    for i in range(1, len(crop_positions)):
        t = i * frame_interval
        if abs(crop_positions[i] - keyframes[-1][1]) > 25 or i == len(crop_positions) - 1:
            keyframes.append((t, crop_positions[i]))

    # If nearly static (range < 30px), use simple single mode
    min_x = min(kf[1] for kf in keyframes)
    max_x_val = max(kf[1] for kf in keyframes)
    if max_x_val - min_x < 30:
        avg_x = int(statistics.mean([kf[1] for kf in keyframes]))
        avg_x = max(0, min(avg_x, max_crop_x))
        return {"mode": "single", "crop_x": avg_x, "detections": n_detected}

    print(f"    📹 Tracking: {len(keyframes)} keyframes, x range {min_x}-{max_x_val}px (deadzone={deadzone_px}px)")
    return {"mode": "tracking", "keyframes": keyframes}


def build_tracking_crop_expr(keyframes: list, max_crop_x: int) -> str:
    """Build ffmpeg crop x expression for dynamic face tracking.
    keyframes: list of (time_in_seconds, crop_x_pixels).
    Returns escaped expression string for use in ffmpeg -vf crop filter.
    Commas are escaped as \\, for ffmpeg filter-graph parsing.
    """
    if len(keyframes) <= 1:
        return str(int(keyframes[0][1]) if keyframes else max_crop_x // 2)

    # Build nested if() with piecewise linear interpolation
    expr = str(int(round(keyframes[-1][1])))
    for i in range(len(keyframes) - 2, -1, -1):
        t0, x0 = keyframes[i]
        t1, x1 = keyframes[i + 1]
        dt = t1 - t0
        if dt < 0.01 or abs(x1 - x0) < 3:
            seg = str(int(round(x0)))
        else:
            dx = int(round(x1 - x0))
            base = int(round(x0))
            if dx >= 0:
                seg = f"{base}+{dx}*(t-{t0:.2f})/{dt:.2f}"
            else:
                seg = f"{base}-{abs(dx)}*(t-{t0:.2f})/{dt:.2f}"
        # Escape commas for ffmpeg filter context
        expr = f"if(lt(t\\,{t1:.2f})\\,{seg}\\,{expr})"

    # Clamp to valid range
    expr = f"clip({expr}\\,0\\,{max_crop_x})"
    return expr



# ─────────────────────────── step 5: generate shorts ─────────────
def generate_ass_subtitles(words: list[dict], start_time: float,
                           end_time: float, out_w: int, out_h: int) -> str:
    """Generate ASS subtitle file with word-grouped lines."""
    clip_words = [
        w for w in words
        if w["start"] >= start_time - 0.5 and w["end"] <= end_time + 0.5
    ]

    font_size = int(out_h * 0.052)
    outline = int(out_h * 0.004)
    shadow = int(out_h * 0.003)
    margin_v = int(out_h * 0.15)

    ass = f"""[Script Info]
Title: ShortsCut Subtitles
ScriptType: v4.00+
PlayResX: {out_w}
PlayResY: {out_h}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial Black,{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{outline},{shadow},2,40,40,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    # Group words into lines of ~4 words
    words_per_line = 4
    for i in range(0, len(clip_words), words_per_line):
        group = clip_words[i:i + words_per_line]
        if not group:
            continue

        line_start = max(0, group[0]["start"] - start_time)
        line_end = group[-1]["end"] - start_time
        if line_end <= line_start:
            line_end = line_start + 0.5

        def fmt_ass_time(t):
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            s = int(t % 60)
            cs = int((t % 1) * 100)
            return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

        text = " ".join(w["word"].strip() for w in group)
        text = text.replace("{", "").replace("}", "").replace("\\", "")
        text = text.upper()

        ass += f"Dialogue: 0,{fmt_ass_time(line_start)},{fmt_ass_time(line_end)},Default,,0,0,0,,{text}\n"

    return ass


def generate_shorts(video_path: Path, clips: list[dict], transcript: dict,
                    output_dir: Path, python_path: str,
                    skip_face: bool = False) -> list[Path]:
    """Generate final 9:16 shorts with face crop (or split-screen) and burned subtitles."""
    # Get video dimensions
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-of", "csv=s=x:p=0", str(video_path)],
        capture_output=True, text=True,
    )
    dims = result.stdout.strip().split("x")
    src_w, src_h = int(dims[0]), int(dims[1])

    # 9:16 crop dimensions
    crop_w = int(src_h * 9 / 16)
    crop_h = src_h
    out_w, out_h = 1080, 1920

    # Ensure crop_w doesn't exceed source width
    if crop_w > src_w:
        crop_w = src_w

    print(f"\n🎯 Step 4/5: Face detection...")
    print(f"  📐 Source: {src_w}×{src_h} → Crop: {crop_w}×{crop_h} → Output: {out_w}×{out_h}")

    # Detect face positions for each clip
    face_results = []
    for i, clip in enumerate(clips):
        dur = clip["endTime"] - clip["startTime"]
        print(f"  🎯 Clip {i+1}/{len(clips)}...", end="", flush=True)

        if skip_face:
            face_results.append({"mode": "center", "crop_x": (src_w - crop_w) // 2})
            print(f" center crop (skipped)")
        else:
            result = detect_faces_for_clip(
                python_path, video_path,
                clip["startTime"], dur, src_w, src_h, crop_w,
            )
            face_results.append(result)
            if result["mode"] == "dual":
                print(f" 👥 TWO faces → split-screen")
            elif result["mode"] == "tracking":
                nk = len(result["keyframes"])
                print(f" 📹 face tracking ({nk} keyframes)")
            elif result["mode"] == "single":
                det = result.get('detections', '?')
                print(f" 🎯 face detected ({det} frames) → static crop_x={result['crop_x']}")
            else:
                print(f" ❌ no face → center crop")

    # Generate shorts
    print(f"\n✂️  Step 5/5: Cutting {len(clips)} shorts...")
    output_files = []

    for i, clip in enumerate(clips):
        start = clip["startTime"]
        dur = clip["endTime"] - clip["startTime"]
        face_info = face_results[i]

        # Clean title for filename
        clean_title = re.sub(r"[^\w\s-]", "", clip["title"])[:45]
        clean_title = re.sub(r"\s+", "_", clean_title).strip("_")
        output_file = output_dir / f"{i+1:02d}_{clean_title}.mp4"

        m, s = divmod(start, 60)
        h, m = divmod(int(m), 60)
        ts_str = f"{h}:{m:02d}:{int(s):02d}" if h else f"{m:02d}:{int(s):02d}"
        print(f"\n  ✂️  Short {i+1}/{len(clips)}: {clip['title']}")
        print(f"     {ts_str} → +{dur}s (Score: {clip['viralScore']}/10)")

        # Generate ASS subtitles from Whisper transcript
        ass_content = generate_ass_subtitles(
            transcript["words"], start, start + dur, out_w, out_h,
        )
        ass_path = output_dir / f"sub_{i+1:02d}.ass"
        with open(ass_path, "w", encoding="utf-8") as f:
            f.write(ass_content)

        # Escape the ASS path for ffmpeg filter graph
        # ffmpeg filter escaping: \ → \\, : → \:, ' → \', [ → \[, ] → \]
        ass_esc = (str(ass_path)
                   .replace("\\", "\\\\")
                   .replace(":", "\\:")
                   .replace("'", "\\'")
                   .replace("[", "\\[")
                   .replace("]", "\\]"))

        is_dual = face_info["mode"] == "dual"
        is_tracking = face_info["mode"] == "tracking"

        if is_dual:
            # ── SPLIT-SCREEN: two faces stacked vertically ──
            f1_x = face_info["face1_x"]  # normalized 0-1
            f2_x = face_info["face2_x"]  # normalized 0-1

            # Each half: crop around each face, scale to 1080x960
            half_h = out_h // 2  # 960
            cx1 = int(f1_x * src_w - crop_w / 2)
            cx1 = max(0, min(cx1, src_w - crop_w))
            cx2 = int(f2_x * src_w - crop_w / 2)
            cx2 = max(0, min(cx2, src_w - crop_w))
            print(f"     👥 Split-screen: face1@{cx1}, face2@{cx2}")
        elif is_tracking:
            # ── DYNAMIC FACE TRACKING ──
            tracking_expr = build_tracking_crop_expr(
                face_info["keyframes"], src_w - crop_w)
            print(f"     📹 Dynamic crop: {len(face_info['keyframes'])} keyframes")
        else:
            crop_x = face_info.get("crop_x", (src_w - crop_w) // 2)

        # ── Build ffmpeg command with subtitle burning ──
        # Try ass= filter first, then subtitles= filter, then no subs
        success = False
        for sub_filter_name in ["ass", "subtitles", None]:
            if success:
                break

            if is_dual:
                base_filter = (
                    f"[0:v]split=2[top][bot];"
                    f"[top]crop={crop_w}:{crop_h}:{cx1}:0,scale={out_w}:{half_h}[t];"
                    f"[bot]crop={crop_w}:{crop_h}:{cx2}:0,scale={out_w}:{half_h}[b];"
                    f"[t][b]vstack[stacked]"
                )
                if sub_filter_name:
                    vf = f"{base_filter};[stacked]{sub_filter_name}={ass_esc}[v]"
                else:
                    vf = (
                        f"[0:v]split=2[top][bot];"
                        f"[top]crop={crop_w}:{crop_h}:{cx1}:0,scale={out_w}:{half_h}[t];"
                        f"[bot]crop={crop_w}:{crop_h}:{cx2}:0,scale={out_w}:{half_h}[b];"
                        f"[t][b]vstack[v]"
                    )

                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(start), "-i", str(video_path), "-t", str(dur),
                    "-filter_complex", vf,
                    "-map", "[v]", "-map", "0:a",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "23",
                    "-c:a", "aac", "-b:a", "128k",
                    "-movflags", "+faststart",
                    str(output_file),
                ]
            elif is_tracking:
                # Dynamic face-following crop — expression-based crop x
                crop_filter = f"crop={crop_w}:{crop_h}:{tracking_expr}:0,scale={out_w}:{out_h}"
                if sub_filter_name:
                    vf = f"{crop_filter},{sub_filter_name}={ass_esc}"
                else:
                    vf = crop_filter

                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(start), "-i", str(video_path), "-t", str(dur),
                    "-vf", vf,
                    "-c:v", "libx264", "-preset", "medium", "-crf", "23",
                    "-c:a", "aac", "-b:a", "128k",
                    "-movflags", "+faststart",
                    str(output_file),
                ]
            else:
                # Single/center mode — static crop
                crop_filter = f"crop={crop_w}:{crop_h}:{crop_x}:0,scale={out_w}:{out_h}"
                if sub_filter_name:
                    vf = f"{crop_filter},{sub_filter_name}={ass_esc}"
                else:
                    vf = crop_filter

                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(start), "-i", str(video_path), "-t", str(dur),
                    "-vf", vf,
                    "-c:v", "libx264", "-preset", "medium", "-crf", "23",
                    "-c:a", "aac", "-b:a", "128k",
                    "-movflags", "+faststart",
                    str(output_file),
                ]

            label = f"[{sub_filter_name}]" if sub_filter_name else "[no subs]"
            try:
                proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
                if proc.returncode != 0:
                    raise subprocess.CalledProcessError(proc.returncode, cmd, stderr=proc.stderr)
                size_mb = output_file.stat().st_size / (1024 * 1024)
                suffix = "" if sub_filter_name == "ass" else f" {label}"
                print(f"     ✅ {output_file.name} ({size_mb:.1f} MB){suffix}")
                output_files.append(output_file)
                success = True
            except subprocess.CalledProcessError as e:
                stderr_msg = getattr(e, 'stderr', '') or ''
                if stderr_msg:
                    err_lines = [l for l in stderr_msg.strip().split('\n') if l.strip()][-2:]
                    for el in err_lines:
                        print(f"     ⚠️  {el.strip()}")
                if sub_filter_name == "ass":
                    print(f"     ⚠️  ass filter failed, trying subtitles= filter...")
                elif sub_filter_name == "subtitles":
                    print(f"     ⚠️  subtitles filter failed, rendering without subs...")
                else:
                    print(f"     ❌ Failed even without subtitles: {e}")

    return output_files


# ─────────────────────────── main ────────────────────────────────
def get_video_title(url: str) -> str:
    """Get video title via yt-dlp."""
    try:
        result = subprocess.run(
            ["yt-dlp", "--get-title", "--no-playlist", url],
            capture_output=True, text=True, timeout=30,
        )
        return result.stdout.strip() or "Podcast"
    except Exception:
        return "Podcast"


def main():
    parser = argparse.ArgumentParser(
        description="ShortsCut — Turn podcasts into viral YouTube Shorts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python3 shortscut.py https://youtube.com/watch?v=... --api-key sk-...
              python3 shortscut.py /path/to/video.mp4 --api-key sk-...
              python3 shortscut.py https://youtube.com/watch?v=... --cookies cookies.txt
              OPENAI_API_KEY=sk-... python3 shortscut.py https://youtube.com/watch?v=...
        """),
    )
    parser.add_argument("source", help="YouTube URL or local video file path")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--cookies", help="YouTube cookies file (Netscape format) for bot detection bypass")
    parser.add_argument("--language", default="en", help="Language for titles/analysis (default: en)")
    parser.add_argument("--num-shorts", type=int, default=NUM_SHORTS, help=f"Number of shorts to generate (default: {NUM_SHORTS})")
    parser.add_argument("--min-duration", type=int, default=MIN_DURATION, help=f"Min clip duration in seconds (default: {MIN_DURATION})")
    parser.add_argument("--max-duration", type=int, default=MAX_DURATION, help=f"Max clip duration in seconds (default: {MAX_DURATION}, no real limit)")
    parser.add_argument("--output-dir", help="Output directory (default: shorts_YYYYMMDD_HHMMSS/)")
    parser.add_argument("--skip-face-detection", action="store_true", help="Skip face detection (center crop)")
    args = parser.parse_args()

    # Validate API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API key required.")
        print("   Pass --api-key sk-... or set OPENAI_API_KEY env var.")
        sys.exit(1)

    # Override module-level config from CLI args
    _g = globals()
    _g["MIN_DURATION"] = args.min_duration
    _g["MAX_DURATION"] = args.max_duration
    _g["NUM_SHORTS"] = args.num_shorts

    is_youtube = is_url(args.source)

    print("=" * 60)
    print("🎬 ShortsCut — Podcast → YouTube Shorts")
    print("=" * 60)

    # Setup venv (includes static-ffmpeg)
    python_path = ensure_venv()
    print(f"✅ Environment ready ({VENV_DIR})")

    # Setup bundled ffmpeg from static-ffmpeg (in venv)
    setup_ffmpeg(python_path)

    # Check tools (ffmpeg should now be available via static-ffmpeg or system)
    ensure_tools(need_ytdlp=is_youtube)

    # Get video title
    if is_youtube:
        print(f"\n📋 Fetching video info...")
        video_title = get_video_title(args.source)
    else:
        video_title = Path(args.source).stem.replace("_", " ").replace("-", " ").title()
    print(f"   Title: {video_title}")

    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"shorts_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"   Output: {output_dir}/")

    # Step 1: Get video
    if is_youtube:
        video_path = download_video(args.source, output_dir, args.cookies)
    else:
        video_path = use_local_file(args.source, output_dir)

    # Step 2: Transcribe
    audio_path = extract_audio(video_path, output_dir)
    transcript = transcribe_with_whisper(audio_path, api_key, output_dir)

    # Step 3: AI analysis
    clips = analyze_transcript(api_key, transcript, video_title, args.language, args.num_shorts)
    if not clips:
        print("\n❌ No clips found. Try a different video or lower the duration thresholds.")
        sys.exit(1)

    # Save clips metadata
    with open(output_dir / "clips.json", "w", encoding="utf-8") as f:
        json.dump(clips, f, indent=2, ensure_ascii=False)

    # Steps 4 & 5: Face detection + Generate shorts
    output_files = generate_shorts(
        video_path, clips, transcript, output_dir, python_path,
        skip_face=args.skip_face_detection,
    )

    # Summary
    print("\n" + "=" * 60)
    print(f"🎉 Done! {len(output_files)} shorts generated in {output_dir}/")
    print("=" * 60)
    for f in output_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  📹 {f.name} ({size_mb:.1f} MB)")

    # Cleanup temp files
    audio_path.unlink(missing_ok=True)
    for chunk in output_dir.glob("audio_chunk_*.mp3"):
        chunk.unlink(missing_ok=True)
    for sub in output_dir.glob("sub_*.ass"):
        sub.unlink(missing_ok=True)

    print(f"\n💡 Upload to YouTube Shorts and watch them go viral! 🚀")


if __name__ == "__main__":
    main()
