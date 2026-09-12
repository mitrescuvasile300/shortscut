"""
Whisper-based, word-grouped subtitles for the VPS processor.

Mirrors the local pipeline (src/lib/shortscut_pipeline.py):
  * transcribe_with_whisper()  -> OpenAI whisper-1, word-level timestamps,
                                  language auto-detected (= original language)
  * generate_ass_subtitles()   -> 4 words per line, UPPERCASE, Arial Black,
                                  sizes as fractions of the output height

Only the clip's own audio is sent to Whisper (a few seconds to a few
minutes), so this is cheap and fast and never hits the 25 MB upload limit
for normal Shorts lengths. For very long clips the audio is chunked.
"""
import json
import logging
import os
import subprocess

import requests

log = logging.getLogger("shortscut-processor")

WHISPER_URL = "https://api.openai.com/v1/audio/transcriptions"
MAX_CHUNK_BYTES = 24 * 1024 * 1024
CHUNK_SECONDS = 600  # 10 min of 64 kbps mono mp3 ~= 4.8 MB


def extract_clip_audio(input_path, start_time, duration, work_dir, index=0):
    """Extract the clip's audio as mono 16 kHz mp3 (what the local script feeds Whisper)."""
    out = os.path.join(work_dir, f"whisper_audio_{index}.mp3")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start_time:.3f}", "-i", input_path, "-t", f"{duration:.3f}",
        "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k", out,
    ]
    subprocess.run(cmd, check=True, timeout=300, capture_output=True)
    if not os.path.exists(out) or os.path.getsize(out) < 1000:
        raise RuntimeError("audio extraction produced an empty file")
    return out


def _split_if_needed(audio_path, duration, work_dir, index):
    """Return [(path, offset_seconds), ...]; split only if over Whisper's size limit."""
    if os.path.getsize(audio_path) <= MAX_CHUNK_BYTES:
        return [(audio_path, 0.0)]
    parts = []
    t = 0.0
    n = 0
    while t < duration:
        p = os.path.join(work_dir, f"whisper_audio_{index}_{n}.mp3")
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{t:.3f}", "-i", audio_path, "-t", str(CHUNK_SECONDS),
            "-c", "copy", p,
        ], check=True, timeout=120, capture_output=True)
        parts.append((p, t))
        t += CHUNK_SECONDS
        n += 1
    return parts


def transcribe_words(audio_path, api_key, duration, work_dir, index=0, language=None):
    """Whisper word-level transcription. Returns (words, detected_language).

    `language` is intentionally None by default so Whisper detects the
    video's original language instead of being forced to the UI language.
    """
    words = []
    detected = None
    for part_path, offset in _split_if_needed(audio_path, duration, work_dir, index):
        data = {
            "model": "whisper-1",
            "response_format": "verbose_json",
            "timestamp_granularities[]": ["word", "segment"],
        }
        if language:
            data["language"] = language
        with open(part_path, "rb") as f:
            resp = requests.post(
                WHISPER_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                data=data,
                files={"file": (os.path.basename(part_path), f, "audio/mpeg")},
                timeout=300,
            )
        if resp.status_code != 200:
            raise RuntimeError(f"Whisper HTTP {resp.status_code}: {resp.text[:300]}")
        payload = resp.json()
        detected = detected or payload.get("language")
        for w in payload.get("words", []):
            words.append({
                "start": float(w["start"]) + offset,
                "end": float(w["end"]) + offset,
                "word": w["word"],
            })
    return words, detected


def _fmt_ass_time(t):
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    cs = int((t % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def generate_ass_subtitles(words, start_time, end_time, out_w, out_h,
                           words_per_line=4, font_name="Arial Black"):
    """Identical layout to the local script's generate_ass_subtitles()."""
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
Style: Default,{font_name},{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{outline},{shadow},2,40,40,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    for i in range(0, len(clip_words), words_per_line):
        group = clip_words[i:i + words_per_line]
        if not group:
            continue
        line_start = max(0, group[0]["start"] - start_time)
        line_end = group[-1]["end"] - start_time
        if line_end <= line_start:
            line_end = line_start + 0.5
        text = " ".join(w["word"].strip() for w in group)
        text = text.replace("{", "").replace("}", "").replace("\\", "")
        text = text.upper()
        ass += f"Dialogue: 0,{_fmt_ass_time(line_start)},{_fmt_ass_time(line_end)},Default,,0,0,0,,{text}\n"

    return ass


def build_whisper_ass(input_path, start_time, duration, out_w, out_h,
                      api_key, work_dir, index=0, language=None):
    """Full pipeline for one clip: audio -> Whisper words -> ASS. Returns (ass, lang, n_words)."""
    audio = extract_clip_audio(input_path, start_time, duration, work_dir, index)
    try:
        words, lang = transcribe_words(audio, api_key, duration, work_dir, index, language)
    finally:
        try:
            os.unlink(audio)
        except OSError:
            pass
    if not words:
        raise RuntimeError("Whisper returned no words")
    ass = generate_ass_subtitles(words, 0.0, duration, out_w, out_h)
    return ass, lang, len(words)
