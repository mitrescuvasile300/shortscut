#!/bin/bash
# Shortscut VPS - Add /transcript endpoint
# Run this on the VPS: curl -sL <url> | bash

echo "=== Updating Shortscut VPS server ==="

# Find current server.py location
SERVER_PY=$(find /root /opt /srv /home -name "server.py" -path "*shortscut*" 2>/dev/null | head -1)
if [ -z "$SERVER_PY" ]; then
    SERVER_PY="/root/shortscut/server.py"
fi

echo "Server: $SERVER_PY"

# Backup current
cp "$SERVER_PY" "${SERVER_PY}.bak" 2>/dev/null

# Check if /transcript endpoint already exists
if grep -q "/transcript" "$SERVER_PY" 2>/dev/null; then
    echo "/transcript endpoint already exists!"
    exit 0
fi

# Add the /transcript endpoint before the 404 handler
python3 << 'PYEOF'
import sys

server_py = sys.argv[1] if len(sys.argv) > 1 else "/root/shortscut/server.py"

with open(server_py, "r") as f:
    content = f.read()

transcript_code = '''
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
                out_template = os.path.join(work_dir, "subs")
                cmd = [
                    "yt-dlp", "--skip-download",
                    "--write-auto-sub", "--write-sub",
                    "--sub-lang", lang, "--sub-format", "json3",
                    "--output", out_template, video_url,
                ]
                stdout, stderr, rc = run_cmd(cmd, timeout=120, cwd=work_dir)
                if rc != 0:
                    log.error(f"[transcript] yt-dlp failed: {stderr}")
                    self._json_response(500, {"error": f"yt-dlp failed (rc={rc})", "details": stderr[:500]})
                    return

                sub_files = list(Path(work_dir).glob("subs*.json3"))
                if not sub_files:
                    cmd2 = [
                        "yt-dlp", "--skip-download",
                        "--write-auto-sub", "--write-sub",
                        "--sub-lang", lang, "--sub-format", "vtt",
                        "--output", out_template, video_url,
                    ]
                    run_cmd(cmd2, timeout=120, cwd=work_dir)
                    sub_files = list(Path(work_dir).glob("subs*.vtt"))

                if not sub_files:
                    self._json_response(404, {"error": "No subtitles found", "details": stderr[:500]})
                    return

                sub_file = sub_files[0]
                sub_content = sub_file.read_text(encoding="utf-8")
                sub_format = sub_file.suffix.lstrip(".")
                segments = []

                if sub_format == "json3":
                    try:
                        json3_data = json.loads(sub_content)
                        for event in json3_data.get("events", []):
                            if "segs" not in event:
                                continue
                            start_ms = event.get("tStartMs", 0)
                            dur_ms = event.get("dDurationMs", 0)
                            text = "".join(seg.get("utf8", "") for seg in event["segs"]).strip()
                            if text and text != "\\n":
                                segments.append({"start": start_ms / 1000.0, "end": (start_ms + dur_ms) / 1000.0, "text": text})
                    except json.JSONDecodeError:
                        pass

                if sub_format == "vtt" and not segments:
                    import re as re_mod
                    for block in sub_content.split("\\n\\n"):
                        lines = block.strip().split("\\n")
                        for line in lines:
                            m = re_mod.match(r"(\\d{2}):(\\d{2}):(\\d{2})\\.(\\d{3})\\s*-->\\s*(\\d{2}):(\\d{2}):(\\d{2})\\.(\\d{3})", line)
                            if m:
                                g = m.groups()
                                start = int(g[0])*3600 + int(g[1])*60 + int(g[2]) + int(g[3])/1000
                                end = int(g[4])*3600 + int(g[5])*60 + int(g[6]) + int(g[7])/1000
                                text_lines = []
                                found = False
                                for l in lines:
                                    if "-->" in l: found = True; continue
                                    if found and l.strip():
                                        text_lines.append(re_mod.sub(r"<[^>]+>", "", l).strip())
                                text = " ".join(text_lines).strip()
                                if text:
                                    segments.append({"start": start, "end": end, "text": text})
                                break

                log.info(f"[transcript] Extracted {len(segments)} segments ({sub_format})")
                self._json_response(200, {
                    "success": True, "format": sub_format, "language": lang,
                    "segments": segments, "segment_count": len(segments),
                })
            except Exception as e:
                log.error(f"[transcript] Error: {e}")
                self._json_response(500, {"error": str(e)})
            finally:
                shutil.rmtree(work_dir, ignore_errors=True)
            return

'''

# Insert before the 404 handler
marker = '        self._json_response(404, {"error": "Not found"})'
if marker in content:
    content = content.replace(marker, transcript_code + marker)
    with open(server_py, "w") as f:
        f.write(content)
    print(f"Updated {server_py}")
else:
    print("ERROR: Could not find insertion point in server.py")
    sys.exit(1)
PYEOF "$SERVER_PY"

# Restart the server
echo "Restarting server..."
pkill -f "python3.*server.py.*3458" 2>/dev/null || true
sleep 2

# Find and restart
cd "$(dirname "$SERVER_PY")"
nohup python3 "$SERVER_PY" > /var/log/shortscut-server.log 2>&1 &
sleep 2

# Verify
curl -s http://localhost:3458/health
echo ""
echo "=== Done! Testing /transcript endpoint ==="
curl -s -X POST http://localhost:3458/transcript \
  -H "Content-Type: application/json" \
  -H "X-API-Key: shortcut-vps-2026" \
  -d '{"video_url":"https://www.youtube.com/watch?v=dQw4w9WgXcQ","lang":"en"}' | python3 -c "
import sys, json
d = json.load(sys.stdin)
if d.get('success'):
    print(f'SUCCESS: {d[\"segment_count\"]} segments extracted')
else:
    print(f'Error: {d.get(\"error\",\"unknown\")}')
"
