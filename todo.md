# ShortsCut — Current Status

## ✅ Working
- Long videos (podcasts) fully in browser — no 150MB limit: per-clip video
  AND audio segment downloads via HTTP Range + mp4box (no python/sh needed)
- No more hanging downloads: chunked Range downloads (YouTube throttles
  plain GETs), timeouts + retries everywhere, stall watchdog
- Frame-accurate clip timing: single input -ss (old two-stage seek offset
  ALL filter times by +0.5s — subs, tracking pans, silence cuts)
- Dual-mode crop uses exact canvas dims (was estimated → skewed framing)
- Smooth camera re-anchor (was a visible jump every 3s)
- VPS fallback for browser generation: if the direct download fails/stalls,
  the VPS downloads via yt-dlp (/fetch) and serves it back with Range
  support (/file/<id>) through the Convex proxy — generation continues
  in the browser automatically
- Full E2E pipeline for normal YouTube videos (tested with Rick Astley)
- Auto-generation (no manual "Generează" button needed)
- Audio in shorts (separate video+audio download, ffmpeg.wasm muxing)
- Face detection (MediaPipe browser-based crop)
- Chunked 8MB Range downloads (no more 79% failures)
- Cookie support in Settings page (UI complete)
- Cookie parsing + SAPISIDHASH auth for InnerTube

## ❌ Blocked
- Bot-protected videos (e.g. Andrew Tate podcast vsp69jYlYsg)
  - YouTube blocks datacenter IPs regardless of cookies
  - Cookies get rotated by YouTube quickly
  - All libraries tested (InnerTube, ytdl-core, youtubei.js, yt-dlp) fail for restricted content from datacenter IPs
  - Works locally because residential IP + real browser fingerprint

## 📋 Planned (awaiting user input)
1. Video file upload fallback — for restricted videos
2. Better error UX — suggest upload when video blocked
3. Consider residential proxy for future
