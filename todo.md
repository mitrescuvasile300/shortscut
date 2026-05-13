# ShortsCut — Current Status

## ✅ Working
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
