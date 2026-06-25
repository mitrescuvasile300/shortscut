/**
 * MP4 Segment Downloader
 *
 * Downloads only the clip portion of a remote MP4 video using mp4box.js
 * to parse the MP4 structure and produce valid fragmented MP4 (fMP4) segments.
 *
 * This solves the browser memory issue where downloading a full 200MB+ video
 * causes "Array buffer allocation failed" in WebAssembly.
 *
 * Flow:
 *  1. Download first ~2MB (moov atom, typically at start for web videos)
 *  2. mp4box.js parses the MP4 structure
 *  3. seek() finds the byte offset for the clip's start time
 *  4. Download only the bytes needed for the clip (+ padding)
 *  5. mp4box.js produces valid fMP4 segments
 *  6. Return concatenated init + media segments → valid fMP4 for ffmpeg
 */

import { type MP4BoxBuffer, createFile } from "mp4box";
import type { Movie, Track } from "mp4box";

// ── Constants ────────────────────────────────────────────────────────────
const HEADER_CHUNK = 2 * 1024 * 1024; // 2 MB initial download for moov
const PADDING_SEC = 10; // seconds of padding before/after clip
const DL_CHUNK = 4 * 1024 * 1024; // 4 MB download chunks
const BITRATE_SAFETY = 1.5; // over-estimate byte range for safety

// ── Helpers ──────────────────────────────────────────────────────────────

async function fetchRange(
  url: string,
  start: number,
  end: number,
): Promise<ArrayBuffer> {
  const resp = await fetch(url, {
    headers: { Range: `bytes=${start}-${end}` },
  });
  if (!resp.ok && resp.status !== 206) {
    throw new Error(`Range fetch failed: HTTP ${resp.status}`);
  }
  return resp.arrayBuffer();
}

function asMP4Buf(ab: ArrayBuffer, fileStart: number): MP4BoxBuffer {
  const buf = ab as MP4BoxBuffer;
  buf.fileStart = fileStart;
  return buf;
}

// ── Public API ───────────────────────────────────────────────────────────

export interface SegmentResult {
  /** Valid fMP4 data (init segment + media segments) */
  data: Uint8Array;
  /** Video track width in pixels */
  videoWidth: number;
  /** Video track height in pixels */
  videoHeight: number;
}

/**
 * Probe whether a URL supports HTTP Range requests.
 */
export async function supportsRangeRequests(url: string): Promise<boolean> {
  try {
    const resp = await fetch(url, { headers: { Range: "bytes=0-0" } });
    return resp.status === 206;
  } catch {
    return false;
  }
}

/**
 * Get total file size via Range probe. Returns 0 if unknown.
 */
export async function getFileSize(url: string): Promise<number> {
  try {
    const resp = await fetch(url, { headers: { Range: "bytes=0-0" } });
    if (resp.status === 206) {
      const m = resp.headers.get("Content-Range")?.match(/\/(\d+)/);
      if (m) return Number(m[1]);
    }
    const cl = resp.headers.get("Content-Length");
    if (cl) return Number(cl);
    return 0;
  } catch {
    return 0;
  }
}

/**
 * Download only a clip's portion of a remote MP4 video.
 *
 * Uses mp4box.js to parse the MP4 structure, seek to the clip's time range,
 * download only the required bytes, and produce valid fMP4 output.
 *
 * @param url           Direct URL to the MP4 (must support Range requests)
 * @param clipStartTime Clip start time in seconds
 * @param clipEndTime   Clip end time in seconds
 * @param onProgress    Optional progress callback (0–100)
 * @returns             SegmentResult with fMP4 data and video dimensions
 */
export async function downloadClipSegment(
  url: string,
  clipStartTime: number,
  clipEndTime: number,
  onProgress?: (pct: number) => void,
): Promise<SegmentResult> {
  // Probe file size
  const fileSize = await getFileSize(url);
  if (!fileSize) throw new Error("Cannot determine video file size");

  onProgress?.(5);

  return new Promise<SegmentResult>((resolve, reject) => {
    const file = createFile();
    const mediaSegs: ArrayBuffer[] = [];
    let initSegData: Uint8Array | null = null;
    let vW = 0;
    let vH = 0;
    let done = false;

    // ── Finalise: combine init + media segments into one fMP4 ─────
    const finish = (err?: Error) => {
      if (done) return;
      done = true;
      if (err) return reject(err);

      if (!initSegData || mediaSegs.length === 0) {
        return reject(
          new Error(
            "No segments produced — file may not be a compatible MP4",
          ),
        );
      }

      let totalLen = initSegData.byteLength;
      for (const s of mediaSegs) totalLen += s.byteLength;

      const out = new Uint8Array(totalLen);
      let off = 0;
      out.set(initSegData, off);
      off += initSegData.byteLength;
      for (const s of mediaSegs) {
        out.set(new Uint8Array(s), off);
        off += s.byteLength;
      }

      console.log(
        `[segDL] Final fMP4: ${(totalLen / 1e6).toFixed(1)} MB, ` +
          `${mediaSegs.length} media segment(s)`,
      );
      onProgress?.(100);
      resolve({ data: out, videoWidth: vW, videoHeight: vH });
    };

    // ── mp4box callbacks ──────────────────────────────────────────
    file.onError = (e: string) => finish(new Error(`MP4 parse: ${e}`));

    file.onSegment = (
      _id: number,
      _user: unknown,
      buffer: ArrayBuffer,
    ) => {
      mediaSegs.push(buffer);
    };

    file.onReady = async (info: Movie) => {
      try {
        // Find video & audio tracks
        let vTrack = 0;
        let aTrack = 0;
        for (const t of info.tracks as Track[]) {
          if (t.type === "video" && !vTrack) {
            vTrack = t.id;
            vW = t.video?.width || t.track_width || 0;
            vH = t.video?.height || t.track_height || 0;
          }
          if (t.type === "audio" && !aTrack) aTrack = t.id;
        }
        if (!vTrack) return finish(new Error("No video track in MP4"));

        console.log(
          `[segDL] Video track ${vTrack}: ${vW}×${vH}` +
            (aTrack ? `, Audio track ${aTrack}` : ""),
        );

        // Configure segmentation
        file.setSegmentOptions(vTrack, null, { nbSamples: 5000 });
        if (aTrack) {
          file.setSegmentOptions(aTrack, null, { nbSamples: 10000 });
        }

        const initResult = file.initializeSegmentation();
        if (initResult) {
          // Combined mode returns { tracks, buffer }
          const buf =
            "buffer" in initResult
              ? (initResult as { buffer: ArrayBuffer }).buffer
              : null;
          if (buf) {
            initSegData = new Uint8Array(buf);
          }
        }

        // Seek to just before the clip start (with padding for keyframe)
        const seekTime = Math.max(0, clipStartTime - PADDING_SEC);
        const seekInfo = file.seek(seekTime, true);

        // Estimate how many bytes we need for the clip
        const durSec = info.duration / info.timescale;
        const byterate = fileSize / Math.max(durSec, 1);
        const clipDur =
          clipEndTime - clipStartTime + PADDING_SEC * 2 + 5;
        const estBytes = Math.ceil(clipDur * byterate * BITRATE_SAFETY);
        const bStart = Math.max(0, seekInfo.offset);
        const bEnd = Math.min(bStart + estBytes - 1, fileSize - 1);
        const dlTotal = bEnd - bStart + 1;

        console.log(
          `[segDL] Clip ${clipStartTime.toFixed(0)}→${clipEndTime.toFixed(0)}s, ` +
            `downloading ${(dlTotal / 1e6).toFixed(1)} MB ` +
            `(${bStart}→${bEnd} of ${(fileSize / 1e6).toFixed(1)} MB)`,
        );
        onProgress?.(25);

        // Download clip data in chunks
        const chunks: Uint8Array[] = [];
        let dlDone = 0;
        while (dlDone < dlTotal) {
          const cs = bStart + dlDone;
          const ce = Math.min(cs + DL_CHUNK - 1, bEnd);
          const chunk = new Uint8Array(await fetchRange(url, cs, ce));
          chunks.push(chunk);
          dlDone += chunk.byteLength;
          onProgress?.(25 + Math.round((dlDone / dlTotal) * 55));
        }

        // Combine chunks into one buffer
        const combined = new Uint8Array(dlDone);
        let co = 0;
        for (const c of chunks) {
          combined.set(c, co);
          co += c.byteLength;
        }

        // Feed to mp4box → triggers onSegment callbacks
        const clipBuf = combined.buffer.slice(
          combined.byteOffset,
          combined.byteOffset + combined.byteLength,
        );
        file.appendBuffer(asMP4Buf(clipBuf, bStart));
        file.flush();

        onProgress?.(90);
        finish();
      } catch (err) {
        finish(err instanceof Error ? err : new Error(String(err)));
      }
    };

    // ── Download header to trigger moov parse ────────────────────
    (async () => {
      try {
        const hEnd = Math.min(HEADER_CHUNK - 1, fileSize - 1);
        const hData = await fetchRange(url, 0, hEnd);
        onProgress?.(15);

        file.appendBuffer(asMP4Buf(hData, 0));

        // If moov wasn't in the first chunk, try end of file
        if (
          !done &&
          file.nextParsePosition &&
          file.nextParsePosition > hEnd + 1
        ) {
          console.log(
            `[segDL] moov not at start, trying end of file...`,
          );
          const tailStart = Math.max(0, fileSize - HEADER_CHUNK * 2);
          const tailData = await fetchRange(
            url,
            tailStart,
            fileSize - 1,
          );
          file.appendBuffer(asMP4Buf(tailData, tailStart));
        }

        // Still not ready? Download more from the beginning
        if (!done) {
          await new Promise((r) => setTimeout(r, 50));
          if (!done) {
            const moreEnd = Math.min(
              HEADER_CHUNK * 4 - 1,
              fileSize - 1,
            );
            if (moreEnd > hEnd) {
              const moreData = await fetchRange(url, hEnd + 1, moreEnd);
              file.appendBuffer(asMP4Buf(moreData, hEnd + 1));
            }
          }
        }

        // Final check
        await new Promise((r) => setTimeout(r, 100));
        if (!done) {
          finish(new Error("Could not parse MP4 header"));
        }
      } catch (err) {
        finish(err instanceof Error ? err : new Error(String(err)));
      }
    })();
  });
}
