"use node";

import { action } from "./_generated/server";
import { internal } from "./_generated/api";
import { v } from "convex/values";
import { getAuthUserId } from "@convex-dev/auth/server";

// ── VPS Processing Server ─────────────────────────────────────────────
// Sends clips to external VPS with native ffmpeg for processing.
// No browser memory limits, handles any video size.

const VPS_URL = "http://76.13.133.153:3458";
const VPS_API_KEY = "shortcut-vps-2026";

// ── Subtitle generation (mirror of src/lib/subtitles.ts) ──────────────

interface SubtitleSegment {
  start: number;
  end: number;
  text: string;
}

function formatAssTime(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  const sInt = Math.floor(s);
  const cs = Math.floor((s - sInt) * 100);
  return `${h}:${m.toString().padStart(2, "0")}:${sInt
    .toString()
    .padStart(2, "0")}.${cs.toString().padStart(2, "0")}`;
}

function getSegmentsForClip(
  allSegments: SubtitleSegment[],
  clipStart: number,
  clipEnd: number,
): SubtitleSegment[] {
  return allSegments
    .filter((seg) => seg.end > clipStart && seg.start < clipEnd)
    .map((seg) => ({
      start: Math.max(0, seg.start - clipStart),
      end: Math.min(clipEnd - clipStart, seg.end - clipStart),
      text: seg.text.replace(/\n/g, " ").trim(),
    }))
    .filter((seg) => seg.text.length > 0);
}

function generateAssSubtitles(
  segments: SubtitleSegment[],
  width = 720,
  height = 1280,
): string {
  // Fallback only (used when the VPS can't run Whisper). Same look as the
  // local script's generate_ass_subtitles(): Arial Black, sizes relative to
  // the output height, ~4 UPPERCASE words per line. Caption segments have no
  // word timings, so each segment is split into 4-word groups with times
  // distributed proportionally.
  const fontSize = Math.floor(height * 0.052);
  const outline = Math.floor(height * 0.004);
  const shadow = Math.floor(height * 0.003);
  const marginV = Math.floor(height * 0.15);
  const wordsPerLine = 4;

  let ass = `[Script Info]
Title: ShortsCut Subtitles
ScriptType: v4.00+
PlayResX: ${width}
PlayResY: ${height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial Black,${fontSize},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,${outline},${shadow},2,40,40,${marginV},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
`;

  for (const seg of segments) {
    const words = seg.text
      .replace(/[{}\\]/g, "")
      .split(/\s+/)
      .filter((w) => w.length > 0);
    if (words.length === 0) continue;
    const groups: string[][] = [];
    for (let i = 0; i < words.length; i += wordsPerLine) {
      groups.push(words.slice(i, i + wordsPerLine));
    }
    const segDur = Math.max(0.5, seg.end - seg.start);
    let t = seg.start;
    for (const group of groups) {
      const dur = segDur * (group.length / words.length);
      let lineEnd = t + dur;
      if (lineEnd <= t) lineEnd = t + 0.5;
      const text = group.join(" ").toUpperCase();
      ass += `Dialogue: 0,${formatAssTime(t)},${formatAssTime(lineEnd)},Default,,0,0,0,,${text}\n`;
      t = lineEnd;
    }
  }

  return ass;
}

// ── Main server processing action ─────────────────────────────────────

export const processJobOnServer = action({
  args: { jobId: v.id("jobs") },
  returns: v.null(),
  handler: async (ctx, { jobId }) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) throw new Error("Not authenticated");

    // Get job data
    const job = await ctx.runQuery(internal.processing.getJobInternal, {
      jobId,
    });
    if (!job) throw new Error("Job not found");

    // Get clips for this job
    const clips = await ctx.runQuery(internal.processing.getClipsInternal, {
      jobId,
    });
    if (!clips || clips.length === 0) {
      throw new Error(
        "Nu există clipuri pentru acest job. Rulează mai întâi analiza.",
      );
    }

    // Get video download URL.
    // NOTE (2026-09-03): the Piped/InnerTube-provided signed googlevideo.com
    // `videoDownloadUrl` is fragile — even though it's IP-locked to the VPS's
    // IP, direct curl fetches from the VPS get HTTP 403 (likely needs the
    // exact request fingerprint used when the URL was minted), which used to
    // silently save the error body as "source.mp4" and fail ffmpeg with
    // "moov atom not found". The VPS's own yt-dlp+cookies download (used
    // successfully for transcripts) is reliable, so always pass youtube_url
    // and let the VPS prefer that path (`vps/server.py` tries yt-dlp first
    // for youtube.com URLs, falling back to the direct URL only if yt-dlp
    // fails).
    const videoUrl = job.videoDownloadUrl;
    const audioUrl = job.audioDownloadUrl || null;
    const youtubeUrl = job.videoUrl;

    const settingsForVps = await ctx.runQuery(
      internal.processing.getUserSettings,
      { userId },
    );
    const vpsCookies = settingsForVps?.youtubeCookies || undefined;
    // The VPS transcribes each clip with Whisper (word timestamps, language
    // auto-detected = original language of the video) and builds the same
    // 4-words-per-line UPPERCASE Arial Black subtitles as the local script.
    // The YouTube-caption ASS below is only a fallback if Whisper fails.
    const vpsOpenaiKey = settingsForVps?.openaiApiKey || undefined;

    if (!videoUrl && !youtubeUrl) {
      throw new Error(
        "Nu există URL de download. Reîmprospătează URL-ul video-ului.",
      );
    }

    // Get transcript segments for subtitle generation
    let allSegments: SubtitleSegment[] = [];
    if (job.transcriptSegments) {
      try {
        allSegments = JSON.parse(job.transcriptSegments);
      } catch {
        console.log("[serverProcessing] Could not parse transcript segments");
      }
    }

    // Update status to generating
    await ctx.runMutation(internal.processing.updateJobStatus, {
      jobId,
      status: "generating",
    });

    try {
      // Build clip configs for VPS
      const clipConfigs = clips.map((clip, index) => {
        const clipSegments = getSegmentsForClip(
          allSegments,
          clip.startTime,
          clip.endTime,
        );
        const assContent =
          clipSegments.length > 0 ? generateAssSubtitles(clipSegments) : null;

        return {
          index,
          start_time: clip.startTime,
          end_time: clip.endTime,
          ass_subtitles: assContent,
          remove_silence: true,
          crop_plan: null,
          auto_face_track: true, // VPS runs real MediaPipe face tracking (see PR #1 on the source repo); falls back to center crop automatically if it can't detect a face
        };
      });

      // Call VPS processing API
      console.log(
        `[serverProcessing] Sending ${clipConfigs.length} clips to VPS...`,
      );

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 600000); // 10 min timeout

      const response = await fetch(`${VPS_URL}/process`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-API-Key": VPS_API_KEY,
        },
        body: JSON.stringify({
          video_url: videoUrl,
          audio_url: audioUrl,
          youtube_url: youtubeUrl,
          cookies: vpsCookies,
          openai_api_key: vpsOpenaiKey,
          clips: clipConfigs,
        }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(
          `VPS processing failed (HTTP ${response.status}): ${errorText}`,
        );
      }

      const result = await response.json();
      if (!result.success) {
        throw new Error(`VPS processing failed: ${result.error || "Unknown"}`);
      }

      console.log(
        `[serverProcessing] VPS returned ${result.clips?.length || 0} clips`,
      );

      // Download each processed clip from VPS and upload to Convex storage
      let successCount = 0;
      for (const clipResult of result.clips || []) {
        if (!clipResult.success) {
          console.error(
            `[serverProcessing] Clip ${clipResult.index} failed: ${clipResult.error}`,
          );
          continue;
        }

        const clip = clips[clipResult.index];
        if (!clip) continue;

        try {
          // Download processed clip from VPS
          const downloadUrl = `${VPS_URL}${clipResult.download_url}`;
          console.log(
            `[serverProcessing] Downloading clip ${clipResult.index} (${(clipResult.size / 1e6).toFixed(1)} MB)...`,
          );

          const clipResp = await fetch(downloadUrl);
          if (!clipResp.ok) {
            throw new Error(`Download failed: HTTP ${clipResp.status}`);
          }

          const clipBlob = await clipResp.blob();

          // Upload to Convex storage
          const uploadUrl = await ctx.storage.generateUploadUrl();
          const uploadResp = await fetch(uploadUrl, {
            method: "POST",
            headers: { "Content-Type": "video/mp4" },
            body: clipBlob,
          });

          if (!uploadResp.ok) {
            throw new Error(`Convex upload failed: HTTP ${uploadResp.status}`);
          }

          const { storageId } = (await uploadResp.json()) as {
            storageId: string;
          };

          // Create short entry
          const safeTitle = clip.title
            .replace(/[^a-zA-Z0-9\s-]/g, "")
            .replace(/\s+/g, "_")
            .substring(0, 40);
          const fileName = `${String(clipResult.index + 1).padStart(2, "0")}_${safeTitle}.mp4`;

          await ctx.runMutation(internal.processing.upsertShort, {
            clipId: clip._id,
            jobId,
            userId,
            storageId: storageId as any,
            fileName,
            duration: clip.endTime - clip.startTime,
            fileSize: clipResult.size || clipBlob.size,
            hasSubtitles: allSegments.length > 0,
          });

          successCount++;
          console.log(
            `[serverProcessing] Clip ${clipResult.index} uploaded to Convex ✓`,
          );
        } catch (err) {
          console.error(
            `[serverProcessing] Clip ${clipResult.index} failed:`,
            err instanceof Error ? err.message : String(err),
          );
        }
      }

      if (successCount === 0) {
        throw new Error("Niciun clip nu a putut fi procesat pe server.");
      }

      // Mark job as completed
      await ctx.runMutation(internal.processing.updateJobStatus, {
        jobId,
        status: "completed",
      });

      console.log(
        `[serverProcessing] ✓ Job completed: ${successCount}/${clips.length} clips`,
      );
    } catch (error) {
      console.error("[serverProcessing] Failed:", error);
      await ctx.runMutation(internal.processing.updateJobStatus, {
        jobId,
        status: "failed",
        error:
          error instanceof Error
            ? error.message
            : "Eroare procesare server necunoscută",
      });
    }

    return null;
  },
});
