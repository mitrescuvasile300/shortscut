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
  width = 1080,
  height = 1920,
): string {
  const fontSize = 48;
  const marginV = 120;
  const outlineSize = 3;

  let ass = `[Script Info]
Title: ShortsCut Subtitles
ScriptType: v4.00+
PlayResX: ${width}
PlayResY: ${height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,${fontSize},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,${outlineSize},1,2,20,20,${marginV},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
`;

  for (const seg of segments) {
    let text = seg.text
      .replace(/\\/g, "\\\\")
      .replace(/{/g, "\\{")
      .replace(/}/g, "\\}");

    const words = text.split(" ");
    const lines: string[] = [];
    let currentLine = "";
    for (const word of words) {
      if (currentLine.length + word.length + 1 > 30 && currentLine.length > 0) {
        lines.push(currentLine);
        currentLine = word;
      } else {
        currentLine = currentLine ? `${currentLine} ${word}` : word;
      }
    }
    if (currentLine) lines.push(currentLine);
    text = lines.join("\\N");

    ass += `Dialogue: 0,${formatAssTime(seg.start)},${formatAssTime(seg.end)},Default,,0,0,0,,${text}\n`;
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

    // Get video download URL
    const videoUrl = job.videoDownloadUrl;
    const audioUrl = job.audioDownloadUrl || null;
    const youtubeUrl = job.videoUrl;

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
          crop_plan: null, // center crop by default (face tracking is browser-only)
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
          youtube_url: !videoUrl ? youtubeUrl : undefined,
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

// ═══════════════════════════════════════════════════════════════════
// VPS source-video fetch — lets the BROWSER pipeline source the video
// from the VPS when direct YouTube/Piped downloads hang or fail.
// The VPS downloads with yt-dlp (reliable) and serves the file back
// with HTTP Range support at /file/<id>; the browser reads it through
// the Convex video-proxy (the VPS is plain http → mixed content).
// ═══════════════════════════════════════════════════════════════════

export const startVpsFetch = action({
  args: { jobId: v.id("jobs") },
  returns: v.object({
    fetchId: v.union(v.string(), v.null()),
    error: v.optional(v.string()),
  }),
  handler: async (ctx, { jobId }): Promise<{ fetchId: string | null; error?: string }> => {
    const userId = await getAuthUserId(ctx);
    if (!userId) throw new Error("Not authenticated");

    const job: { videoUrl: string } | null = await ctx.runQuery(internal.processing.getJobInternal, {
      jobId,
    });
    if (!job) throw new Error("Job not found");

    try {
      const resp: Response = await fetch(`${VPS_URL}/fetch`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-API-Key": VPS_API_KEY,
        },
        body: JSON.stringify({ video_url: job.videoUrl }),
      });
      const data = (await resp.json()) as {
        fetch_id?: string;
        error?: string;
      };
      if (!resp.ok || !data.fetch_id) {
        return {
          fetchId: null,
          error: data.error || `VPS /fetch HTTP ${resp.status}`,
        };
      }
      return { fetchId: data.fetch_id };
    } catch (e) {
      return {
        fetchId: null,
        error: e instanceof Error ? e.message : "VPS unreachable",
      };
    }
  },
});

export const getVpsFetchStatus = action({
  args: { fetchId: v.string() },
  returns: v.object({
    status: v.string(),
    fileUrl: v.union(v.string(), v.null()),
    size: v.optional(v.number()),
    width: v.optional(v.number()),
    height: v.optional(v.number()),
    error: v.optional(v.string()),
  }),
  handler: async (ctx, { fetchId }): Promise<{ status: string; fileUrl: string | null; size?: number; width?: number; height?: number; error?: string }> => {
    const userId = await getAuthUserId(ctx);
    if (!userId) throw new Error("Not authenticated");

    try {
      const resp: Response = await fetch(
        `${VPS_URL}/fetch-status?id=${encodeURIComponent(fetchId)}`,
      );
      const data = (await resp.json()) as {
        status?: string;
        size?: number;
        width?: number;
        height?: number;
        error?: string;
      };
      const status = data.status || "error";
      return {
        status,
        fileUrl: status === "ready" ? `${VPS_URL}/file/${fetchId}` : null,
        size: data.size ?? undefined,
        width: data.width ?? undefined,
        height: data.height ?? undefined,
        error: data.error ?? undefined,
      };
    } catch (e) {
      return {
        status: "error",
        fileUrl: null,
        error: e instanceof Error ? e.message : "VPS unreachable",
      };
    }
  },
});
