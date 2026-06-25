"use node";

/**
 * Server-side video processing using ffmpeg.wasm in Node.js.
 *
 * This runs ffmpeg on the Convex server, avoiding browser memory limits.
 * Each clip is processed independently:
 *   1. Download video segment via Range requests
 *   2. Process with ffmpeg.wasm (crop + subtitles)
 *   3. Upload result to Convex file storage
 *
 * Limitations:
 *   - No face detection (no canvas/DOM on server) → uses center crop
 *   - Convex action timeout: ~10 min for internal actions
 *   - Memory: 512 MB max
 */

import { action, internalAction } from "./_generated/server";
import { internal } from "./_generated/api";
import { v } from "convex/values";
import { getAuthUserId } from "@convex-dev/auth/server";
// Id type used indirectly via v.id() validators

// ── Helpers ───────────────────────────────────────────────────────────

/** Download full file */
async function downloadFull(url: string): Promise<Uint8Array> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Download failed: ${resp.status}`);
  return new Uint8Array(await resp.arrayBuffer());
}

// downloadVideoSegment removed — server currently uses downloadFull
// Can be re-added when range-based downloading is needed

// ── Process a single clip on server ──────────────────────────────────

export const processClipOnServer = internalAction({
  args: {
    jobId: v.id("jobs"),
    clipId: v.id("clips"),
    videoUrl: v.string(),
    audioUrl: v.optional(v.string()),
    startTime: v.number(),
    endTime: v.number(),
    clipIndex: v.number(),
    assSubtitles: v.optional(v.string()),
  },
  returns: v.null(),
  handler: async (ctx, args) => {
    try {
      console.log(
        `[serverProc] Processing clip ${args.clipIndex}: ` +
          `${args.startTime.toFixed(0)}→${args.endTime.toFixed(0)}s`,
      );

      // Dynamic import ffmpeg.wasm (Node.js version)
      const { FFmpeg } = await import("@ffmpeg/ffmpeg");
      const { toBlobURL } = await import("@ffmpeg/util");

      const ffmpeg = new FFmpeg();
      ffmpeg.on("log", ({ message }) => {
        if (
          message.includes("frame=") ||
          message.includes("error") ||
          message.includes("Error")
        ) {
          console.log("[ffmpeg]", message);
        }
      });

      // Load ffmpeg core
      const baseURL =
        "https://unpkg.com/@ffmpeg/core@0.12.10/dist/esm";
      await ffmpeg.load({
        coreURL: await toBlobURL(
          `${baseURL}/ffmpeg-core.js`,
          "text/javascript",
        ),
        wasmURL: await toBlobURL(
          `${baseURL}/ffmpeg-core.wasm`,
          "application/wasm",
        ),
      });

      // Download video
      const videoData = await downloadFull(args.videoUrl);
      console.log(
        `[serverProc] Video downloaded: ${(videoData.length / 1e6).toFixed(1)}MB`,
      );

      // Download and mux audio if separate
      let sourceFile = "source.mp4";
      await ffmpeg.writeFile(sourceFile, videoData);

      if (args.audioUrl) {
        const audioData = await downloadFull(args.audioUrl);
        console.log(
          `[serverProc] Audio downloaded: ${(audioData.length / 1e6).toFixed(1)}MB`,
        );
        await ffmpeg.writeFile("audio.m4a", audioData);

        // Mux video + audio
        await ffmpeg.exec([
          "-i",
          "source.mp4",
          "-i",
          "audio.m4a",
          "-c",
          "copy",
          "-movflags",
          "+faststart",
          "-y",
          "muxed.mp4",
        ]);
        await ffmpeg.deleteFile("source.mp4");
        await ffmpeg.deleteFile("audio.m4a");
        sourceFile = "muxed.mp4";
      }

      // Write subtitles if available
      const hasSubs = !!args.assSubtitles;
      if (hasSubs) {
        const encoder = new TextEncoder();
        await ffmpeg.writeFile(
          "subs.ass",
          encoder.encode(args.assSubtitles!),
        );
      }

      // Get video dimensions (probe with ffmpeg)
      // Default to 1920x1080 as fallback
      const outW = 1080;
      const outH = 1920;

      // Build ffmpeg command for center crop + subtitles
      const duration = args.endTime - args.startTime;
      const fastSeek = Math.max(0, args.startTime - 0.5);
      const innerSeek = args.startTime - fastSeek;

      // Center crop (no face detection on server)
      const vf = [
        "crop=ih*9/16:ih:(iw-ih*9/16)/2:0",
        `scale=${outW}:${outH}`,
      ];
      if (hasSubs) {
        vf.push("ass=subs.ass");
      }

      const ffmpegArgs = [
        "-ss",
        fastSeek.toFixed(3),
        "-i",
        sourceFile,
        "-ss",
        innerSeek.toFixed(3),
        "-t",
        duration.toFixed(3),
        "-vf",
        vf.join(","),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        "-y",
        "output.mp4",
      ];

      console.log(`[serverProc] Running ffmpeg...`);
      await ffmpeg.exec(ffmpegArgs);

      // Read output
      const outputData = (await ffmpeg.readFile(
        "output.mp4",
      )) as Uint8Array;
      console.log(
        `[serverProc] Output: ${(outputData.length / 1e6).toFixed(1)}MB`,
      );

      if (!outputData || outputData.length === 0) {
        throw new Error("ffmpeg produced empty output");
      }

      // Upload to Convex storage
      const blob = new Blob([outputData.buffer as ArrayBuffer], { type: "video/mp4" });
      const storageId = await ctx.storage.store(blob);

      // Save the short record
      const safeTitle = `clip_${args.clipIndex + 1}`;
      const fileName = `${String(args.clipIndex + 1).padStart(2, "0")}_${safeTitle}.mp4`;

      await ctx.runAction(internal.serverProcessing.saveServerShort, {
        clipId: args.clipId,
        jobId: args.jobId,
        storageId,
        fileName,
        duration,
        fileSize: outputData.length,
        hasSubtitles: hasSubs,
      });

      console.log(
        `[serverProc] Clip ${args.clipIndex + 1} saved to storage`,
      );
    } catch (error) {
      console.error(
        `[serverProc] Clip ${args.clipIndex + 1} failed:`,
        error,
      );
      throw error;
    }

    return null;
  },
});

// Internal mutation to save a server-processed short
export const saveServerShort = internalAction({
  args: {
    clipId: v.id("clips"),
    jobId: v.id("jobs"),
    storageId: v.id("_storage"),
    fileName: v.string(),
    duration: v.number(),
    fileSize: v.number(),
    hasSubtitles: v.boolean(),
  },
  returns: v.null(),
  handler: async (ctx, args) => {
    // Get the clip to find the userId
    const clip = await ctx.runQuery(internal.processing.getClipInternal, {
      clipId: args.clipId,
    });
    if (!clip) throw new Error("Clip not found");

    // Check for existing short
    await ctx.runMutation(internal.processing.upsertShort, {
      clipId: args.clipId,
      jobId: args.jobId,
      userId: clip.userId,
      storageId: args.storageId,
      fileName: args.fileName,
      duration: args.duration,
      fileSize: args.fileSize,
      hasSubtitles: args.hasSubtitles,
    });

    return null;
  },
});

// upsertShort mutation lives in processing.ts (non-Node file)

// ── Orchestrator: process all clips for a job on server ─────────────

export const processJobOnServer = action({
  args: { jobId: v.id("jobs") },
  returns: v.null(),
  handler: async (ctx, { jobId }) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) throw new Error("Not authenticated");

    const job = await ctx.runQuery(
      internal.processing.getJobInternal,
      { jobId },
    );
    if (!job) throw new Error("Job not found");
    if (!job.videoDownloadUrl) {
      throw new Error("No video download URL available");
    }

    // Get clips for the job
    const clips = await ctx.runQuery(
      internal.processing.getClipsInternal,
      { jobId },
    );
    if (!clips || clips.length === 0) {
      throw new Error("No clips found for this job");
    }

    console.log(
      `[serverProc] Starting server-side processing for ${clips.length} clips`,
    );

    // Process each clip sequentially (to stay within memory limits)
    let processed = 0;
    for (const clip of clips) {
      try {
        // Build ASS subtitles for this clip (simplified server version)
        // The client will pass subtitles via the UI if available
        await ctx.runAction(
          internal.serverProcessing.processClipOnServer,
          {
            jobId,
            clipId: clip._id,
            videoUrl: job.videoDownloadUrl,
            audioUrl: job.audioDownloadUrl || undefined,
            startTime: clip.startTime,
            endTime: clip.endTime,
            clipIndex: processed,
            // No subtitles in basic server mode (would need transcript segments)
          },
        );
        processed++;
      } catch (err) {
        console.error(
          `[serverProc] Clip "${clip.title}" failed:`,
          err instanceof Error ? err.message : err,
        );
      }
    }

    // Mark job as completed if any clips were processed
    if (processed > 0) {
      await ctx.runMutation(internal.processing.updateJobStatus, {
        jobId,
        status: "completed",
      });
    }

    console.log(
      `[serverProc] Done: ${processed}/${clips.length} clips processed`,
    );
    return null;
  },
});
