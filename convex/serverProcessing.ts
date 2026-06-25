"use node";

import { action } from "./_generated/server";
import { internal } from "./_generated/api";
import { v } from "convex/values";
import { getAuthUserId } from "@convex-dev/auth/server";

// ── Note on server-side processing ────────────────────────────────────
// ffmpeg.wasm does NOT support Node.js (Convex runtime).
// Server-side processing requires a real ffmpeg binary on an external server.
// For now, this module provides a placeholder that returns a clear error.
// TODO: Integrate with external ffmpeg API endpoint (VPS, Cloud Run, Modal, etc.)

export const processJobOnServer = action({
  args: { jobId: v.id("jobs") },
  returns: v.null(),
  handler: async (ctx, { jobId }) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) throw new Error("Not authenticated");

    // Verify job exists and belongs to user
    const job = await ctx.runQuery(internal.processing.getJobInternal, {
      jobId,
    });
    if (!job) throw new Error("Job not found");

    throw new Error(
      "Procesarea pe server nu este disponibilă momentan. " +
      "ffmpeg nu poate rula în mediul Convex Node.js. " +
      "Folosește browser-ul cu un video de calitate mai mică (≤720p) " +
      "sau reîmprospătează URL-ul video-ului."
    );
  },
});
