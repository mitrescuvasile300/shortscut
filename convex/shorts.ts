import { v } from "convex/values";
import { mutation, query } from "./_generated/server";
import { getAuthUserId } from "@convex-dev/auth/server";

// Generate an upload URL for storing a short video
export const generateUploadUrl = mutation({
  args: {},
  returns: v.string(),
  handler: async (ctx) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) throw new Error("Not authenticated");
    return await ctx.storage.generateUploadUrl();
  },
});

// Save a short video record after upload
export const save = mutation({
  args: {
    clipId: v.id("clips"),
    jobId: v.id("jobs"),
    storageId: v.id("_storage"),
    fileName: v.string(),
    duration: v.number(),
    fileSize: v.number(),
    hasSubtitles: v.boolean(),
  },
  returns: v.id("shorts"),
  handler: async (ctx, args) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) throw new Error("Not authenticated");

    // Verify the clip belongs to the user
    const clip = await ctx.db.get(args.clipId);
    if (!clip || clip.userId !== userId) throw new Error("Clip not found");

    // Delete any existing short for this clip
    const existing = await ctx.db
      .query("shorts")
      .withIndex("by_clipId", (q) => q.eq("clipId", args.clipId))
      .unique();
    if (existing) {
      await ctx.storage.delete(existing.storageId);
      await ctx.db.delete(existing._id);
    }

    return await ctx.db.insert("shorts", {
      clipId: args.clipId,
      jobId: args.jobId,
      userId,
      storageId: args.storageId,
      fileName: args.fileName,
      duration: args.duration,
      fileSize: args.fileSize,
      hasSubtitles: args.hasSubtitles,
    });
  },
});

// List all shorts for a job
export const listByJob = query({
  args: { jobId: v.id("jobs") },
  returns: v.array(
    v.object({
      _id: v.id("shorts"),
      _creationTime: v.number(),
      clipId: v.id("clips"),
      jobId: v.id("jobs"),
      userId: v.id("users"),
      storageId: v.id("_storage"),
      fileName: v.string(),
      duration: v.number(),
      fileSize: v.number(),
      hasSubtitles: v.boolean(),
      url: v.union(v.string(), v.null()),
    })
  ),
  handler: async (ctx, args) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) return [];

    const shorts = await ctx.db
      .query("shorts")
      .withIndex("by_jobId", (q) => q.eq("jobId", args.jobId))
      .collect();

    return await Promise.all(
      shorts.map(async (s) => ({
        ...s,
        url: await ctx.storage.getUrl(s.storageId),
      }))
    );
  },
});

// Get a single short's download URL
export const getUrl = query({
  args: { shortId: v.id("shorts") },
  returns: v.union(v.string(), v.null()),
  handler: async (ctx, args) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) return null;

    const short = await ctx.db.get(args.shortId);
    if (!short || short.userId !== userId) return null;

    return await ctx.storage.getUrl(short.storageId);
  },
});

// Mark job as completed after all shorts have been generated
export const markJobCompleted = mutation({
  args: { jobId: v.id("jobs") },
  returns: v.null(),
  handler: async (ctx, args) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) throw new Error("Not authenticated");

    const job = await ctx.db.get(args.jobId);
    if (!job || job.userId !== userId) throw new Error("Job not found");

    // Only transition from "generating" to "completed"
    if (job.status === "generating") {
      await ctx.db.patch(args.jobId, { status: "completed" });
    }
    return null;
  },
});

// Delete a short
export const remove = mutation({
  args: { shortId: v.id("shorts") },
  returns: v.null(),
  handler: async (ctx, args) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) throw new Error("Not authenticated");

    const short = await ctx.db.get(args.shortId);
    if (!short || short.userId !== userId) throw new Error("Not found");

    await ctx.storage.delete(short.storageId);
    await ctx.db.delete(args.shortId);
    return null;
  },
});
