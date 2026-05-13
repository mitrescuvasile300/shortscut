import { v } from "convex/values";
import { mutation, query } from "./_generated/server";
import { getAuthUserId } from "@convex-dev/auth/server";

export const list = query({
  args: {},
  returns: v.array(
    v.object({
      _id: v.id("jobs"),
      _creationTime: v.number(),
      userId: v.id("users"),
      videoUrl: v.string(),
      videoTitle: v.optional(v.string()),
      videoDuration: v.optional(v.string()),
      videoThumbnail: v.optional(v.string()),
      language: v.string(),
      numShorts: v.number(),
      minDuration: v.number(),
      maxDuration: v.number(),
      status: v.union(
        v.literal("pending"),
        v.literal("downloading"),
        v.literal("transcribing"),
        v.literal("analyzing"),
        v.literal("generating"),
        v.literal("completed"),
        v.literal("failed"),
        v.literal("waiting_transcript")
      ),
      transcript: v.optional(v.string()),
      transcriptSegments: v.optional(v.string()),
      videoDownloadUrl: v.optional(v.string()),
      audioDownloadUrl: v.optional(v.string()),
      videoDownloadExpiry: v.optional(v.number()),
      error: v.optional(v.string()),
    })
  ),
  handler: async (ctx) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) return [];
    return await ctx.db
      .query("jobs")
      .withIndex("by_userId", (q) => q.eq("userId", userId))
      .order("desc")
      .collect();
  },
});

export const get = query({
  args: { id: v.id("jobs") },
  returns: v.union(
    v.object({
      _id: v.id("jobs"),
      _creationTime: v.number(),
      userId: v.id("users"),
      videoUrl: v.string(),
      videoTitle: v.optional(v.string()),
      videoDuration: v.optional(v.string()),
      videoThumbnail: v.optional(v.string()),
      language: v.string(),
      numShorts: v.number(),
      minDuration: v.number(),
      maxDuration: v.number(),
      status: v.union(
        v.literal("pending"),
        v.literal("downloading"),
        v.literal("transcribing"),
        v.literal("analyzing"),
        v.literal("generating"),
        v.literal("completed"),
        v.literal("failed"),
        v.literal("waiting_transcript")
      ),
      transcript: v.optional(v.string()),
      transcriptSegments: v.optional(v.string()),
      videoDownloadUrl: v.optional(v.string()),
      audioDownloadUrl: v.optional(v.string()),
      videoDownloadExpiry: v.optional(v.number()),
      error: v.optional(v.string()),
    }),
    v.null()
  ),
  handler: async (ctx, args) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) return null;
    const job = await ctx.db.get(args.id);
    if (!job || job.userId !== userId) return null;
    return job;
  },
});

export const create = mutation({
  args: {
    videoUrl: v.string(),
    language: v.string(),
    numShorts: v.number(),
    minDuration: v.number(),
    maxDuration: v.number(),
  },
  returns: v.id("jobs"),
  handler: async (ctx, args) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) throw new Error("Not authenticated");

    return await ctx.db.insert("jobs", {
      userId,
      videoUrl: args.videoUrl,
      language: args.language,
      numShorts: args.numShorts,
      minDuration: args.minDuration,
      maxDuration: args.maxDuration,
      status: "pending",
    });
  },
});

export const remove = mutation({
  args: { id: v.id("jobs") },
  returns: v.null(),
  handler: async (ctx, args) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) throw new Error("Not authenticated");
    const job = await ctx.db.get(args.id);
    if (!job || job.userId !== userId) throw new Error("Not found");

    // Delete associated clips
    const clips = await ctx.db
      .query("clips")
      .withIndex("by_jobId", (q) => q.eq("jobId", args.id))
      .collect();
    for (const clip of clips) {
      await ctx.db.delete(clip._id);
    }

    await ctx.db.delete(args.id);
    return null;
  },
});
