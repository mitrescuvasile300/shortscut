import { v } from "convex/values";
import { query } from "./_generated/server";
import { getAuthUserId } from "@convex-dev/auth/server";

export const listByJob = query({
  args: { jobId: v.id("jobs") },
  returns: v.array(
    v.object({
      _id: v.id("clips"),
      _creationTime: v.number(),
      jobId: v.id("jobs"),
      userId: v.id("users"),
      title: v.string(),
      description: v.string(),
      hashtags: v.array(v.string()),
      startTime: v.number(),
      endTime: v.number(),
      transcriptExcerpt: v.string(),
      viralScore: v.number(),
      reason: v.string(),
      hookLine: v.optional(v.string()),
      caption: v.optional(v.string()),
      topicTags: v.optional(v.array(v.string())),
    })
  ),
  handler: async (ctx, args) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) return [];
    const job = await ctx.db.get(args.jobId);
    if (!job || job.userId !== userId) return [];
    return await ctx.db
      .query("clips")
      .withIndex("by_jobId", (q) => q.eq("jobId", args.jobId))
      .collect();
  },
});
