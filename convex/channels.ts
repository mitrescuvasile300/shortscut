import { v } from "convex/values";
import { mutation, query } from "./_generated/server";
import { getAuthUserId } from "@convex-dev/auth/server";

export const list = query({
  args: {},
  handler: async (ctx) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) return [];
    return ctx.db
      .query("channels")
      .withIndex("by_userId", (q) => q.eq("userId", userId))
      .order("desc")
      .collect();
  },
});

export const get = query({
  args: { id: v.id("channels") },
  handler: async (ctx, { id }) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) return null;
    const channel = await ctx.db.get(id);
    if (!channel || channel.userId !== userId) return null;
    return channel;
  },
});

export const add = mutation({
  args: {
    channelUrl: v.string(),
    channelName: v.optional(v.string()),
    checkIntervalMinutes: v.number(),
    autoLanguage: v.string(),
    autoNumShorts: v.number(),
    autoMinDuration: v.number(),
    autoMaxDuration: v.number(),
  },
  handler: async (ctx, args) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) throw new Error("Not authenticated");

    return ctx.db.insert("channels", {
      userId,
      channelUrl: args.channelUrl,
      channelName: args.channelName,
      isActive: true,
      checkIntervalMinutes: args.checkIntervalMinutes,
      autoLanguage: args.autoLanguage,
      autoNumShorts: args.autoNumShorts,
      autoMinDuration: args.autoMinDuration,
      autoMaxDuration: args.autoMaxDuration,
      processedVideoIds: [],
    });
  },
});

export const update = mutation({
  args: {
    id: v.id("channels"),
    channelName: v.optional(v.string()),
    channelThumbnail: v.optional(v.string()),
    isActive: v.optional(v.boolean()),
    checkIntervalMinutes: v.optional(v.number()),
    autoLanguage: v.optional(v.string()),
    autoNumShorts: v.optional(v.number()),
    autoMinDuration: v.optional(v.number()),
    autoMaxDuration: v.optional(v.number()),
    lastCheckedAt: v.optional(v.number()),
    lastVideoId: v.optional(v.string()),
    processedVideoIds: v.optional(v.array(v.string())),
  },
  handler: async (ctx, { id, ...updates }) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) throw new Error("Not authenticated");
    const channel = await ctx.db.get(id);
    if (!channel || channel.userId !== userId) throw new Error("Not found");

    const patch: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(updates)) {
      if (value !== undefined) patch[key] = value;
    }
    if (Object.keys(patch).length > 0) {
      await ctx.db.patch(id, patch);
    }
  },
});

export const toggleActive = mutation({
  args: { id: v.id("channels") },
  handler: async (ctx, { id }) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) throw new Error("Not authenticated");
    const channel = await ctx.db.get(id);
    if (!channel || channel.userId !== userId) throw new Error("Not found");
    await ctx.db.patch(id, { isActive: !channel.isActive });
  },
});

export const remove = mutation({
  args: { id: v.id("channels") },
  handler: async (ctx, { id }) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) throw new Error("Not authenticated");
    const channel = await ctx.db.get(id);
    if (!channel || channel.userId !== userId) throw new Error("Not found");
    await ctx.db.delete(id);
  },
});
