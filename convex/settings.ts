import { v } from "convex/values";
import { mutation, query } from "./_generated/server";
import { getAuthUserId } from "@convex-dev/auth/server";

export const get = query({
  args: {},
  returns: v.union(
    v.object({
      _id: v.id("userSettings"),
      _creationTime: v.number(),
      userId: v.id("users"),
      youtubeApiKey: v.optional(v.string()),
      openaiApiKey: v.optional(v.string()),
      youtubeCookies: v.optional(v.string()),
      defaultLanguage: v.optional(v.string()),
      defaultShortDuration: v.optional(v.number()),
      defaultNumShorts: v.optional(v.number()),
    }),
    v.null()
  ),
  handler: async (ctx) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) return null;
    return await ctx.db
      .query("userSettings")
      .withIndex("by_userId", (q) => q.eq("userId", userId))
      .unique();
  },
});

export const save = mutation({
  args: {
    youtubeApiKey: v.optional(v.string()),
    openaiApiKey: v.optional(v.string()),
    youtubeCookies: v.optional(v.string()),
    defaultLanguage: v.optional(v.string()),
    defaultShortDuration: v.optional(v.number()),
    defaultNumShorts: v.optional(v.number()),
  },
  returns: v.null(),
  handler: async (ctx, args) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) throw new Error("Not authenticated");

    const existing = await ctx.db
      .query("userSettings")
      .withIndex("by_userId", (q) => q.eq("userId", userId))
      .unique();

    if (existing) {
      await ctx.db.patch(existing._id, args);
    } else {
      await ctx.db.insert("userSettings", { userId, ...args });
    }
    return null;
  },
});
