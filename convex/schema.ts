import { authTables } from "@convex-dev/auth/server";
import { defineSchema, defineTable } from "convex/server";
import { v } from "convex/values";

const schema = defineSchema({
  ...authTables,

  // User settings (API keys, preferences)
  userSettings: defineTable({
    userId: v.id("users"),
    youtubeApiKey: v.optional(v.string()),
    openaiApiKey: v.optional(v.string()),
    youtubeCookies: v.optional(v.string()), // Netscape cookies.txt or raw Cookie header for YouTube
    defaultLanguage: v.optional(v.string()),
    defaultShortDuration: v.optional(v.number()), // seconds
    defaultNumShorts: v.optional(v.number()),
  }).index("by_userId", ["userId"]),

  // Processing jobs
  jobs: defineTable({
    userId: v.id("users"),
    videoUrl: v.string(),
    videoTitle: v.optional(v.string()),
    videoDuration: v.optional(v.string()),
    videoThumbnail: v.optional(v.string()),
    language: v.string(),
    numShorts: v.number(),
    minDuration: v.number(), // seconds
    maxDuration: v.number(), // seconds
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
    transcriptSegments: v.optional(v.string()), // JSON array of {start, end, text} for subtitles
    videoDownloadUrl: v.optional(v.string()), // Direct download URL (video-only or muxed)
    audioDownloadUrl: v.optional(v.string()), // Direct download URL (audio-only, for muxing)
    videoDownloadExpiry: v.optional(v.number()), // Expiry timestamp
    error: v.optional(v.string()),
  })
    .index("by_userId", ["userId"])
    .index("by_status", ["status"]),

  // Monitored YouTube channels
  channels: defineTable({
    userId: v.id("users"),
    channelUrl: v.string(), // YouTube channel URL
    channelName: v.optional(v.string()),
    channelThumbnail: v.optional(v.string()),
    isActive: v.boolean(),
    checkIntervalMinutes: v.number(), // how often to check (e.g. 60, 360, 1440)
    lastCheckedAt: v.optional(v.number()), // timestamp
    lastVideoId: v.optional(v.string()), // last known video ID to detect new ones
    // Auto-processing settings
    autoLanguage: v.string(),
    autoNumShorts: v.number(),
    autoMinDuration: v.number(),
    autoMaxDuration: v.number(),
    processedVideoIds: v.optional(v.array(v.string())), // track which videos we already processed
  })
    .index("by_userId", ["userId"])
    .index("by_isActive", ["isActive"]),

  // Identified clips/shorts from each job
  clips: defineTable({
    jobId: v.id("jobs"),
    userId: v.id("users"),
    title: v.string(),
    description: v.string(),
    hashtags: v.array(v.string()),
    startTime: v.number(), // seconds
    endTime: v.number(), // seconds
    transcriptExcerpt: v.string(),
    viralScore: v.number(), // 1-10
    reason: v.string(), // why this clip was selected
    // New fields from enhanced prompt
    hookLine: v.optional(v.string()), // exact opening sentence that grabs attention
    caption: v.optional(v.string()), // 1-line post description with hashtags
    topicTags: v.optional(v.array(v.string())), // 3-5 keywords for categorization
  })
    .index("by_jobId", ["jobId"])
    .index("by_userId", ["userId"]),

  // Generated short videos (stored in Convex file storage)
  shorts: defineTable({
    clipId: v.id("clips"),
    jobId: v.id("jobs"),
    userId: v.id("users"),
    storageId: v.id("_storage"), // Convex file storage reference
    fileName: v.string(),
    duration: v.number(), // seconds
    fileSize: v.number(), // bytes
    hasSubtitles: v.boolean(),
  })
    .index("by_clipId", ["clipId"])
    .index("by_jobId", ["jobId"])
    .index("by_userId", ["userId"]),
});

export default schema;
