import { v } from "convex/values";
import { action } from "./_generated/server";
import { api } from "./_generated/api";

declare const process: { env: Record<string, string | undefined> };

interface ChannelData {
  _id: string;
  channelUrl: string;
  processedVideoIds?: string[];
  autoLanguage: string;
  autoNumShorts: number;
  autoMinDuration: number;
  autoMaxDuration: number;
}

// Check a single channel for new videos
export const checkChannel = action({
  args: { channelId: v.id("channels") },
  handler: async (ctx, { channelId }): Promise<{ checked: boolean; newVideos: number }> => {
    // Get channel info
    const channel = (await ctx.runQuery(api.channels.get, { id: channelId })) as ChannelData | null;
    if (!channel) throw new Error("Channel not found");

    const channelUrl: string = channel.channelUrl;
    const processedIds: string[] = channel.processedVideoIds || [];

    // Use Viktor tool gateway to search for recent videos from this channel
    const toolGatewayUrl = process.env.TOOL_GATEWAY_URL;
    const toolGatewayToken = process.env.TOOL_GATEWAY_TOKEN;

    if (!toolGatewayUrl || !toolGatewayToken) {
      throw new Error("Tool gateway not configured");
    }

    // Extract channel identifier for search
    const channelIdentifier = extractChannelInfo(channelUrl);

    // Search for recent videos from this channel
    const searchResponse = await fetch(toolGatewayUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${toolGatewayToken}`,
      },
      body: JSON.stringify({
        tool: "quick_ai_search",
        args: {
          query: `site:youtube.com "${channelIdentifier}" latest videos uploaded in the last week`,
          response_format: `Return a JSON array of the 5 most recent YouTube video URLs from this channel. Format: [{"url": "https://youtube.com/watch?v=...", "title": "...", "videoId": "..."}]. Only include actual video URLs, no shorts, no playlists.`,
        },
      }),
    });

    if (!searchResponse.ok) {
      throw new Error(`Search failed: ${searchResponse.status}`);
    }

    const searchResult = await searchResponse.json();
    const resultText = searchResult?.result || searchResult?.text || "";

    // Parse video results
    const videos = parseVideoResults(resultText);

    // Filter out already-processed videos
    const newVideos = videos.filter(
      (v) => v.videoId && !processedIds.includes(v.videoId)
    );

    // Update channel's last checked time
    await ctx.runMutation(api.channels.update, {
      id: channelId,
      lastCheckedAt: Date.now(),
      ...(newVideos.length > 0
        ? { lastVideoId: newVideos[0].videoId }
        : {}),
    });

    if (newVideos.length === 0) {
      return { checked: true, newVideos: 0 };
    }

    // Create jobs for each new video
    const newProcessedIds = [...processedIds];

    for (const video of newVideos) {
      try {
        const jobId = await ctx.runMutation(api.jobs.create, {
          videoUrl: video.url,
          language: channel.autoLanguage,
          numShorts: channel.autoNumShorts,
          minDuration: channel.autoMinDuration,
          maxDuration: channel.autoMaxDuration,
        });

        // Start processing
        await ctx.runAction(api.processing.processJob, { jobId });

        newProcessedIds.push(video.videoId);
      } catch (e) {
        console.error(`Failed to process video ${video.url}:`, e);
      }
    }

    // Update processed video IDs (keep last 100 to avoid unbounded growth)
    await ctx.runMutation(api.channels.update, {
      id: channelId,
      processedVideoIds: newProcessedIds.slice(-100),
    });

    return { checked: true, newVideos: newVideos.length };
  },
});

// Check all active channels (called by cron or manually)
export const checkAllChannels = action({
  args: {},
  handler: async (_ctx) => {
    // We need an internal way to get all active channels
    // For now, this would be triggered per-user from the frontend
    return { message: "Use checkChannel for individual channel checks" };
  },
});

function extractChannelInfo(url: string): string {
  // Extract channel name or ID from various URL formats
  // https://www.youtube.com/@channelname
  // https://www.youtube.com/channel/UC...
  // https://www.youtube.com/c/channelname
  const atMatch = url.match(/@([a-zA-Z0-9_-]+)/);
  if (atMatch) return `@${atMatch[1]}`;

  const channelMatch = url.match(/\/channel\/([a-zA-Z0-9_-]+)/);
  if (channelMatch) return channelMatch[1];

  const cMatch = url.match(/\/c\/([a-zA-Z0-9_-]+)/);
  if (cMatch) return cMatch[1];

  // Return the URL as-is if we can't parse it
  return url;
}

function parseVideoResults(
  text: string
): Array<{ url: string; title: string; videoId: string }> {
  try {
    const jsonMatch = text.match(/\[[\s\S]*\]/);
    if (jsonMatch) {
      const parsed = JSON.parse(jsonMatch[0]);
      return parsed
        .filter(
          (v: Record<string, string>) => v.url && typeof v.url === "string"
        )
        .map((v: Record<string, string>) => ({
          url: v.url,
          title: v.title || "",
          videoId:
            v.videoId || extractVideoId(v.url) || "",
        }));
    }
  } catch {
    // Fall back to URL extraction
  }

  // Try to extract YouTube URLs from text
  const urlRegex =
    /https?:\/\/(?:www\.)?(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]{11})/g;
  const results: Array<{ url: string; title: string; videoId: string }> = [];
  let match;
  while ((match = urlRegex.exec(text)) !== null) {
    results.push({
      url: match[0],
      title: "",
      videoId: match[1],
    });
  }
  return results;
}

function extractVideoId(url: string): string {
  const match = url.match(
    /(?:v=|youtu\.be\/|shorts\/)([a-zA-Z0-9_-]{11})/
  );
  return match?.[1] || "";
}
