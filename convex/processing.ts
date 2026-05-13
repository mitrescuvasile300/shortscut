import { v } from "convex/values";
import { action, internalMutation, internalAction, internalQuery } from "./_generated/server";
import { internal } from "./_generated/api";
import { getAuthUserId } from "@convex-dev/auth/server";

declare const process: { env: Record<string, string | undefined> };

const VIKTOR_API_URL = process.env.VIKTOR_SPACES_API_URL!;
const PROJECT_NAME = process.env.VIKTOR_SPACES_PROJECT_NAME!;
const PROJECT_SECRET = process.env.VIKTOR_SPACES_PROJECT_SECRET!;

async function callTool<T>(
  role: string,
  args: Record<string, unknown> = {}
): Promise<T> {
  const response = await fetch(
    `${VIKTOR_API_URL}/api/viktor-spaces/tools/call`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        project_name: PROJECT_NAME,
        project_secret: PROJECT_SECRET,
        role,
        arguments: args,
      }),
    }
  );

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`HTTP ${response.status}: ${text}`);
  }

  const json = await response.json();
  if (!json.success) {
    throw new Error(json.error ?? "Tool call failed");
  }
  return json.result as T;
}

// --- YouTube oEmbed API (direct fetch, no AI) ---
async function fetchVideoInfo(
  videoUrl: string,
  videoId: string
): Promise<{ title: string; author: string; thumbnail: string }> {
  try {
    const oembedUrl = `https://www.youtube.com/oembed?url=${encodeURIComponent(videoUrl)}&format=json`;
    const resp = await fetchWithTimeout(oembedUrl, {
      headers: { "User-Agent": "Mozilla/5.0" },
      timeout: 10000,
    });
    if (resp.ok) {
      const data = await resp.json();
      return {
        title: data.title || `YouTube Video ${videoId}`,
        author: data.author_name || "Unknown",
        thumbnail:
          data.thumbnail_url ||
          `https://img.youtube.com/vi/${videoId}/maxresdefault.jpg`,
      };
    }
  } catch {
    // fall through to fallback
  }
  return {
    title: `YouTube Video ${videoId}`,
    author: "Unknown",
    thumbnail: `https://img.youtube.com/vi/${videoId}/maxresdefault.jpg`,
  };
}

// ═══════════════════════════════════════════════════════════════════
// YouTube Transcript Fetching (no Whisper needed!)
// Fetches auto-generated captions from YouTube's innertube API
// ═══════════════════════════════════════════════════════════════════

interface TranscriptSegment {
  start: number;
  end: number;
  text: string;
}

async function fetchYouTubeTranscript(
  videoId: string,
  language: string,
  cookieHeader?: string
): Promise<{ segments: TranscriptSegment[]; text: string } | null> {
  try {
    // Step 1: Fetch YouTube page to get the initial data
    const headers: Record<string, string> = {
      "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
      "Accept-Language": "en-US,en;q=0.9",
    };
    if (cookieHeader) {
      headers["Cookie"] = cookieHeader;
    }
    const watchResp = await fetchWithTimeout(`https://www.youtube.com/watch?v=${videoId}`, {
      headers,
      timeout: 15000,
    });
    if (!watchResp.ok) return null;
    const html = await watchResp.text();

    // Step 2: Extract the player response JSON
    const match = html.match(/ytInitialPlayerResponse\s*=\s*({.+?});/);
    if (!match) return null;

    const playerData = JSON.parse(match[1]);
    const captions = playerData?.captions?.playerCaptionsTracklistRenderer?.captionTracks;
    if (!captions || captions.length === 0) return null;

    // Step 3: Find best caption track
    // Priority: user's language > English > first available
    let captionUrl: string | null = null;
    const langPriority = [language, "en", ""];

    for (const lang of langPriority) {
      if (!lang) {
        captionUrl = captions[0]?.baseUrl;
        break;
      }
      const track = captions.find(
        (t: { languageCode: string }) => t.languageCode === lang || t.languageCode.startsWith(lang)
      );
      if (track) {
        captionUrl = track.baseUrl;
        break;
      }
    }

    if (!captionUrl) return null;

    // Step 4: Fetch the transcript in JSON3 format (has timestamps)
    const transcriptResp = await fetchWithTimeout(`${captionUrl}&fmt=json3`, { timeout: 15000 });
    if (!transcriptResp.ok) return null;
    const transcriptData = await transcriptResp.json();

    const events = transcriptData?.events;
    if (!events || events.length === 0) return null;

    // Step 5: Parse into segments
    const segments: TranscriptSegment[] = [];
    for (const event of events) {
      if (!event.segs || event.tStartMs === undefined) continue;

      const startMs = event.tStartMs;
      const durationMs = event.dDurationMs || 3000;
      const text = event.segs
        .map((s: { utf8: string }) => s.utf8 || "")
        .join("")
        .trim();

      if (!text || text === "\n") continue;

      segments.push({
        start: startMs / 1000,
        end: (startMs + durationMs) / 1000,
        text,
      });
    }

    if (segments.length === 0) return null;

    // Build formatted transcript text with timestamps
    const lines = segments.map((seg) => {
      const h = Math.floor(seg.start / 3600);
      const m = Math.floor((seg.start % 3600) / 60);
      const s = Math.floor(seg.start % 60);
      const ts = h > 0 ? `[${h}:${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}]` : `[${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}]`;
      return `${ts} ${seg.text}`;
    });

    return {
      segments,
      text: lines.join("\n"),
    };
  } catch (e) {
    console.error("YouTube transcript fetch error:", e);
    return null;
  }
}

// ═══════════════════════════════════════════════════════════════════
// Whisper API transcription (fallback when YouTube captions unavailable)
// ═══════════════════════════════════════════════════════════════════

async function transcribeWithWhisper(
  audioUrl: string,
  openaiApiKey: string
): Promise<{ segments: TranscriptSegment[]; text: string } | null> {
  try {
    // Download audio from URL
    const audioResp = await fetch(audioUrl);
    if (!audioResp.ok) return null;
    const audioBuffer = await audioResp.arrayBuffer();
    const audioBlob = new Blob([audioBuffer], { type: "audio/mpeg" });

    // Check if file needs splitting (Whisper limit: 25MB)
    const MAX_SIZE = 24 * 1024 * 1024; // 24MB to be safe
    const allSegments: TranscriptSegment[] = [];

    if (audioBlob.size <= MAX_SIZE) {
      // Single file — send directly
      const result = await callWhisperApi(audioBlob, openaiApiKey);
      if (result) allSegments.push(...result);
    } else {
      // Split into chunks by byte size
      const numChunks = Math.ceil(audioBlob.size / MAX_SIZE);
      const totalDuration = estimateAudioDuration(audioBlob.size);
      for (let i = 0; i < numChunks; i++) {
        const start = i * MAX_SIZE;
        const end = Math.min(start + MAX_SIZE, audioBlob.size);
        const chunk = audioBlob.slice(start, end, "audio/mpeg");

        // Estimate time offset based on chunk position
        const timeOffset = (start / audioBlob.size) * totalDuration;
        const result = await callWhisperApi(chunk, openaiApiKey, timeOffset);
        if (result) {
          allSegments.push(...result);
        }
      }
    }

    if (allSegments.length === 0) return null;

    // Build formatted transcript text
    const lines = allSegments.map((seg) => {
      const h = Math.floor(seg.start / 3600);
      const m = Math.floor((seg.start % 3600) / 60);
      const s = Math.floor(seg.start % 60);
      const ts = h > 0 ? `[${h}:${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}]` : `[${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}]`;
      return `${ts} ${seg.text}`;
    });

    return {
      segments: allSegments,
      text: lines.join("\n"),
    };
  } catch (e) {
    console.error("Whisper transcription error:", e);
    return null;
  }
}

function estimateAudioDuration(sizeBytes: number): number {
  // Rough estimate assuming ~128kbps MP3
  return (sizeBytes * 8) / 128000;
}

async function callWhisperApi(
  audioBlob: Blob,
  apiKey: string,
  timeOffset = 0
): Promise<TranscriptSegment[] | null> {
  const formData = new FormData();
  formData.append("file", audioBlob, "audio.mp3");
  formData.append("model", "whisper-1");
  formData.append("response_format", "verbose_json");
  formData.append("timestamp_granularities[]", "word");
  formData.append("timestamp_granularities[]", "segment");

  const resp = await fetch("https://api.openai.com/v1/audio/transcriptions", {
    method: "POST",
    headers: { Authorization: `Bearer ${apiKey}` },
    body: formData,
  });

  if (!resp.ok) {
    console.error("Whisper API error:", resp.status, await resp.text());
    return null;
  }
  const result = await resp.json();

  // Use word-level timestamps if available (better for subtitles), else fall back to segments
  if (result.words && result.words.length > 0) {
    // Group words into ~3-5 second subtitle segments
    const segments: TranscriptSegment[] = [];
    let currentWords: Array<{ word: string; start: number; end: number }> = [];
    let segStart = 0;

    for (const word of result.words) {
      if (currentWords.length === 0) {
        segStart = word.start + timeOffset;
      }
      currentWords.push(word);

      // Split at natural breaks (~4 seconds or after punctuation)
      const segDuration = (word.end + timeOffset) - segStart;
      const endsWithPunctuation = /[.!?,;:]$/.test(word.word);
      if (segDuration >= 3.5 || (segDuration >= 2 && endsWithPunctuation)) {
        segments.push({
          start: segStart,
          end: word.end + timeOffset,
          text: currentWords.map(w => w.word).join(" ").trim(),
        });
        currentWords = [];
      }
    }

    // Remaining words
    if (currentWords.length > 0) {
      segments.push({
        start: segStart,
        end: currentWords[currentWords.length - 1].end + timeOffset,
        text: currentWords.map(w => w.word).join(" ").trim(),
      });
    }

    return segments;
  }

  // Fallback to segment-level timestamps
  return (result.segments || []).map((seg: { start: number; end: number; text: string }) => ({
    start: seg.start + timeOffset,
    end: seg.end + timeOffset,
    text: seg.text.trim(),
  }));
}

// ═══════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════
// Cookie Parsing — convert Netscape cookies.txt or raw Cookie header
// ═══════════════════════════════════════════════════════════════════

function parseCookiesToHeader(raw: string): string {
  if (!raw || !raw.trim()) return "";
  
  const trimmed = raw.trim();
  
  // If it already looks like a Cookie header (key=value; key=value format)
  // and doesn't contain tabs (which would indicate Netscape format)
  if (!trimmed.includes("\t") && trimmed.includes("=") && !trimmed.startsWith("#")) {
    return trimmed;
  }
  
  // Parse Netscape cookies.txt format
  // Lines: domain  include_subdomains  path  secure  expiry  name  value
  const pairs: string[] = [];
  for (const line of trimmed.split("\n")) {
    const l = line.trim();
    if (!l || l.startsWith("#")) continue;
    const parts = l.split("\t");
    if (parts.length >= 7) {
      const name = parts[5].trim();
      const value = parts[6].trim();
      if (name) pairs.push(`${name}=${value}`);
    }
  }
  
  return pairs.join("; ");
}

// YouTube InnerTube API — get direct download URLs via multiple clients
// ═══════════════════════════════════════════════════════════════════

function extractVideoId(url: string): string | null {
  const match = url.match(
    /(?:v=|youtu\.be\/|shorts\/|embed\/)([a-zA-Z0-9_-]{11})/
  );
  return match?.[1] ?? null;
}

interface YouTubeStreamInfo {
  audioUrl: string | null;
  videoUrl: string | null;
  title: string;
  duration: number;
}

// Compute SAPISIDHASH for YouTube authenticated API requests
// YouTube uses this to verify cookie ownership:
// SAPISIDHASH = SHA1(timestamp + " " + SAPISID + " " + origin)
async function computeSapisidHash(sapisid: string, origin: string): Promise<string> {
  const timestamp = Math.floor(Date.now() / 1000);
  const input = `${timestamp} ${sapisid} ${origin}`;
  const encoder = new TextEncoder();
  const data = encoder.encode(input);
  const hashBuffer = await crypto.subtle.digest("SHA-1", data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  const hashHex = hashArray.map(b => b.toString(16).padStart(2, "0")).join("");
  return `${timestamp}_${hashHex}`;
}

// Extract SAPISID from a Cookie header string
function extractSapisid(cookieHeader: string): string | null {
  // Try SAPISID first, then __Secure-3PAPISID
  for (const name of ["SAPISID", "__Secure-3PAPISID"]) {
    const match = cookieHeader.match(new RegExp(`(?:^|;\\s*)${name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}=([^;]+)`));
    if (match) return match[1];
  }
  return null;
}

async function tryInnerTubeClient(
  videoId: string,
  clientConfig: {
    name: string;
    clientName: string;
    clientVersion: string;
    userAgent: string;
    clientNum: string;
    extraContext?: Record<string, unknown>;
  },
  cookieHeader?: string
): Promise<{ status: string; reason?: string; data: Record<string, unknown> }> {
  const payload = {
    context: {
      client: {
        clientName: clientConfig.clientName,
        clientVersion: clientConfig.clientVersion,
        hl: "en",
        gl: "US",
        ...(clientConfig.extraContext || {}),
      },
    },
    videoId,
    playbackContext: { contentPlaybackContext: { html5Preference: "HTML5_PREF_WANTS" } },
    contentCheckOk: true,
    racyCheckOk: true,
  };

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    "User-Agent": clientConfig.userAgent,
    "X-YouTube-Client-Name": clientConfig.clientNum,
    "X-YouTube-Client-Version": clientConfig.clientVersion,
  };

  // Add cookies if provided
  if (cookieHeader) {
    headers["Cookie"] = cookieHeader;
    
    // SAPISIDHASH is needed for web-based clients (browser auth)
    // Mobile clients (ANDROID_VR, IOS etc.) reject it with HTTP 400
    const isWebClient = clientConfig.clientName.startsWith("WEB") || clientConfig.clientName.startsWith("TV");
    if (isWebClient) {
      const sapisid = extractSapisid(cookieHeader);
      if (sapisid) {
        const origin = "https://www.youtube.com";
        const hash = await computeSapisidHash(sapisid, origin);
        headers["Authorization"] = `SAPISIDHASH ${hash}`;
        headers["Origin"] = origin;
        headers["X-Origin"] = origin;
        console.log(`[YouTube] Added SAPISIDHASH auth for ${clientConfig.name}`);
      }
    }
  }

  const resp = await fetchWithTimeout(
    "https://www.youtube.com/youtubei/v1/player?key=AIzaSyA8eiZmM1FaDVjRy-df2KTyQ_vz_yYM39w&prettyPrint=false",
    {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
      timeout: 15000,
    }
  );

  if (!resp.ok) {
    return { status: "HTTP_ERROR", reason: `HTTP ${resp.status}`, data: {} };
  }

  const data = await resp.json();
  return {
    status: data.playabilityStatus?.status || "UNKNOWN",
    reason: data.playabilityStatus?.reason,
    data,
  };
}

// InnerTube clients to try in order
// Clients used with cookies — WEB needs SAPISIDHASH auth
const WEB_CREATOR_CLIENT = {
  name: "WEB_CREATOR",
  clientName: "WEB_CREATOR",
  clientVersion: "1.20250506.01.00",
  userAgent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
  clientNum: "62",
  extraContext: {},
};

const WEB_CLIENT = {
  name: "WEB",
  clientName: "WEB",
  clientVersion: "2.20250506.01.00",
  userAgent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
  clientNum: "1",
  extraContext: {},
};

const ANDROID_CLIENT = {
  name: "ANDROID",
  clientName: "ANDROID",
  clientVersion: "19.29.37",
  userAgent: "com.google.android.youtube/19.29.37 (Linux; U; Android 14) gzip",
  clientNum: "3",
  extraContext: {
    osName: "Android",
    osVersion: "14",
    androidSdkVersion: 34,
  },
};

const INNERTUBE_CLIENTS = [
  {
    name: "ANDROID_VR",
    clientName: "ANDROID_VR",
    clientVersion: "1.60.19",
    userAgent: "com.google.android.apps.youtube.vr.oculus/1.60.19 (Linux; U; Android 12L; eureka-user Build/SQ3A.220605.009.A1) gzip",
    clientNum: "28",
    extraContext: {
      deviceMake: "Oculus",
      deviceModel: "Quest 3",
      osName: "Android",
      osVersion: "12L",
      androidSdkVersion: 32,
    },
  },
  {
    name: "ANDROID_TESTSUITE",
    clientName: "ANDROID_TESTSUITE",
    clientVersion: "1.9",
    userAgent: "com.google.android.youtube/1.9 (Linux; U; Android 14) gzip",
    clientNum: "30",
    extraContext: {
      osName: "Android",
      osVersion: "14",
      androidSdkVersion: 34,
    },
  },
  {
    name: "IOS",
    clientName: "IOS",
    clientVersion: "20.03.02",
    userAgent: "com.google.ios.youtube/20.03.02 (iPhone16,2; U; CPU iOS 18_3_2 like Mac OS X;)",
    clientNum: "5",
    extraContext: {
      deviceMake: "Apple",
      deviceModel: "iPhone16,2",
      osName: "iOS",
      osVersion: "18.3.2.22D82",
    },
  },
];

async function getYouTubeStreamUrls(
  videoUrl: string,
  mode: "audio" | "video" | "both",
  cookieHeader?: string
): Promise<YouTubeStreamInfo> {
  const videoId = extractVideoId(videoUrl);
  if (!videoId) throw new Error("Invalid YouTube URL");

  if (cookieHeader) {
    console.log(`[YouTube] Using user-supplied cookies (${cookieHeader.length} chars)`);
  }

  // When cookies are available, try clients that work with auth
  // WEB_CREATOR returns direct URLs for authenticated users (YouTube Studio client)
  // WEB also works but may return signatureCipher URLs
  // ANDROID with cookies might bypass bot-detection too
  const clients = cookieHeader
    ? [WEB_CREATOR_CLIENT, WEB_CLIENT, ANDROID_CLIENT, ...INNERTUBE_CLIENTS]
    : INNERTUBE_CLIENTS;

  // Try each InnerTube client until one returns streams
  let lastError = "";
  for (const client of clients) {
    console.log(`[YouTube] Trying ${client.name} for ${videoId}...`);
    const result = await tryInnerTubeClient(videoId, client, cookieHeader);

    if (result.status === "OK") {
      console.log(`[YouTube] ${client.name} succeeded!`);
      const streams = extractStreamsFromPlayerData(result.data, mode);
      if (streams.videoUrl || streams.audioUrl) {
        return streams;
      }
      console.log(`[YouTube] ${client.name} returned OK but no usable streams`);
    } else {
      console.log(`[YouTube] ${client.name} failed: ${result.status} - ${result.reason || "?"}`);
      lastError = result.reason || result.status;
    }
  }

  // All clients failed
  const cookieHint = cookieHeader
    ? " Cookies-urile furnizate nu au funcționat — încearcă să le re-exportezi."
    : " Adaugă cookies YouTube în Setări pentru a debloca video-uri protejate.";
  throw new Error(
    `Nu am putut accesa video-ul de pe YouTube. ${lastError}.${cookieHint}`
  );
}

function extractStreamsFromPlayerData(
  data: Record<string, unknown>,
  mode: "audio" | "video" | "both"
): YouTubeStreamInfo {
  const streamingData = (data as { streamingData?: { adaptiveFormats?: Array<Record<string, unknown>> } }).streamingData;
  const videoDetails = (data as { videoDetails?: { title?: string; lengthSeconds?: string } }).videoDetails;
  
  const adaptive: Array<{
    itag: number;
    mimeType: string;
    url?: string;
    signatureCipher?: string;
    averageBitrate?: number;
    bitrate?: number;
    height?: number;
    width?: number;
    contentLength?: string;
  }> = (streamingData?.adaptiveFormats || []) as typeof adaptive;

  const title = videoDetails?.title || "Unknown";
  const duration = parseInt(videoDetails?.lengthSeconds || "0", 10);

  // Log format counts for debugging
  const withUrl = adaptive.filter(f => f.url).length;
  const withCipher = adaptive.filter(f => f.signatureCipher && !f.url).length;
  if (adaptive.length > 0) {
    console.log(`[YouTube] Formats: ${adaptive.length} total, ${withUrl} with direct URL, ${withCipher} with signatureCipher`);
  }

  let audioUrl: string | null = null;
  let videoDownloadUrl: string | null = null;

  if (mode === "audio" || mode === "both") {
    const audioFormats = adaptive
      .filter((f) => f.mimeType?.startsWith("audio/") && f.url)
      .sort((a, b) => (b.averageBitrate || b.bitrate || 0) - (a.averageBitrate || a.bitrate || 0));

    const m4aFormat = audioFormats.find((f) => f.mimeType?.includes("mp4a"));
    audioUrl = m4aFormat?.url || audioFormats[0]?.url || null;
  }

  if (mode === "video" || mode === "both") {
    const videoFormats = adaptive
      .filter((f) => f.mimeType?.startsWith("video/") && f.url)
      .sort((a, b) => (b.height || 0) - (a.height || 0));

    const mp4_720 = videoFormats.find(
      (f) => f.height === 720 && f.mimeType?.includes("avc1")
    );
    const any720 = videoFormats.find((f) => f.height === 720);
    const best = videoFormats.find((f) => (f.height || 0) <= 1080);

    videoDownloadUrl =
      mp4_720?.url || any720?.url || best?.url || videoFormats[0]?.url || null;
  }

  return { audioUrl, videoUrl: videoDownloadUrl, title, duration };
}

// Backward-compatible wrapper (replaces getCobaltDownloadUrl)
async function getCobaltDownloadUrl(
  videoUrl: string,
  mode: "audio" | "video",
  cookieHeader?: string
): Promise<string | null> {
  try {
    const streams = await getYouTubeStreamUrls(videoUrl, mode, cookieHeader);
    return mode === "audio" ? streams.audioUrl : streams.videoUrl;
  } catch (err) {
    console.error("YouTube stream extraction failed:", err);
    return null;
  }
}

// ═══════════════════════════════════════════════════════════════════
// Piped API — Reliable YouTube proxy (bypasses bot detection)
// Uses public Piped instances that proxy YouTube content with CORS
// ═══════════════════════════════════════════════════════════════════

const PIPED_INSTANCES = [
  "https://api.piped.private.coffee",
  "https://pipedapi.kavin.rocks",
  "https://pipedapi.r4fo.com",
];

// Piped proxy on user's server (fetches Piped API from a residential/VPS IP)
const PIPED_PROXY_URL = "http://76.13.133.153:3457";
const PIPED_PROXY_KEY = "shortcut-piped-2026";

/**
 * Fetch with timeout using AbortController (Convex fetch has no built-in timeout)
 */
async function fetchWithTimeout(
  url: string,
  options: RequestInit & { timeout?: number } = {}
): Promise<Response> {
  const { timeout = 30000, ...fetchOptions } = options;
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(url, {
      ...fetchOptions,
      signal: controller.signal,
    });
    return response;
  } finally {
    clearTimeout(id);
  }
}

interface PipedStreamInfo {
  videoUrl: string | null;
  audioUrl: string | null;
  title: string;
  duration: number;
  thumbnail: string;
  subtitleUrl: string | null;
  subtitleFormat: string | null;
}

async function fetchFromPiped(videoId: string): Promise<PipedStreamInfo | null> {
  // Try proxy server first (most reliable from Convex), then direct Piped instances
  const sources = [
    { url: `${PIPED_PROXY_URL}/api/streams/${videoId}`, headers: { "X-API-Key": PIPED_PROXY_KEY }, name: "proxy" },
    ...PIPED_INSTANCES.map(i => ({ url: `${i}/streams/${videoId}`, headers: { "User-Agent": "Mozilla/5.0" }, name: i })),
  ];

  for (const source of sources) {
    try {
      console.log(`[Piped] Trying ${source.name} for ${videoId}...`);
      const resp = await fetchWithTimeout(source.url, {
        headers: source.headers,
        timeout: 20000,
      });
      if (!resp.ok) {
        console.log(`[Piped] ${source.name} returned ${resp.status}`);
        continue;
      }
      const data = await resp.json();

      // Extract best video stream (720p mp4 preferred, then any ≤1080p)
      const videoStreams: Array<{
        url: string; quality: string; mimeType: string;
        videoOnly: boolean; width?: number; height?: number;
      }> = data.videoStreams || [];

      const audioStreams: Array<{
        url: string; quality: string; mimeType: string;
        bitrate: number;
      }> = data.audioStreams || [];

      // Find best video: prefer mp4 720p, then mp4 1080p, then any 720p
      let videoUrl: string | null = null;
      const mp4Videos = videoStreams.filter(
        (s) => s.mimeType?.startsWith("video/mp4") && s.videoOnly
      );
      const mp4_720 = mp4Videos.find((s) => s.quality === "720p");
      const mp4_1080 = mp4Videos.find((s) => s.quality === "1080p");
      const mp4_480 = mp4Videos.find((s) => s.quality === "480p");
      // Also check muxed (videoOnly=false)
      const muxedMp4 = videoStreams.find(
        (s) => s.mimeType?.startsWith("video/mp4") && !s.videoOnly
      );
      videoUrl = mp4_720?.url || mp4_1080?.url || mp4_480?.url || muxedMp4?.url || videoStreams[0]?.url || null;

      // Find best audio: prefer mp4 audio (m4a), highest bitrate
      let audioUrl: string | null = null;
      const mp4Audio = audioStreams
        .filter((s) => s.mimeType?.startsWith("audio/mp4"))
        .sort((a, b) => (b.bitrate || 0) - (a.bitrate || 0));
      const webmAudio = audioStreams
        .filter((s) => s.mimeType?.startsWith("audio/webm"))
        .sort((a, b) => (b.bitrate || 0) - (a.bitrate || 0));
      audioUrl = mp4Audio[0]?.url || webmAudio[0]?.url || audioStreams[0]?.url || null;

      // If we have a muxed stream (video+audio), we might not need separate audio
      if (muxedMp4 && !audioUrl) {
        audioUrl = muxedMp4.url;
      }

      // Subtitles
      const subs: Array<{
        url: string; code: string; name: string;
        autoGenerated: boolean; mimeType: string;
      }> = data.subtitles || [];
      const enSub = subs.find((s) => s.code === "en") || subs[0];

      console.log(
        `[Piped] Success! ${videoStreams.length} video, ${audioStreams.length} audio streams, ` +
        `${subs.length} subtitles. Title: "${data.title}"`
      );

      return {
        videoUrl,
        audioUrl,
        title: data.title || `YouTube Video ${videoId}`,
        duration: data.duration || 0,
        thumbnail: data.thumbnailUrl || `https://img.youtube.com/vi/${videoId}/maxresdefault.jpg`,
        subtitleUrl: enSub?.url || null,
        subtitleFormat: enSub?.mimeType || null,
      };
    } catch (err) {
      console.log(`[Piped] ${source.name} error: ${err instanceof Error ? err.message : String(err)}`);
    }
  }
  return null;
}

/**
 * Parse TTML (Piped subtitle format) into transcript segments
 */
function parseTTML(ttml: string): { segments: TranscriptSegment[]; text: string } {
  const segments: TranscriptSegment[] = [];
  // Match <p begin="HH:MM:SS.mmm" end="HH:MM:SS.mmm" ...>text</p>
  const pRegex = /<p\s+begin="([^"]+)"\s+end="([^"]+)"[^>]*>([\s\S]*?)<\/p>/g;
  let match: RegExpExecArray | null;

  function parseTime(t: string): number {
    const parts = t.split(":");
    if (parts.length === 3) {
      const [h, m, s] = parts;
      return parseFloat(h) * 3600 + parseFloat(m) * 60 + parseFloat(s);
    }
    if (parts.length === 2) {
      const [m, s] = parts;
      return parseFloat(m) * 60 + parseFloat(s);
    }
    return parseFloat(t);
  }

  function stripTags(html: string): string {
    return html
      .replace(/<[^>]+>/g, "")
      .replace(/&#39;/g, "'")
      .replace(/&amp;/g, "&")
      .replace(/&lt;/g, "<")
      .replace(/&gt;/g, ">")
      .replace(/&quot;/g, '"')
      .replace(/\s+/g, " ")
      .trim();
  }

  while ((match = pRegex.exec(ttml)) !== null) {
    const start = parseTime(match[1]);
    const end = parseTime(match[2]);
    const text = stripTags(match[3]);
    if (text && text !== "[Music]" && text !== "[Applause]") {
      segments.push({ start, end, text });
    }
  }

  // Deduplicate overlapping segments (TTML often has overlapping entries)
  const deduped: TranscriptSegment[] = [];
  const seen = new Set<string>();
  for (const seg of segments) {
    const key = `${seg.start.toFixed(1)}_${seg.text.substring(0, 30)}`;
    if (!seen.has(key)) {
      seen.add(key);
      deduped.push(seg);
    }
  }

  const text = deduped.map((s) => s.text).join(" ");
  return { segments: deduped, text };
}

// Internal mutation to update job status
export const updateJobStatus = internalMutation({
  args: {
    jobId: v.id("jobs"),
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
    videoTitle: v.optional(v.string()),
    videoDuration: v.optional(v.string()),
    videoThumbnail: v.optional(v.string()),
    transcript: v.optional(v.string()),
    transcriptSegments: v.optional(v.string()),
    videoDownloadUrl: v.optional(v.string()),
    audioDownloadUrl: v.optional(v.string()),
    videoDownloadExpiry: v.optional(v.number()),
    error: v.optional(v.string()),
  },
  returns: v.null(),
  handler: async (ctx, args) => {
    const { jobId, ...updates } = args;
    const cleanUpdates: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(updates)) {
      if (value !== undefined) cleanUpdates[key] = value;
    }
    await ctx.db.patch(jobId, cleanUpdates);
    return null;
  },
});

// Internal mutation to save clips
export const saveClips = internalMutation({
  args: {
    jobId: v.id("jobs"),
    userId: v.id("users"),
    clips: v.array(
      v.object({
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
  },
  returns: v.null(),
  handler: async (ctx, args) => {
    for (const clip of args.clips) {
      await ctx.db.insert("clips", {
        jobId: args.jobId,
        userId: args.userId,
        ...clip,
      });
    }
    return null;
  },
});

// Internal mutation to delete existing clips for a job
export const deleteClipsByJob = internalMutation({
  args: { jobId: v.id("jobs") },
  returns: v.null(),
  handler: async (ctx, { jobId }) => {
    const clips = await ctx.db
      .query("clips")
      .withIndex("by_jobId", (q) => q.eq("jobId", jobId))
      .collect();
    for (const clip of clips) {
      await ctx.db.delete(clip._id);
    }
    return null;
  },
});

// ═══════════════════════════════════════════════════════════════════
// AI Clip Analysis — shared logic used by processJob & HTTP
// ═══════════════════════════════════════════════════════════════════

const CHUNK_SIZE = 25000;
const CHUNK_OVERLAP = 2000;
const CANDIDATES_PER_CHUNK = 3;

interface CandidateClip {
  title: string;
  hookLine: string;
  caption: string;
  startTime: number;
  endTime: number;
  viralScore: number;
  viralReasoning: string;
  topicTags: string[];
  chunkIndex: number;
}

function splitTranscript(transcript: string): string[] {
  const chunks: string[] = [];
  let pos = 0;
  while (pos < transcript.length) {
    const end = Math.min(pos + CHUNK_SIZE, transcript.length);
    chunks.push(transcript.substring(pos, end));
    if (end >= transcript.length) break;
    pos = end - CHUNK_OVERLAP;
  }
  return chunks;
}

async function scanChunkForCandidates(
  chunk: string,
  chunkIndex: number,
  totalChunks: number,
  videoTitle: string,
  videoAuthor: string,
  targetLang: string,
  minDuration: number,
  maxDuration: number
): Promise<CandidateClip[]> {
  const n = CANDIDATES_PER_CHUNK;
  const properties: Record<string, { type: string; description: string }> = {};
  const required: string[] = [];

  for (let i = 1; i <= n; i++) {
    properties[`c${i}_title`] = { type: "string", description: `5-8 word title in ${targetLang} that maximizes curiosity` };
    properties[`c${i}_hookLine`] = { type: "string", description: "Exact opening sentence copied verbatim from transcript" };
    properties[`c${i}_startSeconds`] = { type: "integer", description: "Start time in total seconds where the hook begins" };
    properties[`c${i}_endSeconds`] = { type: "integer", description: `End time in total seconds, ${minDuration}-${maxDuration}s after start` };
    properties[`c${i}_viralScore`] = { type: "integer", description: "Honest viral potential 1-10. Most=5-7, exceptional=9-10" };
    properties[`c${i}_viralReasoning`] = { type: "string", description: `1-2 sentences in ${targetLang} explaining why this would perform well` };
    properties[`c${i}_topicTags`] = { type: "string", description: "3-5 keyword tags, comma-separated" };
    required.push(`c${i}_title`, `c${i}_startSeconds`, `c${i}_endSeconds`, `c${i}_viralScore`);
  }

  properties["noGoodMoments"] = { type: "string", description: "Set to 'true' if this section has NO viral-worthy moments (score >= 5). Otherwise leave empty." };

  const result = await callTool<{ result: Record<string, unknown> | null; error: string | null }>("ai_structured_output", {
    prompt: `You are scanning SECTION ${chunkIndex + 1} of ${totalChunks} from the podcast "${videoTitle}" by ${videoAuthor}.

Find up to ${n} potential viral short clips (${minDuration}-${maxDuration}s) in THIS section.

# WHAT MAKES A CLIP VIRAL
1. STRONG HOOK — bold claim, surprising statement, question, or pattern interrupt in first 3 seconds
2. SELF-CONTAINED — understandable without the rest of the podcast
3. EMOTIONAL PAYLOAD — surprise, controversy, humor, vulnerability, insight
4. CLEAR PAYOFF — delivers on the hook's promise
5. QUOTABLE LINE — something people would repeat or screenshot

# VIRAL SIGNAL DETECTION
## 🎤 Punchlines & Reactions — "what?!", "no way", swearing = engagement magnet
## 😂 Laughter & Comedy — self-roasts, one-liners, comedic timing
## 🔄 Reversal Moments — setup → unexpected twist
## 😬 Awkward / Tension — uncomfortable truths, long pauses + comebacks
## ⚡ High-Energy Exchanges — rapid back-and-forth, passionate disagreements
## 💎 Standalone Gold — hot takes, confessions, quotable lines

# WHAT TO SKIP
- Needs backstory, generic advice, slow build-ups, mid-sentence cuts, small talk

# SCORING: Most clips are 5-7. Only 1 per podcast can be 9-10.

# TIMESTAMP RULES: Use [MM:SS] or [HH:MM:SS] → convert to total seconds. hookLine = exact sentence from transcript.

# LANGUAGE: titles, reasoning in ${targetLang}. hookLine verbatim from transcript.`,
    output_schema: { type: "object", properties, required },
    input_text: chunk,
    intelligence_level: "smart",
  });

  if (result.error || !result.result) return [];
  const data = result.result;
  if (String(data["noGoodMoments"]) === "true") return [];

  const candidates: CandidateClip[] = [];
  for (let i = 1; i <= n; i++) {
    const title = data[`c${i}_title`];
    const startTime = Number(data[`c${i}_startSeconds`] || 0);
    let endTime = Number(data[`c${i}_endSeconds`] || 0);
    const viralScore = Math.min(10, Math.max(1, Number(data[`c${i}_viralScore`] || 0)));
    if (!title || startTime === 0 || viralScore < 5) continue;
    if (endTime <= startTime) endTime = startTime + maxDuration;
    const dur = endTime - startTime;
    if (dur > maxDuration) endTime = startTime + maxDuration;
    else if (dur < minDuration) endTime = startTime + minDuration;
    const topicTagsStr = String(data[`c${i}_topicTags`] || "");
    candidates.push({
      title: String(title),
      hookLine: String(data[`c${i}_hookLine`] || ""),
      caption: "",
      startTime,
      endTime,
      viralScore,
      viralReasoning: String(data[`c${i}_viralReasoning`] || ""),
      topicTags: topicTagsStr.split(",").map(t => t.trim()).filter(t => t.length > 0),
      chunkIndex,
    });
  }
  return candidates;
}

async function selectBestClips(
  candidates: CandidateClip[],
  videoTitle: string,
  videoAuthor: string,
  targetLang: string,
  numShorts: number,
  _minDuration: number,
  _maxDuration: number
): Promise<Array<{
  title: string; description: string; hashtags: string[]; startTime: number; endTime: number;
  transcriptExcerpt: string; viralScore: number; reason: string; hookLine: string; caption: string; topicTags: string[];
}>> {
  const candidateSummary = candidates
    .map((c, idx) => `[${idx + 1}] "${c.title}" | ${c.startTime}s-${c.endTime}s | Score: ${c.viralScore}/10 | Hook: "${c.hookLine.substring(0, 100)}" | Why: ${c.viralReasoning.substring(0, 120)} | Tags: ${c.topicTags.join(", ")}`)
    .join("\n");

  const properties: Record<string, { type: string; description: string }> = {};
  const required: string[] = [];
  for (let i = 1; i <= numShorts; i++) {
    properties[`pick${i}_candidateNumber`] = { type: "integer", description: `Candidate number [N] for clip ${i}` };
    properties[`pick${i}_title`] = { type: "string", description: `Polished 5-8 word title in ${targetLang}` };
    properties[`pick${i}_caption`] = { type: "string", description: `1-line social media post in ${targetLang} with 2-3 hashtags` };
    properties[`pick${i}_finalScore`] = { type: "integer", description: "Final viral score 1-10" };
    required.push(`pick${i}_candidateNumber`, `pick${i}_title`, `pick${i}_finalScore`);
  }

  const result = await callTool<{ result: Record<string, unknown> | null; error: string | null }>("ai_structured_output", {
    prompt: `Select the BEST ${numShorts} clips from ${candidates.length} candidates for "${videoTitle}" by ${videoAuthor}.
Consider: VARIETY, QUALITY, NO OVERLAP, BEST HOOKS, HUMOR, REVERSALS, CONTROVERSY.
TITLE: 5-8 words, curiosity gap, power words, no spoilers.
SCORING: Be honest. Most 6-8, max 1 at 9-10. All text in ${targetLang}.

Candidates:
${candidateSummary}`,
    output_schema: { type: "object", properties, required },
    input_text: candidateSummary,
    intelligence_level: "smart",
  });

  if (result.error || !result.result) {
    const sorted = [...candidates].sort((a, b) => b.viralScore - a.viralScore);
    return sorted.slice(0, numShorts).map((c) => ({
      title: c.title, description: c.caption, hashtags: c.topicTags.map(t => `#${t}`),
      startTime: c.startTime, endTime: c.endTime, transcriptExcerpt: c.hookLine,
      viralScore: c.viralScore, reason: c.viralReasoning, hookLine: c.hookLine,
      caption: c.caption, topicTags: c.topicTags,
    }));
  }

  const data = result.result;
  const finalClips: Array<{
    title: string; description: string; hashtags: string[]; startTime: number; endTime: number;
    transcriptExcerpt: string; viralScore: number; reason: string; hookLine: string; caption: string; topicTags: string[];
  }> = [];

  const usedIndices = new Set<number>();
  for (let i = 1; i <= numShorts; i++) {
    const candidateNum = Number(data[`pick${i}_candidateNumber`] || 0);
    if (candidateNum < 1 || candidateNum > candidates.length) continue;
    if (usedIndices.has(candidateNum)) continue;
    usedIndices.add(candidateNum);
    const original = candidates[candidateNum - 1];
    const title = String(data[`pick${i}_title`] || original.title);
    const captionStr = String(data[`pick${i}_caption`] || "");
    const finalScore = Math.min(10, Math.max(1, Number(data[`pick${i}_finalScore`] || original.viralScore)));
    const hashtagMatches = captionStr.match(/#\w+/g) || [];
    finalClips.push({
      title, description: captionStr,
      hashtags: hashtagMatches.length > 0 ? hashtagMatches : original.topicTags.map(t => `#${t}`),
      startTime: original.startTime, endTime: original.endTime,
      transcriptExcerpt: original.hookLine, viralScore: finalScore,
      reason: original.viralReasoning, hookLine: original.hookLine,
      caption: captionStr, topicTags: original.topicTags,
    });
  }
  return finalClips;
}

// Timestamp correction
function parseTimestamp(ts: string): number {
  const parts = ts.split(":").map(Number);
  if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
  if (parts.length === 2) return parts[0] * 60 + parts[1];
  return 0;
}

function correctTimestamps(
  transcript: string,
  clips: Array<{
    title: string; description: string; hashtags: string[]; startTime: number; endTime: number;
    transcriptExcerpt: string; viralScore: number; reason: string; hookLine: string; caption: string; topicTags: string[];
  }>
): typeof clips {
  const tsRegex = /\[(\d{1,2}:\d{2}(?::\d{2})?)\]/g;
  const timestampIndex: Array<{ charPos: number; seconds: number }> = [];
  let match: RegExpExecArray | null;
  while ((match = tsRegex.exec(transcript)) !== null) {
    timestampIndex.push({ charPos: match.index, seconds: parseTimestamp(match[1]) });
  }
  if (timestampIndex.length === 0) return clips;

  return clips.map((clip) => {
    if (!clip.hookLine || clip.hookLine.length < 10) return clip;
    let hookPos = -1;
    for (const variant of [clip.hookLine, clip.hookLine.substring(0, 80), clip.hookLine.substring(0, 50), clip.hookLine.substring(0, 30)]) {
      if (variant.length < 10) continue;
      const idx = transcript.toLowerCase().indexOf(variant.toLowerCase());
      if (idx >= 0) { hookPos = idx; break; }
    }
    if (hookPos < 0) return clip;
    let nearestTs: { seconds: number } | null = null;
    for (let i = timestampIndex.length - 1; i >= 0; i--) {
      if (timestampIndex[i].charPos <= hookPos) { nearestTs = timestampIndex[i]; break; }
    }
    if (!nearestTs) return clip;
    const dur = clip.endTime - clip.startTime;
    return { ...clip, startTime: nearestTs.seconds, endTime: nearestTs.seconds + dur };
  });
}

function deduplicateCandidates(candidates: CandidateClip[]): CandidateClip[] {
  const sorted = [...candidates].sort((a, b) => a.startTime - b.startTime);
  const result: CandidateClip[] = [];
  for (const c of sorted) {
    const existingIdx = result.findIndex(existing => Math.abs(existing.startTime - c.startTime) < 15);
    if (existingIdx >= 0) {
      if (c.viralScore > result[existingIdx].viralScore) result[existingIdx] = c;
    } else {
      result.push(c);
    }
  }
  return result;
}

async function analyzeTranscriptWithAI(
  transcript: string, videoTitle: string, videoAuthor: string,
  language: string, numShorts: number, minDuration: number, maxDuration: number
): Promise<Array<{
  title: string; description: string; hashtags: string[]; startTime: number; endTime: number;
  transcriptExcerpt: string; viralScore: number; reason: string; hookLine: string; caption: string; topicTags: string[];
}>> {
  const langMap: Record<string, string> = {
    ro: "Romanian", en: "English", es: "Spanish", fr: "French",
    de: "German", it: "Italian", pt: "Portuguese", ru: "Russian",
  };
  const targetLang = langMap[language] || language;

  const chunks = splitTranscript(transcript);
  const allCandidates: CandidateClip[] = [];
  for (let i = 0; i < chunks.length; i++) {
    const candidates = await scanChunkForCandidates(chunks[i], i, chunks.length, videoTitle, videoAuthor, targetLang, minDuration, maxDuration);
    allCandidates.push(...candidates);
  }
  if (allCandidates.length === 0) return [];

  // Correct timestamps before dedup
  const correctedCandidates = correctTimestamps(transcript, allCandidates.map(c => ({
    title: c.title, description: c.caption, hashtags: [], startTime: c.startTime, endTime: c.endTime,
    transcriptExcerpt: c.hookLine, viralScore: c.viralScore, reason: c.viralReasoning,
    hookLine: c.hookLine, caption: c.caption, topicTags: c.topicTags,
  })));
  for (let i = 0; i < allCandidates.length; i++) {
    allCandidates[i].startTime = correctedCandidates[i].startTime;
    allCandidates[i].endTime = correctedCandidates[i].endTime;
  }

  const deduped = deduplicateCandidates(allCandidates);
  const finalClips = await selectBestClips(deduped, videoTitle, videoAuthor, targetLang, numShorts, minDuration, maxDuration);
  return correctTimestamps(transcript, finalClips);
}

// ═══════════════════════════════════════════════════════════════════
// processJob — NEW: fully automatic pipeline
// 1. Video info (oEmbed)
// 2. Transcript (YouTube captions → Whisper fallback)
// 3. AI analysis
// 4. Video download URL (cobalt)
// No user intervention needed!
// ═══════════════════════════════════════════════════════════════════

export const processJob = action({
  args: { jobId: v.id("jobs") },
  returns: v.null(),
  handler: async (ctx, { jobId }) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) throw new Error("Not authenticated");

    const job = await ctx.runQuery(
      // biome-ignore lint: internal query
      (await import("./_generated/api")).api.jobs.get as any,
      { id: jobId }
    );
    if (!job) throw new Error("Job not found");

    try {
      // ===== STEP 1: Video info + Piped data =====
      await ctx.runMutation(internal.processing.updateJobStatus, { jobId, status: "downloading" });

      const videoId = extractVideoId(job.videoUrl);
      if (!videoId) throw new Error("URL invalidă. Folosește un link YouTube valid.");

      // Get user settings (OpenAI key + YouTube cookies)
      const settings = await ctx.runQuery(internal.processing.getUserSettings, { userId });
      const openaiKey = settings?.openaiApiKey;
      const cookieHeader = settings?.youtubeCookies ? parseCookiesToHeader(settings.youtubeCookies) : undefined;

      // Try Piped first (reliable, bypasses bot detection)
      console.log(`[processJob] Fetching video data via Piped for ${videoId}...`);
      const piped = await fetchFromPiped(videoId);

      // Fallback to oEmbed for title if Piped fails
      const videoTitle = piped?.title || (await fetchVideoInfo(job.videoUrl, videoId)).title;
      const videoThumbnail = piped?.thumbnail || `https://img.youtube.com/vi/${videoId}/maxresdefault.jpg`;

      await ctx.runMutation(internal.processing.updateJobStatus, {
        jobId, status: "transcribing",
        videoTitle,
        videoThumbnail,
      });

      // ===== STEP 2: Get transcript =====
      // Priority: Piped subtitles (fast) → YouTube captions → Whisper (slow, needs audio download)
      let transcriptResult: { segments: TranscriptSegment[]; text: string } | null = null;

      // 2a. Try Piped subtitles first (fast, lightweight TTML download via proxy)
      if (piped?.subtitleUrl) {
        try {
          console.log("[processJob] Fetching Piped subtitles via proxy...");
          const subUrl = `${PIPED_PROXY_URL}/api/subtitles?url=${encodeURIComponent(piped.subtitleUrl)}`;
          const subResp = await fetchWithTimeout(subUrl, {
            headers: { "X-API-Key": PIPED_PROXY_KEY },
            timeout: 30000,
          });
          if (subResp.ok) {
            const ttml = await subResp.text();
            transcriptResult = parseTTML(ttml);
            if (transcriptResult.segments.length > 0) {
              console.log(`[processJob] Piped subtitles OK: ${transcriptResult.segments.length} segments`);
            } else {
              transcriptResult = null;
            }
          } else {
            console.log(`[processJob] Piped subtitles HTTP ${subResp.status}`);
          }
        } catch (err) {
          console.log(`[processJob] Piped subtitles failed: ${err instanceof Error ? err.message : String(err)}`);
        }
      }

      // 2b. Fallback: YouTube InnerTube captions
      if (!transcriptResult) {
        try {
          console.log("[processJob] Trying YouTube InnerTube captions...");
          transcriptResult = await fetchYouTubeTranscript(videoId, job.language, cookieHeader);
          if (transcriptResult) {
            console.log(`[processJob] YouTube captions OK: ${transcriptResult.segments.length} segments`);
          }
        } catch (err) {
          console.log(`[processJob] YouTube captions failed: ${err instanceof Error ? err.message : String(err)}`);
        }
      }

      // 2c. Last resort: Whisper (downloads full audio — slow but works for any video)
      if (!transcriptResult && openaiKey) {
        const audioUrl = piped?.audioUrl || await getCobaltDownloadUrl(job.videoUrl, "audio", cookieHeader);
        if (audioUrl) {
          try {
            console.log("[processJob] Transcribing with Whisper (audio download)...");
            transcriptResult = await transcribeWithWhisper(audioUrl, openaiKey);
          } catch (err) {
            console.log(`[processJob] Whisper failed: ${err instanceof Error ? err.message : String(err)}`);
          }
        }
      }

      if (!transcriptResult || transcriptResult.segments.length === 0) {
        throw new Error(
          "Nu am putut obține transcriptul. Verifică:\n" +
          "1. Video-ul are subtitrări auto-generate pe YouTube\n" +
          "2. Sau ai setat OpenAI API Key în Settings (folosește Whisper ca fallback)"
        );
      }

      // Save transcript segments for subtitle generation
      const segmentsJson = JSON.stringify(transcriptResult.segments);

      await ctx.runMutation(internal.processing.updateJobStatus, {
        jobId, status: "analyzing",
        transcript: transcriptResult.text.substring(0, 49000),
        transcriptSegments: segmentsJson.substring(0, 900000),
      });

      // ===== STEP 3: AI analysis =====
      const videoAuthor = piped ? piped.title.split(" - ")[0] : "Unknown";
      const clips = await analyzeTranscriptWithAI(
        transcriptResult.text, videoTitle, videoAuthor,
        job.language, job.numShorts, job.minDuration, job.maxDuration
      );

      if (clips.length === 0) {
        throw new Error("Nu am putut identifica momente potrivite pentru Shorts.");
      }

      // Delete existing clips
      await ctx.runMutation(internal.processing.deleteClipsByJob, { jobId });
      await ctx.runMutation(internal.processing.saveClips, { jobId, userId, clips });

      // ===== STEP 4: Get video + audio download URLs =====
      // Piped URLs have CORS: * so browser can download directly!
      let videoDownloadUrl = piped?.videoUrl || null;
      let audioDownloadUrl = piped?.audioUrl || null;

      // Fallback to InnerTube if Piped didn't provide URLs
      if (!videoDownloadUrl) {
        try {
          console.log("[processJob] No Piped video URL, trying InnerTube...");
          const streams = await getYouTubeStreamUrls(job.videoUrl, "both", cookieHeader);
          videoDownloadUrl = streams.videoUrl;
          audioDownloadUrl = streams.audioUrl || audioDownloadUrl;
        } catch (err) {
          console.log(`[processJob] InnerTube fallback failed: ${err instanceof Error ? err.message : String(err)}`);
        }
      }

      if (!videoDownloadUrl) {
        throw new Error(
          "Am analizat video-ul dar nu am putut obține URL-ul de download. " +
          "Încearcă din nou sau adaugă cookies în Settings."
        );
      }

      await ctx.runMutation(internal.processing.updateJobStatus, {
        jobId, status: "generating",
        videoDownloadUrl: videoDownloadUrl || undefined,
        audioDownloadUrl: audioDownloadUrl || undefined,
        videoDownloadExpiry: videoDownloadUrl ? Date.now() + 3600000 : undefined, // 1 hour
      });

    } catch (error) {
      await ctx.runMutation(internal.processing.updateJobStatus, {
        jobId, status: "failed",
        error: error instanceof Error ? error.message : "Eroare necunoscută",
      });
    }
    return null;
  },
});

// Internal query to get user settings
export const getUserSettings = internalQuery({
  args: { userId: v.id("users") },
  returns: v.union(
    v.object({
      openaiApiKey: v.optional(v.string()),
      youtubeCookies: v.optional(v.string()),
    }),
    v.null()
  ),
  handler: async (ctx, { userId }) => {
    const settings = await ctx.db
      .query("userSettings")
      .withIndex("by_userId", (q) => q.eq("userId", userId))
      .unique();
    if (!settings) return null;
    return {
      openaiApiKey: settings.openaiApiKey,
      youtubeCookies: settings.youtubeCookies,
    };
  },
});

// ═══════════════════════════════════════════════════════════════════
// refreshVideoUrl — re-fetch download URL when it expires
// ═══════════════════════════════════════════════════════════════════

export const refreshVideoUrl = action({
  args: { jobId: v.id("jobs") },
  returns: v.union(v.string(), v.null()),
  handler: async (ctx, { jobId }) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) throw new Error("Not authenticated");

    const job = await ctx.runQuery(
      // biome-ignore lint: internal query
      (await import("./_generated/api")).api.jobs.get as any,
      { id: jobId }
    );
    if (!job) throw new Error("Job not found");

    const videoId = extractVideoId(job.videoUrl);
    if (!videoId) throw new Error("Invalid YouTube URL");

    // Try Piped first (most reliable)
    const piped = await fetchFromPiped(videoId);
    if (piped?.videoUrl) {
      await ctx.runMutation(internal.processing.updateJobStatus, {
        jobId,
        status: job.status as "generating" | "completed",
        videoDownloadUrl: piped.videoUrl,
        audioDownloadUrl: piped.audioUrl || undefined,
        videoDownloadExpiry: Date.now() + 3600000,
      });
      return piped.videoUrl;
    }

    // Fallback to InnerTube
    const settings = await ctx.runQuery(internal.processing.getUserSettings, { userId });
    const cookieHeader = settings?.youtubeCookies ? parseCookiesToHeader(settings.youtubeCookies) : undefined;
    const streams = await getYouTubeStreamUrls(job.videoUrl, "both", cookieHeader);
    if (streams.videoUrl) {
      await ctx.runMutation(internal.processing.updateJobStatus, {
        jobId,
        status: job.status as "generating" | "completed",
        videoDownloadUrl: streams.videoUrl,
        audioDownloadUrl: streams.audioUrl || undefined,
        videoDownloadExpiry: Date.now() + 3600000,
      });
    }
    return streams.videoUrl;
  },
});

// ═══════════════════════════════════════════════════════════════════
// analyzeTranscript — kept for backward compatibility (HTTP endpoint)
// ═══════════════════════════════════════════════════════════════════

export const analyzeTranscript = internalAction({
  args: { jobId: v.id("jobs"), transcript: v.string() },
  returns: v.object({
    success: v.boolean(),
    clips: v.array(v.object({
      title: v.string(), description: v.string(), hashtags: v.array(v.string()),
      startTime: v.number(), endTime: v.number(), transcriptExcerpt: v.string(),
      viralScore: v.number(), reason: v.string(),
      hookLine: v.optional(v.string()), caption: v.optional(v.string()), topicTags: v.optional(v.array(v.string())),
    })),
    error: v.optional(v.string()),
  }),
  handler: async (ctx, { jobId, transcript }) => {
    const jobData = await ctx.runQuery(internal.processing.getJobInternal, { jobId });
    if (!jobData) return { success: false, clips: [], error: "Job not found" };

    try {
      await ctx.runMutation(internal.processing.updateJobStatus, {
        jobId, status: "analyzing",
        transcript: `[Source: whisper]\n${transcript.substring(0, 49000)}`,
      });

      const clips = await analyzeTranscriptWithAI(
        transcript, jobData.videoTitle || "Unknown Video", "Unknown",
        jobData.language, jobData.numShorts, jobData.minDuration, jobData.maxDuration
      );

      if (clips.length === 0) {
        await ctx.runMutation(internal.processing.updateJobStatus, {
          jobId, status: "failed", error: "Nu am putut identifica momente potrivite.",
        });
        return { success: false, clips: [], error: "No clips found" };
      }

      await ctx.runMutation(internal.processing.deleteClipsByJob, { jobId });
      await ctx.runMutation(internal.processing.saveClips, { jobId, userId: jobData.userId, clips });
      await ctx.runMutation(internal.processing.updateJobStatus, { jobId, status: "generating" });
      return { success: true, clips };
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : "Eroare necunoscută";
      await ctx.runMutation(internal.processing.updateJobStatus, { jobId, status: "failed", error: errorMsg });
      return { success: false, clips: [], error: errorMsg };
    }
  },
});

// Internal queries (kept for HTTP endpoint)
export const getJobInternal = internalQuery({
  args: { jobId: v.id("jobs") },
  returns: v.union(v.object({
    _id: v.id("jobs"), userId: v.id("users"), videoUrl: v.string(),
    videoTitle: v.optional(v.string()), language: v.string(),
    numShorts: v.number(), minDuration: v.number(), maxDuration: v.number(), status: v.string(),
  }), v.null()),
  handler: async (ctx, { jobId }) => {
    const job = await ctx.db.get(jobId);
    if (!job) return null;
    return {
      _id: job._id, userId: job.userId, videoUrl: job.videoUrl,
      videoTitle: job.videoTitle, language: job.language,
      numShorts: job.numShorts, minDuration: job.minDuration, maxDuration: job.maxDuration, status: job.status,
    };
  },
});

export const getClipsInternal = internalQuery({
  args: { jobId: v.id("jobs") },
  returns: v.array(v.object({
    title: v.string(), startTime: v.number(), endTime: v.number(), viralScore: v.number(),
    transcriptExcerpt: v.string(), hookLine: v.optional(v.string()), caption: v.optional(v.string()),
  })),
  handler: async (ctx, { jobId }) => {
    const clips = await ctx.db.query("clips").withIndex("by_jobId", (q) => q.eq("jobId", jobId)).collect();
    return clips.map(c => ({
      title: c.title, startTime: c.startTime, endTime: c.endTime, viralScore: c.viralScore,
      transcriptExcerpt: c.transcriptExcerpt, hookLine: c.hookLine, caption: c.caption,
    }));
  },
});

// extractVideoId defined above
