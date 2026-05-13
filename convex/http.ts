import { httpRouter } from "convex/server";
import { auth } from "./auth";
import { httpAction } from "./_generated/server";
import { internal } from "./_generated/api";
import type { Id } from "./_generated/dataModel";

const http = httpRouter();
auth.addHttpRoutes(http);

// ═══════════════════════════════════════════════════════════════════
// POST /api/transcript — Receives Whisper transcript from local script
// ═══════════════════════════════════════════════════════════════════
http.route({
  path: "/api/transcript",
  method: "POST",
  handler: httpAction(async (ctx, request) => {
    // CORS headers
    const corsHeaders = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
    };

    try {
      const body = await request.json();
      const { jobId, transcript } = body as {
        jobId: string;
        transcript: string;
      };

      if (!jobId || !transcript) {
        return new Response(
          JSON.stringify({ success: false, error: "Missing jobId or transcript" }),
          { status: 400, headers: { "Content-Type": "application/json", ...corsHeaders } }
        );
      }

      if (transcript.length < 50) {
        return new Response(
          JSON.stringify({ success: false, error: "Transcript too short" }),
          { status: 400, headers: { "Content-Type": "application/json", ...corsHeaders } }
        );
      }

      // Verify job exists
      const job = await ctx.runQuery(internal.processing.getJobInternal, {
        jobId: jobId as Id<"jobs">,
      });

      if (!job) {
        return new Response(
          JSON.stringify({ success: false, error: "Job not found" }),
          { status: 404, headers: { "Content-Type": "application/json", ...corsHeaders } }
        );
      }

      // Trigger async analysis — the local script will poll for results
      // Using scheduler to run the analysis asynchronously
      await ctx.runAction(internal.processing.analyzeTranscript, {
        jobId: jobId as Id<"jobs">,
        transcript,
      });

      return new Response(
        JSON.stringify({ success: true, message: "Transcript received and analyzed" }),
        { status: 200, headers: { "Content-Type": "application/json", ...corsHeaders } }
      );
    } catch (error) {
      const msg = error instanceof Error ? error.message : "Unknown error";
      return new Response(
        JSON.stringify({ success: false, error: msg }),
        { status: 500, headers: { "Content-Type": "application/json", ...corsHeaders } }
      );
    }
  }),
});

// CORS preflight for transcript endpoint
http.route({
  path: "/api/transcript",
  method: "OPTIONS",
  handler: httpAction(async () => {
    return new Response(null, {
      status: 204,
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
      },
    });
  }),
});

// ═══════════════════════════════════════════════════════════════════
// GET /api/job-status?jobId=... — Poll job status & clips from script
// ═══════════════════════════════════════════════════════════════════
http.route({
  path: "/api/job-status",
  method: "GET",
  handler: httpAction(async (ctx, request) => {
    const corsHeaders = {
      "Access-Control-Allow-Origin": "*",
      "Content-Type": "application/json",
    };

    const url = new URL(request.url);
    const jobId = url.searchParams.get("jobId");

    if (!jobId) {
      return new Response(
        JSON.stringify({ success: false, error: "Missing jobId" }),
        { status: 400, headers: corsHeaders }
      );
    }

    try {
      const job = await ctx.runQuery(internal.processing.getJobInternal, {
        jobId: jobId as Id<"jobs">,
      });

      if (!job) {
        return new Response(
          JSON.stringify({ success: false, error: "Job not found" }),
          { status: 404, headers: corsHeaders }
        );
      }

      // If completed, also fetch clips
      let clips: Array<{
        title: string;
        startTime: number;
        endTime: number;
        viralScore: number;
        transcriptExcerpt: string;
        hookLine?: string;
        caption?: string;
      }> = [];

      if (job.status === "completed") {
        const allClips = await ctx.runQuery(internal.processing.getClipsInternal, {
          jobId: jobId as Id<"jobs">,
        });
        clips = allClips.map((c) => ({
          title: c.title,
          startTime: c.startTime,
          endTime: c.endTime,
          viralScore: c.viralScore,
          transcriptExcerpt: c.transcriptExcerpt,
          hookLine: c.hookLine,
          caption: c.caption,
        }));
      }

      return new Response(
        JSON.stringify({ success: true, status: job.status, clips }),
        { status: 200, headers: corsHeaders }
      );
    } catch (error) {
      return new Response(
        JSON.stringify({ success: false, error: "Internal error" }),
        { status: 500, headers: corsHeaders }
      );
    }
  }),
});

http.route({
  path: "/api/job-status",
  method: "OPTIONS",
  handler: httpAction(async () => {
    return new Response(null, {
      status: 204,
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
      },
    });
  }),
});

// ═══════════════════════════════════════════════════════════════════
// GET /api/video-proxy?url=... — CORS proxy for YouTube CDN URLs
// Used by the browser-side ffmpeg to download video for processing
// ═══════════════════════════════════════════════════════════════════
http.route({
  path: "/api/video-proxy",
  method: "GET",
  handler: httpAction(async (_ctx, request) => {
    const corsHeaders = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Range",
      "Access-Control-Expose-Headers": "Content-Length, Content-Range, Accept-Ranges",
    };

    const url = new URL(request.url);
    const targetUrl = url.searchParams.get("url");

    if (!targetUrl) {
      return new Response(
        JSON.stringify({ error: "Missing url param" }),
        { status: 400, headers: { "Content-Type": "application/json", ...corsHeaders } }
      );
    }

    // Security: only proxy YouTube CDN URLs
    try {
      const parsed = new URL(targetUrl);
      const isYouTubeCDN =
        parsed.hostname.endsWith(".googlevideo.com") ||
        parsed.hostname.endsWith(".youtube.com") ||
        parsed.hostname.endsWith(".ytimg.com");

      if (!isYouTubeCDN) {
        return new Response(
          JSON.stringify({ error: "Only YouTube CDN URLs are allowed" }),
          { status: 403, headers: { "Content-Type": "application/json", ...corsHeaders } }
        );
      }
    } catch {
      return new Response(
        JSON.stringify({ error: "Invalid URL" }),
        { status: 400, headers: { "Content-Type": "application/json", ...corsHeaders } }
      );
    }

    try {
      // Forward range request headers if present (for partial content)
      const fetchHeaders: Record<string, string> = {
        "User-Agent": "Mozilla/5.0",
      };
      const rangeHeader = request.headers.get("Range");
      if (rangeHeader) {
        fetchHeaders["Range"] = rangeHeader;
      }

      const resp = await fetch(targetUrl, { headers: fetchHeaders });

      if (!resp.ok && resp.status !== 206) {
        return new Response(
          JSON.stringify({ error: `Upstream error: ${resp.status}` }),
          { status: 502, headers: { "Content-Type": "application/json", ...corsHeaders } }
        );
      }

      // Stream the response body through with CORS headers
      const responseHeaders: Record<string, string> = {
        ...corsHeaders,
        "Content-Type": resp.headers.get("Content-Type") || "video/mp4",
      };

      const cl = resp.headers.get("Content-Length");
      if (cl) responseHeaders["Content-Length"] = cl;

      const cr = resp.headers.get("Content-Range");
      if (cr) responseHeaders["Content-Range"] = cr;
      const ar = resp.headers.get("Accept-Ranges");
      if (ar) responseHeaders["Accept-Ranges"] = ar;

      return new Response(resp.body, {
        status: resp.status,
        headers: responseHeaders,
      });
    } catch (err) {
      return new Response(
        JSON.stringify({ error: "Proxy fetch failed" }),
        { status: 502, headers: { "Content-Type": "application/json", ...corsHeaders } }
      );
    }
  }),
});

http.route({
  path: "/api/video-proxy",
  method: "OPTIONS",
  handler: httpAction(async () => {
    return new Response(null, {
      status: 204,
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Range",
        "Access-Control-Expose-Headers": "Content-Length, Content-Range",
      },
    });
  }),
});

export default http;
