"use node";

import { action, internalAction } from "./_generated/server";
import { internal } from "./_generated/api";
import { v } from "convex/values";
import { getAuthUserId } from "@convex-dev/auth/server";
import type { Id } from "./_generated/dataModel";

// ── Full local-script pipeline on the VPS ─────────────────────────────
// The VPS runs the exact `shortscut_pipeline.py` that the downloadable .sh
// script runs locally (same arguments), so server output == local output:
// yt-dlp download → Whisper (word timestamps) → 2-pass AI analysis →
// per-shot face framing (single / tracking / split-screen) → silence removal
// → burned subtitles → 1080x1920 libx264. Convex only starts the run, polls
// its status and pulls the finished MP4s into storage.

const VPS_URL = "http://76.13.133.153:3458";
const VPS_API_KEY = "shortcut-vps-2026";
const POLL_INTERVAL_MS = 10_000;
const MAX_RUNTIME_MS = 90 * 60_000; // give up after 90 minutes

type VpsState = "running" | "completed" | "failed";
type VpsStep = "downloading" | "transcribing" | "analyzing" | "generating";

interface VpsClip {
  title: string;
  hookLine?: string;
  startTime: number;
  endTime: number;
  viralScore: number;
  reason?: string;
}

interface VpsStatus {
  id: string;
  state: VpsState;
  step: VpsStep;
  video_title: string | null;
  clips: VpsClip[] | null;
  outputs: Array<{ index: number; name: string; size: number; download_url: string }>;
  error: string | null;
  log_tail: string;
  elapsed: number;
}

async function vpsFetch(path: string, init: RequestInit = {}, timeoutMs = 30_000) {
  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(`${VPS_URL}${path}`, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": VPS_API_KEY,
        ...(init.headers || {}),
      },
      signal: controller.signal,
    });
  } finally {
    clearTimeout(t);
  }
}

export const startPipeline = action({
  args: { jobId: v.id("jobs") },
  returns: v.null(),
  handler: async (ctx, { jobId }) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) throw new Error("Not authenticated");

    const job = await ctx.runQuery(internal.processing.getJobInternal, { jobId });
    if (!job) throw new Error("Job not found");

    const settings = await ctx.runQuery(internal.processing.getUserSettings, { userId });
    if (!settings?.openaiApiKey) {
      const msg = "Setează OpenAI API Key în Settings (scriptul folosește Whisper + GPT).";
      await ctx.runMutation(internal.processing.updateJobStatus, {
        jobId,
        status: "failed",
        error: msg,
      });
      throw new Error(msg);
    }

    try {
      // Fresh start: drop clips/shorts from any previous attempt
      await ctx.runMutation(internal.processing.deleteClipsByJob, { jobId });

      const resp = await vpsFetch("/pipeline", {
        method: "POST",
        body: JSON.stringify({
          youtube_url: job.videoUrl,
          openai_api_key: settings.openaiApiKey,
          cookies: settings.youtubeCookies || undefined,
          language: job.language || "en",
          num_shorts: job.numShorts,
          min_duration: job.minDuration,
          max_duration: job.maxDuration,
        }),
      });
      if (!resp.ok) {
        throw new Error(`VPS nu a pornit procesarea (HTTP ${resp.status}): ${(await resp.text()).slice(0, 300)}`);
      }
      const data = (await resp.json()) as { success: boolean; pipeline_id: string; error?: string };
      if (!data.success || !data.pipeline_id) {
        throw new Error(`VPS: ${data.error || "răspuns invalid"}`);
      }

      await ctx.runMutation(internal.processing.updateJobStatus, {
        jobId,
        status: "downloading",
        vpsPipelineId: data.pipeline_id,
        clearError: true,
      });

      await ctx.scheduler.runAfter(5_000, internal.vpsPipeline.pollPipeline, {
        jobId,
        userId,
        pipelineId: data.pipeline_id,
        startedAt: Date.now(),
        clipsSaved: false,
      });
    } catch (error) {
      const msg = error instanceof Error ? error.message : "Eroare necunoscută";
      await ctx.runMutation(internal.processing.updateJobStatus, {
        jobId,
        status: "failed",
        error: msg,
      });
      throw error;
    }
    return null;
  },
});

export const pollPipeline = internalAction({
  args: {
    jobId: v.id("jobs"),
    userId: v.id("users"),
    pipelineId: v.string(),
    startedAt: v.number(),
    clipsSaved: v.boolean(),
  },
  returns: v.null(),
  handler: async (ctx, args) => {
    const { jobId, userId, pipelineId, startedAt } = args;
    let clipsSaved = args.clipsSaved;

    const fail = async (msg: string) => {
      await ctx.runMutation(internal.processing.updateJobStatus, {
        jobId,
        status: "failed",
        error: msg,
      });
    };

    let st: VpsStatus;
    try {
      const resp = await vpsFetch(`/pipeline/${pipelineId}`);
      if (resp.status === 404) {
        await fail("VPS-ul a pierdut job-ul (probabil a fost repornit). Încearcă din nou.");
        return null;
      }
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      st = (await resp.json()) as VpsStatus;
    } catch (err) {
      // transient network error → retry unless we've been at it too long
      if (Date.now() - startedAt > MAX_RUNTIME_MS) {
        await fail(`VPS inaccesibil: ${err instanceof Error ? err.message : String(err)}`);
        return null;
      }
      await ctx.scheduler.runAfter(POLL_INTERVAL_MS, internal.vpsPipeline.pollPipeline, {
        ...args,
        clipsSaved,
      });
      return null;
    }

    // Persist the clip list as soon as the AI analysis is done, so the UI
    // shows the selected moments while the shorts are still rendering.
    if (!clipsSaved && st.clips && st.clips.length > 0) {
      await ctx.runMutation(internal.processing.deleteClipsByJob, { jobId });
      await ctx.runMutation(internal.processing.saveClips, {
        jobId,
        userId,
        clips: st.clips.map(c => ({
          title: c.title,
          description: c.reason || "",
          hashtags: [],
          startTime: c.startTime,
          endTime: c.endTime,
          transcriptExcerpt: c.hookLine || "",
          viralScore: c.viralScore,
          reason: c.reason || "",
          hookLine: c.hookLine || undefined,
        })),
      });
      clipsSaved = true;
    }

    if (st.state === "running") {
      if (Date.now() - startedAt > MAX_RUNTIME_MS) {
        await fail("Procesarea pe VPS a depășit 90 de minute.");
        return null;
      }
      await ctx.runMutation(internal.processing.updateJobStatus, {
        jobId,
        status: st.step,
        videoTitle: st.video_title && st.video_title !== "Podcast" && st.video_title !== "Source"
          ? st.video_title
          : undefined,
      });
      await ctx.scheduler.runAfter(POLL_INTERVAL_MS, internal.vpsPipeline.pollPipeline, {
        ...args,
        clipsSaved,
      });
      return null;
    }

    if (st.state === "failed") {
      await fail(st.error || "Scriptul a eșuat pe VPS fără mesaj de eroare.");
      return null;
    }

    // ── completed: pull the MP4s into Convex storage ──────────────────
    const clips = await ctx.runQuery(internal.processing.getClipsInternal, { jobId });
    // The script names files NN_Title.mp4 in the order of clips.json.
    // getClipsInternal returns them in insertion order == clips.json order.
    const byIndex = clips || [];

    let ok = 0;
    for (const out of st.outputs) {
      const clip = byIndex[out.index];
      if (!clip) continue;
      try {
        const fileResp = await fetch(`${VPS_URL}${out.download_url}`);
        if (!fileResp.ok) throw new Error(`download HTTP ${fileResp.status}`);
        const blob = await fileResp.blob();
        const uploadUrl = await ctx.storage.generateUploadUrl();
        const up = await fetch(uploadUrl, {
          method: "POST",
          headers: { "Content-Type": "video/mp4" },
          body: blob,
        });
        if (!up.ok) throw new Error(`storage upload HTTP ${up.status}`);
        const { storageId } = (await up.json()) as { storageId: string };
        await ctx.runMutation(internal.processing.upsertShort, {
          clipId: clip._id,
          jobId,
          userId,
          storageId: storageId as Id<"_storage">,
          fileName: out.name,
          duration: clip.endTime - clip.startTime,
          fileSize: out.size || blob.size,
          hasSubtitles: true,
        });
        ok++;
      } catch (err) {
        console.error(`[vpsPipeline] clip ${out.index} (${out.name}) failed:`, err);
      }
    }

    if (ok === 0) {
      await fail("Scriptul a terminat, dar niciun short nu a putut fi preluat de pe VPS.");
      return null;
    }
    await ctx.runMutation(internal.processing.updateJobStatus, {
      jobId,
      status: "completed",
      clearError: true,
    });
    console.log(`[vpsPipeline] job ${jobId}: ${ok}/${st.outputs.length} shorts stored (${st.elapsed}s on VPS)`);
    return null;
  },
});
