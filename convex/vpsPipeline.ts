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

      const musicMode = job.musicMode === "custom" || job.musicMode === "default" ? job.musicMode : "none";
      let musicUrl: string | undefined;
      if (musicMode === "custom") {
        if (!job.musicStorageId) throw new Error("Lipsește fișierul de muzică încărcat pentru acest job.");
        musicUrl = (await ctx.storage.getUrl(job.musicStorageId)) ?? undefined;
        if (!musicUrl) throw new Error("Fișierul de muzică nu mai există în storage.");
      }

      const resp = await vpsFetch("/pipeline", {
        method: "POST",
        body: JSON.stringify({
          youtube_url: job.videoUrl,
          music_mode: musicMode,
          music_url: musicUrl,
          openai_api_key: settings.openaiApiKey,
          gpt_model: settings.openaiModel || undefined,
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

    // ── completed: pull the MP4s into Convex storage, one file per action
    // step (each step is short and saves progress, so a crash never leaves
    // the job hanging and the watchdog can resume it).
    await ctx.scheduler.runAfter(0, internal.vpsPipeline.pullOutput, {
      jobId,
      userId,
      pipelineId,
      index: 0,
      ok: 0,
      lastErr: "",
    });
    return null;
  },
});

const PULL_TIMEOUT_MS = 8 * 60_000; // VPS → storage upload of one MP4

export const pullOutput = internalAction({
  args: {
    jobId: v.id("jobs"),
    userId: v.id("users"),
    pipelineId: v.string(),
    index: v.number(),
    ok: v.number(),
    lastErr: v.string(),
  },
  returns: v.null(),
  handler: async (ctx, args) => {
    const { jobId, userId, pipelineId, index } = args;
    let { ok, lastErr } = args;

    const fail = async (msg: string) => {
      await ctx.runMutation(internal.processing.updateJobStatus, { jobId, status: "failed", error: msg });
    };

    const resp = await vpsFetch(`/pipeline/${pipelineId}`);
    if (resp.status === 404) {
      await fail("Fișierele nu mai există pe VPS (au expirat). Rulează procesarea din nou.");
      return null;
    }
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const st = (await resp.json()) as VpsStatus;
    const outputs = st.outputs || [];

    if (index >= outputs.length) {
      if (ok === 0) {
        await fail(
          `Scriptul a terminat, dar niciun short nu a putut fi preluat de pe VPS${lastErr ? ` (${lastErr})` : ""}. Fișierele rămân pe VPS 3h — poți reîncerca preluarea.`,
        );
        return null;
      }
      await ctx.runMutation(internal.processing.updateJobStatus, { jobId, status: "completed", clearError: true });
      console.log(`[vpsPipeline] job ${jobId}: ${ok}/${outputs.length} shorts stored (${st.elapsed}s on VPS)`);
      return null;
    }

    // heartbeat so the watchdog knows we're alive
    await ctx.runMutation(internal.processing.updateJobStatus, { jobId, status: "generating" });

    const out = outputs[index];
    const clips = (await ctx.runQuery(internal.processing.getClipsInternal, { jobId })) || [];
    const clip = clips[out.index];
    if (clip) {
      try {
        // The VPS streams the file directly into Convex storage; the action
        // never holds the MP4 in memory (85 MB blobs killed the old approach).
        const uploadUrl = await ctx.storage.generateUploadUrl();
        const up = await vpsFetch(
          `/pipeline/${pipelineId}/upload`,
          { method: "POST", body: JSON.stringify({ name: out.name, upload_url: uploadUrl }) },
          PULL_TIMEOUT_MS,
        );
        if (!up.ok) throw new Error(`VPS upload HTTP ${up.status}: ${(await up.text()).slice(0, 200)}`);
        const data = (await up.json()) as { success: boolean; storage_id?: string; size?: number; error?: string };
        if (!data.success || !data.storage_id) throw new Error(data.error || "VPS nu a returnat storageId");
        await ctx.runMutation(internal.processing.upsertShort, {
          clipId: clip._id,
          jobId,
          userId,
          storageId: data.storage_id as Id<"_storage">,
          fileName: out.name,
          duration: clip.endTime - clip.startTime,
          fileSize: out.size || data.size || 0,
          hasSubtitles: true,
        });
        ok++;
      } catch (err) {
        lastErr = err instanceof Error ? err.message : String(err);
        console.error(`[vpsPipeline] clip ${out.index} (${out.name}) failed:`, err);
      }
    }

    await ctx.scheduler.runAfter(0, internal.vpsPipeline.pullOutput, {
      jobId,
      userId,
      pipelineId,
      index: index + 1,
      ok,
      lastErr,
    });
    return null;
  },
});

// ── Watchdog (cron, every 5 min): jobs with no heartbeat for 15 min mean the
// polling/pull action died. Re-attach to the VPS instead of hanging forever.
export const watchdog = internalAction({
  args: {},
  returns: v.number(),
  handler: async (ctx): Promise<number> => {
    const stuck = await ctx.runQuery(internal.processing.listStuckJobs, { staleMs: 15 * 60_000 });
    for (const j of stuck) {
      const fail = (msg: string) =>
        ctx.runMutation(internal.processing.updateJobStatus, { jobId: j.jobId, status: "failed", error: msg });
      try {
        const resp = await vpsFetch(`/pipeline/${j.pipelineId}`);
        if (resp.status === 404) {
          await fail("Jobul s-a blocat și VPS-ul nu îl mai are. Încearcă din nou.");
          continue;
        }
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const st = (await resp.json()) as VpsStatus;
        console.log(`[watchdog] job ${j.jobId} stale (${Math.round((Date.now() - j.lastSeen) / 60000)} min), VPS state=${st.state}`);
        if (st.state === "failed") {
          await fail(st.error || "Scriptul a eșuat pe VPS fără mesaj de eroare.");
        } else {
          // running → resume polling; completed → re-pull (pollPipeline hands off to pullOutput)
          await ctx.runMutation(internal.processing.updateJobStatus, { jobId: j.jobId, status: "generating" });
          await ctx.scheduler.runAfter(0, internal.vpsPipeline.pollPipeline, {
            jobId: j.jobId,
            userId: j.userId,
            pipelineId: j.pipelineId,
            startedAt: Date.now(),
            clipsSaved: true,
          });
        }
      } catch (err) {
        console.error(`[watchdog] job ${j.jobId}:`, err);
        if (Date.now() - j.lastSeen > 3 * 3600_000) await fail("Jobul s-a blocat și VPS-ul nu răspunde.");
      }
    }
    return stuck.length;
  },
});

// ── Ops helpers: re-pull finished outputs for jobs that failed only at the
// "copy MP4s from VPS into storage" step (files stay on the VPS for 3h). ──
export const retryPull = internalAction({
  args: { jobId: v.optional(v.id("jobs")) },
  returns: v.number(),
  handler: async (ctx, { jobId }): Promise<number> => {
    type Target = { jobId: Id<"jobs">; userId: Id<"users">; pipelineId: string };
    const all: Target[] = await ctx.runQuery(internal.processing.listFailedPulls, {});
    const targets = jobId ? all.filter((j) => j.jobId === jobId) : all;
    for (const t of targets) {
      await ctx.runMutation(internal.processing.updateJobStatus, {
        jobId: t.jobId,
        status: "generating",
        clearError: true,
      });
      await ctx.scheduler.runAfter(0, internal.vpsPipeline.pollPipeline, {
        jobId: t.jobId,
        userId: t.userId,
        pipelineId: t.pipelineId,
        startedAt: Date.now(),
        clipsSaved: true,
      });
    }
    return targets.length;
  },
});

// User-facing: re-pull the finished MP4s for my own job (files stay on the VPS 3h).
export const retryPullForJob = action({
  args: { jobId: v.id("jobs") },
  returns: v.null(),
  handler: async (ctx, { jobId }) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) throw new Error("Not authenticated");
    const job = await ctx.runQuery(internal.processing.getJobInternal, { jobId });
    if (!job || job.userId !== userId) throw new Error("Job not found");
    if (!job.vpsPipelineId) throw new Error("Jobul nu a rulat pe VPS.");
    const resp = await vpsFetch(`/pipeline/${job.vpsPipelineId}`);
    if (resp.status === 404) {
      throw new Error("Fișierele nu mai există pe VPS (au expirat). Rulează procesarea din nou.");
    }
    await ctx.runMutation(internal.processing.updateJobStatus, {
      jobId,
      status: "generating",
      clearError: true,
    });
    await ctx.scheduler.runAfter(0, internal.vpsPipeline.pollPipeline, {
      jobId,
      userId,
      pipelineId: job.vpsPipelineId,
      startedAt: Date.now(),
      clipsSaved: true,
    });
    return null;
  },
});
