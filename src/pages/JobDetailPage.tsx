import { useAction, useMutation, useQuery } from "convex/react";
import {
  ArrowLeft,
  CheckCircle2,
  ClipboardCopy,
  Download,
  ExternalLink,
  Film,
  Flame,
  Loader2,
  MessageSquareQuote,
  Mic,
  Play,
  RefreshCw,
  Server,
  Sparkles,
  Terminal,
  Video,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Progress } from "@/components/ui/progress";
import { downloadScript } from "@/lib/scriptGenerator";
import {
  generateAssSubtitles,
  getSegmentsForClip,
  type SubtitleSegment,
} from "@/lib/subtitles";
import {
  type ClipConfig,
  type ProcessingProgress,
  processAllClips,
} from "@/lib/videoProcessor";
import { api } from "../../convex/_generated/api";
import type { Id } from "../../convex/_generated/dataModel";

/**
 * Hosts the Convex /api/video-proxy allowlists. Previously only
 * googlevideo.com URLs were routed through the proxy — Piped/VPS URLs
 * were fetched directly by the browser, where missing CORS headers or a
 * slow instance made the download hang at 0% forever.
 */
function shouldProxy(rawUrl: string): boolean {
  try {
    const h = new URL(rawUrl).hostname;
    return (
      h.endsWith(".googlevideo.com") ||
      h.endsWith(".youtube.com") ||
      h.endsWith(".ytimg.com") ||
      h === "piped.private.coffee" ||
      h.endsWith(".piped.private.coffee") ||
      h.endsWith(".kavin.rocks") ||
      h.endsWith(".r4fo.com") ||
      h.endsWith(".adminforge.de") ||
      h === "76.13.133.153"
    );
  } catch {
    return false;
  }
}

function formatSeconds(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0)
    return `${h}:${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function ViralScoreBadge({ score }: { score: number }) {
  const color =
    score >= 8
      ? "text-green-400 bg-green-400/10"
      : score >= 5
        ? "text-amber-400 bg-amber-400/10"
        : "text-muted-foreground bg-muted";
  return (
    <span
      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold ${color}`}
    >
      <Flame className="size-3" />
      {score}/10
    </span>
  );
}

function StatusProgress({
  status,
  generatingProgress,
}: {
  status: string;
  generatingProgress?: ProcessingProgress | null;
}) {
  const steps = [
    { key: "downloading", label: "Obținere info video" },
    { key: "transcribing", label: "Transcriere automată" },
    { key: "analyzing", label: "Analiză AI & Selecție" },
    { key: "generating", label: "Generare Shorts" },
    { key: "completed", label: "Finalizat" },
  ];
  const stepOrder = [
    "pending",
    "downloading",
    "transcribing",
    "analyzing",
    "generating",
    "completed",
  ];
  const currentIndex = stepOrder.indexOf(status);

  return (
    <div className="border rounded-xl p-6 bg-card">
      <h3 className="text-sm font-semibold text-muted-foreground mb-4 uppercase tracking-wider">
        Progres Procesare
      </h3>
      <div className="space-y-3">
        {steps.map(step => {
          const stepIdx = stepOrder.indexOf(step.key);
          const isComplete = currentIndex > stepIdx;
          const isCurrent = currentIndex === stepIdx;
          return (
            <div key={step.key}>
              <div className="flex items-center gap-3">
                {isComplete ? (
                  <CheckCircle2 className="size-5 text-green-400 shrink-0" />
                ) : isCurrent ? (
                  <Loader2 className="size-5 text-primary animate-spin shrink-0" />
                ) : (
                  <div className="size-5 rounded-full border-2 border-muted shrink-0" />
                )}
                <span
                  className={`text-sm ${isCurrent ? "text-foreground font-medium" : isComplete ? "text-muted-foreground" : "text-muted-foreground/60"}`}
                >
                  {step.label}
                </span>
              </div>
              {/* Show generating progress inline */}
              {step.key === "generating" && isCurrent && generatingProgress && (
                <div className="ml-8 mt-2 space-y-2">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">
                      {generatingProgress.message}
                    </span>
                    <span className="font-mono text-primary">
                      {generatingProgress.percent}%
                    </span>
                  </div>
                  <Progress
                    value={generatingProgress.percent}
                    className="h-1.5"
                  />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// Video player component for generated shorts
function ShortVideoPlayer({
  url,
  title: _title,
}: {
  url: string;
  title: string;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);

  return (
    <div className="relative aspect-[9/16] max-h-[400px] rounded-lg overflow-hidden bg-black group">
      <video
        ref={videoRef}
        src={url}
        className="w-full h-full object-contain"
        playsInline
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
        onEnded={() => setIsPlaying(false)}
        controls
      />
      {!isPlaying && (
        <div
          className="absolute inset-0 flex items-center justify-center bg-black/30 cursor-pointer group-hover:bg-black/20 transition-colors"
          onClick={() => videoRef.current?.play()}
        >
          <div className="size-16 rounded-full bg-white/90 flex items-center justify-center shadow-lg">
            <Play className="size-7 text-black ml-1" />
          </div>
        </div>
      )}
    </div>
  );
}

export function JobDetailPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const job = useQuery(
    api.jobs.get,
    jobId ? { id: jobId as Id<"jobs"> } : "skip",
  );
  const clips = useQuery(
    api.clips.listByJob,
    jobId ? { jobId: jobId as Id<"jobs"> } : "skip",
  );
  const shorts = useQuery(
    api.shorts.listByJob,
    jobId ? { jobId: jobId as Id<"jobs"> } : "skip",
  );
  const settings = useQuery(api.settings.get);
  const generateUploadUrl = useMutation(api.shorts.generateUploadUrl);
  const saveShort = useMutation(api.shorts.save);
  const markJobCompleted = useMutation(api.shorts.markJobCompleted);
  const refreshDownloadUrls = useAction(api.processing.refreshDownloadUrls);
  const startProcessing = useAction(api.processing.processJob);
  const startServerProcessing = useAction(api.serverProcessing.processJobOnServer);
  const startVpsFetch = useAction(api.serverProcessing.startVpsFetch);
  const getVpsFetchStatus = useAction(api.serverProcessing.getVpsFetchStatus);


  const [processing, setProcessing] = useState(false);
  const [startingBackend, setStartingBackend] = useState(false);
  const [serverProcessing, setServerProcessing] = useState(false);
  const serverModeRef = useRef(false);
  const [progress, setProgress] = useState<ProcessingProgress | null>(null);
  const [generatedBlobs, setGeneratedBlobs] = useState<
    Map<number, { blob: Blob; url: string }>
  >(new Map());
  const autoGenerateTriggered = useRef(false);

  const copyToClipboard = (text: string, label: string) => {
    navigator.clipboard.writeText(text);
    toast.success(`${label} copiat!`);
  };

  // Parse transcript segments from job data
  const getTranscriptSegments = useCallback((): SubtitleSegment[] => {
    if (!job?.transcriptSegments) return [];
    try {
      return JSON.parse(job.transcriptSegments);
    } catch {
      return [];
    }
  }, [job?.transcriptSegments]);

  // Generate shorts in the browser
  const handleGenerateShorts = useCallback(async () => {
    if (!job || !clips || clips.length === 0 || !jobId) return;
    if (processing) return;

    setProcessing(true);
    setProgress(null);

    try {
      // Get video + audio download URLs
      let videoUrl = job.videoDownloadUrl;
      let audioUrl = job.audioDownloadUrl;
      const isExpired =
        !videoUrl ||
        (job.videoDownloadExpiry && Date.now() > job.videoDownloadExpiry);

      if (isExpired) {
        setProgress({
          clipIndex: 0,
          totalClips: clips.length,
          stage: "loading",
          percent: 0,
          message: "Se obține link-ul de descărcare...",
        });
        const fresh = await refreshDownloadUrls({
          jobId: jobId as Id<"jobs">,
        });
        videoUrl = fresh.videoUrl ?? undefined;
        audioUrl = fresh.audioUrl ?? undefined;
      }

      if (!videoUrl) {
        toast.error("Nu s-a putut obține link-ul de descărcare al video-ului.");
        setProcessing(false);
        return;
      }

      // Route through CORS proxy (YouTube CDN / Piped don't allow browser fetch)
      const convexSiteUrl = import.meta.env.VITE_CONVEX_SITE_URL;
      if (convexSiteUrl && videoUrl && shouldProxy(videoUrl)) {
        videoUrl = `${convexSiteUrl}/api/video-proxy?url=${encodeURIComponent(videoUrl)}`;
      }
      if (convexSiteUrl && audioUrl && shouldProxy(audioUrl)) {
        audioUrl = `${convexSiteUrl}/api/video-proxy?url=${encodeURIComponent(audioUrl)}`;
      }

      // Get transcript segments for subtitles
      const allSegments = getTranscriptSegments();

      // Prepare clip configs with subtitles
      const sortedClips = [...clips].sort(
        (a, b) => b.viralScore - a.viralScore,
      );
      const clipConfigs: ClipConfig[] = sortedClips.map((clip, i) => {
        const clipSegments = getSegmentsForClip(
          allSegments,
          clip.startTime,
          clip.endTime,
        );
        const assContent =
          clipSegments.length > 0
            ? generateAssSubtitles(clipSegments)
            : undefined;

        return {
          index: i,
          title: clip.title,
          startTime: clip.startTime,
          endTime: clip.endTime,
          assSubtitles: assContent,
        };
      });

      // Process all clips (pass audio URL for muxing if available).
      // If the direct download fails or stalls (YouTube blocking, expired
      // URL, dead Piped instance...), fall back to sourcing the video
      // through the VPS: it downloads with yt-dlp and serves the file
      // back with Range support, relayed through the Convex proxy.
      let results: Map<number, Blob>;
      try {
        results = await processAllClips(
          videoUrl,
          clipConfigs,
          setProgress,
          audioUrl,
        );
      } catch (directErr) {
        console.warn(
          "[generate] Direct download failed → VPS fallback:",
          directErr,
        );
        toast.info("Descărcarea directă a eșuat — trec prin VPS...");
        setProgress({
          clipIndex: 0,
          totalClips: clips.length,
          stage: "downloading",
          percent: 0,
          message: "Se descarcă video-ul pe server (VPS)...",
        });

        const { fetchId, error: fetchErr } = await startVpsFetch({
          jobId: jobId as Id<"jobs">,
        });
        if (!fetchId) {
          throw new Error(
            `Descărcare directă eșuată (${directErr instanceof Error ? directErr.message : "eroare"}), ` +
              `iar VPS-ul nu răspunde: ${fetchErr}`,
          );
        }

        // Poll until the VPS finished downloading (max 15 min)
        let vpsFileUrl: string | null = null;
        const deadline = Date.now() + 15 * 60 * 1000;
        while (Date.now() < deadline) {
          await new Promise(r => setTimeout(r, 4000));
          const st = await getVpsFetchStatus({ fetchId });
          if (st.status === "ready" && st.fileUrl) {
            vpsFileUrl = st.fileUrl;
            break;
          }
          if (st.status === "error") {
            throw new Error(`Descărcare VPS eșuată: ${st.error || "necunoscut"}`);
          }
          setProgress({
            clipIndex: 0,
            totalClips: clips.length,
            stage: "downloading",
            percent: 5,
            message: `Se descarcă pe server (VPS)...${st.size ? ` ${(st.size / 1e6).toFixed(0)}MB` : ""}`,
          });
        }
        if (!vpsFileUrl) {
          throw new Error("Descărcarea pe VPS a durat prea mult (15 min)");
        }

        // Relay through the Convex proxy (VPS is plain http — the browser
        // can't fetch it directly from an https page)
        const vpsProxied = convexSiteUrl
          ? `${convexSiteUrl}/api/video-proxy?url=${encodeURIComponent(vpsFileUrl)}`
          : vpsFileUrl;

        // VPS file is already muxed (video+audio) → no separate audio URL
        results = await processAllClips(
          vpsProxied,
          clipConfigs,
          setProgress,
          null,
        );
      }

      // Create blob URLs for preview
      const newBlobs = new Map<number, { blob: Blob; url: string }>();
      for (const [index, blob] of results) {
        newBlobs.set(index, { blob, url: URL.createObjectURL(blob) });
      }
      setGeneratedBlobs(newBlobs);

      // Upload each short to Convex storage
      for (const [index, blob] of results) {
        const clip = sortedClips[index];
        if (!clip) continue;

        try {
          const uploadUrl = await generateUploadUrl();
          const uploadResp = await fetch(uploadUrl, {
            method: "POST",
            headers: { "Content-Type": "video/mp4" },
            body: blob,
          });
          const { storageId } = await uploadResp.json();

          const safeTitle = clip.title
            .replace(/[^a-zA-Z0-9\s-]/g, "")
            .replace(/\s+/g, "_")
            .substring(0, 40);
          const fileName = `${String(index + 1).padStart(2, "0")}_${safeTitle}.mp4`;

          await saveShort({
            clipId: clip._id,
            jobId: jobId as Id<"jobs">,
            storageId,
            fileName,
            duration: clip.endTime - clip.startTime,
            fileSize: blob.size,
            hasSubtitles: allSegments.length > 0,
          });
        } catch (e) {
          console.error(`Failed to save short ${index}:`, e);
        }
      }

      // Mark job as completed now that shorts are generated
      await markJobCompleted({ jobId: jobId as Id<"jobs"> });

      toast.success(`${results.size} shorts generate și salvate!`);
    } catch (error) {
      console.error("Processing error:", error);
      toast.error(
        `Eroare: ${error instanceof Error ? error.message : "Eroare necunoscută"}`,
      );
    } finally {
      setProcessing(false);
    }
  }, [
    job,
    clips,
    jobId,
    processing,
    getTranscriptSegments,
    generateUploadUrl,
    saveShort,
    markJobCompleted,
    refreshDownloadUrls,
    startVpsFetch,
    getVpsFetchStatus,
  ]);

  // Auto-trigger generation when job reaches "generating" status
  useEffect(() => {
    if (
      job?.status === "generating" &&
      clips &&
      clips.length > 0 &&
      shorts !== undefined &&
      shorts.length === 0 &&
      !processing &&
      !autoGenerateTriggered.current &&
      !serverModeRef.current
    ) {
      autoGenerateTriggered.current = true;
      console.log("[AutoGenerate] Starting automatic short generation...");
      handleGenerateShorts();
    }
  }, [job?.status, clips, shorts, processing, handleGenerateShorts]);

  // Reset auto-trigger flag when job changes
  useEffect(() => {
    autoGenerateTriggered.current = false;
  }, [jobId]);

  if (job === undefined || clips === undefined) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="size-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (!job) {
    return (
      <div className="p-6 md:p-8 max-w-4xl mx-auto text-center py-20">
        <h2 className="text-xl font-semibold mb-2">Job negăsit</h2>
        <Button asChild variant="outline">
          <Link to="/dashboard">
            <ArrowLeft className="size-4 mr-1" />
            Înapoi la Dashboard
          </Link>
        </Button>
      </div>
    );
  }

  const isProcessing = [
    "pending",
    "downloading",
    "transcribing",
    "analyzing",
    "generating",
  ].includes(job.status);
  const isGenerating = job.status === "generating";
  const isWaitingTranscript = job.status === "waiting_transcript";
  const sortedClips = clips
    ? [...clips].sort((a, b) => b.viralScore - a.viralScore)
    : [];

  // Map shorts by clipId for easy lookup
  const shortsByClipId = new Map((shorts || []).map(s => [s.clipId, s]));

  const hasAnyShorts = (shorts || []).length > 0;

  return (
    <div className="p-6 md:p-8 max-w-4xl mx-auto">
      {/* Header */}
      <Button
        asChild
        variant="ghost"
        size="sm"
        className="mb-4 -ml-2 text-muted-foreground"
      >
        <Link to="/dashboard">
          <ArrowLeft className="size-4 mr-1" />
          Dashboard
        </Link>
      </Button>

      {/* Video info */}
      <div className="border rounded-xl overflow-hidden mb-6">
        {job.videoThumbnail && (
          <div className="relative aspect-video max-h-[200px] bg-muted overflow-hidden">
            <img
              src={job.videoThumbnail}
              alt=""
              className="w-full h-full object-cover"
              onError={e => {
                (e.target as HTMLImageElement).parentElement!.style.display =
                  "none";
              }}
            />
            <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent" />
          </div>
        )}
        <div className="p-5">
          <h1 className="text-xl md:text-2xl font-bold mb-2">
            {job.videoTitle || "Se procesează..."}
          </h1>
          <div className="flex flex-wrap items-center gap-3 text-sm text-muted-foreground">
            <a
              href={job.videoUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 hover:text-primary transition-colors"
            >
              <ExternalLink className="size-3.5" />
              Deschide pe YouTube
            </a>
            <span>•</span>
            <span>{job.numShorts} shorts solicitate</span>
            <span>•</span>
            <span>
              {job.minDuration}s – {job.maxDuration}s
            </span>
            <span>•</span>
            <span>{job.language.toUpperCase()}</span>
          </div>
        </div>
      </div>

      {/* Pending: start processing or download script */}
      {job.status === "pending" && (
        <div className="space-y-4 mb-6">
          {/* Primary: Process in Browser */}
          <div className="border-2 border-primary/40 bg-primary/5 rounded-xl p-6">
            <div className="flex flex-col items-center text-center gap-4">
              <div className="size-14 rounded-full bg-primary/10 flex items-center justify-center">
                <Sparkles className="size-7 text-primary" />
              </div>
              <div>
                <h3 className="text-lg font-bold mb-1">
                  Procesează în Browser
                </h3>
                <p className="text-sm text-muted-foreground max-w-md">
                  Totul se face automat: transcriere → analiză AI → face
                  detection → crop 9:16 → subtitrări. Funcționează și pentru
                  video-uri lungi (podcast-uri) — se descarcă doar segmentele
                  clipurilor. Nu ai nevoie de nimic instalat.
                </p>
              </div>
              <Button
                size="lg"
                className="w-full max-w-xs h-12 text-base"
                disabled={startingBackend}
                onClick={async () => {
                  if (!jobId) return;
                  setStartingBackend(true);
                  try {
                    const res = await startProcessing({
                      jobId: jobId as Id<"jobs">,
                    });
                    if (res && res.success === false) {
                      toast.error(`Analiza a eșuat: ${res.error || "eroare necunoscută"}`);
                    }
                  } catch (err) {
                    console.error("processJob failed:", err);
                    toast.error(
                      `Eroare: ${err instanceof Error ? err.message : "Eroare necunoscută"}`,
                    );
                  } finally {
                    setStartingBackend(false);
                  }
                }}
              >
                {startingBackend ? (
                  <>
                    <Loader2 className="size-5 mr-2 animate-spin" />
                    Se pornește...
                  </>
                ) : (
                  <>
                    <Sparkles className="size-5 mr-2" />
                    Start Procesare
                  </>
                )}
              </Button>
            </div>
          </div>

          {/* Secondary: Process on Server */}
          <div className="border border-blue-500/30 rounded-xl p-4 hover:bg-blue-500/5 transition-colors">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Server className="size-5 text-blue-400" />
                <div>
                  <span className="text-sm font-medium">Procesează pe Server</span>
                  <p className="text-xs text-muted-foreground">
                    ffmpeg nativ pe VPS — fără limită de memorie, orice dimensiune video
                  </p>
                </div>
              </div>
              <Button
                size="sm"
                variant="outline"
                className="border-blue-500/50 text-blue-400 hover:bg-blue-500/10"
                disabled={startingBackend || serverProcessing}
                onClick={async () => {
                  if (!jobId) return;
                  serverModeRef.current = true;
                  setStartingBackend(true);
                  try {
                    // First trigger backend pipeline (transcribe + AI analysis)
                    const res = await startProcessing({
                      jobId: jobId as Id<"jobs">,
                    });
                    if (res && res.success === false) {
                      // Analysis failed → there are no clips; VPS processing
                      // would only throw "Nu există clipuri pentru acest job"
                      throw new Error(res.error || "Analiza a eșuat");
                    }
                    toast.info("Analiză completă — pornesc procesarea pe server...");
                    // Now trigger VPS processing
                    setServerProcessing(true);
                    setStartingBackend(false);
                    await startServerProcessing({ jobId: jobId as Id<"jobs"> });
                    toast.success("Shorts-urile au fost generate pe server! 🎉");
                  } catch (_err) {
                    console.error("Server processing failed:", _err);
                    const msg = _err instanceof Error ? _err.message : "Eroare necunoscută";
                    toast.error(`Eroare procesare server: ${msg}`);
                  } finally {
                    setStartingBackend(false);
                    setServerProcessing(false);
                    serverModeRef.current = false;
                  }
                }}
              >
                {startingBackend ? (
                  <>
                    <Loader2 className="size-4 mr-1.5 animate-spin" />
                    Analiză...
                  </>
                ) : serverProcessing ? (
                  <>
                    <Loader2 className="size-4 mr-1.5 animate-spin" />
                    Procesare VPS...
                  </>
                ) : (
                  <>
                    <Server className="size-4 mr-1.5" />
                    Start Server
                  </>
                )}
              </Button>
            </div>
          </div>

          {/* Tertiary: Download Script */}
          <Collapsible>
            <CollapsibleTrigger className="w-full">
              <div className="border rounded-xl p-4 hover:bg-accent/30 transition-colors flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Terminal className="size-5 text-muted-foreground" />
                  <span className="text-sm font-medium text-muted-foreground">
                    Sau descarcă scriptul pentru procesare locală
                  </span>
                </div>
                <ArrowLeft className="size-4 text-muted-foreground -rotate-90" />
              </div>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <div className="border border-t-0 rounded-b-xl p-5 -mt-2 pt-6">
                <div className="flex flex-col items-center text-center gap-3">
                  <p className="text-sm text-muted-foreground max-w-md">
                    Scriptul procesează totul local pe calculatorul tău:
                    download video → transcriere Whisper → analiză AI → face
                    crop 9:16 → burn subtitrări.
                  </p>
                  <div className="flex gap-2 w-full max-w-xs">
                    <Button
                      size="lg"
                      variant="outline"
                      className="flex-1 h-11"
                      onClick={() => {
                        if (!job) return;
                        const videoId =
                          job.videoUrl.match(
                            /(?:v=|youtu\.be\/|shorts\/)([a-zA-Z0-9_-]{11})/,
                          )?.[1] || "video";
                        downloadScript(
                          {
                            videoUrl: job.videoUrl,
                            language: job.language,
                            numShorts: job.numShorts,
                            minDuration: job.minDuration,
                            maxDuration: job.maxDuration,
                            openaiApiKey: settings?.openaiApiKey || undefined,
                          },
                          "py",
                          `shortscut_${videoId}.py`,
                        );
                        toast.success(
                          "Script descărcat! Rulează: python shortscut_*.py",
                        );
                      }}
                    >
                      <Download className="size-5 mr-2" />
                      Descarcă .py
                    </Button>
                    <Button
                      size="lg"
                      variant="ghost"
                      className="h-11 px-3 text-muted-foreground"
                      onClick={() => {
                        if (!job) return;
                        const videoId =
                          job.videoUrl.match(
                            /(?:v=|youtu\.be\/|shorts\/)([a-zA-Z0-9_-]{11})/,
                          )?.[1] || "video";
                        downloadScript(
                          {
                            videoUrl: job.videoUrl,
                            language: job.language,
                            numShorts: job.numShorts,
                            minDuration: job.minDuration,
                            maxDuration: job.maxDuration,
                            openaiApiKey: settings?.openaiApiKey || undefined,
                          },
                          "sh",
                          `shortscut_${videoId}.sh`,
                        );
                        toast.success(
                          "Script descărcat! Rulează: chmod +x shortscut_*.sh && ./shortscut_*.sh",
                        );
                      }}
                    >
                      .sh
                    </Button>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Necesită: python3, ffmpeg, yt-dlp
                    {!settings?.openaiApiKey && " + OpenAI API key"}
                  </p>
                  <div className="text-xs text-muted-foreground bg-muted rounded-lg p-3 w-full max-w-md font-mono">
                    python shortscut_*.py
                  </div>
                </div>
              </div>
            </CollapsibleContent>
          </Collapsible>
        </div>
      )}

      {/* Processing status (including generating step) */}
      {isProcessing && job.status !== "pending" && (
        <StatusProgress
          status={job.status}
          generatingProgress={isGenerating ? progress : null}
        />
      )}

      {/* Waiting for transcript (legacy fallback) */}
      {isWaitingTranscript && (
        <div className="border border-amber-500/30 bg-amber-500/5 rounded-xl p-5">
          <div className="flex items-center gap-2 mb-2">
            <Mic className="size-5 text-amber-400" />
            <h3 className="font-bold">Transcriere în așteptare</h3>
          </div>
          <p className="text-sm text-muted-foreground">
            Video-ul nu are subtitrări auto-generate pe YouTube. Adaugă un{" "}
            <Link to="/settings" className="text-primary underline">
              OpenAI API Key
            </Link>{" "}
            în Settings pentru transcriere automată cu Whisper, sau procesează
            din nou.
          </p>
        </div>
      )}

      {/* Error */}
      {job.status === "failed" && (
        <div className="border border-destructive/30 bg-destructive/5 rounded-xl p-5">
          <h3 className="font-semibold text-destructive mb-1">Eroare</h3>
          <p className="text-sm text-destructive/80 mb-3">{job.error}</p>
          <Button
            size="sm"
            variant="outline"
            disabled={startingBackend}
            onClick={async () => {
              if (!jobId) return;
              setStartingBackend(true);
              try {
                const res = await startProcessing({
                  jobId: jobId as Id<"jobs">,
                });
                if (res && res.success === false) {
                  toast.error(`Eroare: ${res.error || "necunoscută"}`);
                } else {
                  toast.success("Procesarea a reînceput!");
                }
              } catch (err) {
                toast.error(
                  `Eroare: ${err instanceof Error ? err.message : "Eroare necunoscută"}`,
                );
              } finally {
                setStartingBackend(false);
              }
            }}
          >
            {startingBackend ? (
              <>
                <Loader2 className="size-4 mr-1.5 animate-spin" />
                Se repornește...
              </>
            ) : (
              <>
                <Sparkles className="size-4 mr-1.5" />
                Încearcă din nou
              </>
            )}
          </Button>
        </div>
      )}

      {/* Show clips when generating or completed */}
      {(isGenerating || job.status === "completed") &&
        sortedClips.length > 0 && (
          <>
            {/* Generating info banner */}
            {isGenerating && !processing && !hasAnyShorts && (
              <div className="border border-primary/30 bg-primary/5 rounded-xl p-5 mb-6 mt-4">
                <div className="flex items-center gap-3">
                  <Loader2 className="size-5 text-primary animate-spin" />
                  <div>
                    <h3 className="font-bold">Se generează Shorts...</h3>
                    <p className="text-sm text-muted-foreground">
                      Procesarea se face local în browser — descărcare video →
                      face detection → crop 9:16 → subtitrări → encode MP4.
                    </p>
                  </div>
                </div>
                {/* Refresh URL / server processing fallbacks */}
                <div className="mt-3 pt-3 border-t border-primary/20 flex items-center justify-between gap-2 flex-wrap">
                  <span className="text-xs text-muted-foreground">
                    Blocat? Reîmprospătează URL-ul sau procesează pe server.
                  </span>
                  <Button
                    size="sm"
                    variant="outline"
                    className="text-xs h-7 border-blue-500/50 text-blue-400 hover:bg-blue-500/10"
                    disabled={serverProcessing}
                    onClick={async () => {
                      if (!jobId) return;
                      serverModeRef.current = true;
                      setServerProcessing(true);
                      try {
                        await startServerProcessing({ jobId: jobId as Id<"jobs"> });
                        toast.success("Shorts-urile au fost generate pe server! 🎉");
                      } catch (_err) {
                        console.error("Server processing failed:", _err);
                        toast.error(
                          `Eroare procesare server: ${_err instanceof Error ? _err.message : "necunoscută"}`,
                        );
                      } finally {
                        setServerProcessing(false);
                        serverModeRef.current = false;
                      }
                    }}
                  >
                    <Server className="size-3 mr-1" />
                    Procesează pe Server
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    className="text-xs h-7"
                    disabled={serverProcessing}
                    onClick={async () => {
                      if (!jobId) return;
                      setServerProcessing(true);
                      try {
                        const fresh = await refreshDownloadUrls({ jobId: jobId as Id<"jobs"> });
                        if (fresh.videoUrl) {
                          toast.success("URL reîmprospătat! Reîncearcă generarea.");
                        } else {
                          toast.error("Nu s-a putut obține un URL nou.");
                        }
                      } catch (_err) {
                        console.error("Refresh failed:", _err);
                        toast.error("Eroare la reîmprospătare URL");
                      } finally {
                        setServerProcessing(false);
                      }
                    }}
                  >
                    {serverProcessing ? (
                      <><Loader2 className="size-3 mr-1 animate-spin" /> Se reîmprospătează...</>
                    ) : (
                      <><RefreshCw className="size-3 mr-1" /> Reîmprospătează URL</>
                    )}
                  </Button>
                </div>
              </div>
            )}

            {/* Transcript info */}
            {job.transcript && (
              <div className="border border-green-500/30 bg-green-500/5 rounded-xl p-3 mb-4 mt-4">
                <div className="flex items-center gap-2">
                  <Mic className="size-4 text-green-400" />
                  <span className="text-sm text-green-400 font-medium">
                    {job.transcript.startsWith("[Source: whisper]")
                      ? "Transcris cu Whisper — timestamp-uri precise"
                      : "Transcript YouTube — subtitrări auto-generate"}
                  </span>
                </div>
              </div>
            )}

            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-bold">
                {sortedClips.length} Clips Identificate
              </h2>
            </div>

            <div className="space-y-4 mb-8">
              {sortedClips.map((clip, i) => {
                const duration = clip.endTime - clip.startTime;
                const existingShort = shortsByClipId.get(clip._id);
                const generatedBlob = generatedBlobs.get(i);

                return (
                  <Collapsible key={clip._id}>
                    <div className="border rounded-xl overflow-hidden">
                      {/* Clip header */}
                      <CollapsibleTrigger className="w-full p-4 md:p-5 text-left hover:bg-accent/30 transition-colors">
                        <div className="flex items-start justify-between gap-3">
                          <div className="flex-1">
                            <div className="flex items-center gap-2 mb-1 flex-wrap">
                              <span className="text-xs font-mono text-muted-foreground bg-muted px-2 py-0.5 rounded">
                                #{i + 1}
                              </span>
                              <ViralScoreBadge score={clip.viralScore} />
                              <span className="text-xs text-muted-foreground">
                                {formatSeconds(clip.startTime)} →{" "}
                                {formatSeconds(clip.endTime)} ({duration}s)
                              </span>
                              {existingShort && (
                                <Badge
                                  variant="secondary"
                                  className="text-xs bg-green-500/10 text-green-400 border-green-500/20"
                                >
                                  <Film className="size-3 mr-1" />
                                  Video generat
                                </Badge>
                              )}
                            </div>
                            <h3 className="font-semibold">{clip.title}</h3>
                            <p className="text-sm text-muted-foreground mt-1 line-clamp-1">
                              {clip.reason}
                            </p>
                          </div>
                        </div>
                      </CollapsibleTrigger>

                      {/* Clip details */}
                      <CollapsibleContent>
                        <div className="border-t p-4 md:p-5 space-y-4 bg-muted/20">
                          {/* Generated video preview */}
                          {(existingShort?.url || generatedBlob?.url) && (
                            <div>
                              <div className="text-xs font-semibold text-muted-foreground mb-2 uppercase tracking-wider flex items-center gap-1.5">
                                <Video className="size-3" />
                                Preview Short
                              </div>
                              <div className="flex gap-4 items-start">
                                <ShortVideoPlayer
                                  url={
                                    existingShort?.url ||
                                    generatedBlob?.url ||
                                    ""
                                  }
                                  title={clip.title}
                                />
                                <div className="flex flex-col gap-2">
                                  <Button
                                    size="sm"
                                    variant="outline"
                                    onClick={() => {
                                      const url =
                                        existingShort?.url ||
                                        generatedBlob?.url;
                                      if (!url) return;
                                      const a = document.createElement("a");
                                      a.href = url;
                                      a.download = `${String(i + 1).padStart(2, "0")}_${clip.title.replace(/[^a-zA-Z0-9]/g, "_").substring(0, 30)}.mp4`;
                                      a.click();
                                    }}
                                  >
                                    <Download className="size-3.5 mr-1" />
                                    Descarcă
                                  </Button>
                                  {existingShort && (
                                    <span className="text-xs text-muted-foreground">
                                      {(
                                        existingShort.fileSize /
                                        1024 /
                                        1024
                                      ).toFixed(1)}{" "}
                                      MB
                                      {existingShort.hasSubtitles &&
                                        " • Cu subtitrări"}
                                    </span>
                                  )}
                                </div>
                              </div>
                            </div>
                          )}

                          {/* Hook Line */}
                          <div>
                            <div className="flex items-center gap-1.5 text-xs font-semibold text-muted-foreground mb-2 uppercase tracking-wider">
                              <MessageSquareQuote className="size-3" />
                              Hook — Prima propoziție
                            </div>
                            <blockquote className="text-sm italic border-l-2 border-primary/40 pl-3">
                              "{clip.hookLine || clip.transcriptExcerpt}"
                            </blockquote>
                          </div>

                          {/* Caption */}
                          <div>
                            <div className="flex items-center justify-between mb-1.5">
                              <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                                Caption pentru Post
                              </div>
                              <Button
                                size="sm"
                                variant="ghost"
                                className="h-7 text-xs"
                                onClick={e => {
                                  e.stopPropagation();
                                  copyToClipboard(
                                    clip.caption || clip.description,
                                    "Caption",
                                  );
                                }}
                              >
                                <ClipboardCopy className="size-3 mr-1" />
                                Copiază
                              </Button>
                            </div>
                            <p className="text-sm bg-background rounded-lg p-3 border">
                              {clip.caption || clip.description}
                            </p>
                          </div>

                          {/* Tags */}
                          <div className="flex flex-wrap gap-1.5">
                            {(clip.topicTags && clip.topicTags.length > 0
                              ? clip.topicTags
                              : clip.hashtags
                            ).map(tag => (
                              <Badge
                                key={tag}
                                variant="secondary"
                                className="text-xs"
                              >
                                {tag}
                              </Badge>
                            ))}
                          </div>

                          {/* Viral Reasoning */}
                          <div>
                            <div className="text-xs font-semibold text-muted-foreground mb-1.5 uppercase tracking-wider">
                              De ce e viral
                            </div>
                            <p className="text-sm text-muted-foreground">
                              {clip.reason}
                            </p>
                          </div>
                        </div>
                      </CollapsibleContent>
                    </div>
                  </Collapsible>
                );
              })}
            </div>

            {/* Download all shorts */}
            {hasAnyShorts && (
              <div className="border rounded-xl p-5 bg-card mb-6">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="font-bold flex items-center gap-2">
                      <Film className="size-4 text-primary" />
                      {shorts?.length} Shorts Generate
                    </h3>
                    <p className="text-sm text-muted-foreground mt-1">
                      Toate shorts-urile sunt salvate și pot fi descărcate
                      oricând.
                    </p>
                  </div>
                  <Button
                    size="sm"
                    onClick={() => {
                      for (const short of shorts || []) {
                        if (!short.url) continue;
                        const a = document.createElement("a");
                        a.href = short.url;
                        a.download = short.fileName;
                        a.click();
                      }
                      toast.success("Descărcare inițiată!");
                    }}
                  >
                    <Download className="size-3.5 mr-1" />
                    Descarcă Toate
                  </Button>
                </div>
              </div>
            )}
          </>
        )}

      {/* No clips */}
      {job.status === "completed" && sortedClips.length === 0 && (
        <div className="text-center py-12">
          <Video className="size-10 text-muted-foreground mx-auto mb-3" />
          <h3 className="font-semibold mb-1">Niciun clip identificat</h3>
          <p className="text-sm text-muted-foreground">
            AI-ul nu a putut identifica momente potrivite pentru Shorts.
          </p>
        </div>
      )}
    </div>
  );
}
