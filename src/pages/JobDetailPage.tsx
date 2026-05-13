import { useQuery, useMutation, useAction } from "convex/react";
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
  Terminal,
  Video,
} from "lucide-react";
import { useState, useCallback, useRef, useEffect } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../../convex/_generated/api";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { toast } from "sonner";
import type { Id } from "../../convex/_generated/dataModel";
import { processAllClips, type ProcessingProgress, type ClipConfig } from "@/lib/videoProcessor";
import { generateAssSubtitles, getSegmentsForClip, type SubtitleSegment } from "@/lib/subtitles";
import { downloadScript } from "@/lib/scriptGenerator";

function formatSeconds(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}:${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
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
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold ${color}`}>
      <Flame className="size-3" />
      {score}/10
    </span>
  );
}

function StatusProgress({ status, generatingProgress }: { status: string; generatingProgress?: ProcessingProgress | null }) {
  const steps = [
    { key: "downloading", label: "Obținere info video" },
    { key: "transcribing", label: "Transcriere automată" },
    { key: "analyzing", label: "Analiză AI & Selecție" },
    { key: "generating", label: "Generare Shorts" },
    { key: "completed", label: "Finalizat" },
  ];
  const stepOrder = ["pending", "downloading", "transcribing", "analyzing", "generating", "completed"];
  const currentIndex = stepOrder.indexOf(status);

  return (
    <div className="border rounded-xl p-6 bg-card">
      <h3 className="text-sm font-semibold text-muted-foreground mb-4 uppercase tracking-wider">
        Progres Procesare
      </h3>
      <div className="space-y-3">
        {steps.map((step) => {
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
                <span className={`text-sm ${isCurrent ? "text-foreground font-medium" : isComplete ? "text-muted-foreground" : "text-muted-foreground/60"}`}>
                  {step.label}
                </span>
              </div>
              {/* Show generating progress inline */}
              {step.key === "generating" && isCurrent && generatingProgress && (
                <div className="ml-8 mt-2 space-y-2">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">{generatingProgress.message}</span>
                    <span className="font-mono text-primary">{generatingProgress.percent}%</span>
                  </div>
                  <Progress value={generatingProgress.percent} className="h-1.5" />
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
function ShortVideoPlayer({ url, title: _title }: { url: string; title: string }) {
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
  const job = useQuery(api.jobs.get, jobId ? { id: jobId as Id<"jobs"> } : "skip");
  const clips = useQuery(api.clips.listByJob, jobId ? { jobId: jobId as Id<"jobs"> } : "skip");
  const shorts = useQuery(api.shorts.listByJob, jobId ? { jobId: jobId as Id<"jobs"> } : "skip");
  const settings = useQuery(api.settings.get);
  const generateUploadUrl = useMutation(api.shorts.generateUploadUrl);
  const saveShort = useMutation(api.shorts.save);
  const markJobCompleted = useMutation(api.shorts.markJobCompleted);
  const refreshVideoUrl = useAction(api.processing.refreshVideoUrl);

  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState<ProcessingProgress | null>(null);
  const [generatedBlobs, setGeneratedBlobs] = useState<Map<number, { blob: Blob; url: string }>>(new Map());
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
      const isExpired = !videoUrl || (job.videoDownloadExpiry && Date.now() > job.videoDownloadExpiry);

      if (isExpired) {
        setProgress({
          clipIndex: 0, totalClips: clips.length,
          stage: "loading", percent: 0,
          message: "Se obține link-ul de descărcare...",
        });
        videoUrl = await refreshVideoUrl({ jobId: jobId as Id<"jobs"> }) ?? undefined;
        // audioUrl will be refreshed too via the backend
      }

      if (!videoUrl) {
        toast.error("Nu s-a putut obține link-ul de descărcare al video-ului.");
        setProcessing(false);
        return;
      }

      // Route through CORS proxy (YouTube CDN doesn't allow browser fetch)
      const convexSiteUrl = import.meta.env.VITE_CONVEX_SITE_URL;
      if (convexSiteUrl && videoUrl.includes("googlevideo.com")) {
        videoUrl = `${convexSiteUrl}/api/video-proxy?url=${encodeURIComponent(videoUrl)}`;
      }
      if (convexSiteUrl && audioUrl?.includes("googlevideo.com")) {
        audioUrl = `${convexSiteUrl}/api/video-proxy?url=${encodeURIComponent(audioUrl)}`;
      }

      // Get transcript segments for subtitles
      const allSegments = getTranscriptSegments();

      // Prepare clip configs with subtitles
      const sortedClips = [...clips].sort((a, b) => b.viralScore - a.viralScore);
      const clipConfigs: ClipConfig[] = sortedClips.map((clip, i) => {
        const clipSegments = getSegmentsForClip(allSegments, clip.startTime, clip.endTime);
        const assContent = clipSegments.length > 0
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

      // Process all clips (pass audio URL for muxing if available)
      const results = await processAllClips(videoUrl, clipConfigs, setProgress, audioUrl);

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

          const safeTitle = clip.title.replace(/[^a-zA-Z0-9\s-]/g, "").replace(/\s+/g, "_").substring(0, 40);
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
      toast.error(`Eroare: ${error instanceof Error ? error.message : "Eroare necunoscută"}`);
    } finally {
      setProcessing(false);
    }
  }, [job, clips, jobId, processing, getTranscriptSegments, generateUploadUrl, saveShort, markJobCompleted, refreshVideoUrl]);

  // Auto-trigger generation when job reaches "generating" status
  useEffect(() => {
    if (
      job?.status === "generating" &&
      clips && clips.length > 0 &&
      shorts !== undefined &&
      shorts.length === 0 &&
      !processing &&
      !autoGenerateTriggered.current
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
          <Link to="/dashboard"><ArrowLeft className="size-4 mr-1" />Înapoi la Dashboard</Link>
        </Button>
      </div>
    );
  }

  const isProcessing = ["pending", "downloading", "transcribing", "analyzing", "generating"].includes(job.status);
  const isGenerating = job.status === "generating";
  const isWaitingTranscript = job.status === "waiting_transcript";
  const sortedClips = clips ? [...clips].sort((a, b) => b.viralScore - a.viralScore) : [];

  // Map shorts by clipId for easy lookup
  const shortsByClipId = new Map(
    (shorts || []).map(s => [s.clipId, s])
  );

  const hasAnyShorts = (shorts || []).length > 0;

  return (
    <div className="p-6 md:p-8 max-w-4xl mx-auto">
      {/* Header */}
      <Button asChild variant="ghost" size="sm" className="mb-4 -ml-2 text-muted-foreground">
        <Link to="/dashboard"><ArrowLeft className="size-4 mr-1" />Dashboard</Link>
      </Button>

      {/* Video info */}
      <div className="border rounded-xl overflow-hidden mb-6">
        {job.videoThumbnail && (
          <div className="relative aspect-video max-h-[200px] bg-muted overflow-hidden">
            <img src={job.videoThumbnail} alt="" className="w-full h-full object-cover"
              onError={(e) => { (e.target as HTMLImageElement).parentElement!.style.display = "none"; }} />
            <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent" />
          </div>
        )}
        <div className="p-5">
          <h1 className="text-xl md:text-2xl font-bold mb-2">{job.videoTitle || "Se procesează..."}</h1>
          <div className="flex flex-wrap items-center gap-3 text-sm text-muted-foreground">
            <a href={job.videoUrl} target="_blank" rel="noopener noreferrer"
              className="flex items-center gap-1 hover:text-primary transition-colors">
              <ExternalLink className="size-3.5" />Deschide pe YouTube
            </a>
            <span>•</span>
            <span>{job.numShorts} shorts solicitate</span>
            <span>•</span>
            <span>{job.minDuration}s – {job.maxDuration}s</span>
            <span>•</span>
            <span>{job.language.toUpperCase()}</span>
          </div>
        </div>
      </div>

      {/* Download Script — for pending jobs */}
      {job.status === "pending" && (
        <div className="border-2 border-primary/40 bg-primary/5 rounded-xl p-6 mb-6">
          <div className="flex flex-col items-center text-center gap-4">
            <div className="size-14 rounded-full bg-primary/10 flex items-center justify-center">
              <Terminal className="size-7 text-primary" />
            </div>
            <div>
              <h3 className="text-lg font-bold mb-1">Descarcă Scriptul</h3>
              <p className="text-sm text-muted-foreground max-w-md">
                Scriptul procesează totul local pe calculatorul tău:
                download video → transcriere Whisper → analiză AI → face crop 9:16 → burn subtitrări.
              </p>
            </div>
            <div className="flex flex-col gap-2 w-full max-w-xs">
              <Button
                size="lg"
                className="w-full h-12 text-base"
                onClick={() => {
                  if (!job) return;
                  const videoId = job.videoUrl.match(/(?:v=|youtu\.be\/|shorts\/)([a-zA-Z0-9_-]{11})/)?.[1] || "video";
                  downloadScript(
                    {
                      videoUrl: job.videoUrl,
                      language: job.language,
                      numShorts: job.numShorts,
                      minDuration: job.minDuration,
                      maxDuration: job.maxDuration,
                      openaiApiKey: settings?.openaiApiKey || undefined,
                    },
                    `shortscut_${videoId}.sh`
                  );
                  toast.success("Script descărcat! Rulează-l local cu: chmod +x shortscut_*.sh && ./shortscut_*.sh");
                }}
              >
                <Download className="size-5 mr-2" />
                Descarcă .sh
              </Button>
              <p className="text-xs text-muted-foreground">
                Necesită: python3, ffmpeg, yt-dlp{!settings?.openaiApiKey && " + OpenAI API key"}
              </p>
            </div>
            <div className="text-xs text-muted-foreground bg-muted rounded-lg p-3 w-full max-w-md font-mono">
              chmod +x shortscut_*.sh && ./shortscut_*.sh
            </div>
          </div>
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
            Video-ul nu are subtitrări auto-generate pe YouTube.
            Adaugă un <Link to="/settings" className="text-primary underline">OpenAI API Key</Link> în Settings
            pentru transcriere automată cu Whisper, sau procesează din nou.
          </p>
        </div>
      )}

      {/* Error */}
      {job.status === "failed" && (
        <div className="border border-destructive/30 bg-destructive/5 rounded-xl p-5">
          <h3 className="font-semibold text-destructive mb-1">Eroare</h3>
          <p className="text-sm text-destructive/80">{job.error}</p>
        </div>
      )}

      {/* Show clips when generating or completed */}
      {(isGenerating || job.status === "completed") && sortedClips.length > 0 && (
        <>
          {/* Generating info banner */}
          {isGenerating && !processing && !hasAnyShorts && (
            <div className="border border-primary/30 bg-primary/5 rounded-xl p-5 mb-6 mt-4">
              <div className="flex items-center gap-3">
                <Loader2 className="size-5 text-primary animate-spin" />
                <div>
                  <h3 className="font-bold">Se generează Shorts...</h3>
                  <p className="text-sm text-muted-foreground">
                    Procesarea se face local în browser — descărcare video → face detection → crop 9:16 → subtitrări → encode MP4.
                  </p>
                </div>
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
            <h2 className="text-lg font-bold">{sortedClips.length} Clips Identificate</h2>
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
                            <span className="text-xs font-mono text-muted-foreground bg-muted px-2 py-0.5 rounded">#{i + 1}</span>
                            <ViralScoreBadge score={clip.viralScore} />
                            <span className="text-xs text-muted-foreground">
                              {formatSeconds(clip.startTime)} → {formatSeconds(clip.endTime)} ({duration}s)
                            </span>
                            {existingShort && (
                              <Badge variant="secondary" className="text-xs bg-green-500/10 text-green-400 border-green-500/20">
                                <Film className="size-3 mr-1" />Video generat
                              </Badge>
                            )}
                          </div>
                          <h3 className="font-semibold">{clip.title}</h3>
                          <p className="text-sm text-muted-foreground mt-1 line-clamp-1">{clip.reason}</p>
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
                                url={existingShort?.url || generatedBlob?.url || ""}
                                title={clip.title}
                              />
                              <div className="flex flex-col gap-2">
                                <Button
                                  size="sm"
                                  variant="outline"
                                  onClick={() => {
                                    const url = existingShort?.url || generatedBlob?.url;
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
                                    {(existingShort.fileSize / 1024 / 1024).toFixed(1)} MB
                                    {existingShort.hasSubtitles && " • Cu subtitrări"}
                                  </span>
                                )}
                              </div>
                            </div>
                          </div>
                        )}

                        {/* Hook Line */}
                        <div>
                          <div className="flex items-center gap-1.5 text-xs font-semibold text-muted-foreground mb-2 uppercase tracking-wider">
                            <MessageSquareQuote className="size-3" />Hook — Prima propoziție
                          </div>
                          <blockquote className="text-sm italic border-l-2 border-primary/40 pl-3">
                            "{clip.hookLine || clip.transcriptExcerpt}"
                          </blockquote>
                        </div>

                        {/* Caption */}
                        <div>
                          <div className="flex items-center justify-between mb-1.5">
                            <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Caption pentru Post</div>
                            <Button size="sm" variant="ghost" className="h-7 text-xs"
                              onClick={(e) => { e.stopPropagation(); copyToClipboard(clip.caption || clip.description, "Caption"); }}>
                              <ClipboardCopy className="size-3 mr-1" />Copiază
                            </Button>
                          </div>
                          <p className="text-sm bg-background rounded-lg p-3 border">{clip.caption || clip.description}</p>
                        </div>

                        {/* Tags */}
                        <div className="flex flex-wrap gap-1.5">
                          {(clip.topicTags && clip.topicTags.length > 0 ? clip.topicTags : clip.hashtags).map((tag) => (
                            <Badge key={tag} variant="secondary" className="text-xs">{tag}</Badge>
                          ))}
                        </div>

                        {/* Viral Reasoning */}
                        <div>
                          <div className="text-xs font-semibold text-muted-foreground mb-1.5 uppercase tracking-wider">De ce e viral</div>
                          <p className="text-sm text-muted-foreground">{clip.reason}</p>
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
                    Toate shorts-urile sunt salvate și pot fi descărcate oricând.
                  </p>
                </div>
                <Button
                  size="sm"
                  onClick={() => {
                    for (const short of (shorts || [])) {
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
          <p className="text-sm text-muted-foreground">AI-ul nu a putut identifica momente potrivite pentru Shorts.</p>
        </div>
      )}
    </div>
  );
}
