import { useMutation, useQuery } from "convex/react";
import { ArrowLeft, Loader2, Music, Sparkles, Upload, X } from "lucide-react";
import { useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { api } from "../../convex/_generated/api";
import type { Id } from "../../convex/_generated/dataModel";

export function NewJobPage() {
  const navigate = useNavigate();
  const createJob = useMutation(api.jobs.create);
  const generateMusicUploadUrl = useMutation(api.jobs.generateMusicUploadUrl);
  // User picks Browser or Server processing on the job detail page
  const settings = useQuery(api.settings.get);

  const [videoUrl, setVideoUrl] = useState("");
  const [language, setLanguage] = useState(settings?.defaultLanguage || "ro");
  const [numShorts, setNumShorts] = useState(settings?.defaultNumShorts || 5);
  const [durationRange, setDurationRange] = useState<[number, number]>([
    30,
    settings?.defaultShortDuration || 300,
  ]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [musicMode, setMusicMode] = useState<"none" | "default" | "custom">("none");
  const [musicFile, setMusicFile] = useState<File | null>(null);
  const musicInputRef = useRef<HTMLInputElement>(null);

  const isValidUrl =
    videoUrl.includes("youtube.com/") || videoUrl.includes("youtu.be/");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!isValidUrl || isSubmitting) return;

    if (musicMode === "custom" && !musicFile) {
      toast.error("Alege un fișier audio pentru muzica de fundal");
      return;
    }

    setIsSubmitting(true);
    try {
      let musicStorageId: string | undefined;
      if (musicMode === "custom" && musicFile) {
        const uploadUrl = await generateMusicUploadUrl();
        const res = await fetch(uploadUrl, {
          method: "POST",
          headers: { "Content-Type": musicFile.type || "application/octet-stream" },
          body: musicFile,
        });
        if (!res.ok) throw new Error("upload failed");
        musicStorageId = (await res.json()).storageId as string;
      }

      const jobId = await createJob({
        videoUrl: videoUrl.trim(),
        language,
        numShorts,
        minDuration: durationRange[0],
        maxDuration: durationRange[1],
        musicMode,
        musicStorageId: musicStorageId as Id<"_storage"> | undefined,
        musicFileName: musicFile?.name,
      });

      toast.success("Job creat! Alege modul de procesare.");
      navigate(`/job/${jobId}`);
    } catch (_err) {
      toast.error("Eroare la crearea job-ului");
      setIsSubmitting(false);
    }
  };

  return (
    <div className="p-6 md:p-8 max-w-2xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <Button
          asChild
          variant="ghost"
          size="sm"
          className="mb-4 -ml-2 text-muted-foreground"
        >
          <Link to="/dashboard">
            <ArrowLeft className="size-4 mr-1" />
            Înapoi
          </Link>
        </Button>
        <h1 className="text-2xl md:text-3xl font-bold">Job Nou</h1>
        <p className="text-muted-foreground mt-1">
          Configurează procesarea unui podcast YouTube
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-8">
        {/* Video URL */}
        <div className="space-y-3">
          <Label htmlFor="videoUrl" className="text-base font-semibold">
            Link YouTube
          </Label>
          <Input
            id="videoUrl"
            type="url"
            placeholder="https://www.youtube.com/watch?v=..."
            value={videoUrl}
            onChange={e => setVideoUrl(e.target.value)}
            className="h-12 text-base"
            required
          />
          {videoUrl && !isValidUrl && (
            <p className="text-sm text-destructive">
              Introdu un link YouTube valid
            </p>
          )}

          {/* Preview thumbnail */}
          {isValidUrl && (
            <div className="mt-3">
              {(() => {
                const match = videoUrl.match(
                  /(?:v=|youtu\.be\/|shorts\/)([a-zA-Z0-9_-]{11})/,
                );
                const videoId = match?.[1];
                if (!videoId) return null;
                return (
                  <div className="relative rounded-xl overflow-hidden bg-muted aspect-video max-w-sm">
                    <img
                      src={`https://img.youtube.com/vi/${videoId}/mqdefault.jpg`}
                      alt="Video thumbnail"
                      className="w-full h-full object-cover"
                      onError={e => {
                        (e.target as HTMLImageElement).style.display = "none";
                      }}
                    />
                    <div className="absolute inset-0 bg-black/20 flex items-center justify-center">
                      <div className="size-12 rounded-full bg-white/90 flex items-center justify-center">
                        <div className="w-0 h-0 border-t-[8px] border-t-transparent border-b-[8px] border-b-transparent border-l-[14px] border-l-primary ml-1" />
                      </div>
                    </div>
                  </div>
                );
              })()}
            </div>
          )}
        </div>

        {/* Language */}
        <div className="space-y-3">
          <Label className="text-base font-semibold">Limba conținutului</Label>
          <Select value={language} onValueChange={setLanguage}>
            <SelectTrigger className="h-11">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="ro">🇷🇴 Română</SelectItem>
              <SelectItem value="en">🇬🇧 English</SelectItem>
              <SelectItem value="es">🇪🇸 Español</SelectItem>
              <SelectItem value="fr">🇫🇷 Français</SelectItem>
              <SelectItem value="de">🇩🇪 Deutsch</SelectItem>
              <SelectItem value="it">🇮🇹 Italiano</SelectItem>
              <SelectItem value="pt">🇵🇹 Português</SelectItem>
              <SelectItem value="ru">🇷🇺 Русский</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Number of shorts */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <Label className="text-base font-semibold">Număr de Shorts</Label>
            <span className="text-sm font-mono text-primary font-semibold bg-primary/10 px-2.5 py-0.5 rounded-md">
              {numShorts}
            </span>
          </div>
          <Slider
            value={[numShorts]}
            onValueChange={([v]) => setNumShorts(v)}
            min={1}
            max={15}
            step={1}
            className="py-2"
          />
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>1</span>
            <span>15</span>
          </div>
        </div>

        {/* Duration range */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <Label className="text-base font-semibold">
              Durată Short (secunde)
            </Label>
            <span className="text-sm font-mono text-primary font-semibold bg-primary/10 px-2.5 py-0.5 rounded-md">
              {durationRange[0]}s – {durationRange[1]}s
            </span>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-1.5">
              <Label className="text-xs text-muted-foreground">
                Minim (secunde)
              </Label>
              <Input
                type="number"
                min={15}
                max={120}
                value={durationRange[0]}
                onChange={e =>
                  setDurationRange([Number(e.target.value), durationRange[1]])
                }
                className="h-10"
              />
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs text-muted-foreground">
                Maxim (secunde, 300 = fără limită)
              </Label>
              <Input
                type="number"
                min={30}
                max={300}
                value={durationRange[1]}
                onChange={e =>
                  setDurationRange([durationRange[0], Number(e.target.value)])
                }
                className="h-10"
              />
            </div>
          </div>
        </div>

        {/* Background music */}
        <div className="space-y-3">
          <Label className="text-base font-semibold">Muzică de fundal</Label>
          <Select
            value={musicMode}
            onValueChange={v => {
              setMusicMode(v as "none" | "default" | "custom");
              if (v !== "custom") setMusicFile(null);
            }}
          >
            <SelectTrigger className="h-11">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="none">🔇 Fără muzică (implicit)</SelectItem>
              <SelectItem value="default">🎵 Muzică inclusă (royalty-free, volum redus)</SelectItem>
              <SelectItem value="custom">📁 Încarc propriul fișier audio</SelectItem>
            </SelectContent>
          </Select>
          {musicMode === "custom" && (
            <div className="space-y-2">
              <input
                ref={musicInputRef}
                type="file"
                accept="audio/*,.mp3,.wav,.m4a,.aac,.ogg,.flac"
                className="hidden"
                onChange={e => {
                  const f = e.target.files?.[0] ?? null;
                  if (f && f.size > 50 * 1024 * 1024) {
                    toast.error("Fișierul depășește 50 MB");
                    e.target.value = "";
                    return;
                  }
                  setMusicFile(f);
                }}
              />
              {musicFile ? (
                <div className="flex items-center gap-3 rounded-lg border bg-muted/40 px-3 py-2 text-sm">
                  <Music className="size-4 text-primary shrink-0" />
                  <span className="truncate flex-1">{musicFile.name}</span>
                  <span className="text-muted-foreground text-xs">
                    {(musicFile.size / 1024 / 1024).toFixed(1)} MB
                  </span>
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    className="size-7"
                    onClick={() => {
                      setMusicFile(null);
                      if (musicInputRef.current) musicInputRef.current.value = "";
                    }}
                  >
                    <X className="size-4" />
                  </Button>
                </div>
              ) : (
                <Button
                  type="button"
                  variant="outline"
                  className="w-full h-11"
                  onClick={() => musicInputRef.current?.click()}
                >
                  <Upload className="size-4 mr-2" />
                  Alege fișier audio (mp3, wav, m4a… max 50 MB)
                </Button>
              )}
              <p className="text-xs text-muted-foreground">
                Piesa e pusă în buclă sub voce, la volum redus, cu fade in/out.
              </p>
            </div>
          )}
        </div>

        {/* Submit */}
        <Button
          type="submit"
          size="lg"
          className="w-full h-13 text-base"
          disabled={!isValidUrl || isSubmitting}
        >
          {isSubmitting ? (
            <>
              <Loader2 className="size-5 mr-2 animate-spin" />
              Se procesează...
            </>
          ) : (
            <>
              <Sparkles className="size-5 mr-2" />
              Analizează & Generează Shorts
            </>
          )}
        </Button>
      </form>
    </div>
  );
}
