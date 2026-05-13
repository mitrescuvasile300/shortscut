import { useMutation, useQuery } from "convex/react";
import { ArrowLeft, Loader2, Sparkles } from "lucide-react";
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../../convex/_generated/api";
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
import { toast } from "sonner";

export function NewJobPage() {
  const navigate = useNavigate();
  const createJob = useMutation(api.jobs.create);
  const settings = useQuery(api.settings.get);

  const [videoUrl, setVideoUrl] = useState("");
  const [language, setLanguage] = useState(
    settings?.defaultLanguage || "ro"
  );
  const [numShorts, setNumShorts] = useState(
    settings?.defaultNumShorts || 5
  );
  const [durationRange, setDurationRange] = useState<[number, number]>([
    30,
    settings?.defaultShortDuration || 300,
  ]);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const isValidUrl =
    videoUrl.includes("youtube.com/") || videoUrl.includes("youtu.be/");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!isValidUrl || isSubmitting) return;

    setIsSubmitting(true);
    try {
      const jobId = await createJob({
        videoUrl: videoUrl.trim(),
        language,
        numShorts,
        minDuration: durationRange[0],
        maxDuration: durationRange[1],
      });

      toast.success("Job creat! Descarcă scriptul de pe pagina jobului.");
      navigate(`/job/${jobId}`);
    } catch (err) {
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
            onChange={(e) => setVideoUrl(e.target.value)}
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
                  /(?:v=|youtu\.be\/|shorts\/)([a-zA-Z0-9_-]{11})/
                );
                const videoId = match?.[1];
                if (!videoId) return null;
                return (
                  <div className="relative rounded-xl overflow-hidden bg-muted aspect-video max-w-sm">
                    <img
                      src={`https://img.youtube.com/vi/${videoId}/mqdefault.jpg`}
                      alt="Video thumbnail"
                      className="w-full h-full object-cover"
                      onError={(e) => {
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
              <Label className="text-xs text-muted-foreground">Minim (secunde)</Label>
              <Input
                type="number"
                min={15}
                max={120}
                value={durationRange[0]}
                onChange={(e) =>
                  setDurationRange([Number(e.target.value), durationRange[1]])
                }
                className="h-10"
              />
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs text-muted-foreground">Maxim (secunde, 300 = fără limită)</Label>
              <Input
                type="number"
                min={30}
                max={300}
                value={durationRange[1]}
                onChange={(e) =>
                  setDurationRange([durationRange[0], Number(e.target.value)])
                }
                className="h-10"
              />
            </div>
          </div>
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
