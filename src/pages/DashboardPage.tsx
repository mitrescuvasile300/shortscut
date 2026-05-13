import { useQuery, useMutation } from "convex/react";
import {
  Clock,
  ExternalLink,
  Loader2,
  Plus,
  Scissors,
  Trash2,
  Video,
} from "lucide-react";
import { Link } from "react-router-dom";
import { api } from "../../convex/_generated/api";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { toast } from "sonner";
import type { Id } from "../../convex/_generated/dataModel";

const statusConfig: Record<
  string,
  { label: string; variant: "default" | "secondary" | "destructive" | "outline" }
> = {
  pending: { label: "În așteptare", variant: "secondary" },
  downloading: { label: "Se descarcă...", variant: "outline" },
  transcribing: { label: "Se transcrie...", variant: "outline" },
  analyzing: { label: "Se analizează...", variant: "outline" },
  generating: { label: "Se generează shorts...", variant: "outline" },
  completed: { label: "Finalizat ✓", variant: "default" },
  failed: { label: "Eroare ✗", variant: "destructive" },
  waiting_transcript: { label: "Așteaptă transcriere", variant: "outline" },
};

function formatTimeAgo(timestamp: number): string {
  const diff = Date.now() - timestamp;
  const minutes = Math.floor(diff / 60000);
  if (minutes < 1) return "acum";
  if (minutes < 60) return `acum ${minutes} min`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `acum ${hours}h`;
  const days = Math.floor(hours / 24);
  return `acum ${days}z`;
}

export function DashboardPage() {
  const jobs = useQuery(api.jobs.list);
  const removeJob = useMutation(api.jobs.remove);

  const handleDelete = async (id: Id<"jobs">) => {
    try {
      await removeJob({ id });
      toast.success("Job șters");
    } catch {
      toast.error("Eroare la ștergere");
    }
  };

  const isProcessing = (status: string) =>
    ["pending", "downloading", "transcribing", "analyzing"].includes(status);

  return (
    <div className="p-6 md:p-8 max-w-5xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl md:text-3xl font-bold">Dashboard</h1>
          <p className="text-muted-foreground mt-1">
            Toate job-urile tale de procesare
          </p>
        </div>
        <Button asChild>
          <Link to="/new">
            <Plus className="size-4 mr-2" />
            Job Nou
          </Link>
        </Button>
      </div>

      {/* Jobs list */}
      {jobs === undefined ? (
        <div className="flex items-center justify-center py-20">
          <Loader2 className="size-6 animate-spin text-muted-foreground" />
        </div>
      ) : jobs.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-20 text-center">
          <div className="size-16 rounded-2xl bg-muted flex items-center justify-center mb-4">
            <Video className="size-8 text-muted-foreground" />
          </div>
          <h3 className="text-lg font-semibold mb-2">Niciun job încă</h3>
          <p className="text-muted-foreground mb-6 max-w-sm">
            Creează primul tău job pentru a transforma un podcast în YouTube
            Shorts
          </p>
          <Button asChild>
            <Link to="/new">
              <Plus className="size-4 mr-2" />
              Creează Primul Job
            </Link>
          </Button>
        </div>
      ) : (
        <div className="space-y-3">
          {jobs.map((job) => {
            const status = statusConfig[job.status] || statusConfig.pending;
            return (
              <div
                key={job._id}
                className="group border rounded-xl p-4 md:p-5 hover:bg-accent/50 transition-colors"
              >
                <div className="flex items-start gap-4">
                  {/* Thumbnail */}
                  <div className="hidden sm:block shrink-0">
                    {job.videoThumbnail ? (
                      <img
                        src={job.videoThumbnail}
                        alt=""
                        className="w-28 h-16 rounded-lg object-cover bg-muted"
                        onError={(e) => {
                          (e.target as HTMLImageElement).style.display = "none";
                        }}
                      />
                    ) : (
                      <div className="w-28 h-16 rounded-lg bg-muted flex items-center justify-center">
                        <Video className="size-5 text-muted-foreground" />
                      </div>
                    )}
                  </div>

                  {/* Content */}
                  <div className="flex-1">
                    <div className="flex items-start justify-between gap-2">
                      <div>
                        <Link
                          to={`/job/${job._id}`}
                          className="font-semibold hover:text-primary transition-colors line-clamp-1"
                        >
                          {job.videoTitle || "Se procesează..."}
                        </Link>
                        <div className="flex items-center gap-3 mt-1.5 text-sm text-muted-foreground">
                          <span className="flex items-center gap-1">
                            <Clock className="size-3.5" />
                            {formatTimeAgo(job._creationTime)}
                          </span>
                          <a
                            href={job.videoUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center gap-1 hover:text-primary transition-colors"
                          >
                            <ExternalLink className="size-3.5" />
                            YouTube
                          </a>
                          <span>
                            {job.numShorts} shorts · {job.language.toUpperCase()}
                          </span>
                        </div>
                      </div>

                      <div className="flex items-center gap-2 shrink-0">
                        <Badge variant={status.variant}>
                          {isProcessing(job.status) && (
                            <Loader2 className="size-3 animate-spin mr-1" />
                          )}
                          {status.label}
                        </Badge>

                        {job.status === "completed" && (
                          <Button asChild size="sm" variant="outline">
                            <Link to={`/job/${job._id}`}>
                              <Scissors className="size-3.5 mr-1" />
                              Vezi Clipuri
                            </Link>
                          </Button>
                        )}

                        {job.status === "waiting_transcript" && (
                          <Button asChild size="sm" variant="outline">
                            <Link to={`/job/${job._id}`}>
                              <Scissors className="size-3.5 mr-1" />
                              Descarcă Script
                            </Link>
                          </Button>
                        )}

                        <AlertDialog>
                          <AlertDialogTrigger asChild>
                            <Button
                              size="icon"
                              variant="ghost"
                              className="size-8 opacity-0 group-hover:opacity-100 transition-opacity text-muted-foreground hover:text-destructive"
                            >
                              <Trash2 className="size-3.5" />
                            </Button>
                          </AlertDialogTrigger>
                          <AlertDialogContent>
                            <AlertDialogHeader>
                              <AlertDialogTitle>Șterge job-ul?</AlertDialogTitle>
                              <AlertDialogDescription>
                                Această acțiune va șterge job-ul și toate clipurile asociate.
                              </AlertDialogDescription>
                            </AlertDialogHeader>
                            <AlertDialogFooter>
                              <AlertDialogCancel>Anulează</AlertDialogCancel>
                              <AlertDialogAction
                                onClick={() => handleDelete(job._id)}
                                className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                              >
                                Șterge
                              </AlertDialogAction>
                            </AlertDialogFooter>
                          </AlertDialogContent>
                        </AlertDialog>
                      </div>
                    </div>

                    {job.error && (
                      <p className="text-sm text-destructive mt-2 bg-destructive/10 rounded-md px-3 py-1.5">
                        {job.error}
                      </p>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
