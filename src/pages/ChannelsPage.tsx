import { useAction, useMutation, useQuery } from "convex/react";
import {
  CheckCircle2,
  Clock,
  Loader2,
  Pause,
  Play,
  Plus,
  Radio,
  RefreshCw,
  Trash2,
  Tv,
} from "lucide-react";
import { useState } from "react";
import { api } from "../../convex/_generated/api";
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
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
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

function timeAgo(timestamp: number): string {
  const diff = Date.now() - timestamp;
  const minutes = Math.floor(diff / 60000);
  if (minutes < 1) return "acum";
  if (minutes < 60) return `acum ${minutes}m`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `acum ${hours}h`;
  const days = Math.floor(hours / 24);
  return `acum ${days}z`;
}

function intervalLabel(minutes: number): string {
  if (minutes <= 60) return `La fiecare ${minutes} min`;
  if (minutes <= 360) return `La fiecare ${minutes / 60}h`;
  if (minutes === 720) return "De 2 ori pe zi";
  if (minutes === 1440) return "Zilnic";
  return `La fiecare ${Math.round(minutes / 60)}h`;
}

export function ChannelsPage() {
  const channels = useQuery(api.channels.list);
  const settings = useQuery(api.settings.get);
  const addChannel = useMutation(api.channels.add);
  const toggleActive = useMutation(api.channels.toggleActive);
  const removeChannel = useMutation(api.channels.remove);
  const checkChannel = useAction(api.monitor.checkChannel);

  const [isAddOpen, setIsAddOpen] = useState(false);
  const [channelUrl, setChannelUrl] = useState("");
  const [channelName, setChannelName] = useState("");
  const [checkInterval, setCheckInterval] = useState(360);
  const [language, setLanguage] = useState(
    settings?.defaultLanguage || "ro"
  );
  const [numShorts, setNumShorts] = useState(
    settings?.defaultNumShorts || 5
  );
  const [minDuration, setMinDuration] = useState(30);
  const [maxDuration, setMaxDuration] = useState(
    settings?.defaultShortDuration || 60
  );
  const [isAdding, setIsAdding] = useState(false);
  const [checkingId, setCheckingId] = useState<string | null>(null);

  const isValidUrl =
    channelUrl.includes("youtube.com/") || channelUrl.includes("youtu.be/");

  const handleAdd = async () => {
    if (!isValidUrl || isAdding) return;
    setIsAdding(true);
    try {
      await addChannel({
        channelUrl: channelUrl.trim(),
        channelName: channelName.trim() || undefined,
        checkIntervalMinutes: checkInterval,
        autoLanguage: language,
        autoNumShorts: numShorts,
        autoMinDuration: minDuration,
        autoMaxDuration: maxDuration,
      });
      toast.success("Canal adăugat!");
      setIsAddOpen(false);
      setChannelUrl("");
      setChannelName("");
    } catch {
      toast.error("Eroare la adăugarea canalului");
    } finally {
      setIsAdding(false);
    }
  };

  const handleCheck = async (channelId: string) => {
    setCheckingId(channelId);
    try {
      const result = await checkChannel({
        channelId: channelId as any,
      });
      if (result.newVideos > 0) {
        toast.success(
          `${result.newVideos} video-uri noi găsite! Job-urile au fost create.`
        );
      } else {
        toast.info("Niciun video nou găsit.");
      }
    } catch {
      toast.error("Eroare la verificarea canalului");
    } finally {
      setCheckingId(null);
    }
  };

  return (
    <div className="p-6 md:p-8 max-w-4xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl md:text-3xl font-bold flex items-center gap-2">
            <Radio className="size-7 text-primary" />
            Canale Monitorizate
          </h1>
          <p className="text-muted-foreground mt-1">
            Urmărește canale YouTube și procesează automat video-urile noi
          </p>
        </div>

        <Dialog open={isAddOpen} onOpenChange={setIsAddOpen}>
          <DialogTrigger asChild>
            <Button>
              <Plus className="size-4 mr-1" />
              Adaugă Canal
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-lg">
            <DialogHeader>
              <DialogTitle>Adaugă Canal YouTube</DialogTitle>
              <DialogDescription>
                Monitorizează un canal și procesează automat video-urile noi
              </DialogDescription>
            </DialogHeader>

            <div className="space-y-5 py-2">
              {/* Channel URL */}
              <div className="space-y-2">
                <Label>Link Canal YouTube</Label>
                <Input
                  placeholder="https://www.youtube.com/@canal"
                  value={channelUrl}
                  onChange={(e) => setChannelUrl(e.target.value)}
                />
                {channelUrl && !isValidUrl && (
                  <p className="text-xs text-destructive">
                    Introdu un link YouTube valid
                  </p>
                )}
              </div>

              {/* Channel name (optional) */}
              <div className="space-y-2">
                <Label>
                  Nume Canal{" "}
                  <span className="text-muted-foreground">(opțional)</span>
                </Label>
                <Input
                  placeholder="ex: Joe Rogan Experience"
                  value={channelName}
                  onChange={(e) => setChannelName(e.target.value)}
                />
              </div>

              {/* Check interval */}
              <div className="space-y-2">
                <Label>Frecvență verificare</Label>
                <Select
                  value={String(checkInterval)}
                  onValueChange={(v) => setCheckInterval(Number(v))}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="60">La fiecare oră</SelectItem>
                    <SelectItem value="180">La fiecare 3 ore</SelectItem>
                    <SelectItem value="360">La fiecare 6 ore</SelectItem>
                    <SelectItem value="720">De 2 ori pe zi</SelectItem>
                    <SelectItem value="1440">Zilnic</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Language */}
              <div className="space-y-2">
                <Label>Limba conținutului</Label>
                <Select value={language} onValueChange={setLanguage}>
                  <SelectTrigger>
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

              {/* Shorts config */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label>Shorts per video</Label>
                  <span className="text-sm font-mono text-primary font-semibold bg-primary/10 px-2 py-0.5 rounded">
                    {numShorts}
                  </span>
                </div>
                <Slider
                  value={[numShorts]}
                  onValueChange={([v]) => setNumShorts(v)}
                  min={1}
                  max={15}
                  step={1}
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1.5">
                  <Label className="text-xs text-muted-foreground">
                    Durată min. (s)
                  </Label>
                  <Input
                    type="number"
                    min={15}
                    max={55}
                    value={minDuration}
                    onChange={(e) => setMinDuration(Number(e.target.value))}
                  />
                </div>
                <div className="space-y-1.5">
                  <Label className="text-xs text-muted-foreground">
                    Durată max. (s)
                  </Label>
                  <Input
                    type="number"
                    min={20}
                    max={60}
                    value={maxDuration}
                    onChange={(e) => setMaxDuration(Number(e.target.value))}
                  />
                </div>
              </div>
            </div>

            <DialogFooter>
              <Button
                variant="outline"
                onClick={() => setIsAddOpen(false)}
              >
                Anulează
              </Button>
              <Button
                onClick={handleAdd}
                disabled={!isValidUrl || isAdding}
              >
                {isAdding ? (
                  <Loader2 className="size-4 mr-1 animate-spin" />
                ) : (
                  <Plus className="size-4 mr-1" />
                )}
                Adaugă
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {/* Channels list */}
      {channels === undefined ? (
        <div className="flex items-center justify-center py-20">
          <Loader2 className="size-6 animate-spin text-muted-foreground" />
        </div>
      ) : channels.length === 0 ? (
        <div className="text-center py-16 border rounded-xl bg-card">
          <Tv className="size-12 text-muted-foreground mx-auto mb-4" />
          <h3 className="text-lg font-semibold mb-2">
            Niciun canal monitorizat
          </h3>
          <p className="text-muted-foreground mb-6 max-w-sm mx-auto">
            Adaugă un canal YouTube pentru a primi automat shorts din
            fiecare video nou
          </p>
          <Button onClick={() => setIsAddOpen(true)}>
            <Plus className="size-4 mr-1" />
            Adaugă Primul Canal
          </Button>
        </div>
      ) : (
        <div className="space-y-3">
          {channels.map((channel) => {
            const isChecking = checkingId === channel._id;

            return (
              <div
                key={channel._id}
                className={`border rounded-xl p-5 transition-colors ${channel.isActive ? "bg-card" : "bg-card/50 opacity-70"}`}
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 min-w-0">
                    {/* Channel header */}
                    <div className="flex items-center gap-2 mb-1">
                      <h3 className="font-semibold truncate">
                        {channel.channelName ||
                          extractDisplayName(channel.channelUrl)}
                      </h3>
                      {channel.isActive ? (
                        <Badge
                          variant="secondary"
                          className="bg-green-400/10 text-green-400 text-xs"
                        >
                          <Radio className="size-2.5 mr-1" />
                          Activ
                        </Badge>
                      ) : (
                        <Badge variant="secondary" className="text-xs">
                          <Pause className="size-2.5 mr-1" />
                          Pausat
                        </Badge>
                      )}
                    </div>

                    {/* Channel URL */}
                    <a
                      href={channel.channelUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sm text-muted-foreground hover:text-primary transition-colors truncate block"
                    >
                      {channel.channelUrl}
                    </a>

                    {/* Meta info */}
                    <div className="flex flex-wrap items-center gap-3 mt-2 text-xs text-muted-foreground">
                      <span className="flex items-center gap-1">
                        <Clock className="size-3" />
                        {intervalLabel(channel.checkIntervalMinutes)}
                      </span>
                      <span>•</span>
                      <span>
                        {channel.autoNumShorts} shorts,{" "}
                        {channel.autoMinDuration}-{channel.autoMaxDuration}s
                      </span>
                      <span>•</span>
                      <span>{channel.autoLanguage.toUpperCase()}</span>
                      {channel.lastCheckedAt && (
                        <>
                          <span>•</span>
                          <span className="flex items-center gap-1">
                            <CheckCircle2 className="size-3" />
                            Verificat {timeAgo(channel.lastCheckedAt)}
                          </span>
                        </>
                      )}
                      {channel.processedVideoIds &&
                        channel.processedVideoIds.length > 0 && (
                          <>
                            <span>•</span>
                            <span>
                              {channel.processedVideoIds.length} video-uri
                              procesate
                            </span>
                          </>
                        )}
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex items-center gap-1.5 shrink-0">
                    {/* Manual check */}
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleCheck(channel._id)}
                      disabled={isChecking}
                      title="Verifică acum"
                    >
                      {isChecking ? (
                        <Loader2 className="size-3.5 animate-spin" />
                      ) : (
                        <RefreshCw className="size-3.5" />
                      )}
                    </Button>

                    {/* Toggle active */}
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => toggleActive({ id: channel._id })}
                      title={
                        channel.isActive ? "Pauză" : "Activează"
                      }
                    >
                      {channel.isActive ? (
                        <Pause className="size-3.5" />
                      ) : (
                        <Play className="size-3.5" />
                      )}
                    </Button>

                    {/* Delete */}
                    <AlertDialog>
                      <AlertDialogTrigger asChild>
                        <Button
                          variant="outline"
                          size="sm"
                          className="text-destructive hover:text-destructive"
                          title="Șterge"
                        >
                          <Trash2 className="size-3.5" />
                        </Button>
                      </AlertDialogTrigger>
                      <AlertDialogContent>
                        <AlertDialogHeader>
                          <AlertDialogTitle>
                            Șterge canalul?
                          </AlertDialogTitle>
                          <AlertDialogDescription>
                            Canalul va fi eliminat din monitorizare.
                            Job-urile deja create nu vor fi afectate.
                          </AlertDialogDescription>
                        </AlertDialogHeader>
                        <AlertDialogFooter>
                          <AlertDialogCancel>Anulează</AlertDialogCancel>
                          <AlertDialogAction
                            onClick={() =>
                              removeChannel({ id: channel._id })
                            }
                            className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                          >
                            Șterge
                          </AlertDialogAction>
                        </AlertDialogFooter>
                      </AlertDialogContent>
                    </AlertDialog>
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

function extractDisplayName(url: string): string {
  const atMatch = url.match(/@([a-zA-Z0-9_.-]+)/);
  if (atMatch) return `@${atMatch[1]}`;
  const cMatch = url.match(/\/c\/([a-zA-Z0-9_.-]+)/);
  if (cMatch) return cMatch[1];
  const channelMatch = url.match(/\/channel\/([a-zA-Z0-9_-]+)/);
  if (channelMatch) return channelMatch[1].substring(0, 20) + "…";
  return url;
}
