import { useAuthActions } from "@convex-dev/auth/react";
import { useMutation, useQuery } from "convex/react";
import {
  Cookie,
  Eye,
  EyeOff,
  Key,
  Loader2,
  Save,
  Settings,
  Sliders,
  Trash2,
} from "lucide-react";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
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
import { Separator } from "@/components/ui/separator";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "sonner";

export function SettingsPage() {
  const user = useQuery(api.auth.currentUser);
  const settings = useQuery(api.settings.get);
  const saveSettings = useMutation(api.settings.save);
  const deleteAccount = useMutation(api.users.deleteAccount);
  const { signOut } = useAuthActions();
  const navigate = useNavigate();

  const [youtubeApiKey, setYoutubeApiKey] = useState("");
  const [openaiApiKey, setOpenaiApiKey] = useState("");
  const [defaultLanguage, setDefaultLanguage] = useState("ro");
  const [defaultNumShorts, setDefaultNumShorts] = useState(5);
  const [defaultShortDuration, setDefaultShortDuration] = useState(60);
  const [youtubeCookies, setYoutubeCookies] = useState("");
  const [showYoutubeKey, setShowYoutubeKey] = useState(false);
  const [showOpenaiKey, setShowOpenaiKey] = useState(false);
  const [showCookies, setShowCookies] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  // Load settings
  useEffect(() => {
    if (settings) {
      setYoutubeApiKey(settings.youtubeApiKey || "");
      setOpenaiApiKey(settings.openaiApiKey || "");
      setYoutubeCookies(settings.youtubeCookies || "");
      setDefaultLanguage(settings.defaultLanguage || "ro");
      setDefaultNumShorts(settings.defaultNumShorts || 5);
      setDefaultShortDuration(settings.defaultShortDuration || 60);
    }
  }, [settings]);

  const handleSave = async () => {
    setIsSaving(true);
    try {
      await saveSettings({
        youtubeApiKey: youtubeApiKey || undefined,
        openaiApiKey: openaiApiKey || undefined,
        youtubeCookies: youtubeCookies || undefined,
        defaultLanguage,
        defaultNumShorts,
        defaultShortDuration,
      });
      toast.success("Setări salvate!");
    } catch {
      toast.error("Eroare la salvare");
    } finally {
      setIsSaving(false);
    }
  };

  const handleDeleteAccount = async () => {
    try {
      await deleteAccount();
      await signOut();
      navigate("/");
      toast.success("Cont șters");
    } catch {
      toast.error("Eroare la ștergerea contului");
    }
  };

  return (
    <div className="p-6 md:p-8 max-w-2xl mx-auto">
      <div className="mb-8">
        <h1 className="text-2xl md:text-3xl font-bold flex items-center gap-2">
          <Settings className="size-7 text-primary" />
          Setări
        </h1>
        <p className="text-muted-foreground mt-1">
          Configurează API keys și preferințele implicite
        </p>
      </div>

      <div className="space-y-8">
        {/* API Keys Section */}
        <section>
          <div className="flex items-center gap-2 mb-4">
            <Key className="size-4 text-primary" />
            <h2 className="text-lg font-semibold">API Keys</h2>
          </div>
          <p className="text-sm text-muted-foreground mb-4">
            Opțional — cheile sunt stocate securizat și folosite doar pentru
            procesarea video-urilor tale. Dacă nu ai chei proprii, aplicația
            va folosi AI-ul integrat Viktor.
          </p>

          <div className="space-y-4">
            {/* YouTube API Key */}
            <div className="space-y-2">
              <Label htmlFor="youtubeApiKey">YouTube Data API Key</Label>
              <div className="relative">
                <Input
                  id="youtubeApiKey"
                  type={showYoutubeKey ? "text" : "password"}
                  placeholder="AIza..."
                  value={youtubeApiKey}
                  onChange={(e) => setYoutubeApiKey(e.target.value)}
                  className="pr-10"
                />
                <button
                  type="button"
                  onClick={() => setShowYoutubeKey(!showYoutubeKey)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                >
                  {showYoutubeKey ? (
                    <EyeOff className="size-4" />
                  ) : (
                    <Eye className="size-4" />
                  )}
                </button>
              </div>
              <p className="text-xs text-muted-foreground">
                Obțineți de pe{" "}
                <a
                  href="https://console.cloud.google.com/apis/credentials"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary hover:underline"
                >
                  Google Cloud Console
                </a>
              </p>
            </div>

            {/* OpenAI API Key */}
            <div className="space-y-2">
              <Label htmlFor="openaiApiKey">OpenAI API Key</Label>
              <div className="relative">
                <Input
                  id="openaiApiKey"
                  type={showOpenaiKey ? "text" : "password"}
                  placeholder="sk-..."
                  value={openaiApiKey}
                  onChange={(e) => setOpenaiApiKey(e.target.value)}
                  className="pr-10"
                />
                <button
                  type="button"
                  onClick={() => setShowOpenaiKey(!showOpenaiKey)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                >
                  {showOpenaiKey ? (
                    <EyeOff className="size-4" />
                  ) : (
                    <Eye className="size-4" />
                  )}
                </button>
              </div>
              <p className="text-xs text-muted-foreground">
                Obțineți de pe{" "}
                <a
                  href="https://platform.openai.com/api-keys"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary hover:underline"
                >
                  OpenAI Platform
                </a>
              </p>
            </div>
          </div>
        </section>

        <Separator />

        {/* YouTube Cookies Section */}
        <section>
          <div className="flex items-center gap-2 mb-4">
            <Cookie className="size-4 text-primary" />
            <h2 className="text-lg font-semibold">YouTube Cookies</h2>
          </div>
          <p className="text-sm text-muted-foreground mb-4">
            Unele video-uri YouTube sunt protejate de bot-detection și nu pot fi
            accesate de pe server. Exportează cookies-urile tale YouTube din
            browser pentru a debloca aceste video-uri.
          </p>

          <div className="space-y-3">
            <div className="space-y-2">
              <Label htmlFor="youtubeCookies">
                Cookies (format cookies.txt sau Cookie header)
              </Label>
              <div className="relative">
                {showCookies ? (
                  <Textarea
                    id="youtubeCookies"
                    placeholder={`.youtube.com\tTRUE\t/\tTRUE\t0\tVISITOR_INFO1_LIVE\t...\n.youtube.com\tTRUE\t/\tTRUE\t0\tYSC\t...`}
                    value={youtubeCookies}
                    onChange={(e) => setYoutubeCookies(e.target.value)}
                    rows={5}
                    className="font-mono text-xs pr-10"
                  />
                ) : (
                  <div
                    className="flex items-center min-h-[80px] rounded-md border border-input bg-transparent px-3 py-2 text-sm cursor-pointer hover:border-primary/50 transition-colors pr-10"
                    onClick={() => setShowCookies(true)}
                  >
                    <span className="text-muted-foreground">
                      {youtubeCookies
                        ? `●●●●●●●● (${youtubeCookies.split("\n").filter((l: string) => l.trim()).length} linii)`
                        : "Nu sunt setate — click pentru a adăuga"}
                    </span>
                  </div>
                )}
                <button
                  type="button"
                  onClick={() => setShowCookies(!showCookies)}
                  className="absolute right-3 top-3 text-muted-foreground hover:text-foreground transition-colors"
                >
                  {showCookies ? (
                    <EyeOff className="size-4" />
                  ) : (
                    <Eye className="size-4" />
                  )}
                </button>
              </div>
            </div>

            <div className="rounded-lg bg-muted/50 p-3 space-y-2">
              <p className="text-xs font-medium">Cum exportezi cookies:</p>
              <ol className="text-xs text-muted-foreground space-y-1 list-decimal list-inside">
                <li>
                  Instalează extensia{" "}
                  <a
                    href="https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary hover:underline"
                  >
                    Get cookies.txt LOCALLY
                  </a>{" "}
                  (Chrome)
                </li>
                <li>Mergi pe youtube.com (logat cu contul tău)</li>
                <li>Click pe extensie → Export → copiază tot textul</li>
                <li>Lipește aici</li>
              </ol>
              <p className="text-xs text-muted-foreground mt-1">
                ⚠️ Cookies-urile expiră periodic — re-exportează dacă apar
                erori de acces.
              </p>
            </div>
          </div>
        </section>

        <Separator />

        {/* Default Preferences */}
        <section>
          <div className="flex items-center gap-2 mb-4">
            <Sliders className="size-4 text-primary" />
            <h2 className="text-lg font-semibold">Preferințe Implicite</h2>
          </div>

          <div className="space-y-4">
            <div className="space-y-2">
              <Label>Limba implicită</Label>
              <Select value={defaultLanguage} onValueChange={setDefaultLanguage}>
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

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Număr shorts implicit</Label>
                <Input
                  type="number"
                  min={1}
                  max={15}
                  value={defaultNumShorts}
                  onChange={(e) => setDefaultNumShorts(Number(e.target.value))}
                />
              </div>
              <div className="space-y-2">
                <Label>Durată max. implicită (s)</Label>
                <Input
                  type="number"
                  min={20}
                  max={60}
                  value={defaultShortDuration}
                  onChange={(e) =>
                    setDefaultShortDuration(Number(e.target.value))
                  }
                />
              </div>
            </div>
          </div>
        </section>

        {/* Save button */}
        <Button onClick={handleSave} disabled={isSaving} className="w-full">
          {isSaving ? (
            <Loader2 className="size-4 mr-2 animate-spin" />
          ) : (
            <Save className="size-4 mr-2" />
          )}
          Salvează Setările
        </Button>

        <Separator />

        {/* Account info */}
        <section>
          <h2 className="text-lg font-semibold mb-4">Cont</h2>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Nume</span>
              <span className="text-sm font-medium">{user?.name || "—"}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Email</span>
              <span className="text-sm font-medium">{user?.email || "—"}</span>
            </div>
          </div>
        </section>

        <Separator />

        {/* Danger zone */}
        <section>
          <h2 className="text-lg font-semibold text-destructive mb-4">
            Zonă periculoasă
          </h2>
          <AlertDialog>
            <AlertDialogTrigger asChild>
              <Button variant="destructive" className="w-full">
                <Trash2 className="size-4 mr-2" />
                Șterge contul
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Ești sigur?</AlertDialogTitle>
                <AlertDialogDescription>
                  Această acțiune este ireversibilă. Toate datele tale, inclusiv
                  job-urile și clipurile, vor fi șterse permanent.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Anulează</AlertDialogCancel>
                <AlertDialogAction
                  onClick={handleDeleteAccount}
                  className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                >
                  Da, șterge contul
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        </section>
      </div>
    </div>
  );
}
