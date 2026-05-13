import { useConvexAuth } from "convex/react";
import {
  ArrowRight,
  Brain,
  Download,
  Scissors,
  Upload,
  Zap,
} from "lucide-react";
import { Link, Navigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { APP_NAME } from "@/lib/constants";

export function LandingPage() {
  const { isAuthenticated, isLoading } = useConvexAuth();

  if (!isLoading && isAuthenticated) {
    return <Navigate to="/dashboard" replace />;
  }

  return (
    <div className="min-h-[calc(100vh-4rem)] flex flex-col">
      {/* Hero */}
      <section className="flex-1 flex items-center justify-center px-4 py-16 md:py-24">
        <div className="max-w-4xl mx-auto text-center">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-primary/10 text-primary text-sm font-medium mb-8">
            <Zap className="size-3.5" />
            Powered by AI
          </div>

          <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold tracking-tight mb-6">
            Podcast →{" "}
            <span className="text-primary">YouTube Shorts</span>
            <br />
            <span className="text-muted-foreground">Automat.</span>
          </h1>

          <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto mb-10">
            Lipești un link, AI-ul găsește cele mai virale momente,
            primești comenzile FFmpeg gata de tăiat. Zero efort manual.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button asChild size="lg" className="text-base px-8 h-12">
              <Link to="/signup">
                Începe Acum
                <ArrowRight className="size-4 ml-2" />
              </Link>
            </Button>
            <Button
              asChild
              variant="outline"
              size="lg"
              className="text-base px-8 h-12"
            >
              <Link to="/login">Am deja cont</Link>
            </Button>
          </div>
        </div>
      </section>

      {/* How it works */}
      <section className="border-t bg-muted/30 py-20 px-4">
        <div className="max-w-5xl mx-auto">
          <h2 className="text-2xl md:text-3xl font-bold text-center mb-12">
            Cum funcționează
          </h2>

          <div className="grid md:grid-cols-4 gap-8">
            {[
              {
                icon: Upload,
                title: "1. Lipește linkul",
                desc: "Adaugă URL-ul podcastului YouTube",
              },
              {
                icon: Download,
                title: "2. Transcriere AI",
                desc: "Transcrierea automată a conținutului",
              },
              {
                icon: Brain,
                title: "3. Analiză AI",
                desc: "AI-ul identifică momentele virale",
              },
              {
                icon: Scissors,
                title: "4. Taie & Publică",
                desc: "Primești comenzi FFmpeg + titluri SEO",
              },
            ].map((step) => (
              <div key={step.title} className="text-center">
                <div className="size-14 rounded-2xl bg-primary/10 flex items-center justify-center mx-auto mb-4">
                  <step.icon className="size-6 text-primary" />
                </div>
                <h3 className="font-semibold mb-2">{step.title}</h3>
                <p className="text-sm text-muted-foreground">{step.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t py-8 px-4 text-center text-sm text-muted-foreground">
        <p>
          {APP_NAME} — Built with ❤️ by Viktor AI
        </p>
      </footer>
    </div>
  );
}
