import { useConvexAuth } from "convex/react";
import { Scissors } from "lucide-react";
import { Link } from "react-router-dom";
import { APP_NAME } from "@/lib/constants";
import { Button } from "./ui/button";

export function Header() {
  const { isAuthenticated } = useConvexAuth();

  return (
    <header className="sticky top-0 z-50 border-b bg-background/80 backdrop-blur-lg">
      <div className="container flex h-14 items-center justify-between">
        <Link to="/" className="flex items-center gap-2 font-bold text-lg">
          <div className="size-7 rounded-lg bg-primary flex items-center justify-center">
            <Scissors className="size-3.5 text-primary-foreground" />
          </div>
          {APP_NAME}
        </Link>

        <nav className="flex items-center gap-2">
          {isAuthenticated ? (
            <Button asChild size="sm">
              <Link to="/dashboard">Dashboard</Link>
            </Button>
          ) : (
            <>
              <Button asChild variant="ghost" size="sm">
                <Link to="/login">Conectare</Link>
              </Button>
              <Button asChild size="sm">
                <Link to="/signup">Creează Cont</Link>
              </Button>
            </>
          )}
        </nav>
      </div>
    </header>
  );
}
