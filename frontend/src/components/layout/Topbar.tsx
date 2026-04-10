import { Moon, Sun, Volume2, VolumeX } from "lucide-react";

import { Button } from "@/components/ui/Button";
import { useUiStore } from "@/store/ui";

export function Topbar() {
  const {
    online,
    theme,
    voiceEnabled,
    toggleTheme,
    toggleVoice
  } = useUiStore();

  return (
    <header className="sticky top-0 z-30 border-b border-border bg-[var(--topbar-bg)] backdrop-blur">
      <div className="mx-auto grid h-16 w-full max-w-[1380px] grid-cols-[auto_1fr_auto] items-center gap-4 px-4">
        <div className="min-w-[190px]">
          <div className="font-display text-lg font-semibold leading-none tracking-[-0.05em] text-ink-strong">
            Mnemon
          </div>
          <div className="mt-1 font-mono text-[10px] uppercase tracking-[0.22em] text-muted">
            Jarvis · Control Room
          </div>
        </div>

        <div className="hidden h-px bg-gradient-to-r from-transparent via-border-strong to-transparent md:block" />

        <div className="flex items-center justify-end gap-2">
          <Button onClick={toggleTheme} size="sm" variant="default">
            {theme === "dark" ? <Moon size={14} /> : <Sun size={14} />}
            <span className="hidden sm:inline">{theme}</span>
          </Button>
          <Button
            onClick={toggleVoice}
            size="sm"
            variant={voiceEnabled ? "accent" : "default"}
          >
            {voiceEnabled ? <Volume2 size={14} /> : <VolumeX size={14} />}
            <span className="hidden sm:inline">{voiceEnabled ? "Voice on" : "Voice off"}</span>
          </Button>
          <div className="flex items-center gap-2 rounded-full border border-border bg-surface px-3 py-2 shadow-card">
            <span
              className={
                online
                  ? "h-2.5 w-2.5 rounded-full bg-success shadow-[0_0_0_4px_rgba(31,138,101,0.16)]"
                  : "h-2.5 w-2.5 rounded-full bg-danger"
              }
            />
            <div
              className={
                online
                  ? "font-mono text-[11px] font-bold uppercase tracking-[0.14em] text-success"
                  : "font-mono text-[11px] font-bold uppercase tracking-[0.14em] text-danger"
              }
            >
              {online ? "online" : "offline"}
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
