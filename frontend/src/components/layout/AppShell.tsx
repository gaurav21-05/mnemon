import type { ReactNode } from "react";

import { Sidebar } from "@/components/layout/Sidebar";
import { Topbar } from "@/components/layout/Topbar";
import { useUiStore } from "@/store/ui";

export function AppShell({ children }: { children: ReactNode }) {
  const status = useUiStore((state) => state.status);
  const online = useUiStore((state) => state.online);
  const connectionError = status?.connection_error || status?.error;

  return (
    <div className="min-h-screen bg-paper text-ink">
      <Topbar />
      <main className="mx-auto grid h-[calc(100vh-64px)] w-full max-w-[1380px] grid-rows-[auto_minmax(0,1fr)] gap-3 px-4 py-4">
        {!online && (
          <div className="rounded-full border border-dashed border-border-strong bg-danger/10 px-5 py-3 text-center font-mono text-xs uppercase tracking-[0.14em] text-danger">
            Daemon reconnecting{connectionError ? ` · ${connectionError}` : ""}
          </div>
        )}
        <section className="grid min-h-0 grid-cols-[248px_minmax(0,1fr)] gap-4">
          <Sidebar />
          <div className="min-h-0 rounded-2xl border border-border bg-surface-soft shadow-card">
            {children}
          </div>
        </section>
      </main>
    </div>
  );
}
