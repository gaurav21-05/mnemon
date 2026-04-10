import type { ReactNode } from "react";

import { Badge } from "@/components/ui/Badge";

export function Panel({
  title,
  badge,
  children
}: {
  title: string;
  badge?: string;
  children: ReactNode;
}) {
  return (
    <section className="flex h-full min-h-0 flex-col">
      <header className="flex items-center justify-between border-b border-border px-4 py-3">
        <h2 className="font-display text-2xl font-bold tracking-[-0.05em] text-ink-strong">
          {title}
        </h2>
        {badge && <Badge>{badge}</Badge>}
      </header>
      <div className="min-h-0 flex-1 overflow-auto p-4">{children}</div>
    </section>
  );
}
