import { useEffect, useState } from "react";

import { getTimeline } from "@/lib/api";
import { timeAgo } from "@/lib/utils";
import { useUiStore } from "@/store/ui";

import { Panel } from "./Panel";

export function TimelinePanel() {
  const thoughts = useUiStore((state) => state.thoughts);
  const [items, setItems] = useState<Array<Record<string, unknown>>>([]);
  useEffect(() => {
    getTimeline(80).then(setItems).catch(() => setItems([]));
  }, [thoughts.length]);
  const merged = items.length
    ? items
    : thoughts.map((thought) => ({ ...thought, kind: "thought", title: thought.activity }));
  const sorted = [...merged].sort((a, b) => String(b.timestamp || "").localeCompare(String(a.timestamp || "")));

  return (
    <Panel title="Timeline" badge={`${sorted.length} events`}>
      <div className="relative ml-3 grid gap-0 border-l border-border pl-5">
        {sorted.map((item, index) => (
          <article key={`${String(item.timestamp)}-${index}`} className="relative pb-5">
            <span className="absolute -left-[29px] top-1 h-3 w-3 rounded-full border-2 border-surface bg-accent" />
            <div className="rounded-xl border border-border bg-surface p-4">
              <div className="flex justify-between font-mono text-xs uppercase tracking-[0.14em] text-muted">
                <span>{String(item.kind || item.activity || "event")}</span>
                <span>{timeAgo(String(item.timestamp || ""))}</span>
              </div>
              <h3 className="mt-2 font-display font-bold text-ink-strong">
                {String(item.title || item.activity || "thought")}
              </h3>
              <p className="mt-1 text-ink">{String(item.summary || "")}</p>
            </div>
          </article>
        ))}
        {!sorted.length && <div className="text-muted">No timeline events yet.</div>}
      </div>
    </Panel>
  );
}
