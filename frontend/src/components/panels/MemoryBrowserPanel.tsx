import { useEffect, useState } from "react";

import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { getRecentMemories } from "@/lib/api";
import type { MemoryItem } from "@/lib/schemas";
import { timeAgo } from "@/lib/utils";

import { Panel } from "./Panel";

export function MemoryBrowserPanel() {
  const [items, setItems] = useState<MemoryItem[]>([]);
  const [filter, setFilter] = useState<"all" | "important" | "tagged">("all");

  useEffect(() => {
    getRecentMemories(30).then(setItems).catch(() => setItems([]));
  }, []);

  const filtered = items.filter((item) => {
    if (filter === "important") return Number(item.importance || 0) >= 0.6;
    if (filter === "tagged") return Boolean(item.tags?.length);
    return true;
  });

  return (
    <Panel title="Memory Browser" badge={`${filtered.length} items`}>
      <div className="mb-3 flex gap-2">
        {(["all", "important", "tagged"] as const).map((value) => (
          <Button
            key={value}
            size="sm"
            variant={filter === value ? "accent" : "default"}
            onClick={() => setFilter(value)}
          >
            {value}
          </Button>
        ))}
      </div>
      <div className="grid gap-2">
        {filtered.map((item) => (
          <Card key={item.id} className="p-4">
            <div className="flex items-start justify-between gap-3">
              <h3 className="font-display text-base font-bold text-ink-strong">
                {item.preview || item.content || "Untitled memory"}
              </h3>
              <Badge>{Math.round(Number(item.score ?? item.importance ?? 0) * 100)}%</Badge>
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              {(item.tags || ["raw memory"]).slice(0, 4).map((tag) => (
                <Badge key={tag}>{tag}</Badge>
              ))}
            </div>
            <div className="mt-3 font-mono text-xs text-muted">
              {timeAgo(item.timestamp)} · {item.citation || item.id}
            </div>
          </Card>
        ))}
        {!filtered.length && <div className="rounded-xl border border-border p-6">No memories yet.</div>}
      </div>
    </Panel>
  );
}
