import { useEffect, useState, useTransition } from "react";

import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { clearEverything, getDbSnapshot } from "@/lib/api";
import type { DbSnapshot } from "@/lib/schemas";

import { Panel } from "./Panel";

export function TransparencyPanel() {
  const [snapshot, setSnapshot] = useState<DbSnapshot | null>(null);
  const [status, setStatus] = useState("");
  const [, startTransition] = useTransition();

  async function load() {
    setSnapshot(await getDbSnapshot(8));
  }

  useEffect(() => {
    void getDbSnapshot(8)
      .then((data) => startTransition(() => setSnapshot(data)))
      .catch((error: Error) => startTransition(() => setStatus(error.message)));
  }, []);

  async function resetAll() {
    const confirmation = window.prompt(
      "Type CLEAR EVERYTHING to delete all DBs, thoughts, goals, inbox, chat, and Jarvis files."
    );
    if (confirmation !== "CLEAR EVERYTHING") return;
    setStatus("Clearing all daemon state…");
    const result = await clearEverything();
    setStatus(result.ok ? "Everything cleared. Starting fresh." : JSON.stringify(result));
    await load();
  }

  return (
    <Panel title="Data Transparency" badge="testing tools">
      <div className="mb-4 flex flex-wrap gap-2">
        <Button onClick={() => load()} size="sm">Refresh snapshot</Button>
        <Button onClick={resetAll} size="sm" variant="danger">Clear everything</Button>
      </div>
      {status && <div className="mb-4 rounded-xl border border-border bg-surface p-3">{status}</div>}
      <div className="grid gap-3 xl:grid-cols-2">
        {snapshot &&
          Object.entries(snapshot).map(([name, value]) => {
            const store = normalizeStore(value);
            return (
              <Card key={name} className="min-w-0 p-4">
                <div className="flex items-center justify-between gap-3">
                  <h3 className="font-display text-lg font-bold capitalize text-ink-strong">{name}</h3>
                  <Badge>{store.countLabel}</Badge>
                </div>
                <div className="mt-4 grid gap-2">
                  {store.rows.map((row, index) => (
                    <RecordPreview key={`${name}-${index}`} row={row} />
                  ))}
                </div>
              </Card>
            );
          })}
      </div>
    </Panel>
  );
}

function normalizeStore(value: unknown) {
  if (!value || typeof value !== "object") return { countLabel: "0 records", rows: [] };
  const record = value as { count?: number; sample?: unknown[] };
  if (Array.isArray(record.sample)) {
    return {
      countLabel: `${record.count ?? record.sample.length} records`,
      rows: record.sample.slice(0, 8)
    };
  }
  return {
    countLabel: "runtime",
    rows: Object.entries(value).map(([key, item]) => ({ key, value: item }))
  };
}

function RecordPreview({ row }: { row: unknown }) {
  if (!row || typeof row !== "object") {
    return <div className="rounded-xl border border-border bg-surface p-3 text-sm">{String(row)}</div>;
  }
  const data = row as Record<string, unknown>;
  const title = String(data.context || data.label || data.canonical_name || data.key || data.id || "record");
  const subtitle = String(data.action || data.predicate || data.value || data.source || "");
  const meta = [
    data.timestamp ? String(data.timestamp) : "",
    data.lifecycle_state ? String(data.lifecycle_state) : "",
    data.importance != null ? `importance ${Math.round(Number(data.importance) * 100)}%` : ""
  ].filter(Boolean);
  return (
    <article className="rounded-xl border border-border bg-surface p-3">
      <div className="line-clamp-2 font-display font-bold text-ink-strong">{title}</div>
      {subtitle && <div className="mt-1 text-sm text-ink">{subtitle}</div>}
      {meta.length > 0 && (
        <div className="mt-2 font-mono text-[11px] uppercase tracking-[0.1em] text-muted">
          {meta.join(" · ")}
        </div>
      )}
    </article>
  );
}
