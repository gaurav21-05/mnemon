import { Card } from "@/components/ui/Card";
import { useNow } from "@/lib/useNow";
import { formatDuration } from "@/lib/utils";
import { useUiStore } from "@/store/ui";

import { Panel } from "./Panel";

export function OverviewPanel() {
  const { status, online, thoughts, goals } = useUiStore();
  const daemon = status?.daemon;
  const now = useNow();
  const started = daemon?.started_at ? new Date(daemon.started_at).getTime() : Number.NaN;
  const uptime = Number.isFinite(started) ? formatDuration((now - started) / 1000) : "--";

  return (
    <Panel title="Overview" badge={online ? "live" : "syncing"}>
      <div className="grid gap-3 lg:grid-cols-[1.35fr_0.95fr]">
        <Card className="p-5">
          <div className="font-mono text-xs uppercase tracking-[0.16em] text-accent-strong">
            {online ? "Daemon online" : "Daemon reconnecting"}
          </div>
          <h3 className="mt-2 max-w-2xl font-display text-5xl font-bold leading-[0.95] tracking-[-0.07em] text-ink-strong">
            Operate Jarvis from one calm control surface.
          </h3>
          <p className="mt-4 max-w-2xl text-lg leading-relaxed text-ink">
            Review state, approve actions, manage memory, inspect history, and read
            Jarvis' latest idle thoughts without turning them into interruptions.
          </p>
        </Card>
        <Card className="p-5">
          <div className="font-mono text-xs uppercase tracking-[0.16em] text-muted">
            Operator checklist
          </div>
          <p className="mt-3 leading-relaxed">
            Start with graph clusters for memory shape, then inspect goals and inbox for
            what Jarvis wants to do next.
          </p>
        </Card>
      </div>

      <div className="mt-3 grid gap-3 md:grid-cols-4">
        <Stat label="Uptime" value={uptime} />
        <Stat label="Cycles" value={String(daemon?.total_cycles ?? 0)} />
        <Stat label="Thoughts" value={String(thoughts.length)} />
        <Stat label="Goals" value={String(goals.length)} />
      </div>
    </Panel>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <Card className="p-4">
      <div className="font-mono text-xs uppercase tracking-[0.14em] text-muted">{label}</div>
      <div className="mt-2 font-display text-3xl font-bold tracking-[-0.06em] text-ink-strong">
        {value}
      </div>
    </Card>
  );
}
