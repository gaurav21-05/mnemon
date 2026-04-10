import { Panel } from "./Panel";
import { useUiStore } from "@/store/ui";

export function InboxPanel() {
  const inbox = useUiStore((state) => state.status?.proactive_inbox || []);
  return (
    <Panel title="Inbox & History" badge={`${inbox.length} items`}>
      <div className="grid gap-2">
        {inbox.map((item, index) => (
          <article key={String(item.id || index)} className="rounded-xl border border-border bg-surface p-4">
            <div className="font-mono text-xs uppercase tracking-[0.14em] text-muted">
              {String(item.source_activity || "daemon")}
            </div>
            <p className="mt-2">{String(item.content || "")}</p>
          </article>
        ))}
        {!inbox.length && <div className="text-muted">No proactive inbox items.</div>}
      </div>
    </Panel>
  );
}
