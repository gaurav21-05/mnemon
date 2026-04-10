import {
  Bot,
  BrainCircuit,
  Clock3,
  GitGraph,
  Inbox,
  Database,
  MessageSquare,
  Settings,
  Target
} from "lucide-react";

import { cn } from "@/lib/utils";
import { type TabId, useUiStore } from "@/store/ui";

const navItems: Array<{ id: TabId; title: string; copy: string; icon: typeof BrainCircuit }> = [
  { id: "chat", title: "Chat", copy: "Talk with Jarvis first", icon: MessageSquare },
  { id: "memory", title: "Memory", copy: "Browse what Jarvis remembers", icon: BrainCircuit },
  { id: "graph", title: "Graph", copy: "Zoom, group, expand relationships", icon: GitGraph },
  { id: "goals", title: "Goals", copy: "Current work and priorities", icon: Target },
  { id: "timeline", title: "Timeline", copy: "Thoughts and recent activity", icon: Clock3 },
  { id: "inbox", title: "Inbox", copy: "Proactive messages", icon: Inbox },
  { id: "transparency", title: "Data", copy: "Inspect DBs and reset test state", icon: Database },
  { id: "overview", title: "Overview", copy: "Daemon status at a glance", icon: Bot },
  { id: "settings", title: "Settings", copy: "Channels and runtime", icon: Settings }
];

export function Sidebar() {
  const activeTab = useUiStore((state) => state.activeTab);
  const setActiveTab = useUiStore((state) => state.setActiveTab);

  return (
    <aside className="flex min-h-0 flex-col gap-3 rounded-2xl border border-border bg-surface p-4 shadow-card">
      <div>
        <h1 className="font-display text-xl font-bold tracking-[-0.04em] text-ink-strong">
          Jarvis workspace
        </h1>
        <p className="mt-1 text-sm text-muted">Switch modes without cramming the room.</p>
      </div>
      <nav className="grid gap-2 overflow-auto pr-1">
        {navItems.map((item) => {
          const Icon = item.icon;
          const active = activeTab === item.id;
          return (
            <button
              key={item.id}
              className={cn(
                "rounded-xl border p-3 text-left transition",
                active
                  ? "border-border-strong bg-surface-sand text-ink-strong"
                  : "border-border bg-surface-muted text-ink hover:bg-surface-sand"
              )}
              onClick={() => setActiveTab(item.id)}
              type="button"
            >
              <span className="flex items-center gap-2 font-display text-sm font-bold">
                <Icon size={15} />
                {item.title}
              </span>
              <span className="mt-1 block text-xs text-muted">{item.copy}</span>
            </button>
          );
        })}
      </nav>
    </aside>
  );
}
