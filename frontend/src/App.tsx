import { useEffect } from "react";

import { AppShell } from "@/components/layout/AppShell";
import { ChatPanel } from "@/components/panels/ChatPanel";
import { GoalsPanel } from "@/components/panels/GoalsPanel";
import { InboxPanel } from "@/components/panels/InboxPanel";
import { MemoryBrowserPanel } from "@/components/panels/MemoryBrowserPanel";
import { MemoryGraphPanel } from "@/components/panels/MemoryGraphPanel";
import { OverviewPanel } from "@/components/panels/OverviewPanel";
import { SettingsPanel } from "@/components/panels/SettingsPanel";
import { TransparencyPanel } from "@/components/panels/TransparencyPanel";
import { TimelinePanel } from "@/components/panels/TimelinePanel";
import { useDaemonEvents } from "@/lib/useDaemonEvents";
import { useUiStore } from "@/store/ui";

export function App() {
  useDaemonEvents();
  const activeTab = useUiStore((state) => state.activeTab);
  const theme = useUiStore((state) => state.theme);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("mnemon-theme", theme);
  }, [theme]);

  return (
    <AppShell>
      {activeTab === "overview" && <OverviewPanel />}
      {activeTab === "memory" && <MemoryBrowserPanel />}
      {activeTab === "graph" && <MemoryGraphPanel />}
      {activeTab === "chat" && <ChatPanel />}
      {activeTab === "goals" && <GoalsPanel />}
      {activeTab === "timeline" && <TimelinePanel />}
      {activeTab === "inbox" && <InboxPanel />}
      {activeTab === "transparency" && <TransparencyPanel />}
      {activeTab === "settings" && <SettingsPanel />}
    </AppShell>
  );
}
