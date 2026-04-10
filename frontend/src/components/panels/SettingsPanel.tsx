import { useEffect, useState } from "react";

import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import {
  getTelegramInfo,
  sendTelegramTest,
  setAutonomy,
  unpairTelegram
} from "@/lib/api";
import { useUiStore } from "@/store/ui";

import { Panel } from "./Panel";

const autonomyLevels = ["passive", "suggest", "semi_auto", "autonomous"];

export function SettingsPanel() {
  const status = useUiStore((state) => state.status);
  const [telegram, setTelegram] = useState<Record<string, unknown> | null>(null);
  const [message, setMessage] = useState("");
  const currentAutonomy = status?.daemon?.autonomy_level || "suggest";
  const config = status?.config || {};
  const telegramStatus = (
    status?.channels && typeof status.channels.telegram === "object" && status.channels.telegram
      ? status.channels.telegram
      : {}
  ) as Record<string, unknown>;

  useEffect(() => {
    getTelegramInfo().then(setTelegram).catch((error: Error) => setMessage(error.message));
  }, []);

  async function updateAutonomy(level: string) {
    const result = await setAutonomy(level);
    setMessage(result.error ? String(result.error) : `Autonomy set to ${level}`);
  }

  async function testTelegram() {
    const result = await sendTelegramTest();
    setMessage(result.error ? String(result.error) : "Telegram test sent.");
  }

  async function disconnectTelegram() {
    const result = await unpairTelegram();
    setMessage(result.error ? String(result.error) : "Telegram unpaired.");
    setTelegram(await getTelegramInfo().catch(() => null));
  }

  return (
    <Panel title="Settings & Channels" badge="controls">
      <div className="grid gap-3 xl:grid-cols-3">
        <Card className="p-4">
          <h3 className="font-display text-lg font-bold text-ink-strong">Autonomy</h3>
          <p className="mt-1 text-sm text-muted">Choose how freely Jarvis may act.</p>
          <div className="mt-4 grid gap-2">
            {autonomyLevels.map((level) => (
              <Button
                key={level}
                onClick={() => updateAutonomy(level)}
                variant={currentAutonomy === level ? "accent" : "default"}
              >
                {level.replace("_", " ")}
              </Button>
            ))}
          </div>
        </Card>

        <Card className="p-4">
          <h3 className="font-display text-lg font-bold text-ink-strong">Telegram</h3>
          <div className="mt-4 grid gap-2 text-sm">
            <Row label="Configured" value={String(telegramStatus.configured ?? false)} />
            <Row label="Paired" value={String(telegramStatus.paired ?? false)} />
            <Row label="Bot" value={String(telegram?.username || "unknown")} />
            <Row label="Poll" value={`${String(telegramStatus.poll_interval_s || "?")}s`} />
          </div>
          <div className="mt-4 flex flex-wrap gap-2">
            <Button onClick={() => getTelegramInfo().then(setTelegram)} size="sm">Refresh</Button>
            <Button onClick={testTelegram} size="sm">Send test</Button>
            <Button onClick={disconnectTelegram} size="sm" variant="danger">Unpair</Button>
          </div>
        </Card>

        <Card className="p-4">
          <h3 className="font-display text-lg font-bold text-ink-strong">Runtime</h3>
          <div className="mt-4 grid gap-2 text-sm">
            <Row label="Socket" value={String(config.socket_path || "~/.mnemon/daemon.sock")} />
            <Row label="Web UI" value={`${String(config.webui_host || "0.0.0.0")}:${String(config.webui_port || 7777)}`} />
            <Row label="State dir" value={String(config.state_path || "~/.mnemon/state")} />
          </div>
        </Card>
      </div>
      {message && (
        <div className="mt-4 rounded-xl border border-border bg-surface p-3 text-sm text-ink-strong">
          {message}
        </div>
      )}
    </Panel>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-start justify-between gap-3 border-b border-border pb-2 last:border-b-0">
      <span className="font-mono text-xs uppercase tracking-[0.14em] text-muted">{label}</span>
      <strong className="max-w-[65%] break-words text-right text-ink-strong">{value}</strong>
    </div>
  );
}
