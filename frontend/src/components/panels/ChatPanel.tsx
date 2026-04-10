import { Send } from "lucide-react";
import type { FormEvent } from "react";
import { useState } from "react";

import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { sendChat } from "@/lib/api";
import { useUiStore } from "@/store/ui";

import { Panel } from "./Panel";

export function ChatPanel() {
  const messages = useUiStore((state) => state.chatMessages);
  const addChatMessage = useUiStore((state) => state.addChatMessage);
  const speakAssistantReply = useUiStore((state) => state.speakAssistantReply);
  const [sending, setSending] = useState(false);
  const [draft, setDraft] = useState("");

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const message = draft.trim();
    if (!message || sending) return;
    setSending(true);
    addChatMessage({ role: "You", text: message });
    setDraft("");
    const response: { reply?: string; error?: string; result?: string; message?: string } = await sendChat(message).catch((error: Error) => ({
      error: error.message
    }));
    const replyText = response.reply || response.result || response.message || response.error || "No response.";
    addChatMessage({ role: "Jarvis", text: replyText });
    if (!response.error) speakAssistantReply(replyText);
    setSending(false);
  }

  return (
    <Panel title="Chat with Jarvis" badge="memory context">
      <div className="flex h-full flex-col gap-3">
        <div className="min-h-0 flex-1 overflow-auto rounded-xl border border-border bg-surface p-4">
          {messages.length ? (
            messages.map((message, index) => (
              <div key={`${message.role}-${index}`} className="mb-4">
                <div className="font-mono text-xs uppercase tracking-[0.14em] text-muted">
                  {message.role}
                </div>
                <div className="mt-1 rounded-xl border border-border bg-surface-muted p-3">
                  {message.text}
                </div>
              </div>
            ))
          ) : (
            <div className="text-muted">Say something when the daemon is online.</div>
          )}
        </div>
        <form className="flex gap-2" data-testid="chat-form" onSubmit={onSubmit}>
          <Input
            className="flex-1"
            onChange={(event) => setDraft(event.target.value)}
            placeholder="Ask Jarvis anything…"
            value={draft}
          />
          <Button disabled={sending} type="submit" variant="accent">
            <Send size={16} />
            {sending ? "Thinking…" : "Send"}
          </Button>
        </form>
      </div>
    </Panel>
  );
}
