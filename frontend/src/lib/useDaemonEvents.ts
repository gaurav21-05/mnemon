import { useEffect } from "react";

import { GoalSchema, ThoughtSchema } from "@/lib/schemas";
import { parseStatus } from "@/lib/api";
import { useUiStore } from "@/store/ui";

export function useDaemonEvents() {
  const setOnline = useUiStore((state) => state.setOnline);
  const setStatus = useUiStore((state) => state.setStatus);
  const setThoughts = useUiStore((state) => state.setThoughts);
  const setGoals = useUiStore((state) => state.setGoals);

  useEffect(() => {
    let retry: number | undefined;
    let source: EventSource | undefined;

    const connect = () => {
      source?.close();
      source = new EventSource("/events");

      source.addEventListener("status", (event) => {
        const status = parseStatus(JSON.parse(event.data));
        setStatus(status);
        setOnline(!(status.error || status.connection_error));
      });

      source.addEventListener("thoughts", (event) => {
        const thoughts = JSON.parse(event.data);
        setThoughts(ThoughtSchema.array().parse(thoughts));
      });

      source.addEventListener("goals", (event) => {
        const goals = JSON.parse(event.data);
        setGoals(GoalSchema.array().parse(goals));
      });

      source.addEventListener("error", () => {
        setOnline(false);
        source?.close();
        retry = window.setTimeout(connect, 3000);
      });
    };

    connect();
    return () => {
      source?.close();
      if (retry) window.clearTimeout(retry);
    };
  }, [setGoals, setOnline, setStatus, setThoughts]);
}
