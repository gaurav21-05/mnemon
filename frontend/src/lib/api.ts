import {
  GoalSchema,
  MemoryItemSchema,
  StatusSchema,
  DbSnapshotSchema,
  type Goal,
  type GraphEdge,
  type GraphNode,
  type MemoryItem,
  type Status,
  type DbSnapshot
} from "@/lib/schemas";

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  const data: unknown = await response.json();
  if (!response.ok || (typeof data === "object" && data && "error" in data)) {
    const error = typeof data === "object" && data && "error" in data ? String(data.error) : url;
    throw new Error(error);
  }
  return data as T;
}

export async function getRecentMemories(limit = 20): Promise<MemoryItem[]> {
  const data = await fetchJson<{ items?: unknown[] }>(`/api/memory/recent?limit=${limit}`);
  return (data.items || []).map((item) => MemoryItemSchema.parse(item));
}

export async function getMemoryGraph(limit = 80): Promise<{
  nodes: GraphNode[];
  edges: GraphEdge[];
}> {
  const data = await fetchJson<{ nodes?: unknown[]; edges?: unknown[] }>(
    `/api/memory/graph?limit=${limit}`
  );
  return {
    nodes: (data.nodes || []).flatMap((node) => {
      if (!node || typeof node !== "object") return [];
      const raw = node as Record<string, unknown>;
      return [{
        id: String(raw.id || ""),
        label: raw.label == null ? undefined : String(raw.label),
        kind: raw.kind == null ? undefined : String(raw.kind),
        memory_type: raw.memory_type == null ? undefined : String(raw.memory_type),
        memory_id: raw.memory_id == null ? undefined : String(raw.memory_id),
        count: typeof raw.count === "number" ? raw.count : undefined,
        importance: typeof raw.importance === "number" ? raw.importance : undefined
      }].filter((item) => item.id);
    }),
    edges: (data.edges || []).flatMap((edge) => {
      if (!edge || typeof edge !== "object") return [];
      const raw = edge as Record<string, unknown>;
      const source = normalizeEndpoint(raw.source);
      const target = normalizeEndpoint(raw.target);
      if (!source || !target) return [];
      return [{
        id: raw.id == null ? undefined : String(raw.id),
        source,
        target,
        kind: raw.kind == null ? undefined : String(raw.kind)
      }];
    })
  };
}

function normalizeEndpoint(value: unknown) {
  if (typeof value === "string") return value;
  if (value && typeof value === "object" && "id" in value) {
    return String((value as { id?: unknown }).id || "");
  }
  return "";
}

export async function sendChat(message: string): Promise<{ reply?: string; error?: string }> {
  return fetchJson("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message })
  });
}

export async function getTimeline(limit = 60): Promise<Array<Record<string, unknown>>> {
  const data = await fetchJson<{ items?: Array<Record<string, unknown>> }>(
    `/api/timeline?limit=${limit}`
  );
  return data.items || [];
}

export async function getDbSnapshot(limit = 5): Promise<DbSnapshot> {
  const data = await fetchJson<unknown>(`/api/debug/dbs?limit=${limit}`);
  return DbSnapshotSchema.parse(data);
}

export async function clearEverything(): Promise<Record<string, unknown>> {
  return fetchJson("/api/debug/clear-all", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ confirm: true })
  });
}

export async function setAutonomy(level: string): Promise<Record<string, unknown>> {
  return fetchJson("/api/settings/autonomy", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ level })
  });
}

export async function getTelegramInfo(): Promise<Record<string, unknown>> {
  return fetchJson("/api/telegram/info");
}

export async function sendTelegramTest(): Promise<Record<string, unknown>> {
  return fetchJson("/api/telegram/test", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: "Jarvis test message from Mnemon settings." })
  });
}

export async function unpairTelegram(): Promise<Record<string, unknown>> {
  return fetchJson("/api/telegram/unpair", { method: "POST" });
}

export async function addGoal(description: string, priority: number): Promise<Goal> {
  const data = await fetchJson<unknown>("/api/goals", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ description, priority })
  });
  return GoalSchema.parse(data);
}

export function parseStatus(data: unknown): Status {
  return StatusSchema.parse(data);
}
