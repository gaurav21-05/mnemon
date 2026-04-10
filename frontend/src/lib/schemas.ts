import { z } from "zod";

export const ThoughtSchema = z.object({
  timestamp: z.string().optional(),
  activity: z.string(),
  summary: z.string(),
  details: z.record(z.string(), z.unknown()).optional()
});

export const GoalSchema = z.object({
  id: z.string(),
  description: z.string(),
  priority: z.number().optional(),
  status: z.string().optional(),
  progress: z.number().optional(),
  parent_id: z.string().nullable().optional(),
  subgoals: z.array(z.string()).optional(),
  success_criteria: z.string().optional()
});

export const MemoryItemSchema = z.object({
  id: z.string(),
  preview: z.string().optional(),
  content: z.string().optional(),
  score: z.number().nullable().optional(),
  source: z.string().optional(),
  timestamp: z.string().optional(),
  importance: z.number().optional(),
  tags: z.array(z.string()).optional(),
  citation: z.string().optional(),
  scope_type: z.string().optional(),
  scope_id: z.string().optional(),
  repo_name: z.string().optional(),
  summary_of_count: z.number().optional(),
  source_episode_ids: z.array(z.string()).optional()
});

export const GraphNodeSchema = z.object({
  id: z.string(),
  label: z.string().optional(),
  kind: z.string().optional(),
  memory_type: z.string().optional(),
  memory_id: z.string().optional(),
  count: z.number().optional(),
  importance: z.number().optional()
});

export const GraphEdgeSchema = z.object({
  id: z.string().optional(),
  source: z.string(),
  target: z.string(),
  kind: z.string().optional()
});

export const StatusSchema = z.object({
  daemon: z
    .object({
      started_at: z.string().nullable().optional(),
      total_cycles: z.number().optional(),
      total_idle_ticks: z.number().optional(),
      autonomy_level: z.string().optional()
    })
    .optional(),
  brain: z.object({ active_goals: z.array(GoalSchema).optional() }).optional(),
  pending_approvals: z.array(z.record(z.string(), z.unknown())).optional(),
  proactive_inbox: z.array(z.record(z.string(), z.unknown())).optional(),
  chat_history: z.array(z.record(z.string(), z.string())).optional(),
  channels: z.record(z.string(), z.unknown()).optional(),
  config: z.record(z.string(), z.unknown()).optional(),
  connection_error: z.string().optional(),
  error: z.string().optional()
});

export const DbSnapshotSchema = z.record(
  z.string(),
  z.object({
    count: z.number().optional(),
    sample: z.array(z.unknown()).optional()
  }).or(z.record(z.string(), z.unknown()))
);

export type Thought = z.infer<typeof ThoughtSchema>;
export type Goal = z.infer<typeof GoalSchema>;
export type MemoryItem = z.infer<typeof MemoryItemSchema>;
export type GraphNode = z.infer<typeof GraphNodeSchema>;
export type GraphEdge = z.infer<typeof GraphEdgeSchema>;
export type Status = z.infer<typeof StatusSchema>;
export type DbSnapshot = z.infer<typeof DbSnapshotSchema>;
