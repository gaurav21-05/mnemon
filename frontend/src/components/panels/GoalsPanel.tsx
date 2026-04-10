import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { z } from "zod";

import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { Input } from "@/components/ui/Input";
import { addGoal } from "@/lib/api";
import { useUiStore } from "@/store/ui";

import { Panel } from "./Panel";

const GoalFormSchema = z.object({
  description: z.string().min(3),
  priority: z.coerce.number().min(0).max(1)
});
type GoalFormValues = z.input<typeof GoalFormSchema>;

export function GoalsPanel() {
  const goals = useUiStore((state) => state.goals);
  const form = useForm<GoalFormValues>({
    resolver: zodResolver(GoalFormSchema),
    defaultValues: { description: "", priority: 0.7 }
  });

  async function onSubmit(values: GoalFormValues) {
    const parsed = GoalFormSchema.parse(values);
    await addGoal(parsed.description, parsed.priority);
    form.reset({ description: "", priority: 0.7 });
  }

  return (
    <Panel title="Goals" badge={`${goals.length} goals`}>
      <form className="mb-4 grid gap-2 md:grid-cols-[1fr_120px_auto]" onSubmit={form.handleSubmit(onSubmit)}>
        <Input placeholder="New goal…" {...form.register("description")} />
        <Input step="0.1" type="number" {...form.register("priority")} />
        <Button type="submit" variant="accent">Add</Button>
      </form>
      <div className="grid gap-2">
        {goals.map((goal) => (
          <Card key={goal.id} className="p-4">
            <div className="flex justify-between gap-3">
              <h3 className="font-display font-bold text-ink-strong">{goal.description}</h3>
              <span className="text-sm text-muted">{goal.status || "active"}</span>
            </div>
            <div className="mt-3 h-2 rounded-full bg-surface">
              <div
                className="h-full rounded-full bg-accent"
                style={{ width: `${Math.round((goal.progress || 0) * 100)}%` }}
              />
            </div>
          </Card>
        ))}
      </div>
    </Panel>
  );
}
