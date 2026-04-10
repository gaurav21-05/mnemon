import type { InputHTMLAttributes } from "react";

import { cn } from "@/lib/utils";

export function Input({ className, ...props }: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      className={cn(
        "h-10 rounded-xl border border-border bg-surface px-3 text-sm text-ink-strong",
        "placeholder:text-muted focus:border-accent focus:outline-none focus:ring-4 focus:ring-[var(--focus)]",
        className
      )}
      {...props}
    />
  );
}
