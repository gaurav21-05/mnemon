import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import type { ButtonHTMLAttributes } from "react";

import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 rounded-full border text-sm font-semibold transition focus-visible:outline-none focus-visible:ring-4 disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default: "border-border bg-surface-muted text-ink-strong hover:text-accent-strong",
        accent: "border-accent bg-accent text-[var(--text-on-accent)] hover:bg-accent-strong",
        ghost: "border-border bg-transparent text-muted hover:text-ink-strong",
        danger: "border-danger/40 bg-danger/10 text-danger hover:bg-danger/15"
      },
      size: {
        sm: "h-8 px-3",
        md: "h-10 px-4",
        icon: "h-9 w-9 px-0"
      }
    },
    defaultVariants: {
      variant: "default",
      size: "md"
    }
  }
);

export type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> &
  VariantProps<typeof buttonVariants> & {
    asChild?: boolean;
  };

export function Button({ className, variant, size, asChild, ...props }: ButtonProps) {
  const Comp = asChild ? Slot : "button";
  return <Comp className={cn(buttonVariants({ variant, size }), className)} {...props} />;
}
