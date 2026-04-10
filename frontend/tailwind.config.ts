import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["selector", '[data-theme="dark"]'],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        paper: "var(--paper)",
        surface: "var(--surface)",
        "surface-muted": "var(--surface-muted)",
        "surface-soft": "var(--surface-soft)",
        ink: "var(--ink)",
        "ink-strong": "var(--ink-strong)",
        muted: "var(--muted)",
        accent: "var(--accent)",
        "accent-strong": "var(--accent-strong)",
        danger: "var(--danger)",
        success: "var(--success)"
      },
      fontFamily: {
        display: "var(--font-display)",
        sans: "var(--font-sans)",
        mono: "var(--font-mono)"
      },
      boxShadow: {
        card: "var(--shadow-card)",
        soft: "var(--shadow-soft)"
      }
    }
  },
  plugins: []
};

export default config;
