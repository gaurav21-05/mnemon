import react from "@vitejs/plugin-react";
import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url))
    }
  },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:7777",
      "/chat": "http://127.0.0.1:7777",
      "/events": "http://127.0.0.1:7777"
    }
  }
});
