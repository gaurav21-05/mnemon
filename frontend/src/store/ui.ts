import { create } from "zustand";

import type { Goal, Status, Thought } from "@/lib/schemas";

export type TabId =
  | "chat"
  | "overview"
  | "memory"
  | "graph"
  | "goals"
  | "timeline"
  | "inbox"
  | "transparency"
  | "settings";

type UiState = {
  activeTab: TabId;
  theme: "light" | "dark";
  voiceEnabled: boolean;
  voiceName: string;
  online: boolean;
  status: Status | null;
  thoughts: Thought[];
  goals: Goal[];
  chatMessages: Array<{ role: string; text: string }>;
  setActiveTab: (tab: TabId) => void;
  toggleTheme: () => void;
  toggleVoice: () => void;
  setVoiceName: (name: string) => void;
  setOnline: (online: boolean) => void;
  setStatus: (status: Status) => void;
  setThoughts: (thoughts: Thought[]) => void;
  setGoals: (goals: Goal[]) => void;
  addChatMessage: (message: { role: string; text: string }) => void;
  speakAssistantReply: (text: string) => void;
  clearChatMessages: () => void;
};

function initialTheme(): "light" | "dark" {
  if (typeof window === "undefined") return "light";
  const saved = window.localStorage.getItem("mnemon-theme");
  if (saved === "dark" || saved === "light") return saved;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export const useUiStore = create<UiState>((set, get) => ({
  activeTab: "chat",
  theme: initialTheme(),
  voiceEnabled:
    typeof window !== "undefined" && window.localStorage.getItem("mnemon-voice") === "on",
  voiceName:
    typeof window !== "undefined" ? window.localStorage.getItem("mnemon-voice-name") || "" : "",
  online: false,
  status: null,
  thoughts: [],
  goals: [],
  chatMessages: [],
  setActiveTab: (activeTab) => set({ activeTab }),
  toggleTheme: () =>
    set((state) => ({ theme: state.theme === "dark" ? "light" : "dark" })),
  toggleVoice: () =>
    set((state) => {
      const next = !state.voiceEnabled;
      window.localStorage.setItem("mnemon-voice", next ? "on" : "off");
      if (!next && "speechSynthesis" in window) window.speechSynthesis.cancel();
      return { voiceEnabled: next };
    }),
  setVoiceName: (voiceName) => {
    window.localStorage.setItem("mnemon-voice-name", voiceName);
    set({ voiceName });
  },
  setOnline: (online) => set({ online }),
  setStatus: (status) => set({ status }),
  setThoughts: (thoughts) => set({ thoughts }),
  setGoals: (goals) => set({ goals }),
  addChatMessage: (message) =>
    set((state) => ({ chatMessages: [...state.chatMessages, message] })),
  speakAssistantReply: (text) => {
    if (!text || !get().voiceEnabled || !("speechSynthesis" in window)) return;
    const preferredName = get().voiceName;
    const voices = window.speechSynthesis.getVoices();
    const selected =
      voices.find((voice) => voice.name === preferredName) ||
      voices.find((voice) => /samantha|serena|natural|google uk english female/i.test(voice.name)) ||
      voices.find((voice) => /female|english/i.test(voice.name));
    const utterance = new SpeechSynthesisUtterance(text);
    if (selected) utterance.voice = selected;
    utterance.rate = 0.96;
    utterance.pitch = 0.92;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);
  },
  clearChatMessages: () => set({ chatMessages: [] })
}));
