"use client";

/**
 * Suppress hydration mismatch warnings caused by browser extensions
 * (e.g. Bitdefender, Grammarly) injecting attributes like `bis_skin_checked`.
 *
 * Patches console.error at MODULE LOAD TIME — before React hydration runs.
 * This must be imported early in the component tree (layout.tsx).
 */

if (typeof window !== "undefined") {
  const orig = console.error;
  console.error = (...args: unknown[]) => {
    const msg = typeof args[0] === "string" ? args[0] : String(args[0] ?? "");
    if (msg.includes("hydrat") && msg.includes("match")) return;
    if (msg.includes("bis_skin_checked")) return;
    orig.apply(console, args);
  };
}

export function SuppressHydrationWarning() {
  return null;
}
