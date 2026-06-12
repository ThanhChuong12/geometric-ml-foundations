"use client";

import { useEffect, useState } from "react";
import { ArrowUp } from "lucide-react";

export function ScrollProgress() {
  const [progress, setProgress] = useState(0);
  const [showTop, setShowTop] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      const scrollTop = window.scrollY;
      const docHeight =
        document.documentElement.scrollHeight - window.innerHeight;
      const pct = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0;
      setProgress(pct);
      setShowTop(scrollTop > 200);
    };

    window.addEventListener("scroll", handleScroll, { passive: true });
    handleScroll(); // initialize progress state on mount
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <>
      {/* Scroll progress bar — fixed at the bottom of the viewport */}
      <div className="fixed bottom-0 left-0 w-full h-1.5 bg-stone-200 z-50">
        <div
          className="h-full bg-stone-900"
          style={{ width: `${progress}%`, transition: "width 80ms linear" }}
        />
      </div>

      {/* Scroll-to-top button — visible after scrolling more than 200px */}
      <button
        onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
        aria-label="Scroll to top"
        className={`fixed bottom-4 right-4 z-50 w-10 h-10 bg-stone-900 text-white border-2 border-stone-900
          flex items-center justify-center
          shadow-[3px_3px_0px_0px_rgba(28,25,23,0.4)]
          hover:bg-stone-700 active:shadow-none active:translate-y-0.5 active:translate-x-0.5
          transition-all duration-200
          ${showTop ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4 pointer-events-none"}`}
      >
        <ArrowUp className="w-5 h-5" />
      </button>
    </>
  );
}
