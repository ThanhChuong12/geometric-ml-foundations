interface MathTermProps {
  term: string;
  definition: string;
}

/**
 * Renders an inline mathematical or ML term with a CSS-only hover tooltip.
 * No external tooltip library is required.
 */
export function MathTerm({ term, definition }: MathTermProps) {
  return (
    <span className="relative group inline-block">
      <span className="border-b-2 border-dashed border-stone-500 cursor-help font-bold">
        {term}
      </span>
      {/* Tooltip bubble */}
      <span
        className="
          absolute bottom-full left-1/2 -translate-x-1/2 mb-2
          w-60 bg-stone-900 text-white text-[11px] font-sans leading-relaxed
          px-3 py-2 z-50 pointer-events-none
          border border-stone-600
          shadow-[4px_4px_0px_0px_rgba(0,0,0,0.5)]
          opacity-0 group-hover:opacity-100
          translate-y-1 group-hover:translate-y-0
          transition-all duration-150
          whitespace-normal
        "
      >
        {definition}
        {/* Arrow */}
        <span className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-stone-900" />
      </span>
    </span>
  );
}
