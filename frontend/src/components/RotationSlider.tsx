"use client";
import * as Slider from '@radix-ui/react-slider';

interface RotationSliderProps {
  value: number;
  onChange: (value: number) => void;
}

export function RotationSlider({ value, onChange }: RotationSliderProps) {
  return (
    <div className="flex flex-col gap-6">
      <div className="flex justify-between items-end border-b-2 border-stone-900 pb-2">
        <label className="text-sm font-bold text-stone-900 uppercase tracking-widest font-sans">Góc xoay θ</label>
        <div className="flex items-baseline gap-1">
          <span className="text-5xl font-black text-stone-900 font-serif">{value}</span>
          <span className="text-xl text-stone-900 font-bold font-serif">°</span>
        </div>
      </div>

      <Slider.Root
        className="relative flex items-center select-none touch-none w-full h-10 mt-4"
        value={[value]}
        onValueChange={(values) => onChange(values[0])}
        max={360}
        min={0}
        step={1}
      >
        <Slider.Track className="bg-stone-200 relative grow h-4 border-2 border-stone-900">
          <Slider.Range className="absolute bg-sky-400 h-full border-r-2 border-stone-900" />
        </Slider.Track>
        <Slider.Thumb
          className="block w-8 h-8 bg-yellow-400 border-4 border-stone-900 shadow-[2px_2px_0px_0px_rgba(28,25,23,1)] hover:bg-yellow-300 focus:outline-none focus:ring-0 active:bg-yellow-500 transition-none cursor-grab active:cursor-grabbing"
          aria-label="Rotation"
        />
      </Slider.Root>

      <div className="flex justify-between text-xs text-stone-600 font-bold font-sans mt-2">
        <span className="relative before:absolute before:w-0.5 before:h-2 before:bg-stone-900 before:-top-4 before:left-1/2 before:-translate-x-1/2">0°</span>
        <span className="relative before:absolute before:w-0.5 before:h-2 before:bg-stone-900 before:-top-4 before:left-1/2 before:-translate-x-1/2">90°</span>
        <span className="relative before:absolute before:w-0.5 before:h-2 before:bg-stone-900 before:-top-4 before:left-1/2 before:-translate-x-1/2">180°</span>
        <span className="relative before:absolute before:w-0.5 before:h-2 before:bg-stone-900 before:-top-4 before:left-1/2 before:-translate-x-1/2">270°</span>
        <span className="relative before:absolute before:w-0.5 before:h-2 before:bg-stone-900 before:-top-4 before:left-1/2 before:-translate-x-1/2">360°</span>
      </div>
    </div>
  );
}
