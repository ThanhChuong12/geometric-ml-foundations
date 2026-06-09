"use client";

// Dùng Radix Slider nhất quán với RotationSlider.tsx (Part 1)
import * as Slider from '@radix-ui/react-slider';

interface NumPointsSliderProps {
  value: number;
  onChange: (value: number) => void;
}

const STEPS = [64, 128, 256, 512, 1024] as const;

export function NumPointsSlider({ value, onChange }: NumPointsSliderProps) {
  // Convert current value to step index
  const stepIndex = STEPS.indexOf(value as typeof STEPS[number]);
  const displayIndex = stepIndex === -1 ? STEPS.length - 1 : stepIndex;

  const handleChange = (values: number[]) => {
    onChange(STEPS[values[0]]);
  };

  return (
    // Same container structure as RotationSlider.tsx
    <div className="flex flex-col gap-6">
      <div className="flex justify-between items-end border-b-2 border-stone-900 pb-2">
        <label className="text-sm font-bold text-stone-900 uppercase tracking-widest font-sans">
          So diem N
        </label>
        <div className="flex items-baseline gap-1">
          {/* Big number display — same as RotationSlider */}
          <span className="text-5xl font-black text-stone-900 font-serif">{value}</span>
          <span className="text-base text-stone-700 font-bold font-sans ml-1">pts</span>
        </div>
      </div>

      {/* Radix Slider — same styles as RotationSlider */}
      <Slider.Root
        className="relative flex items-center select-none touch-none w-full h-10 mt-4"
        value={[displayIndex]}
        onValueChange={handleChange}
        min={0}
        max={STEPS.length - 1}
        step={1}
      >
        <Slider.Track className="bg-stone-200 relative grow h-4 border-2 border-stone-900">
          <Slider.Range className="absolute bg-violet-400 h-full border-r-2 border-stone-900" />
        </Slider.Track>
        <Slider.Thumb
          className="block w-8 h-8 bg-yellow-400 border-4 border-stone-900 shadow-[2px_2px_0px_0px_rgba(28,25,23,1)] hover:bg-yellow-300 focus:outline-none focus:ring-0 active:bg-yellow-500 transition-none cursor-grab active:cursor-grabbing"
          aria-label="Number of points"
        />
      </Slider.Root>

      {/* Step labels — same pattern as RotationSlider tick marks */}
      <div className="flex justify-between text-xs text-stone-600 font-bold font-sans mt-2">
        {STEPS.map((step) => (
          <span
            key={step}
            className={`relative before:absolute before:w-0.5 before:h-2 before:bg-stone-900 before:-top-4 before:left-1/2 before:-translate-x-1/2 ${
              value === step ? 'text-stone-900' : 'text-stone-400'
            }`}
          >
            {step}
          </span>
        ))}
      </div>

      {/* Context hint */}
      <p className="text-xs text-stone-500 font-sans leading-relaxed border-t border-stone-200 pt-3">
        <span className="font-bold text-stone-700">Thi nghiem mo rong:</span>{' '}
        Thay doi so luong diem de xem do ben cua PointNet (Theorem 2).
      </p>
    </div>
  );
}
