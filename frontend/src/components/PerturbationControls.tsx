"use client";
import * as Slider from '@radix-ui/react-slider';
import { RotateCcw, Wind, Layers, FlaskConical, AlertTriangle } from 'lucide-react';

export interface PerturbationState {
  rotation_x: number;
  rotation_y: number;
  rotation_z: number;
  noise_level: number;
  drop_ratio: number;
}

interface PerturbationControlsProps {
  value: PerturbationState;
  onChange: (v: PerturbationState) => void;
  onReset: () => void;
}

// Nhất quán với RotationSlider.tsx (Part 1) và NumPointsSlider.tsx
function SliderRow({
  label, value, min, max, step, unit, accentClass,
  onChange, hint,
}: {
  label: string; value: number; min: number; max: number;
  step: number; unit: string; accentClass: string;
  onChange: (v: number) => void; hint: string;
}) {
  return (
    <div className="space-y-2">
      <div className="flex justify-between items-end border-b border-stone-200 pb-1">
        <span className="text-xs font-bold text-stone-900 uppercase tracking-widest font-sans">{label}</span>
        <div className="flex items-baseline gap-1">
          <span className="text-2xl font-black text-stone-900 font-serif">{value % 1 === 0 ? value : value.toFixed(2)}</span>
          <span className="text-sm font-bold text-stone-700 font-sans">{unit}</span>
        </div>
      </div>
      <Slider.Root
        className="relative flex items-center select-none touch-none w-full h-8"
        value={[value]}
        onValueChange={(vals) => onChange(vals[0])}
        min={min} max={max} step={step}
      >
        <Slider.Track className="bg-stone-200 relative grow h-3 border-2 border-stone-900">
          <Slider.Range className={`absolute ${accentClass} h-full border-r-2 border-stone-900`} />
        </Slider.Track>
        <Slider.Thumb className="block w-6 h-6 bg-yellow-400 border-4 border-stone-900 shadow-[2px_2px_0px_0px_rgba(28,25,23,1)] hover:bg-yellow-300 focus:outline-none cursor-grab active:cursor-grabbing" />
      </Slider.Root>
      <p className="text-[10px] text-stone-400 font-sans leading-tight">{hint}</p>
    </div>
  );
}

export const DEFAULT_PERTURBATION: PerturbationState = {
  rotation_x: 0, rotation_y: 0, rotation_z: 0,
  noise_level: 0, drop_ratio: 0,
};

export function PerturbationControls({ value, onChange, onReset }: PerturbationControlsProps) {
  const isActive = value.rotation_x !== 0 || value.rotation_y !== 0 ||
    value.rotation_z !== 0 || value.noise_level !== 0 || value.drop_ratio !== 0;

  return (
    <div className="bg-stone-50 border-2 border-stone-900 shadow-[8px_8px_0px_0px_rgba(28,25,23,1)]">
      {/* Header */}
      <div className="bg-violet-400 border-b-2 border-stone-900 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-stone-900 flex items-center justify-center text-violet-400"><FlaskConical className="w-4 h-4" /></div>
            <div>
              <h3 className="font-black text-stone-900 uppercase tracking-tight font-sans text-base">
                Kiểm Chứng Robustness
              </h3>
              <p className="text-xs font-medium text-stone-800 font-sans">Section 5.2 — PointNet paper</p>
            </div>
          </div>
          {isActive && (
            <button
              onClick={onReset}
              className="text-[10px] font-black font-sans uppercase tracking-widest bg-stone-900 text-white px-3 py-1.5 hover:bg-stone-700 transition-colors"
            >
              Đặt lại
            </button>
          )}
        </div>
      </div>

      <div className="p-6 space-y-6">
        {/* Rotation Group */}
        <div className="space-y-3">
          <div className="flex items-center gap-2 mb-2">
            <RotateCcw className="w-4 h-4 text-stone-700 flex-shrink-0" />
            <span className="text-xs font-black font-sans uppercase tracking-widest text-stone-900">
              Xoay 3D
            </span>
            <span className="text-[10px] font-sans text-stone-500 ml-1">
              Kiểm tra T-Net có giúp Full model bất biến với xoay không?
            </span>
          </div>
          <SliderRow label="Xoay trục X" value={value.rotation_x} min={0} max={360} step={5}
            unit="°" accentClass="bg-red-400"
            onChange={(v) => onChange({ ...value, rotation_x: v })}
            hint="Lật ngửa / úp vật thể" />
          <SliderRow label="Xoay trục Y" value={value.rotation_y} min={0} max={360} step={5}
            unit="°" accentClass="bg-emerald-400"
            onChange={(v) => onChange({ ...value, rotation_y: v })}
            hint="Xoay sang trái / phải (phổ biến nhất)" />
          <SliderRow label="Xoay trục Z" value={value.rotation_z} min={0} max={360} step={5}
            unit="°" accentClass="bg-blue-400"
            onChange={(v) => onChange({ ...value, rotation_z: v })}
            hint="Nghiêng vật thể" />
        </div>

        <div className="border-t border-stone-200" />

        {/* Noise */}
        <div className="space-y-3">
          <div className="flex items-center gap-2 mb-2">
            <Wind className="w-4 h-4 text-stone-700 flex-shrink-0" />
            <span className="text-xs font-black font-sans uppercase tracking-widest text-stone-900">
              Thêm nhiễu Gaussian
            </span>
            <span className="text-[10px] font-sans text-stone-500 ml-1">
              PointNet có robust với nhiễu không?
            </span>
          </div>
          <SliderRow label="Mức nhiễu σ" value={value.noise_level} min={0} max={0.15} step={0.005}
            unit="" accentClass="bg-amber-400"
            onChange={(v) => onChange({ ...value, noise_level: v })}
            hint="0 = không nhiễu | 0.05 = vừa | 0.10+ = nhiễu mạnh" />
        </div>

        <div className="border-t border-stone-200" />

        {/* Drop ratio */}
        <div className="space-y-3">
          <div className="flex items-center gap-2 mb-2">
            <Layers className="w-4 h-4 text-stone-700 flex-shrink-0" />
            <span className="text-xs font-black font-sans uppercase tracking-widest text-stone-900">
              Bỏ điểm ngẫu nhiên
            </span>
            <span className="text-[10px] font-sans text-stone-500 ml-1">
              Theorem 2: critical points quyết định kết quả
            </span>
          </div>
          <SliderRow label="Tỷ lệ bỏ điểm" value={Math.round(value.drop_ratio * 100)} min={0} max={75} step={5}
            unit="%" accentClass="bg-pink-400"
            onChange={(v) => onChange({ ...value, drop_ratio: v / 100 })}
            hint="0% = giữ nguyên | 50% = bỏ một nửa | 75% = chỉ còn 25% điểm" />
        </div>

        {/* Active indicator */}
        {isActive && (
          <div className="bg-violet-100 border border-violet-400 px-4 py-2 flex items-center gap-2">
            <AlertTriangle className="w-3.5 h-3.5 text-violet-800 flex-shrink-0" />
            <p className="text-xs font-bold font-sans text-violet-800">
              Đang áp dụng biến đổi. Nhấn <strong>Chạy Phân Loại</strong> để xem tác động.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
