"use client";

import { useState, useEffect } from 'react';
import { Loader2, CircleHelp } from 'lucide-react';

interface EnergyResultCardProps {
  energy: number | null;
  isLoading: boolean;
  error: string | null;
  durationMs?: number;
}

export function EnergyResultCard({
  energy,
  isLoading,
  error,
  durationMs
}: EnergyResultCardProps) {
  const [displayEnergy, setDisplayEnergy] = useState<number | null>(null);

  // Animate the energy value from zero to the actual result on each new prediction
  useEffect(() => {
    if (energy === null) { setDisplayEnergy(null); return; }
    const target = energy;
    const duration = 900;
    const startTime = performance.now();

    const tick = (now: number) => {
      const progress = Math.min((now - startTime) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
      setDisplayEnergy(target * eased);
      if (progress < 1) requestAnimationFrame(tick);
      else setDisplayEnergy(target);
    };

    requestAnimationFrame(tick);
  }, [energy]);

  return (
    <div className="flex flex-col border-2 border-stone-900 shadow-[4px_4px_0px_0px_rgba(28,25,23,1)] bg-stone-50 h-full">
      {/* Header */}
      <div className="bg-indigo-400 border-b-2 border-stone-900 px-6 py-4">
        <div className="flex items-center gap-3 mb-2">
          <span className="text-2xl font-black text-stone-900 font-serif">*</span>
          <h3 className="font-black text-lg text-stone-900 uppercase tracking-tight font-sans">
            Kết quả Dự đoán Năng lượng
          </h3>
          <span className="ml-auto text-[10px] font-black font-sans uppercase tracking-widest bg-stone-900 text-white px-2 py-1">
            NEQUIP GNN
          </span>
        </div>
        <p className="text-sm font-medium text-stone-900 font-sans">
          Năng lượng cơ học lượng tử (U0) được tính toán bởi mạng đẳng biến E(3).
        </p>
      </div>

      {/* Body */}
      <div className="bg-stone-50 p-6 flex-1 flex flex-col justify-center gap-6 min-h-[200px]">
        {isLoading ? (
          <div className="flex flex-col items-center justify-center py-12 gap-3">
            <Loader2 className="w-10 h-10 animate-spin text-indigo-600" />
            <span className="text-xs font-bold text-stone-500 uppercase tracking-widest font-sans">
              Đang chạy inference mạng NequIP...
            </span>
          </div>
        ) : error ? (
          <div className="bg-rose-100 border-2 border-rose-900 p-4 text-rose-950 text-sm font-bold font-sans shadow-[3px_3px_0px_0px_rgba(225,29,72,0.1)]">
            <span className="block text-xs font-extrabold uppercase tracking-wider mb-1 text-rose-800">
              Lỗi hệ thống:
            </span>
            {error}
          </div>
        ) : energy !== null ? (
          <div className="space-y-4">
            {/* Energy value card */}
            <div className="flex flex-col items-center gap-3 w-full border-2 border-stone-900 p-6 bg-stone-100 shadow-[2px_2px_0px_0px_rgba(28,25,23,1)]">
              <span className="text-xs font-bold text-stone-600 uppercase tracking-widest font-sans">
                Nội năng U₀ (eV)
              </span>
              <div className="text-4xl md:text-5xl font-black text-stone-900 font-serif tracking-tight text-center tabular-nums">
                {displayEnergy !== null ? displayEnergy.toFixed(4) : '...'}
              </div>
            </div>

            {/* Inference metadata */}
            <div className="flex items-center gap-3 bg-stone-100 border border-stone-300 px-4 py-2.5 text-xs font-sans">
              <span className="text-[10px] font-bold text-stone-500 uppercase tracking-widest">
                Độ trễ (Latency):
              </span>
              <span className="font-black text-stone-900">
                {durationMs !== undefined ? `${durationMs.toFixed(0)} ms` : 'N/A'}
              </span>
              <span className="ml-auto text-[10px] font-semibold text-stone-500 uppercase tracking-wider">
                CPU Inference
              </span>
            </div>
          </div>
        ) : (
          <div className="text-stone-400 text-sm text-center py-12 font-sans font-medium uppercase tracking-widest flex flex-col items-center">
            <CircleHelp className="w-12 h-12 mb-6 opacity-30 text-stone-500" />
            <div>Chờ chạy dự đoán</div>
          </div>
        )}
      </div>
    </div>
  );
}
