"use client";
import { PredictionResult } from '@/types';

interface PredictionCardProps {
  title: string;
  description: string;
  result: PredictionResult | null;
  variant: 'baseline' | 'augmentation' | 'averaging';
}

const variantStyles = {
  baseline: {
    bg: 'bg-stone-50',
    headerBg: 'bg-red-500',
    badge: 'bg-red-500',
    icon: '(1)',
  },
  augmentation: {
    bg: 'bg-stone-50',
    headerBg: 'bg-amber-400',
    badge: 'bg-amber-400',
    icon: '(2)',
  },
  averaging: {
    bg: 'bg-stone-50',
    headerBg: 'bg-emerald-400',
    badge: 'bg-emerald-400',
    icon: '(3)',
  },
};

export function PredictionCard({ title, description, result, variant }: PredictionCardProps) {
  const styles = variantStyles[variant];

  return (
    <div className={`flex flex-col border-2 border-stone-900 shadow-[4px_4px_0px_0px_rgba(28,25,23,1)] bg-stone-50`}>
      <div className={`${styles.headerBg} border-b-2 border-stone-900 px-6 py-4`}>
        <div className="flex items-center gap-3 mb-2">
          <span className="text-2xl font-black text-stone-900 font-serif">{styles.icon}</span>
          <h3 className="font-black text-lg text-stone-900 uppercase tracking-tight font-sans">{title}</h3>
        </div>
        <p className="text-sm font-medium text-stone-900 font-sans">{description}</p>
      </div>

      <div className={`${styles.bg} p-8 flex-1 flex flex-col items-center justify-center gap-8`}>
        {result ? (
          <>
            <div className="flex flex-col items-center gap-4 w-full border-2 border-stone-900 p-6 bg-stone-100">
              <span className="text-xs font-bold text-stone-900 uppercase tracking-widest font-sans">Lớp dự đoán</span>
              <div className={`text-8xl font-black text-stone-900 font-serif`}>
                {result.digit}
              </div>
            </div>

            <div className="w-full space-y-3">
              <div className="flex justify-between items-baseline border-b-2 border-stone-900 pb-1">
                <span className="text-sm font-bold text-stone-900 uppercase tracking-wider font-sans">Độ tự tin</span>
                <span className={`text-2xl font-black text-stone-900 font-sans`}>{(result.confidence * 100).toFixed(1)}%</span>
              </div>
              <div className="relative w-full bg-stone-200 h-6 border-2 border-stone-900 overflow-hidden">
                <div
                  className={`absolute inset-y-0 left-0 ${styles.badge} border-r-2 border-stone-900 transition-all duration-700`}
                  style={{ width: `${result.confidence * 100}%` }}
                />
              </div>
            </div>
          </>
        ) : (
          <div className="text-stone-400 text-sm text-center py-12 font-sans font-medium uppercase tracking-widest">
            <div className="text-6xl mb-6 font-serif opacity-30 font-light">?</div>
            <div>Chờ phân loại</div>
          </div>
        )}
      </div>
    </div>
  );
}
