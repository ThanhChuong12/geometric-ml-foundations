"use client";
import { PointNetModelResult } from '@/types/pointnet';

interface PointNetResultCardProps {
  title: string;
  description: string;
  result: PointNetModelResult | null;
  variant: 'basic' | 'full';
}

// Match design language của PredictionCard.tsx (Part 1)
const variantStyles = {
  basic: {
    headerBg: 'bg-red-500',
    barColor: 'bg-red-500',
    badge: 'BASIC',
    icon: '(1)',
  },
  full: {
    headerBg: 'bg-emerald-400',
    barColor: 'bg-emerald-400',
    badge: 'FULL + T-NET',
    icon: '(2)',
  },
};

// Emoji icons cho 40 class ModelNet40 (top common)
const CLASS_EMOJI: Record<string, string> = {
  airplane: '✈', bathtub: '🛁', bed: '🛏', bench: '🪑', bookshelf: '📚',
  bottle: '🍶', bowl: '🥣', car: '🚗', chair: '🪑', cone: '📐',
  cup: '☕', curtain: '🪟', desk: '🪑', door: '🚪', dresser: '🗄',
  flower_pot: '🌺', glass_box: '📦', guitar: '🎸', keyboard: '⌨', lamp: '💡',
  laptop: '💻', mantel: '🏛', monitor: '🖥', night_stand: '🗄', person: '🧍',
  piano: '🎹', plant: '🌿', radio: '📻', range_hood: '🍳', sink: '🚿',
  sofa: '🛋', stairs: '🪜', stool: '🪑', table: '🪵', tent: '⛺',
  toilet: '🚽', tv_stand: '📺', vase: '🏺', wardrobe: '🚪', xbox: '🎮',
};

export function PointNetResultCard({
  title, description, result, variant
}: PointNetResultCardProps) {
  const styles = variantStyles[variant];

  return (
    // Same border + shadow pattern as PredictionCard.tsx
    <div className="flex flex-col border-2 border-stone-900 shadow-[4px_4px_0px_0px_rgba(28,25,23,1)] bg-stone-50">

      {/* Header — same structure as PredictionCard */}
      <div className={`${styles.headerBg} border-b-2 border-stone-900 px-6 py-4`}>
        <div className="flex items-center gap-3 mb-2">
          <span className="text-2xl font-black text-stone-900 font-serif">{styles.icon}</span>
          <h3 className="font-black text-lg text-stone-900 uppercase tracking-tight font-sans">{title}</h3>
          <span className="ml-auto text-[10px] font-black font-sans uppercase tracking-widest bg-stone-900 text-white px-2 py-1">
            {styles.badge}
          </span>
        </div>
        <p className="text-sm font-medium text-stone-900 font-sans">{description}</p>
      </div>

      {/* Body */}
      <div className="bg-stone-50 p-8 flex-1 flex flex-col gap-6">
        {result ? (
          <>
            {/* Prediction class */}
            <div className="flex flex-col items-center gap-3 w-full border-2 border-stone-900 p-6 bg-stone-100">
              <span className="text-xs font-bold text-stone-900 uppercase tracking-widest font-sans">
                Lớp dự đoán
              </span>
              <div className="text-5xl font-black text-stone-900 font-serif">
                {CLASS_EMOJI[result.label] ?? '?'}
              </div>
              <div className="text-xl font-black text-stone-900 font-sans uppercase tracking-wider">
                {result.label}
              </div>
            </div>

            {/* Confidence bar — same pattern as PredictionCard */}
            <div className="w-full space-y-3">
              <div className="flex justify-between items-baseline border-b-2 border-stone-900 pb-1">
                <span className="text-sm font-bold text-stone-900 uppercase tracking-wider font-sans">
                  Do tu tin
                </span>
                <span className="text-2xl font-black text-stone-900 font-sans">
                  {(result.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <div className="relative w-full bg-stone-200 h-6 border-2 border-stone-900 overflow-hidden">
                <div
                  className={`absolute inset-y-0 left-0 ${styles.barColor} border-r-2 border-stone-900 transition-all duration-700`}
                  style={{ width: `${result.confidence * 100}%` }}
                />
              </div>
            </div>

            {/* Top-3 mini breakdown */}
            <div className="space-y-2">
              <span className="text-xs font-bold text-stone-900 uppercase tracking-widest font-sans block border-b border-stone-200 pb-1">
                Top-3 Predictions
              </span>
              {result.top3.map((t, i) => (
                <div key={t.class_id} className="flex items-center gap-3">
                  <span className="text-xs font-bold font-sans text-stone-500 w-4">{i + 1}.</span>
                  <span className="text-xs font-sans text-stone-700 w-20 truncate">{t.label}</span>
                  <div className="flex-1 relative h-3 bg-stone-200 border border-stone-300 overflow-hidden">
                    <div
                      className={`absolute inset-y-0 left-0 ${i === 0 ? styles.barColor : 'bg-stone-400'} transition-all duration-500`}
                      style={{ width: `${t.confidence * 100}%` }}
                    />
                  </div>
                  <span className="text-xs font-bold font-sans text-stone-900 w-12 text-right">
                    {(t.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>

            {/* Critical points info */}
            <div className="flex items-center gap-3 bg-stone-100 border border-stone-300 px-4 py-2">
              <span className="text-[10px] font-bold font-sans uppercase tracking-widest text-stone-500">
                Critical Points
              </span>
              <span className="text-xs font-black font-sans text-stone-900">
                {result.num_critical} / {result.critical_points.length + result.num_critical - result.num_critical} unique
              </span>
              <span className="ml-auto text-[10px] font-sans text-stone-500">
                (Theorem 2)
              </span>
            </div>
          </>
        ) : (
          // Same empty state as PredictionCard
          <div className="text-stone-400 text-sm text-center py-12 font-sans font-medium uppercase tracking-widest">
            <div className="text-6xl mb-6 font-serif opacity-30 font-light">?</div>
            <div>Cho phan loai</div>
          </div>
        )}
      </div>
    </div>
  );
}
