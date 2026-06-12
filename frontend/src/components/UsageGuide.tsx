import { Circle, RotateCcw, Layers, Box, Wind, FlaskConical, Flame } from 'lucide-react';

type TabType = 'part1' | 'part2' | 'part3';

interface Step {
  icon: React.ReactNode;
  label: string;
}

interface GuideConfig {
  steps: Step[];
  badgeColor: string;
}

const GUIDES: Record<TabType, GuideConfig> = {
  part1: {
    badgeColor: 'bg-rose-400',
    steps: [
      { icon: <Circle className="w-3.5 h-3.5 flex-shrink-0" />, label: 'Vẽ chữ số lên canvas' },
      { icon: <RotateCcw className="w-3.5 h-3.5 flex-shrink-0" />, label: 'Kéo slider để xoay ảnh' },
      { icon: <Layers className="w-3.5 h-3.5 flex-shrink-0" />, label: 'Nhấn "Chạy Phân loại"' },
    ],
  },
  part2: {
    badgeColor: 'bg-emerald-400',
    steps: [
      { icon: <Box className="w-3.5 h-3.5 flex-shrink-0" />, label: 'Chọn loại vật thể 3D' },
      { icon: <Wind className="w-3.5 h-3.5 flex-shrink-0" />, label: 'Điều chỉnh nhiễu & số điểm' },
      { icon: <Layers className="w-3.5 h-3.5 flex-shrink-0" />, label: 'Nhấn "Phân loại"' },
    ],
  },
  part3: {
    badgeColor: 'bg-cyan-400',
    steps: [
      { icon: <FlaskConical className="w-3.5 h-3.5 flex-shrink-0" />, label: 'Chọn phân tử mẫu' },
      { icon: <Circle className="w-3.5 h-3.5 flex-shrink-0" />, label: 'Chỉnh nguyên tử nếu muốn' },
      { icon: <Flame className="w-3.5 h-3.5 flex-shrink-0" />, label: 'Nhấn "Dự đoán Năng lượng"' },
    ],
  },
};

export function UsageGuide({ tab }: { tab: TabType }) {
  const { steps, badgeColor } = GUIDES[tab];

  return (
    <div className="w-full px-4 md:px-8 mb-4">
      <div className="border border-stone-200 bg-stone-50 px-4 py-2.5 flex items-center flex-wrap gap-x-3 gap-y-2">

        {/* Label */}
        <span className="text-[10px] font-black uppercase tracking-widest text-stone-400 font-sans border-r border-stone-300 pr-3">
          Hướng dẫn
        </span>

        {/* Steps */}
        {steps.map((step, i) => (
          <div key={i} className="flex items-center gap-2">
            {/* Arrow separator */}
            {i > 0 && (
              <span className="text-stone-300 text-xs font-bold select-none">&#8594;</span>
            )}

            {/* Step */}
            <div className="flex items-center gap-1.5">
              {/* Number badge */}
              <span
                className={`w-4 h-4 ${badgeColor} border border-stone-900 flex items-center justify-center text-stone-900 font-black text-[9px] font-sans flex-shrink-0`}
              >
                {i + 1}
              </span>

              {/* Icon + text */}
              <span className="flex items-center gap-1 text-stone-600 text-[11px] font-sans font-medium">
                {step.icon}
                {step.label}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
