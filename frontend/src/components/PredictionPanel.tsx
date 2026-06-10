"use client";

import { Loader2 } from 'lucide-react';

interface PredictionPanelProps {
  onPredict: () => void;
  onReset: () => void;
  isLoading: boolean;
  isDisabled: boolean;
}

export function PredictionPanel({
  onPredict,
  onReset,
  isLoading,
  isDisabled
}: PredictionPanelProps) {
  return (
    <div className="bg-stone-50 border-2 border-stone-900 p-6 shadow-[8px_8px_0px_0px_rgba(28,25,23,1)] flex flex-col justify-center">
      {/* Title */}
      <div className="flex items-center gap-4 mb-6 pb-4 border-b-2 border-stone-900">
        <div className="w-10 h-10 bg-cyan-400 border-2 border-stone-900 shadow-[2px_2px_0px_0px_rgba(28,25,23,1)] flex items-center justify-center text-stone-900 font-black font-sans text-xl">
          3
        </div>
        <h2 className="text-xl font-black text-stone-900 font-serif uppercase tracking-tight">
          Bộ điều khiển Inference
        </h2>
      </div>

      <div className="flex flex-col gap-4">
        {/* Predict button */}
        <button
          type="button"
          onClick={onPredict}
          disabled={isLoading || isDisabled}
          className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-stone-300 disabled:border-stone-300 disabled:text-stone-500 disabled:cursor-not-allowed text-white font-black py-4 px-6 border-2 border-stone-900 disabled:shadow-none shadow-[4px_4px_0px_0px_rgba(28,25,23,1)] active:shadow-none active:translate-y-1 active:translate-x-1 flex items-center justify-center gap-3 text-base font-sans uppercase tracking-widest transition-all"
        >
          {isLoading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Đang tính toán năng lượng...
            </>
          ) : (
            'Dự đoán Năng lượng'
          )}
        </button>

        {/* Reset button */}
        <button
          type="button"
          onClick={onReset}
          disabled={isLoading}
          className="w-full bg-white hover:bg-stone-50 disabled:bg-stone-100 disabled:text-stone-400 disabled:cursor-not-allowed text-stone-900 font-black py-3 px-6 border-2 border-stone-900 disabled:shadow-none shadow-[4px_4px_0px_0px_rgba(28,25,23,1)] active:shadow-none active:translate-y-1 active:translate-x-1 flex items-center justify-center gap-3 text-sm font-sans uppercase tracking-widest transition-all"
        >
          Đặt lại Phân tử
        </button>
      </div>
    </div>
  );
}
