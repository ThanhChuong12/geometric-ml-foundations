"use client";
import dynamic from 'next/dynamic';

// Three.js dùng browser APIs (window, WebGL) → bắt buộc disable SSR
const PointCloudViewerInner = dynamic(
  () => import('./PointCloudViewerClient').then(mod => ({ default: mod.PointCloudViewer })),
  {
    ssr: false,
    loading: () => (
      <div className="w-full h-full border-2 border-stone-900 bg-stone-50 flex items-center justify-center">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 border-4 border-stone-900 border-t-violet-400 rounded-full animate-spin" />
          <span className="text-xs font-bold text-stone-900 uppercase tracking-widest font-sans">
            Loading 3D...
          </span>
        </div>
      </div>
    ),
  }
);

interface PointCloudViewerProps {
  points: number[][];
  criticalPoints: number[][];
  showCritical: boolean;
  isLoading?: boolean;
}

// Re-export với tên gốc để page.tsx import không cần đổi
export function PointCloudViewer(props: PointCloudViewerProps) {
  return <PointCloudViewerInner {...props} />;
}
