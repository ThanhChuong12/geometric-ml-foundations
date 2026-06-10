"use client";

import dynamic from 'next/dynamic';
import { Atom } from '@/types/molecule';

const MoleculeViewer3DInner = dynamic(
  () => import('./MoleculeViewer3DClient').then(mod => ({ default: mod.MoleculeViewer3DClient })),
  {
    ssr: false,
    loading: () => (
      <div className="w-full h-full border-2 border-stone-900 bg-stone-50 flex items-center justify-center">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 border-4 border-stone-900 border-t-cyan-400 rounded-full animate-spin" />
          <span className="text-xs font-bold text-stone-900 uppercase tracking-widest font-sans">
            Loading 3D Molecule Viewer...
          </span>
        </div>
      </div>
    ),
  }
);

interface MoleculeViewer3DProps {
  atoms: Atom[];
  isLoading?: boolean;
}

export function MoleculeViewer3D(props: MoleculeViewer3DProps) {
  return <MoleculeViewer3DInner {...props} />;
}
