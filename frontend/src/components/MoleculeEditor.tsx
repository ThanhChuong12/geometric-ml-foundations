"use client";

import { useMemo } from 'react';
import { AlertTriangle } from 'lucide-react';
import { Atom } from '@/types/molecule';
import { AtomTable } from './AtomTable';

interface MoleculeEditorProps {
  atoms: Atom[];
  onChangeAtoms: (newAtoms: Atom[]) => void;
}

export function MoleculeEditor({ atoms, onChangeAtoms }: MoleculeEditorProps) {
  // Add a new default Hydrogen atom at origin
  const handleAddAtom = () => {
    // Find a coordinate slightly offset if origin is taken
    const lastAtom = atoms[atoms.length - 1];
    const offset = lastAtom ? lastAtom.x + 1.0 : 0.0;
    onChangeAtoms([
      ...atoms,
      { atomicNumber: 1, x: parseFloat(offset.toFixed(2)), y: 0.0, z: 0.0 }
    ]);
  };

  // Update a single atom
  const handleUpdateAtom = (index: number, updatedAtom: Atom) => {
    const updated = [...atoms];
    updated[index] = updatedAtom;
    onChangeAtoms(updated);
  };

  // Delete an atom
  const handleDeleteAtom = (index: number) => {
    onChangeAtoms(atoms.filter((_, i) => i !== index));
  };

  // Clear all atoms
  const handleClear = () => {
    onChangeAtoms([]);
  };

  // Simple validation to check for overlapping atoms (distance < 0.5 Angstrom)
  const validationError = useMemo(() => {
    if (atoms.length < 2) return null;
    for (let i = 0; i < atoms.length; i++) {
      for (let j = i + 1; j < atoms.length; j++) {
        const dx = atoms[i].x - atoms[j].x;
        const dy = atoms[i].y - atoms[j].y;
        const dz = atoms[i].z - atoms[j].z;
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (dist < 0.5) {
          return `Nguyên tử #${i + 1} và Nguyên tử #${j + 1} đang quá gần nhau (${dist.toFixed(2)} Å). Hãy điều chỉnh tọa độ.`;
        }
      }
    }
    return null;
  }, [atoms]);

  return (
    <div className="flex flex-col h-full bg-stone-50 border-2 border-stone-900 p-6 shadow-[8px_8px_0px_0px_rgba(28,25,23,1)]">
      {/* Title */}
      <div className="flex items-center gap-4 mb-6 pb-4 border-b-2 border-stone-900">
        <div className="w-10 h-10 bg-indigo-500 border-2 border-stone-900 shadow-[2px_2px_0px_0px_rgba(28,25,23,1)] flex items-center justify-center text-white font-black font-sans text-xl">
          2
        </div>
        <h2 className="text-xl font-black text-stone-900 font-serif uppercase tracking-tight">
          Biên tập Tọa độ Phân tử
        </h2>
      </div>

      {/* Editor table */}
      <div className="flex-1 min-h-[200px] overflow-y-auto mb-4">
        <AtomTable
          atoms={atoms}
          onUpdateAtom={handleUpdateAtom}
          onDeleteAtom={handleDeleteAtom}
        />
      </div>

      {/* Validation warning */}
      {validationError && (
        <div className="mb-4 bg-rose-100 border-2 border-rose-900 p-3 text-rose-950 text-xs font-bold font-sans shadow-[2px_2px_0px_0px_rgba(225,29,72,0.2)] flex items-start gap-2">
          <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" />
          <span>{validationError}</span>
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex flex-wrap gap-3 mt-auto">
        <button
          type="button"
          onClick={handleAddAtom}
          className="flex-1 bg-cyan-400 hover:bg-cyan-500 border-2 border-stone-900 text-stone-900 font-black py-2.5 px-4 text-xs font-sans uppercase tracking-widest shadow-[3px_3px_0px_0px_rgba(28,25,23,1)] active:shadow-none active:translate-y-0.5 active:translate-x-0.5 transition-all"
        >
          + Thêm nguyên tử
        </button>
        <button
          type="button"
          onClick={handleClear}
          disabled={atoms.length === 0}
          className="bg-white hover:bg-stone-50 disabled:bg-stone-200 disabled:text-stone-400 disabled:cursor-not-allowed border-2 border-stone-900 text-stone-900 font-black py-2.5 px-4 text-xs font-sans uppercase tracking-widest disabled:shadow-none shadow-[3px_3px_0px_0px_rgba(28,25,23,1)] active:shadow-none active:translate-y-0.5 active:translate-x-0.5 transition-all"
        >
          Xóa tất cả
        </button>
      </div>
    </div>
  );
}
