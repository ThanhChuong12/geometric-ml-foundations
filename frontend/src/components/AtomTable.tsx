"use client";

import { Atom } from '@/types/molecule';
import { AtomRow } from './AtomRow';

interface AtomTableProps {
  atoms: Atom[];
  onUpdateAtom: (index: number, updatedAtom: Atom) => void;
  onDeleteAtom: (index: number) => void;
}

export function AtomTable({ atoms, onUpdateAtom, onDeleteAtom }: AtomTableProps) {
  return (
    <div className="w-full overflow-x-auto border-2 border-stone-900 bg-white shadow-[4px_4px_0px_0px_rgba(28,25,23,1)]">
      <table className="w-full text-left border-collapse">
        <thead>
          <tr className="bg-stone-900 text-white text-xs font-black uppercase tracking-wider font-sans border-b-2 border-stone-900">
            <th className="px-3 py-3 text-center w-12 border-r border-stone-800">#</th>
            <th className="px-3 py-3 border-r border-stone-800">Nguyên tử</th>
            <th className="px-3 py-3 border-r border-stone-800">X (Å)</th>
            <th className="px-3 py-3 border-r border-stone-800">Y (Å)</th>
            <th className="px-3 py-3 border-r border-stone-800">Z (Å)</th>
            <th className="px-3 py-3 text-center">Hành động</th>
          </tr>
        </thead>
        <tbody>
          {atoms.length === 0 ? (
            <tr>
              <td colSpan={6} className="px-4 py-8 text-center text-stone-500 font-sans text-xs italic">
                Chưa có nguyên tử nào. Hãy thêm nguyên tử hoặc chọn phân tử mẫu.
              </td>
            </tr>
          ) : (
            atoms.map((atom, idx) => (
              <AtomRow
                key={`${idx}-${atom.atomicNumber}`}
                atom={atom}
                index={idx}
                onUpdate={onUpdateAtom}
                onDelete={onDeleteAtom}
              />
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}
