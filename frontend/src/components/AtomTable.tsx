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
          <tr className="bg-stone-900 text-white text-[10px] font-black uppercase tracking-wider font-sans border-b-2 border-stone-900">
            <th className="px-3 py-3 text-center w-10 border-r border-stone-700 whitespace-nowrap">#</th>
            <th className="px-3 py-3 border-r border-stone-700 whitespace-nowrap min-w-[150px]">Nguyên tử</th>
            <th className="px-3 py-3 border-r border-stone-700 text-center whitespace-nowrap w-24">X (Å)</th>
            <th className="px-3 py-3 border-r border-stone-700 text-center whitespace-nowrap w-24">Y (Å)</th>
            <th className="px-3 py-3 border-r border-stone-700 text-center whitespace-nowrap w-24">Z (Å)</th>
            <th className="px-3 py-3 text-center whitespace-nowrap w-20">Hành động</th>
          </tr>
        </thead>
        <tbody>
          {atoms.length === 0 ? (
            <tr>
              <td colSpan={6} className="px-4 py-10 text-center text-stone-400 font-sans text-xs italic">
                Chưa có nguyên tử nào. Hãy thêm hoặc chọn phân tử mẫu bên trái.
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

