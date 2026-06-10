"use client";

import { Atom } from '@/types/molecule';

interface AtomRowProps {
  atom: Atom;
  index: number;
  onUpdate: (index: number, updatedAtom: Atom) => void;
  onDelete: (index: number) => void;
}

const COMMON_ELEMENTS = [
  { value: 1, label: 'H (Hydrogen)' },
  { value: 6, label: 'C (Carbon)' },
  { value: 7, label: 'N (Nitrogen)' },
  { value: 8, label: 'O (Oxygen)' },
  { value: 9, label: 'F (Fluorine)' }
];

export function AtomRow({ atom, index, onUpdate, onDelete }: AtomRowProps) {
  const handleElementChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const val = parseInt(e.target.value, 10);
    onUpdate(index, { ...atom, atomicNumber: val });
  };

  const handleCoordinateChange = (axis: 'x' | 'y' | 'z', value: string) => {
    const num = parseFloat(value);
    onUpdate(index, { 
      ...atom, 
      [axis]: isNaN(num) ? 0 : num 
    });
  };

  return (
    <tr className="border-b-2 border-stone-900 font-sans hover:bg-stone-50">
      {/* Index */}
      <td className="px-3 py-2 text-center font-bold text-stone-600 border-r-2 border-stone-900 text-xs">
        {index + 1}
      </td>

      {/* Element Selector */}
      <td className="px-3 py-2 border-r-2 border-stone-900">
        <select
          value={atom.atomicNumber}
          onChange={handleElementChange}
          className="w-full bg-white border-2 border-stone-900 px-2 py-1 text-xs font-bold font-sans rounded-none shadow-[2px_2px_0px_0px_rgba(28,25,23,1)] focus:outline-none focus:bg-stone-50"
        >
          {COMMON_ELEMENTS.map(el => (
            <option key={el.value} value={el.value}>
              {el.label}
            </option>
          ))}
          {!COMMON_ELEMENTS.some(el => el.value === atom.atomicNumber) && (
            <option value={atom.atomicNumber}>
              Z = {atom.atomicNumber}
            </option>
          )}
        </select>
      </td>

      {/* X Coordinate */}
      <td className="px-3 py-2 border-r-2 border-stone-900">
        <input
          type="number"
          step="0.01"
          defaultValue={atom.x}
          key={`x-${index}-${atom.x}`}
          onBlur={(e) => handleCoordinateChange('x', e.target.value)}
          className="w-full bg-white border-2 border-stone-900 px-2 py-1 text-xs font-mono font-bold rounded-none shadow-[2px_2px_0px_0px_rgba(28,25,23,1)] focus:outline-none"
        />
      </td>

      {/* Y Coordinate */}
      <td className="px-3 py-2 border-r-2 border-stone-900">
        <input
          type="number"
          step="0.01"
          defaultValue={atom.y}
          key={`y-${index}-${atom.y}`}
          onBlur={(e) => handleCoordinateChange('y', e.target.value)}
          className="w-full bg-white border-2 border-stone-900 px-2 py-1 text-xs font-mono font-bold rounded-none shadow-[2px_2px_0px_0px_rgba(28,25,23,1)] focus:outline-none"
        />
      </td>

      {/* Z Coordinate */}
      <td className="px-3 py-2 border-r-2 border-stone-900">
        <input
          type="number"
          step="0.01"
          defaultValue={atom.z}
          key={`z-${index}-${atom.z}`}
          onBlur={(e) => handleCoordinateChange('z', e.target.value)}
          className="w-full bg-white border-2 border-stone-900 px-2 py-1 text-xs font-mono font-bold rounded-none shadow-[2px_2px_0px_0px_rgba(28,25,23,1)] focus:outline-none"
        />
      </td>

      {/* Actions */}
      <td className="px-3 py-2 text-center">
        <button
          type="button"
          onClick={() => onDelete(index)}
          className="bg-rose-400 hover:bg-rose-500 border-2 border-stone-900 text-stone-900 font-bold px-3 py-1 text-[10px] uppercase tracking-widest shadow-[2px_2px_0px_0px_rgba(28,25,23,1)] active:shadow-none active:translate-y-0.5 active:translate-x-0.5 transition-all"
        >
          Xóa
        </button>
      </td>
    </tr>
  );
}
