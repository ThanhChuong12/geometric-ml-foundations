"use client";

import { Droplets, Flame, FlaskConical, Wind, Circle, Zap } from 'lucide-react';
import { Atom } from '@/types/molecule';

interface ExampleMoleculesProps {
  selectedMoleculeName: string;
  onSelectMolecule: (name: string, atoms: Atom[]) => void;
}

interface MoleculePreset {
  name: string;
  chemicalFormula: string;
  description: string;
  atoms: Atom[];
}

// Maps each preset name to a representative lucide-react icon node
const PRESET_ICONS: Record<string, React.ReactNode> = {
  'Nước':           <Droplets     className="w-6 h-6" />,
  'Methane':        <Flame        className="w-6 h-6" />,
  'Ammonia':        <FlaskConical className="w-6 h-6" />,
  'Carbon Dioxide': <Wind         className="w-6 h-6" />,
  'Ethanol':        <Droplets     className="w-6 h-6" />,
  'Formaldehyde':   <FlaskConical className="w-6 h-6" />,
  'Fluoromethane':  <Zap          className="w-6 h-6" />,
};

const PRESETS: MoleculePreset[] = [
  {
    name: 'Nước',
    chemicalFormula: 'H₂O',
    description: 'Phân tử phân cực gấp khúc với góc liên kết ~104.5 độ.',
    atoms: [
      { atomicNumber: 8, x: 0.0, y: 0.0, z: 0.1163 },
      { atomicNumber: 1, x: 0.0, y: 0.7583, z: -0.4654 },
      { atomicNumber: 1, x: 0.0, y: -0.7583, z: -0.4654 }
    ]
  },
  {
    name: 'Methane',
    chemicalFormula: 'CH₄',
    description: 'Phân tử hữu cơ hydrocacbon đơn giản nhất dạng tứ diện đều.',
    atoms: [
      { atomicNumber: 6, x: 0.0, y: 0.0, z: 0.0 },
      { atomicNumber: 1, x: 0.63, y: 0.63, z: 0.63 },
      { atomicNumber: 1, x: -0.63, y: -0.63, z: 0.63 },
      { atomicNumber: 1, x: -0.63, y: 0.63, z: -0.63 },
      { atomicNumber: 1, x: 0.63, y: -0.63, z: -0.63 }
    ]
  },
  {
    name: 'Ammonia',
    chemicalFormula: 'NH₃',
    description: 'Phân tử dạng tháp tam giác với cặp electron tự do phân cực.',
    atoms: [
      { atomicNumber: 7, x: 0.0, y: 0.0, z: 0.116 },
      { atomicNumber: 1, x: 0.0, y: 0.94, z: -0.27 },
      { atomicNumber: 1, x: 0.81, y: -0.47, z: -0.27 },
      { atomicNumber: 1, x: -0.81, y: -0.47, z: -0.27 }
    ]
  },
  {
    name: 'Carbon Dioxide',
    chemicalFormula: 'CO₂',
    description: 'Phân tử tuyến tính đối xứng, không phân cực.',
    atoms: [
      { atomicNumber: 6, x: 0.0, y: 0.0, z: 0.0 },
      { atomicNumber: 8, x: 0.0, y: 0.0, z: 1.16 },
      { atomicNumber: 8, x: 0.0, y: 0.0, z: -1.16 }
    ]
  },
  {
    name: 'Ethanol',
    chemicalFormula: 'C₂H₅OH',
    description: 'Rượu etylic — phân tử hữu cơ 9 nguyên tử, nhóm hydroxyl -OH phân cực.',
    atoms: [
      { atomicNumber: 6, x:  0.000, y:  0.000, z:  0.000 },
      { atomicNumber: 6, x:  1.520, y:  0.000, z:  0.000 },
      { atomicNumber: 8, x:  2.080, y:  1.185, z:  0.000 },
      { atomicNumber: 1, x:  2.000, y:  1.736, z:  0.773 },
      { atomicNumber: 1, x: -0.390, y:  1.025, z:  0.000 },
      { atomicNumber: 1, x: -0.390, y: -0.513, z:  0.889 },
      { atomicNumber: 1, x: -0.390, y: -0.513, z: -0.889 },
      { atomicNumber: 1, x:  1.910, y: -0.513, z:  0.889 },
      { atomicNumber: 1, x:  1.910, y: -0.513, z: -0.889 }
    ]
  },
  {
    name: 'Formaldehyde',
    chemicalFormula: 'CH₂O',
    description: 'Aldehyde đơn giản nhất — phân tử phẳng với liên kết đôi C=O.',
    atoms: [
      { atomicNumber: 6, x:  0.000, y: 0.0, z:  0.000 },
      { atomicNumber: 8, x:  0.000, y: 0.0, z:  1.208 },
      { atomicNumber: 1, x:  0.924, y: 0.0, z: -0.541 },
      { atomicNumber: 1, x: -0.924, y: 0.0, z: -0.541 }
    ]
  },
  {
    name: 'Fluoromethane',
    chemicalFormula: 'CH₃F',
    description: 'Methane thay H bằng F — kiểm tra nguyên tố Fluorine (Z=9) trong model.',
    atoms: [
      { atomicNumber: 6, x:  0.000, y:  0.000, z:  0.000 },
      { atomicNumber: 9, x:  0.000, y:  0.000, z:  1.382 },
      { atomicNumber: 1, x:  1.027, y:  0.000, z: -0.371 },
      { atomicNumber: 1, x: -0.513, y:  0.889, z: -0.371 },
      { atomicNumber: 1, x: -0.513, y: -0.889, z: -0.371 }
    ]
  }
];

export function ExampleMolecules({
  selectedMoleculeName,
  onSelectMolecule
}: ExampleMoleculesProps) {
  return (
    <div className="bg-stone-50 border-2 border-stone-900 p-6 shadow-[8px_8px_0px_0px_rgba(28,25,23,1)]">
      {/* Title */}
      <div className="flex items-center gap-4 mb-5 pb-4 border-b-2 border-stone-900">
        <div className="w-10 h-10 bg-amber-400 border-2 border-stone-900 shadow-[2px_2px_0px_0px_rgba(28,25,23,1)] flex items-center justify-center text-stone-900 font-black font-sans text-xl">
          *
        </div>
        <h2 className="text-xl font-black text-stone-900 font-serif uppercase tracking-tight">
          Chọn Cấu trúc Mẫu
        </h2>
      </div>

      {/* Grid selector */}
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-4 gap-3">
        {PRESETS.map((preset) => {
          const isSelected = selectedMoleculeName === preset.name;
          return (
            <button
              key={preset.name}
              onClick={() => onSelectMolecule(preset.name, preset.atoms)}
              className={`flex flex-col text-left p-4 border-2 border-stone-900 font-sans transition-all h-full ${
                isSelected
                  ? 'bg-cyan-300 shadow-[4px_4px_0px_0px_rgba(28,25,23,1)] translate-y-0 translate-x-0'
                  : 'bg-white shadow-none hover:bg-stone-50 translate-y-0.5 translate-x-0.5'
              }`}
            >
              <div className="flex items-center gap-2 mb-2">
                <span className="text-stone-700">
                  {PRESET_ICONS[preset.name] ?? <Circle className="w-6 h-6" />}
                </span>
                <span className="font-black text-stone-900 text-sm uppercase tracking-wide font-sans">
                  {preset.name}
                </span>
                <span className="ml-auto text-xs font-mono font-bold bg-stone-900 text-white px-2 py-0.5">
                  {preset.chemicalFormula}
                </span>
              </div>
              <p className="text-xs text-stone-600 font-medium leading-relaxed font-sans">
                {preset.description}
              </p>
              <div className="mt-auto pt-3 text-[10px] font-bold text-stone-500 uppercase tracking-widest font-sans">
                {preset.atoms.length} Nguyên tử
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
