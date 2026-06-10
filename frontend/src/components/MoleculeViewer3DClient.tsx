"use client";

import { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stage } from '@react-three/drei';
import * as THREE from 'three';
import { Atom } from '@/types/molecule';

interface MoleculeViewer3DClientProps {
  atoms: Atom[];
  isLoading?: boolean;
}

// Map atomic numbers to chemical symbols, colors, and radii
const getAtomStyle = (atomicNumber: number) => {
  switch (atomicNumber) {
    case 1: // Hydrogen
      return { symbol: 'H', color: '#ffffff', radius: 0.3 };
    case 6: // Carbon
      return { symbol: 'C', color: '#374151', radius: 0.5 }; // dark gray
    case 7: // Nitrogen
      return { symbol: 'N', color: '#3b82f6', radius: 0.45 }; // blue
    case 8: // Oxygen
      return { symbol: 'O', color: '#ef4444', radius: 0.45 }; // red
    case 9: // Fluorine
      return { symbol: 'F', color: '#10b981', radius: 0.4 }; // green
    default:
      return { symbol: '?', color: '#a855f7', radius: 0.4 }; // purple
  }
};

// Calculate bonds between atoms based on distance
interface Bond {
  id: string;
  p1: THREE.Vector3;
  p2: THREE.Vector3;
  length: number;
  mid: THREE.Vector3;
  quaternion: THREE.Quaternion;
}

function computeBonds(centeredAtoms: Atom[]): Bond[] {
  const bonds: Bond[] = [];
  const n = centeredAtoms.length;

  for (let i = 0; i < n; i++) {
    const a1 = centeredAtoms[i];
    const p1 = new THREE.Vector3(a1.x, a1.y, a1.z);
    
    // Approximate covalent radius
    const r1 = getAtomStyle(a1.atomicNumber).radius;

    for (let j = i + 1; j < n; j++) {
      const a2 = centeredAtoms[j];
      const p2 = new THREE.Vector3(a2.x, a2.y, a2.z);
      const r2 = getAtomStyle(a2.atomicNumber).radius;

      const dist = p1.distanceTo(p2);
      
      // A chemical bond exists if distance is within logical boundaries
      // min distance 0.5 (prevent overlap bugs), max distance (covalent sum * 1.3)
      const maxBondDist = (r1 + r2) * 2.2; 
      if (dist >= 0.6 && dist <= maxBondDist) {
        const v = new THREE.Vector3().subVectors(p2, p1);
        const length = v.length();
        const mid = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);
        v.normalize();
        const quaternion = new THREE.Quaternion().setFromUnitVectors(
          new THREE.Vector3(0, 1, 0),
          v
        );

        bonds.push({
          id: `${i}-${j}`,
          p1,
          p2,
          length,
          mid,
          quaternion
        });
      }
    }
  }
  return bonds;
}

export function MoleculeViewer3DClient({ atoms, isLoading = false }: MoleculeViewer3DClientProps) {
  // Center the atoms coordinates programmatically
  const { centeredAtoms, cameraDistance } = useMemo(() => {
    if (atoms.length === 0) {
      return { centeredAtoms: [], cameraDistance: 5 };
    }
    
    const xs = atoms.map(a => a.x);
    const ys = atoms.map(a => a.y);
    const zs = atoms.map(a => a.z);
    
    const midX = (Math.max(...xs) + Math.min(...xs)) / 2;
    const midY = (Math.max(...ys) + Math.min(...ys)) / 2;
    const midZ = (Math.max(...zs) + Math.min(...zs)) / 2;
    
    const centered = atoms.map(a => ({
      ...a,
      x: a.x - midX,
      y: a.y - midY,
      z: a.z - midZ
    }));

    // Find the furthest atom from the center to calculate camera distance
    const maxDist = Math.max(...centered.map(a => Math.sqrt(a.x ** 2 + a.y ** 2 + a.z ** 2)));
    // Scaled distance to fit molecule within view frustum
    const cameraDist = Math.max(4, maxDist * 2.5);

    return { centeredAtoms: centered, cameraDistance: cameraDist };
  }, [atoms]);

  // Compute bonds between centered atoms
  const bonds = useMemo(() => computeBonds(centeredAtoms), [centeredAtoms]);

  return (
    <div className="relative w-full h-full border-2 border-stone-900 bg-stone-50 shadow-[4px_4px_0px_0px_rgba(28,25,23,1)] overflow-hidden">
      {atoms.length === 0 ? (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 pointer-events-none">
          <div className="text-7xl font-serif font-light opacity-20 text-stone-900">3D</div>
          <span className="text-xs font-bold text-stone-500 uppercase tracking-widest font-sans">
            Chọn hoặc nhập phân tử để xem
          </span>
        </div>
      ) : (
        <Canvas
          camera={{ position: [0, 0, cameraDistance], fov: 45 }}
          style={{ width: '100%', height: '100%' }}
        >
          {/* Ambient & directional lighting for realistic shadows and shapes */}
          <ambientLight intensity={1.5} />
          <directionalLight position={[10, 10, 10]} intensity={1.5} castShadow />
          <directionalLight position={[-10, -10, -10]} intensity={0.5} />
          
          <group>
            {/* Render bonds as cylinders */}
            {bonds.map(bond => (
              <mesh key={bond.id} position={bond.mid} quaternion={bond.quaternion}>
                <cylinderGeometry args={[0.07, 0.07, bond.length, 8]} />
                <meshStandardMaterial color="#78716c" roughness={0.4} metalness={0.1} />
              </mesh>
            ))}

            {/* Render atoms as spheres */}
            {centeredAtoms.map((atom, idx) => {
              const { color, radius } = getAtomStyle(atom.atomicNumber);
              return (
                <mesh key={idx} position={[atom.x, atom.y, atom.z]}>
                  <sphereGeometry args={[radius, 32, 32]} />
                  <meshStandardMaterial color={color} roughness={0.2} metalness={0.1} />
                </mesh>
              );
            })}
          </group>

          <OrbitControls makeDefault enablePan={true} enableZoom={true} />
        </Canvas>
      )}

      {/* Loading overlay */}
      {isLoading && (
        <div className="absolute inset-0 bg-stone-50/80 flex flex-col items-center justify-center gap-3">
          <div className="w-8 h-8 border-4 border-stone-900 border-t-cyan-500 rounded-full animate-spin" />
          <span className="text-xs font-bold text-stone-900 uppercase tracking-widest font-sans">
            Đang phân tích phân tử...
          </span>
        </div>
      )}

      {/* Atom list stats badge */}
      {atoms.length > 0 && (
        <div className="absolute bottom-2 left-2 flex gap-2">
          <span className="text-[10px] font-bold font-sans uppercase tracking-widest bg-stone-900 text-white px-2 py-1">
            {atoms.length} atoms
          </span>
          <span className="text-[10px] font-bold font-sans uppercase tracking-widest bg-stone-700 text-white px-2 py-1">
            {bonds.length} bonds
          </span>
        </div>
      )}

      {/* Hint */}
      {atoms.length > 0 && (
        <div className="absolute top-2 right-2">
          <span className="text-[10px] font-bold font-sans uppercase tracking-widest bg-white border border-stone-300 text-stone-500 px-2 py-1">
            Drag to Rotate · Scroll to Zoom
          </span>
        </div>
      )}
    </div>
  );
}
