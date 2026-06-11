"use client";

type TabType = 'part1' | 'part2' | 'part3';

// Part 1 — SO(2): two nested squares rotating in opposite directions to illustrate planar rotation symmetry
function SO2Decorator() {
  return (
    <div
      className="relative w-14 h-14 flex-shrink-0 opacity-20"
      aria-hidden="true"
    >
      {/* Outer square — clockwise rotation */}
      <div
        className="absolute inset-0 border-2 border-stone-900"
        style={{ animation: 'rotateSO2 5s linear infinite' }}
      />
      {/* Inner square — counter-clockwise rotation */}
      <div
        className="absolute inset-3 border-2 border-stone-900"
        style={{ animation: 'rotateSO2Reverse 3s linear infinite' }}
      />
      {/* Center dot */}
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="w-1.5 h-1.5 bg-stone-900 rounded-full" />
      </div>
    </div>
  );
}

// Part 2 — PointNet: floating dots at staggered positions and speeds to simulate a sparse 3D point cloud
const DOTS = [
  { x: 18, y: 18, delay: 0,    dur: 1.9 },
  { x: 50, y: 10, delay: 0.3,  dur: 2.1 },
  { x: 82, y: 22, delay: 0.6,  dur: 1.7 },
  { x: 12, y: 50, delay: 0.9,  dur: 2.3 },
  { x: 48, y: 48, delay: 0.15, dur: 1.8 },
  { x: 80, y: 52, delay: 0.5,  dur: 2.0 },
  { x: 22, y: 78, delay: 0.75, dur: 2.2 },
  { x: 55, y: 80, delay: 0.1,  dur: 1.6 },
  { x: 82, y: 78, delay: 0.4,  dur: 2.4 },
];

function PointCloudDecorator() {
  return (
    <div
      className="relative w-14 h-14 flex-shrink-0 opacity-20"
      aria-hidden="true"
    >
      {DOTS.map((d, i) => (
        <div
          key={i}
          className="absolute w-1.5 h-1.5 bg-stone-900 rounded-full"
          style={{
            left: `${d.x}%`,
            top: `${d.y}%`,
            animation: `floatParticle ${d.dur}s ease-in-out infinite`,
            animationDelay: `${d.delay}s`,
          }}
        />
      ))}
    </div>
  );
}

// Part 3 — NequIP: two electrons orbiting an atomic nucleus at different speeds
function AtomDecorator() {
  return (
    <div
      className="relative w-14 h-14 flex-shrink-0 opacity-25 flex items-center justify-center"
      aria-hidden="true"
    >
      {/* Orbit ring 1 */}
      <div className="absolute w-12 h-12 border border-stone-600 rounded-full" />
      {/* Orbit ring 2 — tilted 60 degrees to suggest a second orbital plane */}
      <div
        className="absolute w-12 h-5 border border-stone-600 rounded-full"
        style={{ transform: 'rotate(60deg)' }}
      />
      {/* Nucleus */}
      <div className="w-2.5 h-2.5 bg-stone-900 rounded-full z-10" />
      {/* Electron 1 */}
      <div
        className="absolute w-2 h-2 bg-cyan-500 rounded-full"
        style={{ animation: 'electronOrbit 2s linear infinite' }}
      />
      {/* Electron 2 */}
      <div
        className="absolute w-1.5 h-1.5 bg-indigo-500 rounded-full"
        style={{ animation: 'electronOrbitFast 1.4s linear infinite' }}
      />
    </div>
  );
}

export function HeaderDecorator({ tab }: { tab: TabType }) {
  if (tab === 'part1') return <SO2Decorator />;
  if (tab === 'part2') return <PointCloudDecorator />;
  return <AtomDecorator />;
}
