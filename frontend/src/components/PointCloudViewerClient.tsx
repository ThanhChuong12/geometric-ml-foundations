"use client";
// Three.js implementation — chỉ chạy phía client (browser only)
// Import qua dynamic() trong PointCloudViewer.tsx với ssr: false

import { useRef, useEffect, useCallback } from 'react';
import * as THREE from 'three';

interface PointCloudViewerProps {
  points: number[][];
  criticalPoints: number[][];
  showCritical: boolean;
  isLoading?: boolean;
  /** Góc xoay X (độ) từ slider bên ngoài — đồng bộ 2 chiều */
  rotationX?: number;
  /** Góc xoay Y (độ) từ slider bên ngoài — đồng bộ 2 chiều */
  rotationY?: number;
  /** Góc xoay Z (độ) từ slider bên ngoài — chỉ 1 chiều slider→viewer */
  rotationZ?: number;
  /** Callback khi drag thay đổi góc X/Y — cập nhật slider bên ngoài */
  onRotationChange?: (x: number, y: number) => void;
  /** Callback khi scroll thay đổi góc Z — cập nhật slider bên ngoài */
  onRotationChangeZ?: (z: number) => void;
}

export function PointCloudViewer({
  points,
  criticalPoints,
  showCritical,
  isLoading = false,
  rotationX,
  rotationY,
  rotationZ,
  onRotationChange,
  onRotationChangeZ,
}: PointCloudViewerProps) {
  const mountRef    = useRef<HTMLDivElement>(null);
  const sceneRef    = useRef<THREE.Scene | null>(null);
  const cameraRef   = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const frameRef    = useRef<number>(0);
  const rotationRef = useRef({ x: 0.3, y: 0.5, z: 0 });
  const isDragging  = useRef(false);
  const prevMouse   = useRef({ x: 0, y: 0 });
  // Giữ ref đến callback để tránh stale closure trong animation loop
  const onRotationChangeRef  = useRef(onRotationChange);
  const onRotationChangeZRef = useRef(onRotationChangeZ);
  useEffect(() => { onRotationChangeRef.current  = onRotationChange;  }, [onRotationChange]);
  useEffect(() => { onRotationChangeZRef.current = onRotationChangeZ; }, [onRotationChangeZ]);

  // Đồng bộ từ slider → viewer (X/Y chỉ khi không đang drag; Z luôn cập nhật)
  useEffect(() => {
    if (!isDragging.current) {
      if (rotationX !== undefined) rotationRef.current.x = (rotationX * Math.PI) / 180;
      if (rotationY !== undefined) rotationRef.current.y = (rotationY * Math.PI) / 180;
    }
    if (rotationZ !== undefined) rotationRef.current.z = (rotationZ * Math.PI) / 180;
  }, [rotationX, rotationY, rotationZ]);

  // ── Init scene once ──
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const w = mount.clientWidth  || 400;
    const h = mount.clientHeight || 340;

    const scene    = new THREE.Scene();
    scene.background = new THREE.Color(0xfafaf9); // stone-50
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(50, w / h, 0.01, 100);
    camera.position.set(0, 0, 3.5);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(w, h);
    renderer.setPixelRatio(window.devicePixelRatio);
    mount.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Animate + apply rotation each frame
    const animate = () => {
      frameRef.current = requestAnimationFrame(animate);
      scene.children.forEach(child => {
        if (child instanceof THREE.Points) {
          child.rotation.x = rotationRef.current.x;
          child.rotation.y = rotationRef.current.y;
          child.rotation.z = rotationRef.current.z;
        }
      });
      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      const W = mount.clientWidth;
      const H = mount.clientHeight;
      if (!W || !H) return;
      camera.aspect = W / H;
      camera.updateProjectionMatrix();
      renderer.setSize(W, H);
    };

    // Scroll wheel → xoay trục Z
    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();
      rotationRef.current.z += e.deltaY * 0.005;
      if (onRotationChangeZRef.current) {
        let d = ((rotationRef.current.z * 180) / Math.PI) % 360;
        if (d < 0) d += 360;
        onRotationChangeZRef.current(Math.round(d / 5) * 5);
      }
    };
    mount.addEventListener('wheel', handleWheel, { passive: false });

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      mount.removeEventListener('wheel', handleWheel);
      cancelAnimationFrame(frameRef.current);
      renderer.dispose();
      if (mount.contains(renderer.domElement)) mount.removeChild(renderer.domElement);
    };
  }, []);

  // ── Rebuild point cloud geometry when data/mode changes ──
  const buildGeometry = useCallback((pts: number[][], color: number, size: number, opacity = 1) => {
    const positions = new Float32Array(pts.length * 3);
    pts.forEach(([x, y, z], i) => {
      positions[i * 3]     = x ?? 0;
      positions[i * 3 + 1] = y ?? 0;
      positions[i * 3 + 2] = z ?? 0;
    });
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const mat = new THREE.PointsMaterial({ color, size, sizeAttenuation: true, transparent: opacity < 1, opacity });
    return new THREE.Points(geo, mat);
  }, []);

  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;

    // Clear old point clouds
    const toRemove = scene.children.filter(c => c instanceof THREE.Points);
    toRemove.forEach(c => {
      scene.remove(c);
      (c as THREE.Points).geometry.dispose();
      ((c as THREE.Points).material as THREE.Material).dispose();
    });

    if (points.length === 0) return;

    if (!showCritical) {
      // Normal view: gradient color by Y height
      const geo = new THREE.BufferGeometry();
      const positions = new Float32Array(points.length * 3);
      const colors    = new Float32Array(points.length * 3);
      const colorLow  = new THREE.Color(0x7dd3fc); // sky-300
      const colorHigh = new THREE.Color(0x1d4ed8); // blue-700
      const yVals = points.map(p => p[1] ?? 0);
      const yMin  = Math.min(...yVals);
      const yRange = (Math.max(...yVals) - yMin) || 1;

      points.forEach(([x, y, z], i) => {
        positions[i * 3]     = x ?? 0;
        positions[i * 3 + 1] = y ?? 0;
        positions[i * 3 + 2] = z ?? 0;
        const c = colorLow.clone().lerp(colorHigh, ((y ?? 0) - yMin) / yRange);
        colors[i * 3] = c.r; colors[i * 3 + 1] = c.g; colors[i * 3 + 2] = c.b;
      });

      geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geo.setAttribute('color',    new THREE.BufferAttribute(colors, 3));
      const mat = new THREE.PointsMaterial({ vertexColors: true, size: 0.025, sizeAttenuation: true });
      scene.add(new THREE.Points(geo, mat));
    } else {
      // Critical view: faded background + bright red critical points
      scene.add(buildGeometry(points, 0xd6d3d1, 0.018, 0.3)); // stone-300, faded
      if (criticalPoints.length > 0) {
        scene.add(buildGeometry(criticalPoints, 0xef4444, 0.05, 1)); // red-500
      }
    }
  }, [points, criticalPoints, showCritical, buildGeometry]);

  // ── Mouse drag handlers ──
  const onMouseDown = (e: React.MouseEvent) => {
    isDragging.current = true;
    prevMouse.current  = { x: e.clientX, y: e.clientY };
  };
  const onMouseMove = (e: React.MouseEvent) => {
    if (!isDragging.current) return;
    rotationRef.current.y += (e.clientX - prevMouse.current.x) * 0.01;
    rotationRef.current.x += (e.clientY - prevMouse.current.y) * 0.01;
    prevMouse.current = { x: e.clientX, y: e.clientY };
    // Gửi góc (độ) ra ngoài để cập nhật slider
    if (onRotationChangeRef.current) {
      const toDeg = (rad: number) => {
        let d = ((rad * 180) / Math.PI) % 360;
        if (d < 0) d += 360;
        return Math.round(d / 5) * 5; // snap to step=5
      };
      onRotationChangeRef.current(
        toDeg(rotationRef.current.x),
        toDeg(rotationRef.current.y),
      );
    }
  };
  const onMouseUp = () => { isDragging.current = false; };

  return (
    <div className="relative w-full h-full border-2 border-stone-900 bg-stone-50 shadow-[4px_4px_0px_0px_rgba(28,25,23,1)] overflow-hidden">
      <div
        ref={mountRef}
        className="w-full h-full cursor-grab active:cursor-grabbing"
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
      />

      {/* Loading overlay */}
      {isLoading && (
        <div className="absolute inset-0 bg-stone-50/80 flex flex-col items-center justify-center gap-3">
          <div className="w-8 h-8 border-4 border-stone-900 border-t-blue-500 rounded-full animate-spin" />
          <span className="text-xs font-bold text-stone-900 uppercase tracking-widest font-sans">Processing...</span>
        </div>
      )}

      {/* Empty state */}
      {!isLoading && points.length === 0 && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 pointer-events-none">
          <div className="text-7xl font-serif font-light opacity-20 text-stone-900">3D</div>
          <span className="text-xs font-bold text-stone-500 uppercase tracking-widest font-sans">Chon object de xem</span>
        </div>
      )}

      {/* Stats badge */}
      {points.length > 0 && (
        <div className="absolute bottom-2 left-2 flex gap-2">
          <span className="text-[10px] font-bold font-sans uppercase tracking-widest bg-stone-900 text-white px-2 py-1">
            {points.length} pts
          </span>
          {showCritical && criticalPoints.length > 0 && (
            <span className="text-[10px] font-bold font-sans uppercase tracking-widest bg-red-500 text-white px-2 py-1">
              {criticalPoints.length} critical
            </span>
          )}
        </div>
      )}

      {/* Hint */}
      {points.length > 0 && (
        <div className="absolute top-2 right-2">
          <span className="text-[10px] font-bold font-sans uppercase tracking-widest bg-white border border-stone-300 text-stone-500 px-2 py-1">
            Drag XY · Scroll Z
          </span>
        </div>
      )}
    </div>
  );
}
