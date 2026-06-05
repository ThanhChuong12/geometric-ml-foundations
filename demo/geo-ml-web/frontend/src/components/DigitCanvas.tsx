"use client";
import { useRef, useEffect, useState } from 'react';

interface DigitCanvasProps {
  onImageChange: (imageData: string) => void;
  rotation: number;
}

export function DigitCanvas({ onImageChange, rotation }: DigitCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const displayCanvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }, []);

  useEffect(() => {
    const sourceCanvas = canvasRef.current;
    const displayCanvas = displayCanvasRef.current;
    if (!sourceCanvas || !displayCanvas) return;

    const ctx = displayCanvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, displayCanvas.width, displayCanvas.height);

    ctx.save();
    ctx.translate(displayCanvas.width / 2, displayCanvas.height / 2);
    ctx.rotate((rotation * Math.PI) / 180);
    ctx.drawImage(
      sourceCanvas,
      -sourceCanvas.width / 2,
      -sourceCanvas.height / 2
    );
    ctx.restore();

    const imageData = displayCanvas.toDataURL('image/png');
    onImageChange(imageData);
  }, [rotation, onImageChange]);

  const lastPosRef = useRef<{ x: number; y: number } | null>(null);

  const startDrawing = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDrawing(true);
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    lastPosRef.current = { x, y };
    
    // Vẽ điểm đầu tiên
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.fillStyle = '#1c1917';
      ctx.beginPath();
      ctx.arc(x, y, 8, 0, Math.PI * 2);
      ctx.fill();
    }
  };

  const stopDrawing = () => {
    setIsDrawing(false);
    lastPosRef.current = null;
  };

  const draw = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (lastPosRef.current) {
      ctx.strokeStyle = '#1c1917';
      ctx.lineWidth = 16; // 8 * 2
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.beginPath();
      ctx.moveTo(lastPosRef.current.x, lastPosRef.current.y);
      ctx.lineTo(x, y);
      ctx.stroke();
    }
    
    lastPosRef.current = { x, y };

    const imageData = canvas.toDataURL('image/png');
    onImageChange(imageData);
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const displayCanvas = displayCanvasRef.current;
    if (!canvas || !displayCanvas) return;

    const ctx = canvas.getContext('2d');
    const displayCtx = displayCanvas.getContext('2d');
    if (!ctx || !displayCtx) return;

    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    displayCtx.fillStyle = 'white';
    displayCtx.fillRect(0, 0, displayCanvas.width, displayCanvas.height);

    onImageChange('');
  };

  return (
    <div className="flex flex-col gap-8">
      <div className="grid md:grid-cols-2 gap-8">
        <div className="flex flex-col gap-4">
          <div className="flex items-center justify-between border-b-2 border-stone-900 pb-2">
            <label className="text-sm font-bold text-stone-900 uppercase tracking-widest font-sans">Đầu vào gốc</label>
            <span className="text-xs text-stone-900 bg-sky-200 border border-stone-900 px-2 py-1 font-bold font-sans">280×280</span>
          </div>
          <div className="relative border-2 border-stone-900 shadow-[4px_4px_0px_0px_rgba(28,25,23,1)] bg-white p-2 w-full max-w-[300px] mx-auto">
            <canvas
              ref={canvasRef}
              width={280}
              height={280}
              className="cursor-crosshair w-full h-auto touch-none"
              onMouseDown={startDrawing}
              onMouseMove={draw}
              onMouseUp={stopDrawing}
              onMouseLeave={stopDrawing}
            />
          </div>
        </div>

        <div className="flex flex-col gap-4">
          <div className="flex items-center justify-between border-b-2 border-stone-900 pb-2">
            <label className="text-sm font-bold text-stone-900 uppercase tracking-widest font-sans">Biến đổi xoay</label>
            <span className="text-xs text-stone-900 bg-fuchsia-300 border border-stone-900 px-2 py-1 font-bold font-sans">Live Preview</span>
          </div>
          <div className="relative border-2 border-stone-900 shadow-[4px_4px_0px_0px_rgba(28,25,23,1)] bg-white p-2 w-full max-w-[300px] mx-auto">
            <canvas
              ref={displayCanvasRef}
              width={280}
              height={280}
              className="w-full h-auto pointer-events-none"
            />
          </div>
        </div>
      </div>

      <button
        onClick={clearCanvas}
        className="mx-auto mt-6 px-8 py-3 bg-rose-200 hover:bg-rose-300 text-stone-900 transition-colors text-sm font-black border-2 border-stone-900 uppercase tracking-wider shadow-[4px_4px_0px_0px_rgba(28,25,23,1)] active:translate-y-1 active:translate-x-1 active:shadow-none font-sans"
      >
        [X] Xóa Canvas
      </button>
    </div>
  );
}
