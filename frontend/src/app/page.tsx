"use client";
import { useState, useEffect } from 'react';
import { DigitCanvas } from '@/components/DigitCanvas';
import { RotationSlider } from '@/components/RotationSlider';
import { PredictionCard } from '@/components/PredictionCard';
import { predictDigit } from '@/services/predictionService';
import { ApiResponse } from '@/types';
import { Loader2 } from 'lucide-react';
// ── Part 2: PointNet imports ──
import { PointCloudViewer } from '@/components/PointCloudViewer';
import { PointNetResultCard } from '@/components/PointNetResultCard';
import { NumPointsSlider } from '@/components/NumPointsSlider';
import { PerturbationControls, PerturbationState, DEFAULT_PERTURBATION } from '@/components/PerturbationControls';
import { classifyPointCloud, getSampleCloud } from '@/services/pointnetService';
import { PointNetApiResponse, DEMO_CLASSES, DemoClass } from '@/types/pointnet';
// ── Part 3: NequIP imports ──
import { MoleculeViewer3D } from '@/components/MoleculeViewer3D';
import { MoleculeEditor } from '@/components/MoleculeEditor';
import { PredictionPanel } from '@/components/PredictionPanel';
import { EnergyResultCard } from '@/components/EnergyResultCard';
import { ExampleMolecules } from '@/components/ExampleMolecules';
import { predictEnergy } from '@/services/nequipService';
import { Atom } from '@/types/molecule';

export default function Home() {
  const [activeTab, setActiveTab] = useState<'part1' | 'part2' | 'part3'>('part1');
  // ── Part 1 state ──
  const [rotation, setRotation] = useState(0);
  const [imageData, setImageData] = useState('');
  const [predictions, setPredictions] = useState<ApiResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  // ── Part 2 state ──
  const [selectedClass, setSelectedClass] = useState<DemoClass>('airplane');
  const [numPoints, setNumPoints] = useState(1024);
  const [p2Loading, setP2Loading] = useState(false);
  const [p2Result, setP2Result] = useState<PointNetApiResponse | null>(null);
  const [rawPoints, setRawPoints] = useState<number[][]>([]);
  const [showCritical, setShowCritical] = useState(false);
  const [perturbation, setPerturbation] = useState<PerturbationState>(DEFAULT_PERTURBATION);
  // ── Part 3 state ──
  const [atoms, setAtoms] = useState<Atom[]>([
    { atomicNumber: 8, x: 0.0, y: 0.0, z: 0.1163 },
    { atomicNumber: 1, x: 0.0, y: 0.7583, z: -0.4654 },
    { atomicNumber: 1, x: 0.0, y: -0.7583, z: -0.4654 }
  ]);
  const [selectedMoleculeName, setSelectedMoleculeName] = useState<string>('Nước');
  const [energyResult, setEnergyResult] = useState<number | null>(null);
  const [p3Loading, setP3Loading] = useState(false);
  const [p3Error, setP3Error] = useState<string | null>(null);
  const [latencyMs, setLatencyMs] = useState<number | undefined>(undefined);

  // Chặn SSR render → tránh hydration mismatch từ browser extension
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  const handlePredictEnergy = async () => {
    if (atoms.length === 0) {
      setP3Error('Vui lòng thêm ít nhất một nguyên tử vào phân tử.');
      return;
    }
    setP3Loading(true);
    setP3Error(null);
    setEnergyResult(null);
    const start = performance.now();
    try {
      const input = {
        atomic_numbers: atoms.map(a => a.atomicNumber),
        positions: atoms.map(a => [a.x, a.y, a.z])
      };
      const result = await predictEnergy(input);
      setEnergyResult(result.energy);
      const end = performance.now();
      setLatencyMs(end - start);
    } catch (e: any) {
      setP3Error(e.message || 'Lỗi không xác định.');
    } finally {
      setP3Loading(false);
    }
  };

  const handleResetMolecule = () => {
    setAtoms([
      { atomicNumber: 8, x: 0.0, y: 0.0, z: 0.1163 },
      { atomicNumber: 1, x: 0.0, y: 0.7583, z: -0.4654 },
      { atomicNumber: 1, x: 0.0, y: -0.7583, z: -0.4654 }
    ]);
    setSelectedMoleculeName('Nước');
    setEnergyResult(null);
    setP3Error(null);
    setLatencyMs(undefined);
  };

  const handleSelectExampleMolecule = (name: string, presetAtoms: Atom[]) => {
    setAtoms(presetAtoms);
    setSelectedMoleculeName(name);
    setEnergyResult(null);
    setP3Error(null);
    setLatencyMs(undefined);
  };

  if (!mounted) return null;

  const handleLoadSample = async (cls: DemoClass) => {
    setSelectedClass(cls);
    setP2Result(null);
    setShowCritical(false);
    try {
      const data = await getSampleCloud(cls);
      setRawPoints(data.points);
    } catch { setRawPoints([]); }
  };

  const handleClassify = async () => {
    if (rawPoints.length === 0) return;
    setP2Loading(true);
    try {
      const result = await classifyPointCloud(rawPoints, numPoints, perturbation);
      setP2Result(result);
      // Backend đã apply rotation vào data trả về
      // → reset visual rotation về 0 để không bị double-rotation
      setPerturbation(prev => ({ ...prev, rotation_x: 0, rotation_y: 0, rotation_z: 0 }));
    } catch (e) {
      alert('Lỗi: ' + e);
    } finally { setP2Loading(false); }
  };

  // Cập nhật slider X/Y khi drag trên viewer
  const handleRotationChange = (x: number, y: number) => {
    setPerturbation(prev => ({ ...prev, rotation_x: x, rotation_y: y }));
  };

  // Cập nhật slider Z khi scroll trên viewer
  const handleRotationChangeZ = (z: number) => {
    setPerturbation(prev => ({ ...prev, rotation_z: z }));
  };

  const handlePredict = async () => {
    if (!imageData) {
      alert('Vui lòng vẽ một chữ số trước!');
      return;
    }
    setIsLoading(true);
    try {
      const result = await predictDigit(imageData);
      setPredictions(result);
    } catch (error) {
      console.error('Prediction error:', error);
      alert('Failed to get predictions. Please make sure the backend is running.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-stone-100 font-sans text-stone-900 pb-16" suppressHydrationWarning>
      {/* Header */}
      <header className="w-full px-4 md:px-8 pt-6 pb-4">
        <div className="bg-stone-50 border-2 border-stone-900 p-6 shadow-[6px_6px_0px_0px_rgba(28,25,23,1)] flex flex-col md:flex-row gap-6 md:items-start">
          {/* Left: Title */}
          <div className="flex-1">
            <div className="inline-block mb-3 px-2 py-1 bg-blue-600 border-2 border-stone-900 shadow-[2px_2px_0px_0px_rgba(28,25,23,1)]">
              <span className="text-white text-[10px] font-bold uppercase tracking-widest font-sans">Geometric Deep Learning</span>
            </div>
            <h1 className="text-3xl md:text-4xl font-black text-stone-900 mb-2 font-serif uppercase tracking-tight leading-none">
              {activeTab === 'part1' ? <>Rotation Invariance<br />on MNIST</> : activeTab === 'part2' ? <>Phân Loại 3D<br />với PointNet</> : <>Dự đoán Năng lượng<br />với NequIP</>}
            </h1>
            <p className="text-sm text-stone-700 font-serif mt-3 border-t-2 border-stone-200 pt-3">
              {activeTab === 'part1'
                ? 'Khám phá học máy nhận thức đối xứng qua phân loại chữ số viết tay.'
                : activeTab === 'part2'
                ? 'Phân loại vật thể 3D từ đám mây điểm — Qi et al., CVPR 2017.'
                : 'Dự đoán năng lượng cơ học lượng tử của phân tử bằng mạng thần kinh đồ thị đẳng biến E(3).'}
            </p>
          </div>

          {/* Right: Methods */}
          <div className="flex-[1.5] border-t-2 md:border-t-0 md:border-l-2 border-stone-900 pt-4 md:pt-0 md:pl-6">
            <h2 className="text-sm font-black text-stone-900 mb-3 font-serif uppercase tracking-widest bg-yellow-400 border-2 border-stone-900 shadow-[2px_2px_0px_0px_rgba(28,25,23,1)] inline-block px-2 py-1">Phương pháp Thử nghiệm</h2>
            {activeTab === 'part1' ? (
              <>
                <p className="text-stone-800 text-sm leading-relaxed font-sans mb-3 hidden lg:block">
                  So sánh 3 cách tiếp cận bất biến xoay:
                </p>
                <ul className="text-stone-800 text-sm leading-relaxed font-sans space-y-3">
                  <li><strong className="text-stone-900 bg-red-400 px-1.5 py-0.5 border-2 border-stone-900 shadow-[2px_2px_0px_0px_rgba(28,25,23,1)] mr-2 inline-block">(1) CNN Cơ sở:</strong>Huấn luyện hoàn toàn trên các hướng thẳng đứng chuẩn.</li>
                  <li><strong className="text-stone-900 bg-amber-400 px-1.5 py-0.5 border-2 border-stone-900 shadow-[2px_2px_0px_0px_rgba(28,25,23,1)] mr-2 inline-block">(2) Tăng cường Dữ liệu:</strong>Kết hợp các phép biến đổi xoay đa dạng trong huấn luyện.</li>
                  <li><strong className="text-stone-900 bg-emerald-400 px-1.5 py-0.5 border-2 border-stone-900 shadow-[2px_2px_0px_0px_rgba(28,25,23,1)] mr-2 inline-block">(3) Trung bình Khung:</strong>Tận dụng các dự đoán tương đương nhóm khi kiểm tra.</li>
                </ul>
              </>
            ) : activeTab === 'part2' ? (
              <>
                <p className="text-stone-800 text-sm leading-relaxed font-sans mb-3 hidden lg:block">
                  Phân loại vật thể 3D từ đám mây điểm (Qi et al., CVPR 2017):
                </p>
                <ul className="text-stone-800 text-sm leading-relaxed font-sans space-y-3">
                  <li><strong className="text-stone-900 bg-red-400 px-1.5 py-0.5 border-2 border-stone-900 shadow-[2px_2px_0px_0px_rgba(28,25,23,1)] mr-2 inline-block">(1) PointNet Basic:</strong>Shared MLP + Max Pooling, không có T-Net.</li>
                  <li><strong className="text-stone-900 bg-emerald-400 px-1.5 py-0.5 border-2 border-stone-900 shadow-[2px_2px_0px_0px_rgba(28,25,23,1)] mr-2 inline-block">(2) PointNet Full:</strong>Có Input & Feature Transform (T-Net) và regularization loss.</li>
                  <li><strong className="text-stone-900 bg-violet-400 px-1.5 py-0.5 border-2 border-stone-900 shadow-[2px_2px_0px_0px_rgba(28,25,23,1)] mr-2 inline-block">(3) Critical Points:</strong>Visualize tập điểm quyết định kết quả (Theorem 2).</li>
                </ul>
              </>
            ) : (
              <>
                <p className="text-stone-800 text-sm leading-relaxed font-sans mb-3 hidden lg:block">
                  Dự đoán năng lượng QM9 bằng GNN đẳng biến E(3) (Batzner et al.):
                </p>
                <ul className="text-stone-800 text-sm leading-relaxed font-sans space-y-3">
                  <li><strong className="text-stone-900 bg-cyan-400 px-1.5 py-0.5 border-2 border-stone-900 shadow-[2px_2px_0px_0px_rgba(28,25,23,1)] mr-2 inline-block">(1) Đồ thị Nguyên tử:</strong>Các nguyên tử là nút đồ thị, liên kết cách nhau r &le; r_max là cạnh.</li>
                  <li><strong className="text-stone-900 bg-emerald-400 px-1.5 py-0.5 border-2 border-stone-900 shadow-[2px_2px_0px_0px_rgba(28,25,23,1)] mr-2 inline-block">(2) E(3) Message Passing:</strong>Truyền thông điệp đẳng biến bảo toàn tính bất biến đối xứng quay và tịnh tiến 3D.</li>
                  <li><strong className="text-stone-900 bg-indigo-400 text-white px-1.5 py-0.5 border-2 border-stone-900 shadow-[2px_2px_0px_0px_rgba(28,25,23,1)] mr-2 inline-block">(3) Global Pooling:</strong>Cộng gộp năng lượng của các nút nguyên tử để ra năng lượng tổng phân tử.</li>
                </ul>
              </>
            )}
          </div>
        </div>
      </header>

      {/* TAB SWITCHER */}
      <div className="w-full px-4 md:px-8 mb-6 flex flex-wrap gap-4">
        <button
          onClick={() => setActiveTab('part1')}
          className={`px-6 py-3 font-black text-sm md:text-base uppercase tracking-widest border-2 border-stone-900 transition-all ${
            activeTab === 'part1'
            ? 'bg-rose-400 shadow-[4px_4px_0px_0px_rgba(28,25,23,1)] translate-y-0 translate-x-0'
            : 'bg-white shadow-none translate-y-1 translate-x-1 hover:bg-stone-50'
          }`}
        >
          Nhận diện Hình ảnh (SO(2))
        </button>
        <button
          onClick={() => setActiveTab('part2')}
          className={`px-6 py-3 font-black text-sm md:text-base uppercase tracking-widest border-2 border-stone-900 transition-all ${
            activeTab === 'part2'
            ? 'bg-emerald-400 shadow-[4px_4px_0px_0px_rgba(28,25,23,1)] translate-y-0 translate-x-0'
            : 'bg-white shadow-none translate-y-1 translate-x-1 hover:bg-stone-50'
          }`}
        >
          Phân loại 3D (PointNet)
        </button>
        <button
          onClick={() => setActiveTab('part3')}
          className={`px-6 py-3 font-black text-sm md:text-base uppercase tracking-widest border-2 border-stone-900 transition-all ${
            activeTab === 'part3'
            ? 'bg-cyan-400 shadow-[4px_4px_0px_0px_rgba(28,25,23,1)] translate-y-0 translate-x-0'
            : 'bg-white shadow-none translate-y-1 translate-x-1 hover:bg-stone-50'
          }`}
        >
          Dự đoán Năng lượng (NequIP)
        </button>
      </div>

      <main className="w-full px-4 md:px-8 pb-8">
        {activeTab === 'part1' && (
          <>
            {/* Part 1: Frame Averaging */}
            <div className="grid lg:grid-cols-5 gap-8 mb-12">
              <div className="lg:col-span-3 flex flex-col bg-stone-50 border-2 border-stone-900 p-8 shadow-[8px_8px_0px_0px_rgba(28,25,23,1)]">
                <div className="flex items-center gap-4 mb-8 pb-4 border-b-2 border-stone-900">
                  <div className="w-10 h-10 bg-rose-500 border-2 border-stone-900 shadow-[2px_2px_0px_0px_rgba(28,25,23,1)] flex items-center justify-center text-stone-900 font-black font-sans text-xl">1</div>
                  <h2 className="text-2xl font-black text-stone-900 font-serif uppercase tracking-tight">Vẽ & Mô phỏng</h2>
                </div>
                <div className="flex-1 flex flex-col justify-center">
                  <DigitCanvas onImageChange={setImageData} rotation={rotation} />
                </div>
              </div>

              <div className="lg:col-span-2 space-y-8 flex flex-col">
                <div className="bg-stone-50 border-2 border-stone-900 p-8 shadow-[8px_8px_0px_0px_rgba(28,25,23,1)]">
                  <div className="flex items-center gap-4 mb-8 pb-4 border-b-2 border-stone-900">
                    <div className="w-10 h-10 bg-sky-400 border-2 border-stone-900 shadow-[2px_2px_0px_0px_rgba(28,25,23,1)] flex items-center justify-center text-stone-900 font-black font-sans text-xl">2</div>
                    <h2 className="text-2xl font-black text-stone-900 font-serif uppercase tracking-tight">Xoay Ảnh</h2>
                  </div>
                  <RotationSlider value={rotation} onChange={setRotation} />
                </div>

                <div className="bg-stone-50 border-2 border-stone-900 p-8 shadow-[8px_8px_0px_0px_rgba(28,25,23,1)] flex-1 flex flex-col">
                  <div className="flex items-center gap-4 mb-8 pb-4 border-b-2 border-stone-900">
                    <div className="w-10 h-10 bg-violet-400 border-2 border-stone-900 shadow-[2px_2px_0px_0px_rgba(28,25,23,1)] flex items-center justify-center text-stone-900 font-black font-sans text-xl">3</div>
                    <h2 className="text-2xl font-black text-stone-900 font-serif uppercase tracking-tight">Phân loại</h2>
                  </div>
                  <button
                    onClick={handlePredict}
                    disabled={isLoading || !imageData}
                    className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-stone-300 disabled:border-stone-300 disabled:text-stone-500 disabled:cursor-not-allowed text-white font-black py-5 px-6 transition-colors border-2 border-stone-900 disabled:shadow-none shadow-[6px_6px_0px_0px_rgba(28,25,23,1)] active:shadow-none active:translate-y-1 active:translate-x-1 flex items-center justify-center gap-3 text-lg font-sans uppercase tracking-widest mt-auto mb-auto"
                  >
                    {isLoading ? (<><Loader2 className="w-6 h-6 animate-spin" />Đang xử lý...</>) : 'Chạy Phân loại'}
                  </button>
                  {predictions && (
                    <div className="mt-8 bg-stone-100 border-2 border-stone-900 p-5">
                      <p className="text-sm text-stone-800 leading-relaxed font-sans">
                        <span className="font-bold text-stone-900 uppercase tracking-widest text-xs block mb-1">Quan sát:</span>
                        Mô hình cơ sở bị giảm hiệu suất đáng kể khi ảnh bị xoay, trong khi phương pháp trung bình khung thể hiện tính bất biến mạnh mẽ ở mọi góc độ.
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            <div className="bg-stone-50 border-2 border-stone-900 p-8 shadow-[8px_8px_0px_0px_rgba(28,25,23,1)]">
              <div className="flex items-center gap-4 mb-10 pb-4 border-b-2 border-stone-900">
                <div className="w-12 h-12 bg-pink-400 border-2 border-stone-900 shadow-[2px_2px_0px_0px_rgba(28,25,23,1)] flex items-center justify-center text-stone-900 font-black text-2xl font-serif">*</div>
                <h2 className="text-3xl font-black text-stone-900 font-serif uppercase tracking-tight">Phân tích So sánh</h2>
              </div>
              <div className="grid md:grid-cols-3 gap-8">
                <PredictionCard title="CNN Cơ sở" description="Chỉ huấn luyện trên chữ số thẳng đứng" result={predictions?.baseline || null} variant="baseline" />
                <PredictionCard title="Tăng cường dữ liệu" description="Huấn luyện với dữ liệu tăng cường xoay" result={predictions?.augmentation || null} variant="augmentation" />
                <PredictionCard title="Trung bình khung" description="Trung bình hóa qua nhiều góc xoay" result={predictions?.averaging || null} variant="averaging" />
              </div>
            </div>
          </>
        )}

        {activeTab === 'part2' && (
          // ────────────────────────────────────────────
          // PART 2: PointNet 3D Demo
          // ────────────────────────────────────────────
          <div className="space-y-8">
            {/* Row 1: Viewer + Controls */}
            <div className="grid lg:grid-cols-5 gap-8">
              {/* Left: 3D Viewer */}
              <div className="lg:col-span-3 flex flex-col bg-stone-50 border-2 border-stone-900 p-8 shadow-[8px_8px_0px_0px_rgba(28,25,23,1)]">
                <div className="flex items-center gap-4 mb-6 pb-4 border-b-2 border-stone-900">
                  <div className="w-10 h-10 bg-violet-400 border-2 border-stone-900 shadow-[2px_2px_0px_0px_rgba(28,25,23,1)] flex items-center justify-center text-stone-900 font-black font-sans text-xl">1</div>
                  <h2 className="text-2xl font-black text-stone-900 font-serif uppercase tracking-tight">Đám Mây Điểm 3D</h2>
                  {p2Result && (
                    <button
                      onClick={() => setShowCritical(v => !v)}
                      className={`ml-auto px-4 py-2 text-xs font-black font-sans uppercase tracking-widest border-2 border-stone-900 transition-all ${
                        showCritical
                          ? 'bg-red-500 text-white shadow-[3px_3px_0px_0px_rgba(28,25,23,1)]'
                          : 'bg-white text-stone-900 shadow-none hover:bg-stone-50'
                      }`}
                    >
                      {showCritical ? 'Chế độ Tới hạn' : 'Chế độ Bình thường'}
                    </button>
                  )}
                </div>
                <div className="flex-1" style={{ minHeight: 340 }}>
                  <PointCloudViewer
                    points={p2Result ? p2Result.point_cloud : rawPoints}
                    criticalPoints={p2Result ? (showCritical ? p2Result.full_model.critical_points : []) : []}
                    showCritical={showCritical}
                    isLoading={p2Loading}
                    rotationX={perturbation.rotation_x}
                    rotationY={perturbation.rotation_y}
                    rotationZ={perturbation.rotation_z}
                    onRotationChange={handleRotationChange}
                    onRotationChangeZ={handleRotationChangeZ}
                  />
                </div>
              </div>

              {/* Right: Controls */}
              <div className="lg:col-span-2 space-y-6 flex flex-col">
                {/* Object selector */}
                <div className="bg-stone-50 border-2 border-stone-900 p-6 shadow-[8px_8px_0px_0px_rgba(28,25,23,1)]">
                  <div className="flex items-center gap-4 mb-5 pb-4 border-b-2 border-stone-900">
                    <div className="w-10 h-10 bg-sky-400 border-2 border-stone-900 shadow-[2px_2px_0px_0px_rgba(28,25,23,1)] flex items-center justify-center text-stone-900 font-black font-sans text-xl">2</div>
                    <h2 className="text-xl font-black text-stone-900 font-serif uppercase tracking-tight">Chọn Vật Thể</h2>
                  </div>
                  <div className="grid grid-cols-5 gap-2 mb-4">
                    {DEMO_CLASSES.map((cls) => (
                      <button
                        key={cls}
                        onClick={() => handleLoadSample(cls)}
                        className={`flex flex-col items-center gap-1 py-3 px-1 border-2 border-stone-900 text-xs font-black font-sans uppercase tracking-wide transition-all ${
                          selectedClass === cls
                            ? 'bg-emerald-400 shadow-[3px_3px_0px_0px_rgba(28,25,23,1)]'
                            : 'bg-white shadow-none hover:bg-stone-50 translate-y-0.5 translate-x-0.5'
                        }`}
                      >
                        <span className="text-xl">{cls === 'airplane' ? '✈' : cls === 'chair' ? '🪑' : cls === 'car' ? '🚗' : cls === 'lamp' ? '💡' : '🪵'}</span>
                        <span className="truncate w-full text-center" style={{fontSize:'9px'}}>{cls === 'airplane' ? 'Máy bay' : cls === 'chair' ? 'Ghế' : cls === 'car' ? 'Ô tô' : cls === 'lamp' ? 'Đèn' : 'Bàn'}</span>
                      </button>
                    ))}
                  </div>
                  {rawPoints.length > 0 && (
                    <p className="text-xs text-stone-500 font-sans border-t border-stone-200 pt-2">
                      Đã tải: <span className="font-bold text-stone-900">{selectedClass}</span> — {rawPoints.length} điểm
                    </p>
                  )}
                </div>

                {/* Num Points slider */}
                <div className="bg-stone-50 border-2 border-stone-900 p-6 shadow-[8px_8px_0px_0px_rgba(28,25,23,1)]">
                  <div className="flex items-center gap-4 mb-5 pb-4 border-b-2 border-stone-900">
                    <div className="w-10 h-10 bg-amber-400 border-2 border-stone-900 shadow-[2px_2px_0px_0px_rgba(28,25,23,1)] flex items-center justify-center text-stone-900 font-black font-sans text-xl">3</div>
                    <h2 className="text-xl font-black text-stone-900 font-serif uppercase tracking-tight">Số Điểm</h2>
                  </div>
                  <NumPointsSlider value={numPoints} onChange={setNumPoints} />
                </div>

                {/* Classify button */}
                <button
                  onClick={handleClassify}
                  disabled={p2Loading || rawPoints.length === 0}
                  className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-stone-300 disabled:border-stone-300 disabled:text-stone-500 disabled:cursor-not-allowed text-white font-black py-5 px-6 transition-colors border-2 border-stone-900 disabled:shadow-none shadow-[6px_6px_0px_0px_rgba(28,25,23,1)] active:shadow-none active:translate-y-1 active:translate-x-1 flex items-center justify-center gap-3 text-lg font-sans uppercase tracking-widest"
                >
                  {p2Loading ? (<><Loader2 className="w-6 h-6 animate-spin" />Đang phân tích...</>) : 'Chạy Phân Loại'}
                </button>
              </div>
            </div>

            {/* Row 2: Perturbation Controls */}
            <PerturbationControls
              value={perturbation}
              onChange={setPerturbation}
              onReset={() => setPerturbation(DEFAULT_PERTURBATION)}
            />

            {/* Row 3: Results */}
            <div className="bg-stone-50 border-2 border-stone-900 p-8 shadow-[8px_8px_0px_0px_rgba(28,25,23,1)]">
              <div className="flex items-center gap-4 mb-8 pb-4 border-b-2 border-stone-900">
                <div className="w-12 h-12 bg-pink-400 border-2 border-stone-900 shadow-[2px_2px_0px_0px_rgba(28,25,23,1)] flex items-center justify-center text-stone-900 font-black text-2xl font-serif">*</div>
                <h2 className="text-3xl font-black text-stone-900 font-serif uppercase tracking-tight">Phân tích So sánh</h2>
                {p2Result && (
                  <span className="ml-auto text-xs font-bold font-sans text-stone-500 border border-stone-300 px-3 py-1">
                    {p2Result.processing_time_ms.toFixed(0)} ms
                  </span>
                )}
              </div>
              <div className="grid md:grid-cols-2 gap-8">
                <PointNetResultCard
                  title="PointNet Basic"
                  description="Shared MLP + Max Pool, không có T-Net (baseline)"
                  result={p2Result?.basic_model ?? null}
                  variant="basic"
                />
                <PointNetResultCard
                  title="PointNet Full"
                  description="Có Input & Feature Transform (T-Net), regularization loss"
                  result={p2Result?.full_model ?? null}
                  variant="full"
                />
              </div>
              {p2Result && (
                <div className="mt-6 bg-stone-100 border-2 border-stone-900 p-5">
                  <p className="text-sm text-stone-800 leading-relaxed font-sans">
                    <span className="font-bold text-stone-900 uppercase tracking-widest text-xs block mb-1">Quan sát:</span>
                    Full model (có T-Net) thường có độ tự tin cao hơn Basic model. Chuyển sang{' '}
                    <span className="font-bold text-red-600">Chế độ Tới hạn</span> để xem những điểm nào quyết định kết quả (Theorem 2).
                    Thử điều chỉnh các thanh <span className="font-bold text-violet-700">Kiểm Chứng Robustness</span> bên trên rồi phân loại lại để thấy tác động.
                  </p>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'part3' && (
          // ────────────────────────────────────────────
          // PART 3: NequIP Molecular Energy Prediction
          // ────────────────────────────────────────────
          <div className="space-y-8">
            {/* Row 1: Viewer + Editor */}
            <div className="grid lg:grid-cols-5 gap-8">
              {/* Left: 3D Viewer */}
              <div className="lg:col-span-3 flex flex-col bg-stone-50 border-2 border-stone-900 p-8 shadow-[8px_8px_0px_0px_rgba(28,25,23,1)]">
                <div className="flex items-center gap-4 mb-6 pb-4 border-b-2 border-stone-900">
                  <div className="w-10 h-10 bg-cyan-400 border-2 border-stone-900 shadow-[2px_2px_0px_0px_rgba(28,25,23,1)] flex items-center justify-center text-stone-900 font-black font-sans text-xl">1</div>
                  <h2 className="text-2xl font-black text-stone-900 font-serif uppercase tracking-tight">Trực quan hóa Phân tử 3D</h2>
                </div>
                <div className="flex-1" style={{ minHeight: 400 }}>
                  <MoleculeViewer3D atoms={atoms} isLoading={p3Loading} />
                </div>
              </div>

              {/* Right: Coordinate Editor */}
              <div className="lg:col-span-2">
                <MoleculeEditor atoms={atoms} onChangeAtoms={setAtoms} />
              </div>
            </div>

            {/* Row 2: Controls + Result */}
            <div className="grid lg:grid-cols-5 gap-8">
              {/* Controls */}
              <div className="lg:col-span-2">
                <PredictionPanel
                  onPredict={handlePredictEnergy}
                  onReset={handleResetMolecule}
                  isLoading={p3Loading}
                  isDisabled={atoms.length === 0}
                />
              </div>

              {/* Result Card */}
              <div className="lg:col-span-3">
                <EnergyResultCard
                  energy={energyResult}
                  isLoading={p3Loading}
                  error={p3Error}
                  durationMs={latencyMs}
                />
              </div>
            </div>

            {/* Row 3: Presets selection */}
            <ExampleMolecules
              selectedMoleculeName={selectedMoleculeName}
              onSelectMolecule={handleSelectExampleMolecule}
            />
          </div>
        )}
      </main>
    </div>
  );
}
