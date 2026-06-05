"use client";
import { useState } from 'react';
import { DigitCanvas } from '@/components/DigitCanvas';
import { RotationSlider } from '@/components/RotationSlider';
import { PredictionCard } from '@/components/PredictionCard';
import { predictDigit } from '@/services/predictionService';
import { ApiResponse } from '@/types';
import { Loader2 } from 'lucide-react';

export default function Home() {
  const [activeTab, setActiveTab] = useState<'part1' | 'part2'>('part1');
  const [rotation, setRotation] = useState(0);
  const [imageData, setImageData] = useState('');
  const [predictions, setPredictions] = useState<ApiResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);

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
    <div className="min-h-screen bg-stone-100 font-sans text-stone-900 pb-16">
      {/* Academic Header & Methods Combined */}
      <header className="w-full px-4 md:px-8 pt-6 pb-4">
        <div className="bg-stone-50 border-2 border-stone-900 p-6 shadow-[6px_6px_0px_0px_rgba(28,25,23,1)] flex flex-col md:flex-row gap-6 md:items-start">

          {/* Left Side: Title */}
          <div className="flex-1">
            <div className="inline-block mb-3 px-2 py-1 bg-blue-600 border-2 border-stone-900 shadow-[2px_2px_0px_0px_rgba(28,25,23,1)]">
              <span className="text-white text-[10px] font-bold uppercase tracking-widest font-sans">Geometric Deep Learning</span>
            </div>
            <h1 className="text-3xl md:text-4xl font-black text-stone-900 mb-2 font-serif uppercase tracking-tight leading-none">
              Rotation Invariance<br />on MNIST
            </h1>
            <p className="text-sm text-stone-700 font-serif mt-3 border-t-2 border-stone-200 pt-3">
              Khám phá học máy nhận thức đối xứng qua phân loại chữ số viết tay.
            </p>
          </div>

          {/* Right Side: Methods */}
          <div className="flex-[1.5] border-t-2 md:border-t-0 md:border-l-2 border-stone-900 pt-4 md:pt-0 md:pl-6">
            <h2 className="text-sm font-black text-stone-900 mb-3 font-serif uppercase tracking-widest bg-yellow-400 border-2 border-stone-900 shadow-[2px_2px_0px_0px_rgba(28,25,23,1)] inline-block px-2 py-1">Phương pháp Thử nghiệm</h2>
            {activeTab === 'part1' ? (
              <>
                <p className="text-stone-800 text-sm leading-relaxed font-sans mb-3 hidden lg:block">
                  Bản trình diễn tương tác so sánh 3 cách tiếp cận bất biến xoay:
                </p>
                <ul className="text-stone-800 text-sm leading-relaxed font-sans space-y-3">
                  <li>
                    <strong className="text-stone-900 bg-red-400 px-1.5 py-0.5 border-2 border-stone-900 shadow-[2px_2px_0px_0px_rgba(28,25,23,1)] mr-2 inline-block">(1) CNN Cơ sở:</strong>
                    Huấn luyện hoàn toàn trên các hướng thẳng đứng chuẩn.
                  </li>
                  <li>
                    <strong className="text-stone-900 bg-amber-400 px-1.5 py-0.5 border-2 border-stone-900 shadow-[2px_2px_0px_0px_rgba(28,25,23,1)] mr-2 inline-block">(2) Tăng cường Dữ liệu:</strong>
                    Kết hợp các phép biến đổi xoay đa dạng trong huấn luyện.
                  </li>
                  <li>
                    <strong className="text-stone-900 bg-emerald-400 px-1.5 py-0.5 border-2 border-stone-900 shadow-[2px_2px_0px_0px_rgba(28,25,23,1)] mr-2 inline-block">(3) Trung bình Khung:</strong>
                    Tận dụng các dự đoán tương đương nhóm khi kiểm tra.
                  </li>
                </ul>
              </>
            ) : (
              <>
                <p className="text-stone-800 text-sm leading-relaxed font-sans mb-3 hidden lg:block">
                  Các phương pháp phân tích mạng nơ-ron đồ thị (Đang phát triển):
                </p>
                <ul className="text-stone-800 text-sm leading-relaxed font-sans space-y-3">
                  <li>
                    <strong className="text-stone-900 bg-purple-400 px-1.5 py-0.5 border-2 border-stone-900 shadow-[2px_2px_0px_0px_rgba(28,25,23,1)] mr-2 inline-block">(1) Message Passing:</strong>
                    Truyền tải thông điệp cục bộ qua các cạnh của đồ thị.
                  </li>
                  <li>
                    <strong className="text-stone-900 bg-cyan-400 px-1.5 py-0.5 border-2 border-stone-900 shadow-[2px_2px_0px_0px_rgba(28,25,23,1)] mr-2 inline-block">(2) Graph Convolution:</strong>
                    Tích chập phổ hoặc không gian trên cấu trúc đồ thị (GCN).
                  </li>
                  <li>
                    <strong className="text-stone-900 bg-lime-400 px-1.5 py-0.5 border-2 border-stone-900 shadow-[2px_2px_0px_0px_rgba(28,25,23,1)] mr-2 inline-block">(3) Equivariant GNNs:</strong>
                    Mạng nơ-ron bảo toàn tính đối xứng (xoay, lật, dịch chuyển).
                  </li>
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
          Phân tích Đồ thị (GNN)
        </button>
      </div>

      <main className="w-full px-4 md:px-8 pb-8">
        {activeTab === 'part1' ? (
          <>
        {/* Main Interactive Section */}
        <div className="grid lg:grid-cols-5 gap-8 mb-12">
          {/* Canvas Section */}
          <div className="lg:col-span-3 flex flex-col bg-stone-50 border-2 border-stone-900 p-8 shadow-[8px_8px_0px_0px_rgba(28,25,23,1)]">
            <div className="flex items-center gap-4 mb-8 pb-4 border-b-2 border-stone-900">
              <div className="w-10 h-10 bg-rose-500 border-2 border-stone-900 shadow-[2px_2px_0px_0px_rgba(28,25,23,1)] flex items-center justify-center text-stone-900 font-black font-sans text-xl">
                1
              </div>
              <h2 className="text-2xl font-black text-stone-900 font-serif uppercase tracking-tight">Vẽ & Mô phỏng</h2>
            </div>
            <div className="flex-1 flex flex-col justify-center">
              <DigitCanvas onImageChange={setImageData} rotation={rotation} />
            </div>
          </div>

          {/* Controls Section */}
          <div className="lg:col-span-2 space-y-8 flex flex-col">
            <div className="bg-stone-50 border-2 border-stone-900 p-8 shadow-[8px_8px_0px_0px_rgba(28,25,23,1)]">
              <div className="flex items-center gap-4 mb-8 pb-4 border-b-2 border-stone-900">
                <div className="w-10 h-10 bg-sky-400 border-2 border-stone-900 shadow-[2px_2px_0px_0px_rgba(28,25,23,1)] flex items-center justify-center text-stone-900 font-black font-sans text-xl">
                  2
                </div>
                <h2 className="text-2xl font-black text-stone-900 font-serif uppercase tracking-tight">Xoay Ảnh</h2>
              </div>
              <RotationSlider value={rotation} onChange={setRotation} />
            </div>

            <div className="bg-stone-50 border-2 border-stone-900 p-8 shadow-[8px_8px_0px_0px_rgba(28,25,23,1)] flex-1 flex flex-col">
              <div className="flex items-center gap-4 mb-8 pb-4 border-b-2 border-stone-900">
                <div className="w-10 h-10 bg-violet-400 border-2 border-stone-900 shadow-[2px_2px_0px_0px_rgba(28,25,23,1)] flex items-center justify-center text-stone-900 font-black font-sans text-xl">
                  3
                </div>
                <h2 className="text-2xl font-black text-stone-900 font-serif uppercase tracking-tight">Phân loại</h2>
              </div>

              <button
                onClick={handlePredict}
                disabled={isLoading || !imageData}
                className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-stone-300 disabled:border-stone-300 disabled:text-stone-500 disabled:cursor-not-allowed text-white font-black py-5 px-6 transition-colors border-2 border-stone-900 disabled:shadow-none shadow-[6px_6px_0px_0px_rgba(28,25,23,1)] active:shadow-none active:translate-y-1 active:translate-x-1 flex items-center justify-center gap-3 text-lg font-sans uppercase tracking-widest mt-auto mb-auto"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="w-6 h-6 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  'Run Classification'
                )}
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

        {/* Results Section */}
        <div className="bg-stone-50 border-2 border-stone-900 p-8 shadow-[8px_8px_0px_0px_rgba(28,25,23,1)]">
          <div className="flex items-center gap-4 mb-10 pb-4 border-b-2 border-stone-900">
            <div className="w-12 h-12 bg-pink-400 border-2 border-stone-900 shadow-[2px_2px_0px_0px_rgba(28,25,23,1)] flex items-center justify-center text-stone-900 font-black text-2xl font-serif">
              *
            </div>
            <h2 className="text-3xl font-black text-stone-900 font-serif uppercase tracking-tight">Phân tích So sánh</h2>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            <PredictionCard
              title="CNN Cơ sở"
              description="Chỉ huấn luyện trên chữ số thẳng đứng"
              result={predictions?.baseline || null}
              variant="baseline"
            />
            <PredictionCard
              title="Tăng cường dữ liệu"
              description="Huấn luyện với dữ liệu tăng cường xoay"
              result={predictions?.augmentation || null}
              variant="augmentation"
            />
            <PredictionCard
              title="Trung bình khung"
              description="Trung bình hóa qua nhiều góc xoay"
              result={predictions?.averaging || null}
              variant="averaging"
            />
          </div>
        </div>
          </>
        ) : (
          <div className="bg-stone-50 border-2 border-stone-900 p-12 shadow-[8px_8px_0px_0px_rgba(28,25,23,1)] text-center flex flex-col items-center">
            <div className="mb-6 w-16 h-16 bg-emerald-400 border-2 border-stone-900 shadow-[4px_4px_0px_0px_rgba(28,25,23,1)] flex items-center justify-center text-stone-900 font-black text-3xl font-serif">
              !
            </div>
            <h2 className="text-3xl md:text-4xl font-black text-stone-900 font-serif uppercase tracking-tight mb-6">Giao diện Đang Cập Nhật</h2>
            <p className="text-stone-700 font-sans text-lg max-w-2xl mx-auto border-t-2 border-stone-200 pt-6">
              Phần demo <strong>Mạng Neural Đồ Thị (Equivariant GNNs)</strong> đang được phát triển.<br/>
              Dữ liệu mô phỏng và API Backend sẽ được tích hợp riêng tại đây.
            </p>
          </div>
        )}
      </main>
    </div>
  );
}
