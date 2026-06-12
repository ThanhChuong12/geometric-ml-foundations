import { ExternalLink, GitBranch } from 'lucide-react';

const REFERENCES = [
  {
    label: "PointNet — Qi et al., CVPR 2017",
    href: "https://arxiv.org/abs/1612.00593",
  },
  {
    label: "NequIP — Batzner et al., Nature Comm. 2022",
    href: "https://arxiv.org/abs/2101.03164",
  },
  {
    label: "Frame Averaging — Puny et al., ICML 2022",
    href: "https://arxiv.org/abs/2110.03336",
  },
];

const TECH_STACK = ["Next.js 15", "FastAPI", "PyTorch", "Three.js", "NequIP", "ASE"];

export function Footer() {
  return (
    <footer className="w-full mt-6 border-t-4 border-stone-900 bg-stone-900 text-white">
      <div className="w-full px-4 md:px-8 py-12">
        <div className="grid md:grid-cols-3 gap-8 md:gap-12">

          {/* About */}
          <div>
            <h3 className="text-[10px] font-black uppercase tracking-widest text-stone-400 mb-4 font-sans">
              Về dự án
            </h3>
            <p className="text-sm text-stone-300 font-sans leading-relaxed">
              Demo tương tác 3 hướng nghiên cứu Geometric Machine Learning:
              bất biến xoay SO(2), phân loại đám mây điểm 3D, và dự đoán năng lượng phân tử bằng GNN đẳng biến E(3).
            </p>
          </div>

          {/* References */}
          <div>
            <h3 className="text-[10px] font-black uppercase tracking-widest text-stone-400 mb-4 font-sans">
              Tài liệu tham khảo
            </h3>
            <ul className="space-y-3">
              {REFERENCES.map((ref) => (
                <li key={ref.label}>
                  <a
                    href={ref.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-start gap-2 text-xs font-sans text-stone-300 hover:text-white transition-colors group"
                  >
                    <ExternalLink className="w-3 h-3 mt-0.5 flex-shrink-0 text-stone-500 group-hover:text-stone-300 transition-colors" />
                    {ref.label}
                  </a>
                </li>
              ))}
            </ul>
          </div>

          {/* Tech Stack */}
          <div>
            <h3 className="text-[10px] font-black uppercase tracking-widest text-stone-400 mb-4 font-sans">
              Công nghệ sử dụng
            </h3>
            <div className="flex flex-wrap gap-2">
              {TECH_STACK.map((tech) => (
                <span
                  key={tech}
                  className="text-[10px] font-black font-sans uppercase bg-stone-800 border border-stone-600 text-stone-300 px-2 py-1"
                >
                  {tech}
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="mt-10 pt-6 border-t border-stone-700 flex items-center justify-between flex-wrap gap-4">
          <span className="text-[10px] text-stone-500 font-sans uppercase tracking-widest font-bold">
            Geometric ML Foundations — Academic Demo
          </span>
          <a
            href="https://github.com/ThanhChuong12/geometric-ml-foundations"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 text-[10px] font-black font-sans uppercase tracking-widest text-stone-400 hover:text-white transition-colors border border-stone-600 hover:border-stone-400 px-3 py-1.5"
          >
            <GitBranch className="w-3.5 h-3.5" />
            GitHub
          </a>
        </div>
      </div>
    </footer>
  );
}
