# Recent Developments in Geometric Machine Learning: Foundations, Models, and More

## Thông tin hội nghị
**Diễn giả:** Stefanie Jegelka (TUM & MIT) và Behrooz Tahmasebi (Harvard University)  
**Sự kiện:** NeurIPS 2025, San Diego, CA  
**Thời gian:** December 2025  

## Tổng quan chủ đề

Tutorial này tập trung vào **geometric machine learning**, đặc biệt là các phát triển gần đây về **symmetries (tính đối xứng)** trong machine learning. Hội nghị bao gồm hai phần chính:

### Phần 1: Cơ sở và Kỹ thuật cơ bản
### Phần 2: Symmetries trong neural parameters và các hướng nghiên cứu mới

---

## 1. Symmetries trong Machine Learning

### 1.1 Định nghĩa Symmetry (Tính đối xứng)

**Symmetry của một object:** Các phép biến đổi làm cho thuộc tính của object đó không thay đổi

#### Hai khái niệm chính:

**Invariance (Bất biến):**
- f(T(x)) = f(x)
- Ví dụ: f(shift(x)) = f(x), f(permute(x)) = f(x)
- Output không thay đổi khi input bị transform

**Equivariance (Đồng biến):**
- f(T(x)) = T(f(x))
- Output thay đổi theo cách tương ứng với transformation của input
- Invariance là trường hợp đặc biệt của equivariance với scalar output

### 1.2 Tại sao Symmetries quan trọng?

#### Lợi ích về hiệu suất:
- **Fewer degrees of freedom:** Hiệu quả dữ liệu tốt hơn
- **Better data efficiency:** NequIP model cần ít dữ liệu hơn đến 3 bậc độ lớn
- **Robustness & OOD generalization:** Khả năng tổng quát hóa tốt hơn
- **Computational efficiency:** Cải thiện scaling laws

#### Ứng dụng rộng rãi:
- Learning và sample complexity
- Generalization, robustness, generation
- Optimization, sampling, interpretability
- Model analysis, model merging

---

## 2. Group Theory Basics

### 2.1 Group Definition

Một **group G** là tập hợp các transformations với:
- **Associativity:** (gh)ℓ = g(hℓ)
- **Identity element:** eg = ge = g
- **Inverse:** ∀g ∈ G, ∃g⁻¹ : gg⁻¹ = g⁻¹g = e  
- **Closure:** ∀g,h ∈ G, gh ∈ G

### 2.2 Group Actions và Representations

- **Group action:** G acts on X as g·x for g ∈ G, x ∈ X
- **Group representation:** Biểu diễn group action bằng matrix operations
- **Orbit:** OrbG(x) = {g·x | g ∈ G} - tập tất cả transformations của x

---

## 3. Phương pháp đạt được Equivariant Models

### 3.1 Sử dụng Off-the-shelf Models

#### Data Augmentation
- **Ưu điểm:** Đơn giản, áp dụng được rộng rãi
- **Nhược điểm:** Chỉ approximate symmetry, không đảm bảo invariance ở mọi nơi

#### Canonicalization
- **Ý tưởng:** Đưa data về dạng chuẩn (canonical form)
- **Ưu điểm:** Đơn giản, kế thừa tính chất của base model
- **Nhược điểm:** 
  - Không phải lúc nào cũng dễ thực hiện (ví dụ: graphs)
  - Vấn đề discontinuity
  - Không có continuous canonicalization cho permutations trong không gian >1D

#### Group/Frame Averaging
- **Group Averaging:** Trung bình trên toàn bộ group
  - **Ưu điểm:** Universal approximation, continuous
  - **Nhược điểm:** Expensive cho large groups
  
- **Frame Averaging:** Trung bình trên subset của group
  - **Ví dụ:** PCA frame cho point clouds/proteins
  - Cân bằng giữa cost và performance

### 3.2 Constructing Equivariant Models

#### Linear Equivariant Layers
- **Parameter sharing:** Structure trong weight matrix W
- **Constraint:** WPgx = PgWx cho tất cả g ∈ G
- **Ví dụ:**
  - Shift → Convolution matrix
  - Permutations → DeepSet architecture (2 parameters chỉ)

#### Group Convolution
- Mở rộng regular convolution sang groups bất kỳ
- f ⋆ ψ(g) := ∑h∈G f(h) ψ(g⁻¹h)

#### Invariant Theory
- **Invariant polynomials:** Sử dụng scalar quantities bất biến
- **Ví dụ:** 
  - Point clouds: Sử dụng inner products (v_i^T v_j)
  - Translation invariance: Relative positions

#### Tensor Methods
- Sử dụng tensor powers x⊗k để đạt universality
- **Challenge:** Clebsch-Gordan problem cho finite groups

---

## 4. Applications

### 4.1 Atomistic Systems
- **NequIP:** E(3)-equivariant GNNs cho molecular dynamics
- Prediction energy, forces (via auto-grad)
- **Equiformer:** Equivariant graph transformers

### 4.2 Robotics
- SO(2), SE(3) symmetries trong robot actions
- Optimal policies trong MDPs là equivariant
- Grasp learning, manipulation tasks

### 4.3 Protein Generation
- **Chroma, RFDiffusion:** Sử dụng equivariant GNNs
- SE(3) symmetries cho protein binding
- Local coordinate systems

### 4.4 Symmetry Breaking
- **Vấn đề:** Equivariant architectures không thể break symmetry
- **Giải pháp:** Weight perturbation, gradient methods, sets

---

## 5. Neural Parameter Symmetries

### 5.1 Khái niệm

**Neural parameter symmetries:** Các transformations trên weights mà vẫn giữ nguyên function
- f_θ = f_(g·θ) (same function)
- L(f_θ) = L(f_(g·θ)) (same loss)

### 5.2 Ví dụ các Symmetries

#### MLPs:
- **Scale invariance:** Scaling layers (homogeneous activations)
- **Permutation invariance:** Hoán đổi neurons

#### Matrix Products (LoRAs, Attention):
- **GL(r)-invariance:** UV^T = UR(R^(-1)V^T) cho bất kỳ invertible R

#### Transformers:
- Permutation của attention heads
- RMSNorm symmetries
- Attention và LoRA symmetries

### 5.3 Implications

#### Loss Landscapes:
- **Continuous symmetries:** Mode connectivity qua continuous paths
- **Discrete symmetries:** Multiple basins, cần alignment cho linear connectivity

#### Optimization:
- Different local landscapes ở các symmetry transformations
- Invariant optimization algorithms (Path-SGD)
- Conservative quantities trong training

#### Model Merging:
- Cần alignment trước khi merge models
- Simplification bằng cách remove symmetries

---

## 6. Weight Space Learning

### 6.1 Neural Networks as Data

**Ý tưởng:** Treat neural networks như data points
- **Model populations:** INRs, NeRFs, Hugging Face models (>2M models)
- **Meta-networks:** NNs nhận NNs khác làm input

### 6.2 Applications

#### Faster Evaluation:
- Predict model accuracy không cần evaluate
- LoRA meta networks: nhanh hơn 50,000+ lần

#### Model Analysis:
- Derive information về models
- Model genealogy, training data inference
- IP, licensing, bias estimation

#### Model Editing:
- Domain adaptation, pruning, compression

### 6.3 Techniques

#### Equivariant Meta-networks:
- **Deep Weight Space Networks:** Linear equivariant layers
- **Neural Functional Transformers:** Weight space equivariance + attention

#### Graph-based Methods:
- Treat NN như graphs, sử dụng GNNs
- Parameter graphs cho different architectures

---

## 7. Theoretical Results

### 7.1 Expander Theory

#### Challenges với Large Groups:
- Permutation group S_50: |S_50| ≈ 3×10^64 (lớn hơn số atoms trên Earth!)
- Group averaging trở nên impractical

#### Generating Sets:
- **Theorem:** Bất kỳ finite group nào đều có generating set với size ≤ log₂|G|
- **Random approach:** Random subset size ~2.67×log|G| là generating set với high probability

#### Applications:
- **Computational complexity:** Learning với exact invariances trong polynomial time
- **Data augmentation:** O(log|G|) random elements đủ cho full statistical benefits
- **Approximate group averaging:** Uniform approximation error bounds

### 7.2 Symmetry Testing

#### Statistical Formulation:
- **H₀:** μ ≡ g·μ ∀g ∈ G (distribution is invariant)
- **H₁:** sup_g D(g·μ, μ) ≥ ε (exists violating transformation)

#### Results:
- Finding optimal g là NP-hard
- **Randomized solution:** 𝔼_g[D(g·μ, μ)] ≤ sup_g D(g·μ, μ) ≤ 2𝔼_g[D(g·μ, μ)]
- Deterministic algorithms với reduced covering sizes

### 7.3 Sample Complexity Benefits

#### General Result:
𝔼[‖f̂ - f*‖²] ≤ (C·d·vol(M/G)/n)^(s/(s+d))

- **d:** dim(M) - dim(G) (reduced dimension)
- **vol(M/G):** Volume của quotient space
- **Multiplicative gain:** vol(M/G)
- **Exponential gain:** dimension reduction

#### Proof Technique:
- Extended Weyl's law cho manifolds
- Fourier sparsity → sample complexity gains

---

## 8. New Directions

### 8.1 Flexible Symmetries

#### Adaptive Symmetries:
- **Contextual World Models:** In-context selection của symmetry subsets
- **Any-subgroup Equivariant Networks:** Symmetry-breaking input selects subgroup

#### Symmetry Discovery:
- Learn data transformations/augmentations
- Learn architectural constraints
- Probe trained networks cho symmetries

### 8.2 Any-dimensional Models

#### Ý tưởng:
- Fixed parameters, varying input sizes
- **Representation stability:** Theory từ mathematics
- **Ví dụ:** GNNs train trên small graphs, test trên large graphs

#### Challenges:
- Universality vs expressivity trade-offs
- Evaluation across different sizes

### 8.3 Geometry Beyond Symmetries

#### Topological Deep Learning:
- **Higher-order message passing (HOMP):** Simplicial/cell complexes
- **Multi-cellular Networks (MCN):** Address topological limitations
- Topological features: connected components, loops, orientability

#### Curvature và Learning Theory:
- Manifold hypothesis meets statistical learning
- Positive curvature → efficient learning
- Bounded curvature, unbounded volume → provably hard

---

## 9. Future Discussions

### 9.1 Encode Symmetries or Not?

#### Scale Considerations:
- Inductive bias giúp với smaller/expensive data
- Large models có thể không cần specialized symmetries
- Equivariance vẫn có thể improve scaling laws

#### Symmetry Types:
- Approximate vs exact symmetries
- Data-dependent vs universal symmetries
- Task relevance của symmetries

#### Benchmark Issues:
- Một số datasets đã được canonicalized
- Cần careful evaluation

### 9.2 Flexibility vs Specialization

#### Foundation Models:
- Broad applicability vs targeted performance
- Adaptive symmetry selection
- Tool use cho specialized tasks

#### Generative Models:
- Symmetry có thể restrictive
- Symmetry breaking methods
- Balance giữa structure và creativity

---

## 10. Key Takeaways

1. **Symmetries are fundamental:** Ảnh hưởng đến mọi aspect của ML từ learning đến optimization

2. **Multiple approaches available:** Từ simple data augmentation đến sophisticated equivariant architectures

3. **Theory provides insights:** Expander theory, sample complexity bounds, testing methods

4. **New frontiers emerging:** Neural parameter symmetries, weight space learning, flexible symmetries

5. **Practical impact proven:** Significant gains trong molecular dynamics, robotics, protein generation

6. **Balance is key:** Giữa specialization và generality, exact và approximate symmetries

---

## References và Further Reading

Tutorial này reference đến hàng trăm papers spanning:
- Classic group theory và representation theory
- Modern geometric deep learning architectures  
- Recent theoretical advances
- Cutting-edge applications trong science và robotics

Đây là một lĩnh vực đang phát triển mạnh mẽ với nhiều opportunities cho both theoretical advances và practical applications.