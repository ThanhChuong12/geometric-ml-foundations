# Normalization Strategy Review

This report analyzes the target scaling and coordinate centering strategies employed in the preprocessing pipeline, evaluating them against **NequIP** best practices and physical constraints.

---

## 🔍 1. Physical vs. Pre-Normalized Targets in NequIP

In NequIP, target scaling and shift operations are integrated directly **inside the neural network architecture** using the `PerTypeScaleShift` module. This module shift-scales targets automatically:
- **Shift ($\mu$)**: The potential energy is shifted by the sum of atomic self-energies:
  $$E_{\text{shift}} = E - \sum_{i} E_{z_i}^{(0)}$$
  where $E_{z_i}^{(0)}$ represents the self-energy of atom species $z_i$.
- **Scale ($\sigma$)**: The residual energy is scaled by a global standard deviation:
  $$E_{\text{scaled}} = \frac{E_{\text{shift}}}{\sigma_E}$$

### ⚠️ The Risk of Double Normalization
During dataset initialization, NequIP's `DataStatisticsManager` reads the target values from the `.pt` files to compute $\mu$ and $\sigma_E$.
- **Best Practice**: The values stored in the input dataset must be in **raw physical units** (e.g., eV for energy, eV/$\text{\AA}$ for forces).
- **Double Normalization**: If the preprocessing script scales the energies and forces (using the `--normalize` flag) and then passes them to NequIP, NequIP will scale them **a second time**. This leads to a mismatched target scale, making the trained model output scaled-down and physically incorrect predictions.
- **Recommendation**: For NequIP training, the `--normalize` command-line flag should be **disabled** (left off) in the preprocessing pipeline, preserving raw physical units. The `--normalize` option should only be used for external non-NequIP regression baselines.

---

## 📐 2. Mathematical Consistency in Force Scaling

Forces are the negative spatial gradient of the potential energy:
$$\mathbf{F}_i = -\nabla_{\mathbf{r}_i} E$$

If the energy is scaled by a global factor $\sigma_E$, we must scale the forces by the **same factor** to maintain this physical derivative relation:
$$\mathbf{F}_{i,\text{scaled}} = -\nabla_{\mathbf{r}_i} E_{\text{scaled}} = -\nabla_{\mathbf{r}_i} \left(\frac{E - \mu_E}{\sigma_E}\right) = \frac{\mathbf{F}_i}{\sigma_E}$$

### Analysis of Force Scaling Methods:
1. **`energy_scale` (Physically Consistent)**:
   - *Formula*: $\mathbf{F}_{norm} = \frac{\mathbf{F}}{\sigma_E}$
   - *Result*: Retains the exact gradient relationship. It is the method used by NequIP and is required for models that compute forces as analytical gradients of energy.
2. **`force_std` (Unphysical)**:
   - *Formula*: $\mathbf{F}_{norm} = \frac{\mathbf{F}}{\sigma_F}$
   - *Result*: Scales forces by their own variance, breaking the conservative force gradient relation. This will cause NequIP training to fail or diverge.

---

## 🌎 3. Center of Mass Centering

Centering atomic coordinates around the Center of Mass (COM) translates the molecule so that:
$$\sum_i m_i \mathbf{r}'_i = \mathbf{0}$$
where $m_i$ represents the mass of atom $i$.

- **NequIP Compatibility**: Fully compatible. NequIP's message-passing architecture uses relative vectors $\mathbf{r}_{ij} = \mathbf{r}_i - \mathbf{r}_j$, which are translationally invariant. Centering coordinates does not alter relative distances or vectors, but it helps prevent numerical issues and makes coordinate dimensions comparable.

---

## 🚫 4. Data Leakage Validation

- **Current Implementation**: The pipeline calculates the normalization coefficients ($\mu_E$, $\sigma_E$, $\sigma_F$) using **only** the training split indices:
  ```python
  train_energies = E_flat[train_indices]
  energy_mean = float(np.mean(train_energies))
  energy_std = float(np.std(train_energies))
  ```
  These parameters are then applied to the validation and test sets.
- **Evaluation**: This is correct and prevents data leakage from the validation or test splits into the training parameters.
