# Contributing Guidelines

Thank you for your interest in contributing to the **Geometric ML Foundations** project.

This repository was developed as the final project for the course **CSC14005 – Introduction to Machine Learning** at the Faculty of Information Technology, University of Science (VNU-HCM).

---

# Project Team

| Member                 | Student ID | Main Responsibilities                                                                |
| ---------------------- | ---------- | ------------------------------------------------------------------------------------ |
| **Lê Hà Thanh Chương** | 23120195   | Project management, report integration, NequIP/QM9 research, Part 3 demo system      |
| **Trà Văn Sỹ**         | 23120197   | MNIST rotation-invariant experiments, Part 1 demo, frontend initialization           |
| **Huỳnh Đức Thịnh**    | 23120199   | PointNet architecture research, ModelNet40 implementation, Part 2 demo               |
| **Bùi Trung Hiếu**     | 23120257   | Mathematical foundations, TikZ visualizations, NequIP notebook support, slide design |
| **Lê Công Phúc**       | 23120330   | Advanced theory research, PointNet experimental analysis, slide design               |

---

# Repository Structure

```text
ai_core/
├── notebooks/
├── outputs/
├── datasets/

backend/
frontend/

report/
├── src/
├── output/

docs/
```

---

# Development Workflow

## 1. Create a Feature Branch

```bash
git checkout -b feature/<feature-name>
```

Examples:

```bash
feature/pointnet-demo
feature/nequip-backend
feature/report-update
```

---

## 2. Commit Convention

We follow Conventional Commit style.

### Features

```bash
feat(module): add new functionality
```

Example:

```bash
feat(nequip): add QM9 molecular energy inference pipeline
```

### Bug Fixes

```bash
fix(module): resolve issue
```

Example:

```bash
fix(frontend): correct molecule rendering synchronization
```

### Documentation

```bash
docs(report): update experimental results section
```

### Refactoring

```bash
refactor(module): improve code structure
```

### Maintenance

```bash
chore(repo): update project configuration
```

---

## 3. Pull Request Guidelines

Before creating a Pull Request:

* Ensure the project builds successfully.
* Ensure notebooks execute without errors.
* Verify generated figures are reproducible.
* Update documentation if behavior changes.
* Provide a concise description of modifications.

---

# Report Contributions

Report source files are located in:

```text
report/src/
```

Each contributor is responsible for maintaining the quality and consistency of their assigned sections.

Before merging:

* Compile the report successfully.
* Check references and citations.
* Verify figure numbering.
* Verify cross-references.

---

# Coding Standards

## Python

* Follow PEP 8.
* Use descriptive variable names.
* Add docstrings for public functions.
* Keep modules focused on a single responsibility.

## TypeScript / React

* Prefer functional components.
* Use TypeScript typings.
* Keep components modular and reusable.

---

# Reproducibility

Experiments should be reproducible using:

```bash
pip install -r requirements.txt
```

and the notebooks located in:

```text
ai_core/notebooks/
```

Model checkpoints and generated artifacts should not be committed unless explicitly required.

---

# Acknowledgements

This project explores geometric machine learning through:

* Rotation-Invariant CNNs
* PointNet for Point Cloud Classification
* NequIP for Molecular Energy Prediction

and serves as an educational exploration of symmetry-aware machine learning models.
