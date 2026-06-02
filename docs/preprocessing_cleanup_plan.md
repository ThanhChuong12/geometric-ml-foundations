# Preprocessing Cleanup Plan

This document outlines the audit, classification, and cleanup strategy for the molecular preprocessing documentation in the repository.

---

## 📋 1. Audit and Classification of Reports

We have audited the 13 markdown reports currently located in the `reports/` folder. They are classified below:

| Report File | Classification | Rationale |
| :--- | :--- | :--- |
| `repository_forensics.md` | **ESSENTIAL** | Documents critical NequIP site-package references, files, and definitions. |
| `schema_verification.md` | **ESSENTIAL** | Verifies mandatory vs. optional behavior of the core dataset keys (`atomic_numbers`, `total_energy`, `forces`, `pbc`, `cell`). |
| `loader_compatibility.md` | **ESSENTIAL** | Identifies and proves the unregistered key (`z`, `y`) bug during graph collation. |
| `training_dry_run.md` | **ESSENTIAL** | Verifies successful dataset stats computation and NequIP model initialization. |
| `scalability_analysis.md` | **ESSENTIAL** | Details scaling laws, computational complexity, and storage estimations for larger datasets. |
| `production_readiness_v2.md` | **ESSENTIAL** | Contains the v2 score, issue breakdown, and proposed code fixes. |
| `claim_validation.md` | **USEFUL** | Disproves and verifies historical assumptions using direct code evidence. |
| `final_preprocessing_audit.md` | **USEFUL** | Provides a general overview of script features, physical consistencies, and risks. |
| `graph_construction_review.md` | **REDUNDANT** | Superceded by `final_preprocessing_audit.md` and `loader_compatibility.md`. |
| `normalization_review.md` | **REDUNDANT** | Superceded by `final_preprocessing_audit.md` and `loader_compatibility.md`. |
| `output_verification_report.md` | **REDUNDANT** | Superceded by `final_preprocessing_audit.md` and `loader_compatibility.md`. |
| `dataset_validation_report.md` | **DUPLICATE** | Pre-existing audit document covering details superceded by later reports. |
| `nequip_compatibility_report.md` | **DUPLICATE** | Pre-existing audit document covering details superceded by later reports. |

---

## 🧹 2. Consolidation & Action Plan

### Step 1: Create `docs/preprocessing_audit.md`
Consolidate the key technical details and findings from the **ESSENTIAL** and **USEFUL** reports into a single, comprehensive technical document at `docs/preprocessing_audit.md`.

### Step 2: Remove Obsolete Files
Clean up the `reports/` folder by moving the old markdown files to an archive or removing them to restore cleanliness. We will place them in `archive/preprocessing_audit_history/` as requested to preserve git history and documentation trace.

### Step 3: Restore Root README
Revert the root `README.md` to describe the overall repository rather than a preprocessing manual, and link to the new `docs/preprocessing_audit.md`.

### Step 4: Schema Code Fixes
Modify `scripts/preprocess.py` to fix the unregistered key bug by removing `z` and `y` from the output `.pt` files.
