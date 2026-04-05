# Medical AI Fairness Audit — Code Release

This repository contains all code for the paper:

> **Medical AI Fairness Audit: Generalization Gaps and Metric-Accuracy Dissociation**
> Aditya Sanjeev et al., 2026

We audit four clinical AI architectures across two datasets and show that standard fairness interventions (Group DRO, Adversarial Debiasing, SMOTE) systematically fail under real-world class imbalance. The two core findings are:

- **Source Generalization Gap (SGG)** — demographic performance disparities that persist even when aggregate accuracy looks good
- **Metric-Accuracy Dissociation (MAD)** — standard fairness metrics failing to detect real clinical harm

**Models:** CLIP ViT-L/14, DINOv2, ResNet50, ViT-B/16
**Datasets:** [Fitzpatrick17k](https://github.com/mattgroh/fitzpatrick17k) (skin disease), [NIH ChestX-ray14](https://www.kaggle.com/datasets/nih-chest-xrays/data)

---

## Repository Layout

```
.
├── 01_baselines/        Baseline training — one script per architecture
├── 02_experiments/      Core paper experiments (run in order: exp1 → exp5b)
├── 03_full_pipeline/    End-to-end replication scripts (Fitzpatrick17k and NIH)
├── 04_intersectional/   Intersectional analysis: age × skin tone × sex
├── 05_mechanistic/      Why interventions fail: decision boundaries, loss surfaces, phase transitions
├── 06_supplementary/    Calibration, significance tests, robustness checks
└── utils/               Master runner and CSV patching utilities
```

---

## Setup

All scripts are written as Kaggle notebooks (cells marked with `# ── CELL N`). They run on a single GPU T4 with internet access. Datasets download automatically.

```bash
pip install torch torchvision transformers scikit-learn pandas numpy umap-learn Pillow
```

Datasets needed (add to Kaggle session before running):
- `nazmusresan/fitzpatrick17k`
- `nih-chest-xrays`
- `Fed-ISIC2019` (auto-downloads)

---

## `01_baselines/`

Establishes per-architecture baselines before running the audit.

| Script | What it does | Runtime |
|--------|-------------|---------|
| `clip_linear_probe.py` | CLIP ViT-L/14 linear probe — replicates Table 1 baseline rows | ~20 min |
| `clip_sgg_finetune.py` | Fine-tuned CLIP — tests whether SGG persists beyond linear probing | ~30–40 min |
| `resnet50_sgg_finetune.py` | ResNet50 fine-tune — confirms SGG is architecture-independent | ~25 min |
| `vit_b16_finetune.py` | ViT-B/16 as third architecture across Fitzpatrick17k and NIH | ~20 min |

---

## `02_experiments/`

Core paper experiments. Run these in order to reproduce all main tables and figures.

| Script | What it does | Runtime |
|--------|-------------|---------|
| `exp1_clip_full_backbone_finetune.py` | Full backbone CLIP fine-tune — closes linear probe criticism | ~60 min |
| `exp2_intersectional_age_x_skin.py` | Intersectional analysis: Age × Skin Tone across both datasets | ~30 min |
| `exp3_training_size_ablation.py` | Training set size ablation (0 → 200 dark-skin images) | ~20 min |
| `exp4_dinov2.py` | DINOv2 as fourth architecture | ~20 min |
| `exp5a_dinov2_gap_filling.py` | DINOv2 gap-filling across Fitzpatrick17k, Fed-ISIC, NIH | ~45 min |
| `exp5b_arch_dataset_matrix.py` | Full Architecture × Dataset matrix — all four models, all datasets | ~25 min |
| `cross_dataset_per_class_eval.ipynb` | Primary cross-dataset per-class evaluation pipeline | — |

---

## `03_full_pipeline/`

Self-contained replication scripts for reviewers. Each runs the complete sequence for one dataset: feature extraction → random and demographic-aware splits → SGG computation → DACC calibration → per-class breakdown → result export. These are the canonical scripts for reproducing paper results.

| Script | What it does | Runtime |
|--------|-------------|---------|
| `fitzpatrick17k_sgg_pipeline.py` | Full SGG pipeline on Fitzpatrick17k (skin-tone stratification, DACC, per-class) | ~4–6 hrs |
| `nih_chestxray_sgg_pipeline.py` | Full SGG pipeline on NIH (age-group and sex-stratified experiments, DACC) | ~6–8 hrs |

---

## `04_intersectional/`

Extended intersectional experiments beyond the core ablation.

| Script | What it does | Runtime |
|--------|-------------|---------|
| `ablation_dark_skin_0to500.py` | Extends training ablation from 200 → 500 dark-skin images | ~25 min |
| `nih_sex_architecture_matrix.py` | Completes Architecture × Dataset matrix with NIH sex stratification | ~20 min |
| `fitzpatrick_age_x_skin.py` | Age × Skin Tone intersectional analysis — tests whether the penalty amplifies | ~35 min |

---

## `05_mechanistic/`

Explains *why* fairness interventions fail at the geometric and optimization level.

| Script | What it does |
|--------|-------------|
| `mech1_decision_boundary.py` | Decision boundary in PCA/UMAP space — shows Group DRO moves the boundary *away* from the dark-skin benign cluster |
| `mech2_loss_surface_per_group.py` | Per-group training loss curves — shows dark-skin benign loss spikes back up under DRO minimax |
| `mech3_nc_ng_phase_transition.py` | Synthetic nc/Ng sweep — empirically validates the ~10% class-imbalance phase transition |
| `umap_feature_space.py` | UMAP feature-space geometry — diagnoses why 0% benign accuracy persists after augmentation |
| `decision_boundary_paper_split.py` | Repeats boundary analysis using the exact paper training split |

---

## `06_supplementary/`

Additional validation analyses referenced in supplementary materials.

| Script | What it does | Runtime |
|--------|-------------|---------|
| `calibration_curves_skin_tone.py` | Reliability diagrams by skin tone — shows systematic miscalibration for dark-skin patients | ~15 min |
| `fisher_exact_test_benign.py` | Fisher's Exact Test on the 0% benign finding — confirms statistical significance | ~2 min (CPU) |
| `robustness_larger_dark_skin_sample.py` | Re-runs core experiment with all 2,168 dark-skin images — addresses reviewer n=97 concern | ~15 min |

---

## `utils/`

| Script | What it does |
|--------|-------------|
| `run_all.py` | Master script — runs full feature extraction and MAD theory suite in one shot (~2 hrs on P100) |
| `patch_csvs.py` | Re-aggregates result CSVs from distributed multi-account runs |

---

## Citation

```bibtex
@article{GIGAv9,
  title   = {Medical AI Fairness Audit: Generalization Gaps and Metric-Accuracy Dissociation},
  author  = {Aditya Sanjeev et al.},
  year    = {2026},
  journal = {TBD}
}
```
