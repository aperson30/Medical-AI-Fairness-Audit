# Medical AI Fairness Audit — Code Release

This repository contains all code for the paper:

> **The MAD Risk Score: A Pre-Deployment Diagnostic for Fairness Intervention Failures in Clinical Image Classification**
> Aditya Sanjeev, Eeshan Kavuri, 2026

We audit four clinical AI architectures across multiple datasets and show that standard fairness interventions (Group DRO, Adversarial Debiasing, SMOTE) systematically fail under real-world class imbalance. The two core findings are:

- **Source Generalization Gap (SGG)** — demographic performance disparities that persist even when aggregate accuracy looks good
- **Metric-Accuracy Dissociation (MAD)** — standard fairness metrics failing to detect real clinical harm

We additionally introduce the **MAD Risk Score**, a pre-deployment diagnostic computable from frozen embeddings that predicts whether Group DRO will cause catastrophic per-class accuracy collapse before any training occurs.

**Models:** CLIP ViT-L/14, DINOv2-Base, ResNet-50, ViT-B/16  
**Datasets:** [Fitzpatrick17k](https://github.com/mattgroh/fitzpatrick17k), [NIH ChestX-ray14](https://www.kaggle.com/datasets/nih-chest-xrays/data), Fed-ISIC2019, ISIC 2024 SLICE-3D, DDI, PH2, PAD-UFES-20, CelebA

---

## Repository Layout

```
.
├── 01_baselines/           Baseline training — one script per architecture
├── 02_experiments/         Core paper experiments (run in order: exp1 → exp5c)
├── 03_full_pipeline/       End-to-end replication scripts (Fitzpatrick17k and NIH)
├── 04_intersectional/      Intersectional analysis: age × skin tone × sex; MST groupings; Afrose comparison
├── 05_mechanistic/         Why interventions fail: decision boundaries, loss surfaces, nc/Ng phase transitions
├── 06_supplementary/       Calibration, significance tests, robustness checks, CI validation
├── 07_finetune_suite/      Full fine-tuning robustness suite (nb0 → nb7, run in order)
├── 08_mad_risk_score/      MAD Risk Score: threshold derivation, nc/Ng sweep, external validation
├── 09_smote_recovery/      SMOTE recovery mechanism: cross-architecture + image vs feature space
├── not_used_in_paper/      Exploratory notebooks: extended formula variants not included in final paper
└── utils/                  CSV patching utilities
```

---

## Setup

All scripts are written as Kaggle notebooks (cells marked with `# ── CELL N`). They run on a single GPU T4 with internet access. Datasets download automatically.

```bash
pip install torch torchvision transformers scikit-learn pandas numpy umap-learn Pillow imbalanced-learn
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
| `resnet50_sgg_finetune.py` | ResNet-50 fine-tune — confirms SGG is architecture-independent | ~25 min |
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
| `exp5c_intervention_matrix_all_archs_fixed.ipynb` | **[NEW]** Fixed intervention matrix for all 4 architectures — cu118 GPU wheel fix (P100/sm_60 compatible) | ~3–4 hr |
| `cross_dataset_per_class_eval.ipynb` | Primary cross-dataset per-class evaluation pipeline | — |

---

## `03_full_pipeline/`

Self-contained replication scripts for reviewers. Each runs the complete sequence for one dataset: feature extraction → random and demographic-aware splits → SGG computation → DACC calibration → per-class breakdown → result export.

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
| `mst_skin_tone_grouping_replication.ipynb` | **[NEW]** MST replication — tests whether Group DRO collapse, nc/Ng, and MAD Risk Score change under ITA-derived Monk Skin Tone groupings vs FST | ~60 min |
| `afrose_double_prioritized_comparison.ipynb` | **[NEW]** Afrose et al. double-prioritized bias correction vs SMOTE and Group DRO — the only published method designed for minority-within-minority structure | ~45 min |

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
| `mech4_mad_i_nc_ng_gap_fill.ipynb` | **[NEW]** MAD-I boundary gap fill (nc/Ng 2%–5%) — NIH ChestX-ray14, CLIP ViT-L/14; plugs the gap between confirmed collapse (<1.4%) and safe regime (>5.5%) | ~4–6 hrs |
| `mech5_rho_star_threshold_validation.ipynb` | **[NEW]** ρ* threshold validation — tests nc/Ng at 5, 10, 12, 15, 20, 30, 50% to empirically ground the ρ* ≈ 0.10 claim | ~60 min |
| `mech6_threshold_sweep_collapse.ipynb` | **[NEW]** Critical collapse threshold sweep — varies nc/Ng from 5–50%, runs Group DRO + SMOTE at each level to locate the ρ* transition | ~90 min |
| `mech7_nih_mad_i_boundary_extension.ipynb` | **[NEW]** NIH ChestX-ray14 MAD-I boundary audit — sweeps pathologies to confirm the nc/Ng < 2% contraindication zone | ~60 min |

---

## `06_supplementary/`

Additional validation analyses referenced in supplementary materials.

| Script | What it does | Runtime |
|--------|-------------|---------|
| `calibration_curves_skin_tone.py` | Reliability diagrams by skin tone — shows systematic miscalibration for dark-skin patients | ~15 min |
| `fisher_exact_test_benign.py` | Fisher's Exact Test on the 0% benign finding — confirms statistical significance | ~2 min (CPU) |
| `robustness_larger_dark_skin_sample.py` | Re-runs core experiment with all 2,168 dark-skin images — addresses reviewer n=97 concern | ~15 min |
| `adversarial_debiasing_wilson_ci.ipynb` | **[NEW]** Adversarial debiasing: Wilson 95% CI and statistical comparison to Group DRO — fixes missing CI claim | ~45 min |
| `label_noise_proxy_analysis.ipynb` | **[NEW]** Label noise proxy analysis — tests whether dark-skin benign collapse is explained by higher label noise in that group | ~30 min |
| `ham10000_dermoscopy_isic_contradiction.ipynb` | **[NEW]** HAM10000 dermoscopy validation + ISIC 2019 contradiction resolution — establishes the dermoscopic modality applicability boundary | ~90 min |
| `dagc_sanity_check_full_pool.ipynb` | **[NEW]** DAGC sanity check with full pool (no 80/20 split) — verifies that the high DAGC benign accuracy is not an artifact of pool construction | ~45 min |

---

## `07_finetune_suite/`

Full fine-tuning robustness suite. Run notebooks in order (nb0 → nb6); nb6 reads CSVs produced by nb0–nb5 to build all publication figures. Missing CSVs are silently skipped, so partial figures render after each completed notebook. nb5b, nb5c, and nb7 are independent supplementary notebooks.

| Notebook | What it does | Runtime |
|----------|-------------|---------|
| `nb0_clip_finetune_robustness.ipynb` | CLIP ViT-L/14 full fine-tune robustness check — 5 seeds, single T4, DRO + SMOTE protocol | ~15–20 hrs |
| `nb1_linear_probe.ipynb` | Standalone: feature extraction for all 4 backbones + bug-fixed intervention loop (dark pool fix) | ~2 hrs |
| `nb2_finetune_vit_b16.py` | ViT-B/16 full fine-tune — mirrors nb0 protocol exactly | ~2–2.5 hrs |
| `nb3_finetune_resnet50.py` | ResNet-50 full fine-tune — mirrors nb0 protocol exactly | ~1.5–2 hrs |
| `nb4_finetune_dinov2.py` | DINOv2-Base full fine-tune — mirrors nb0 protocol; DINOv2 showed largest baseline MAD | ~2.5–3 hrs |
| `nb5_clip_eta_ablation.py` | CLIP Group DRO η-ablation: sweeps η ∈ {0.001, 0.01, 0.1, 1.0} to test whether any η rescues benign accuracy | ~3–4 hrs |
| `nb5b_eta_ablation_5seed_replication.ipynb` | **[NEW]** 5-seed η ablation replication — reruns the η sweep with 5 seeds (20 total) for stronger statistical confidence | ~4–5 hrs |
| `nb5c_dro_eta_ablation_5seeds.ipynb` | **[NEW]** Group DRO η ablation (5 seeds, CLIP ViT-L/14) — confirms weight-collapse finding across the full η range with 5 seeds per η | ~50 min |
| `nb6_make_figures.py` | Builds all publication figures from nb0–nb5 CSVs/JSONs (CPU only) | ~5 min |
| `nb7_fill_table4b_frozen_probe_interventions.ipynb` | **[NEW]** Table 4b gap fill — frozen probe Real Oversample, Adversarial Debiasing, and DAGC with bug-fixed pool construction (no test-set leakage) | ~2 hrs |

**Figures produced by nb6:**
- `fig1_dro_weight_collapse.png` — DRO group weight trajectories: weight on dark group collapses to ~0 within 2 epochs across all architectures
- `fig2_per_class_bars.png` — Per-class accuracy by intervention across all 4 architectures (baseline ≈ DRO; SMOTE shows modest gain)
- `fig3_lp_vs_ft.png` — Linear-probe vs fine-tune baseline: 4× swing in benign accuracy from the same training data
- `fig4_eta_ablation.png` — DRO benign accuracy vs η: whether any η rescues dark-skin benign detection

---

## `08_mad_risk_score/`

MAD Risk Score computation, threshold derivation, and external validation across five independent datasets (Section 4.3). The score is computable from frozen embeddings in ~84 seconds on CPU.

**Formula:** `MAD_risk = (1 − μ_cosine) / log(1 + n_minority_train)`  
**Threshold:** 0.086 (derived from Fitzpatrick17k nc/Ng sweep; see `mad_risk_nc_ng_sweep_log_formula.ipynb`)

| Notebook | What it does | Runtime |
|----------|-------------|---------|
| `mad_risk_nc_ng_sweep_log_formula.ipynb` | nc/Ng sweep (9 unique levels, v3 fixed) — derives the 0.086 threshold from Fitzpatrick17k; threshold derivation dataset | ~90 min |
| `mad_risk_recovery_side_validation.ipynb` | Recovery-side validation — validates the score on datasets where Group DRO does *not* collapse, confirming the score is not a constant alarm | ~90 min |
| `mad_risk_external_validation_ddi.ipynb` | External validation on DDI (Stanford) — true negative: MAD_risk = 0.066 < 0.086, DRO did not collapse | ~60 min |
| `mad_risk_external_validation_nih.ipynb` | External validation on NIH ChestX-ray14 — sex × pathology minority-within-minority; true positives confirmed | ~2.5 hrs |
| `mad_risk_external_validation_pad_ufes.ipynb` | External validation on PAD-UFES-20 — false negative case: MAD_risk = 0.062 < 0.086 but DRO collapsed 5/5 seeds | ~60 min |
| `mad_i_isic2024_slice3d_confirmation.ipynb` | MAD-I confirmation on ISIC 2024 SLICE-3D (3D TBP, nc/Ng = 0.097%) — extends MAD-I evidence to a new modality and 401,059-image scale | ~90 min |

---

## `09_smote_recovery/`

Mechanism experiments for SMOTE-based recovery of dark-skin benign accuracy (Section 4.6).

| Notebook | What it does | Runtime |
|----------|-------------|---------|
| `smote_cross_arch_recovery_generalization.ipynb` | Cross-architecture SMOTE recovery (v2 fixed) — tests whether feature-space SMOTE recovery generalizes across CLIP ViT-L/14, DINOv2-Base, DINOv2-Large, CLIP ViT-B/16; bug-fixed pool split | ~3 hrs |
| `smote_image_vs_feature_space_ablation.ipynb` | Image-space vs feature-space SMOTE mechanism ablation — determines whether recovery is geometric (sparse embedding synthesis) or purely quantity-based | ~2 hrs |

---

## `not_used_in_paper/`

Exploratory notebooks that test extended MAD Risk Score formula variants. These were developed during iteration but the extended formulas were not adopted in the final paper (the two-term formula is the published version). Kept for transparency.

| Notebook | What it was testing |
|----------|---------------------|
| `MAD_Risk_Geometry.ipynb` | Geometry fix experiment — replacing the cosine similarity term with Mean Distance to Centroid (MDC) to resolve DDI/PH2 false positives |
| `mad_risk_extended_formula_validation.ipynb` | Extended formula with effective rank normalization (d_eff_norm): `MAD_risk_ext = (1 − μ_cosine) × d_eff_norm / log(1 + n_minority_train)` |

---

## `utils/`

| Script | What it does |
|--------|-------------|
| `patch_csvs.py` | Re-aggregates result CSVs from distributed multi-account runs |

---

## Citation

```bibtex
@article{SanjeevKavuri2026,
  title   = {The MAD Risk Score: A Pre-Deployment Diagnostic for Fairness Intervention Failures in Clinical Image Classification},
  author  = {Sanjeev, Aditya and Kavuri, Eeshan},
  year    = {2026},
  journal = {TBD}
}
```
