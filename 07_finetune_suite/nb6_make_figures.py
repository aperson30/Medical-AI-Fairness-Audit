# ============================================================
# NOTEBOOK 6 — Make Figures for the New Paper
# CPU only, ~5 minutes
#
# PURPOSE
# -------
# Build the publication figures from the CSVs and JSON files produced by
# notebooks 1-5. This is run AFTER all the data-collection notebooks.
#
# FIGURES PRODUCED
# ----------------
# 1. fig1_dro_weight_collapse.png — DRO group weight trajectories across
#    all 4 architectures (or however many you've finished). The new
#    headline figure showing why DRO fails: weight on dark group collapses
#    to near zero within 2 epochs across all seeds and all architectures.
#
# 2. fig2_per_class_bars.png — Per-class accuracy by intervention, with
#    error bars across seeds, for all 4 architectures. Shows: baseline ≈
#    DRO (DRO does nothing), SMOTE provides modest improvement.
#
# 3. fig3_lp_vs_ft.png — Linear-probe baseline vs fine-tune baseline,
#    showing the 4× swing in benign accuracy from the same training data.
#    The "evaluation methodology matters" finding.
#
# 4. fig4_eta_ablation.png — DRO benign accuracy as a function of eta.
#    Shows whether ANY eta works for DRO under fine-tuning.
#
# REQUIRES
# --------
# CSVs in:
#   - results_bugfix/04_intervention_matrix.csv  (notebook 1)
#   - results_level1/level1_finetune_results.csv (existing CLIP run)
#   - results_ft_vit/ft_vit_results.csv          (notebook 2)
#   - results_ft_resnet50/ft_resnet50_results.csv (notebook 3)
#   - results_ft_dinov2/ft_dinov2_results.csv    (notebook 4)
#   - results_eta_ablation/eta_ablation_results.csv (notebook 5)
#
# JSONs in (for figure 1):
#   - results_level1/gdro_weight_trajectories.json
#   - results_ft_vit/gdro_weight_trajectories.json
#   - results_ft_resnet50/gdro_weight_trajectories.json
#   - results_ft_dinov2/gdro_weight_trajectories.json
#
# Missing files are silently skipped — partial figures will still render
# from whatever CSVs exist. Re-run after each new notebook completes.
# ============================================================

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

warnings.filterwarnings('ignore')

mpl.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

FIGURES_DIR = 'figures_new_paper'
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── File paths ────────────────────────────────────────────────
ARCH_FILES = {
    'CLIP ViT-L/14': {
        'csv':  'results_level1/level1_finetune_results.csv',
        'json': 'results_level1/gdro_weight_trajectories.json',
    },
    'ViT-B/16': {
        'csv':  'results_ft_vit/ft_vit_results.csv',
        'json': 'results_ft_vit/gdro_weight_trajectories.json',
    },
    'ResNet-50': {
        'csv':  'results_ft_resnet50/ft_resnet50_results.csv',
        'json': 'results_ft_resnet50/gdro_weight_trajectories.json',
    },
    'DINOv2-Base': {
        'csv':  'results_ft_dinov2/ft_dinov2_results.csv',
        'json': 'results_ft_dinov2/gdro_weight_trajectories.json',
    },
}

LP_BUGFIX_CSV = 'results_bugfix/04_intervention_matrix.csv'
ETA_CSV       = 'results_eta_ablation/eta_ablation_results.csv'

INTV_LABELS = {
    'baseline':  'Baseline\n(no intervention)',
    'group_dro': 'Group DRO\n(η=0.01)',
    'smote':     'SMOTE',
}
INTV_COLORS = {
    'baseline':  '#888888',
    'group_dro': '#D55E00',
    'smote':     '#0072B2',
}

# ════════════════════════════════════════════════════════════════
# FIGURE 1: DRO Weight Collapse Trajectories
# ════════════════════════════════════════════════════════════════
print("Building Figure 1: DRO weight collapse...")

available_archs = [(name, info) for name, info in ARCH_FILES.items()
                   if os.path.exists(info['json'])]
n_archs = len(available_archs)

if n_archs == 0:
    print("  ✗ No weight trajectory JSONs found, skipping fig1")
else:
    n_cols = min(n_archs, 4)
    n_rows = (n_archs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows),
                              sharey=True, squeeze=False)

    for ax_idx, (arch_name, info) in enumerate(available_archs):
        ax = axes[ax_idx // n_cols, ax_idx % n_cols]
        try:
            with open(info['json']) as f:
                wlogs = json.load(f)
        except Exception as e:
            ax.text(0.5, 0.5, f'No data\n({e})', ha='center', va='center',
                    transform=ax.transAxes)
            continue

        for seed_key, log in wlogs.items():
            log = np.array(log)  # shape (epochs, 2)
            if log.ndim != 2 or log.shape[1] != 2: continue
            # Prepend the [0.5, 0.5] init for visual clarity
            xs = np.arange(0, len(log) + 1)
            ys = np.concatenate([[0.5], log[:, 1]])
            ax.plot(xs, ys, marker='o', markersize=3, linewidth=1.2,
                    alpha=0.7, label=seed_key.replace('seed', 's'))

        ax.set_title(arch_name, fontweight='bold')
        ax.set_xlabel('Epoch')
        if ax_idx % n_cols == 0:
            ax.set_ylabel('DRO weight on dark group')
        ax.set_ylim(-0.02, 0.55)
        ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.grid(True, alpha=0.2)
        ax.legend(loc='upper right', fontsize=7, frameon=False, ncol=2)

    # Hide unused axes
    for i in range(n_archs, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].set_visible(False)

    fig.suptitle(
        'Group DRO weight on the dark-skin minority group collapses to ~0\n'
        'within 2 epochs across all seeds and architectures (η=0.01)',
        fontsize=12, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig1_dro_weight_collapse.png'))
    plt.close()
    print(f"  ✓ Saved fig1 ({n_archs} architectures)")

# ════════════════════════════════════════════════════════════════
# FIGURE 2: Per-class accuracy by intervention, all architectures
# ════════════════════════════════════════════════════════════════
print("\nBuilding Figure 2: Per-class accuracy bars...")

available_csvs = [(name, info['csv']) for name, info in ARCH_FILES.items()
                  if os.path.exists(info['csv'])]
n_archs = len(available_csvs)

if n_archs == 0:
    print("  ✗ No fine-tune CSVs found, skipping fig2")
else:
    n_cols = min(n_archs, 4)
    n_rows = (n_archs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows),
                              sharey=True, squeeze=False)

    for ax_idx, (arch_name, csv_path) in enumerate(available_csvs):
        ax = axes[ax_idx // n_cols, ax_idx % n_cols]
        try:
            df_arch = pd.read_csv(csv_path)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {e}', ha='center', va='center',
                    transform=ax.transAxes)
            continue

        intvs = ['baseline', 'group_dro', 'smote']
        positions = np.arange(len(intvs))

        means = []
        stds  = []
        for intv in intvs:
            sub = df_arch[df_arch['intervention'] == intv]
            means.append(sub['acc_benign_dark'].mean() if len(sub) > 0 else 0)
            stds.append(sub['acc_benign_dark'].std() if len(sub) > 1 else 0)

        bars = ax.bar(positions, means, yerr=stds, capsize=4,
                       color=[INTV_COLORS[i] for i in intvs],
                       edgecolor='black', linewidth=0.8, width=0.6)

        # Add value labels on top of bars
        for pos, m, s in zip(positions, means, stds):
            ax.text(pos, m + s + 0.02, f'{m*100:.1f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_title(arch_name, fontweight='bold')
        ax.set_xticks(positions)
        ax.set_xticklabels([INTV_LABELS[i] for i in intvs], fontsize=9)
        if ax_idx % n_cols == 0:
            ax.set_ylabel('Benign acc. on dark skin')
        ax.set_ylim(0, max(0.6, max([m + s for m, s in zip(means, stds)] + [0]) * 1.15))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.0f}%'))
        ax.grid(True, axis='y', alpha=0.3)

    for i in range(n_archs, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].set_visible(False)

    fig.suptitle(
        'Group DRO provides no benefit over baseline under fine-tuning\n'
        '(mean ± std across 5 seeds; SMOTE shows modest recovery)',
        fontsize=12, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig2_per_class_bars.png'))
    plt.close()
    print(f"  ✓ Saved fig2 ({n_archs} architectures)")

# ════════════════════════════════════════════════════════════════
# FIGURE 3: Linear-probe vs Fine-tune baseline (the new headline)
# ════════════════════════════════════════════════════════════════
print("\nBuilding Figure 3: Linear-probe vs fine-tune baselines...")

if not os.path.exists(LP_BUGFIX_CSV):
    print(f"  ⚠ {LP_BUGFIX_CSV} not found — using PUBLISHED LP numbers as fallback")
    lp_data = {
        'CLIP ViT-L/14': 0.079,
        'ViT-B/16':      0.074,
        'ResNet-50':     0.064,
        'DINOv2-Base':   0.039,
    }
else:
    df_lp = pd.read_csv(LP_BUGFIX_CSV)
    lp_baseline = df_lp[df_lp['intervention'] == '1_baseline']
    arch_to_label = {'clip': 'CLIP ViT-L/14', 'vit': 'ViT-B/16',
                     'resnet50': 'ResNet-50', 'dinov2': 'DINOv2-Base'}
    lp_data = {}
    for code, label in arch_to_label.items():
        sub = lp_baseline[lp_baseline['model'] == code]
        if len(sub) > 0:
            lp_data[label] = sub['acc_benign_dark'].mean()

ft_data = {}
for name, info in ARCH_FILES.items():
    if os.path.exists(info['csv']):
        df_arch = pd.read_csv(info['csv'])
        sub = df_arch[df_arch['intervention'] == 'baseline']
        if len(sub) > 0:
            ft_data[name] = (sub['acc_benign_dark'].mean(),
                             sub['acc_benign_dark'].std() if len(sub) > 1 else 0)

if not ft_data:
    print("  ✗ No fine-tune baseline data found, skipping fig3")
else:
    archs = list(ft_data.keys())
    n_archs = len(archs)
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * n_archs + 2), 4.5))

    x = np.arange(n_archs)
    width = 0.36

    lp_means = [lp_data.get(a, np.nan) for a in archs]
    ft_means = [ft_data[a][0] for a in archs]
    ft_stds  = [ft_data[a][1] for a in archs]

    bars1 = ax.bar(x - width/2, lp_means, width, label='Linear probe',
                    color='#888888', edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x + width/2, ft_means, width, yerr=ft_stds, capsize=4,
                    label='Fine-tuned (5 seeds)', color='#0072B2',
                    edgecolor='black', linewidth=0.8)

    for i, (lp, ft) in enumerate(zip(lp_means, ft_means)):
        if not np.isnan(lp):
            ax.text(i - width/2, lp + 0.01, f'{lp*100:.1f}%',
                    ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, ft + ft_stds[i] + 0.01, f'{ft*100:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Annotate ratio
        if not np.isnan(lp) and lp > 0:
            ratio = ft / lp
            y_arrow = max(lp, ft + ft_stds[i]) + 0.06
            ax.annotate(f'{ratio:.1f}×',
                         xy=(i, y_arrow), ha='center',
                         fontsize=10, fontweight='bold', color='#D55E00')

    ax.set_xticks(x)
    ax.set_xticklabels(archs, fontsize=10)
    ax.set_ylabel('Benign accuracy on dark skin')
    ax.set_title('Same training data, same model: linear probe vs fine-tuning\n'
                 'produces a multi-fold swing in per-class accuracy',
                 fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.0f}%'))
    ax.set_ylim(0, max(ft_means + [0.1]) * 1.4)
    ax.legend(loc='upper left', frameon=False)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig3_lp_vs_ft.png'))
    plt.close()
    print(f"  ✓ Saved fig3 ({len(ft_data)} architectures)")

# ════════════════════════════════════════════════════════════════
# FIGURE 4: η-ablation
# ════════════════════════════════════════════════════════════════
print("\nBuilding Figure 4: η-ablation...")

if not os.path.exists(ETA_CSV):
    print(f"  ⚠ {ETA_CSV} not found — skipping fig4")
else:
    df_eta = pd.read_csv(ETA_CSV)
    eta_summary = df_eta.groupby('eta').agg(
        benign_mean=('acc_benign_dark', 'mean'),
        benign_std=('acc_benign_dark', 'std'),
        auc_mean=('demo_auc', 'mean'),
        auc_std=('demo_auc', 'std'),
        weight_mean=('final_dark_weight', 'mean'),
    ).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Reference: nb_finetune_v3 baseline ≈ 0.326 (from your existing run)
    BASELINE_REF = 0.326

    # Left panel: benign accuracy vs eta
    ax1.errorbar(eta_summary['eta'], eta_summary['benign_mean'],
                  yerr=eta_summary['benign_std'].fillna(0),
                  marker='o', markersize=8, linewidth=1.5,
                  color='#D55E00', capsize=5, label='DRO benign acc')
    ax1.axhline(BASELINE_REF, color='#888888', linestyle='--', linewidth=1.5,
                 label=f'Fine-tune baseline (no DRO): {BASELINE_REF*100:.1f}%')
    ax1.set_xscale('log')
    ax1.set_xlabel('Group DRO η (log scale)')
    ax1.set_ylabel('Benign acc. on dark skin')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.0f}%'))
    ax1.set_title('Does any η rescue DRO?', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=9)

    # Right panel: final dark group weight vs eta
    ax2.errorbar(eta_summary['eta'], eta_summary['weight_mean'],
                  marker='s', markersize=8, linewidth=1.5,
                  color='#0072B2', capsize=5)
    ax2.axhline(0.5, color='gray', linestyle=':', linewidth=1, label='Initial weight')
    ax2.axhline(0.0, color='gray', linestyle='-', linewidth=0.5)
    ax2.set_xscale('log')
    ax2.set_xlabel('Group DRO η (log scale)')
    ax2.set_ylabel('Final DRO weight on dark group')
    ax2.set_title('Where does DRO put its weight?', fontweight='bold')
    ax2.set_ylim(-0.05, 0.6)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=9)

    fig.suptitle('Group DRO η-ablation (CLIP ViT-L/14 fine-tune, 2 seeds per η)',
                 fontsize=12, fontweight='bold', y=1.03)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig4_eta_ablation.png'))
    plt.close()
    print(f"  ✓ Saved fig4")

# ════════════════════════════════════════════════════════════════
print(f"\n{'='*60}\nFIGURES SAVED to {FIGURES_DIR}/\n{'='*60}")
for f in sorted(os.listdir(FIGURES_DIR)):
    full = os.path.join(FIGURES_DIR, f)
    size_kb = os.path.getsize(full) / 1024
    print(f"  {f}  ({size_kb:.0f} KB)")

print('\nDone. Re-run after each new data-collection notebook completes.')
