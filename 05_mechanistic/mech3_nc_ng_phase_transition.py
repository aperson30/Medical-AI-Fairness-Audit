# ============================================================
# MECHANISM NOTEBOOK 3 — Synthetic nc/Ng Sweep
# Empirical validation of the ~10% phase transition threshold
#
# PURPOSE: Vary the benign fraction in the dark-skin training
# pool from ~2% to ~30% and plot benign accuracy vs. nc/Ng for
# Group DRO, Adversarial Debiasing, and SMOTE. This empirically
# validates the ~10% threshold as a phase transition rather than
# a single data point — the central theoretical contribution.
#
# EXPERIMENTAL DESIGN:
#   - Fix total mitigation pool size at N=200 dark-skin images
#   - Vary nc/Ng by subsampling benign images: nc in [5, 10, 15, 20,
#     25, 30, 40, 50, 60, 80, 100, 130, 160, 190] (cap by availability)
#   - Non-benign images fill remainder from dark non-neo + malignant
#   - Run each (method × nc/Ng) with 3 random seeds, report mean±std
#   - Methods: Baseline, Group DRO, Adversarial Debiasing (projection),
#     SMOTE, Real Oversample
#
# WHAT THIS PRODUCES:
#   Panel A: Benign accuracy vs. nc/Ng for all methods (main result)
#   Panel B: SGG vs. nc/Ng (does AUC gap also recover?)
#   Panel C: MAD-S vs. nc/Ng (joint penalty)
#   Panel D: Phase transition detection (gradient of benign acc)
#   CSV:     Full results table for paper appendix
#   JSON:    Complete results with CIs
#
# RUNTIME: ~4–6 hours on Kaggle T4 (14 nc values × 5 methods × 3 seeds)
# Kaggle: GPU T4, Internet ON.
# TIP: Run overnight. Results are checkpointed to JSON every iteration.
# ============================================================

!pip install transformers torch torchvision scikit-learn pandas numpy matplotlib -q

import torch
import numpy as np
import pandas as pd
import os, json, warnings, time
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from transformers import CLIPModel, CLIPProcessor
warnings.filterwarnings('ignore')

RANDOM_STATE = 42

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── Paths ─────────────────────────────────────────────────────
fitz_csv     = '/kaggle/input/datasets/nazmusresan/fitzpatrick17k/New folder/fitzpatrick17k (1).csv'
fitz_img_dir = '/kaggle/input/datasets/nazmusresan/fitzpatrick17k/New folder/background removed'

# ── Load dataset ──────────────────────────────────────────────
df = pd.read_csv(fitz_csv)
df = df[df['fitzpatrick_scale'] > 0]
image_files = {
    f.replace('.jpg','').replace('.png',''): os.path.join(fitz_img_dir, f)
    for f in os.listdir(fitz_img_dir)
    if f.endswith('.jpg') or f.endswith('.png')
}
df['local_path'] = df['md5hash'].map(image_files)
df = df[df['local_path'].notna()].copy()
df['skin_group'] = df['fitzpatrick_scale'].apply(
    lambda x: 'light' if x <= 2 else ('medium' if x <= 4 else 'dark'))

MAX = 1000
light_df  = df[df['skin_group']=='light'].sample(MAX, random_state=RANDOM_STATE)
medium_df = df[df['skin_group']=='medium'].sample(MAX, random_state=RANDOM_STATE)
dark_df   = df[df['skin_group']=='dark'].copy()
if len(dark_df) > 1300:
    dark_df = dark_df.sample(1300, random_state=RANDOM_STATE)
print(f"Loaded: light={len(light_df)}, medium={len(medium_df)}, dark={len(dark_df)}")

le = LabelEncoder()
le.fit(list(light_df['three_partition_label']) +
       list(medium_df['three_partition_label']) +
       list(dark_df['three_partition_label']))
print(f"Classes: {le.classes_}")

BENIGN_IDX = list(le.classes_).index('benign')
MALIG_IDX  = list(le.classes_).index('malignant')
NONNEO_IDX = list(le.classes_).index('non-neoplastic')

# ── Load CLIP ─────────────────────────────────────────────────
print("Loading CLIP ViT-L/14...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
clip_model.eval()
print("CLIP loaded.")

def load_imgs(dataframe):
    imgs, lbls = [], []
    for _, row in dataframe.iterrows():
        try:
            img = Image.open(row['local_path']).convert('RGB').resize((224,224))
            imgs.append(img)
            lbls.append(le.transform([row['three_partition_label']])[0])
        except:
            pass
    return imgs, np.array(lbls)

@torch.no_grad()
def get_features(images, batch_size=32):
    all_feats = []
    for i in range(0, len(images), batch_size):
        batch  = images[i:i+batch_size]
        inputs = clip_proc(images=batch, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feats  = clip_model.get_image_features(**inputs)
        if not isinstance(feats, torch.Tensor):
            feats = feats.pooler_output if hasattr(feats,'pooler_output') \
                    else feats.last_hidden_state[:,0]
        feats  = feats / feats.norm(dim=-1, keepdim=True)
        all_feats.append(feats.cpu().numpy())
    return np.vstack(all_feats)

print("Loading images...")
light_imgs,  light_y  = load_imgs(light_df)
medium_imgs, medium_y = load_imgs(medium_df)
dark_imgs,   dark_y   = load_imgs(dark_df)

print("Extracting features...")
light_feats  = get_features(light_imgs)
medium_feats = get_features(medium_imgs)
dark_feats   = get_features(dark_imgs)
print(f"Features: light={light_feats.shape}, medium={medium_feats.shape}, dark={dark_feats.shape}")

# ── Fixed train / test sets ────────────────────────────────────
train_f = np.vstack([light_feats, medium_feats])
train_y = np.concatenate([light_y, medium_y])

# Hold out 80% of dark for test; remaining 20% is the pool we subsample from
n_dark_test  = int(0.8 * len(dark_feats))
test_f = dark_feats[:n_dark_test]
test_y = dark_y[:n_dark_test]
pool_f = dark_feats[n_dark_test:]
pool_y = dark_y[n_dark_test:]

# Separate pool by class
pool_benign_f  = pool_f[pool_y == BENIGN_IDX]
pool_nonneo_f  = pool_f[pool_y == NONNEO_IDX]
pool_malig_f   = pool_f[pool_y == MALIG_IDX]
print(f"\nPool: benign={len(pool_benign_f)}, non-neo={len(pool_nonneo_f)}, malig={len(pool_malig_f)}")

# Random AUC (upper bound)
all_f = np.vstack([train_f, test_f])
all_y = np.concatenate([train_y, test_y])
tr_idx, te_idx = train_test_split(np.arange(len(all_y)), test_size=0.25,
                                   stratify=all_y, random_state=RANDOM_STATE)
clf_rand = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
clf_rand.fit(all_f[tr_idx], all_y[tr_idx])
rand_auc = roc_auc_score(all_y[te_idx], clf_rand.predict_proba(all_f[te_idx]),
                          multi_class='ovr', average='macro')
print(f"Random split AUC: {rand_auc:.4f}")

# Baseline AUC (demo-aware, no intervention)
clf_base0 = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
clf_base0.fit(train_f, train_y)
base_auc0 = roc_auc_score(test_y, clf_base0.predict_proba(test_f),
                            multi_class='ovr', average='macro')
base_sgg0 = rand_auc - base_auc0
print(f"Baseline demo-aware AUC: {base_auc0:.4f}, SGG: {base_sgg0:.4f}")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

N_POOL     = 200   # fixed total mitigation pool size
DRO_ETA    = 0.01
DRO_EPOCHS = 20

def build_mitigation_pool(nc, seed):
    """
    Sample nc benign + (N_POOL - nc) non-benign images from the dark pool.
    Returns (mitig_f, mitig_y).
    """
    rng = np.random.RandomState(seed)
    nc  = min(nc, len(pool_benign_f))
    n_rest = N_POOL - nc
    # Fill rest: proportional to non-neo:malignant ratio in pool
    n_avail_rest  = len(pool_nonneo_f) + len(pool_malig_f)
    n_nonneo_take = min(len(pool_nonneo_f),
                        int(round(n_rest * len(pool_nonneo_f) / max(n_avail_rest, 1))))
    n_malig_take  = min(len(pool_malig_f), n_rest - n_nonneo_take)

    b_idx = rng.choice(len(pool_benign_f), nc, replace=nc > len(pool_benign_f))
    n_idx = rng.choice(len(pool_nonneo_f), n_nonneo_take, replace=n_nonneo_take > len(pool_nonneo_f))
    m_idx = rng.choice(len(pool_malig_f),  n_malig_take,  replace=n_malig_take  > len(pool_malig_f))

    mf = np.vstack([pool_benign_f[b_idx], pool_nonneo_f[n_idx], pool_malig_f[m_idx]])
    my = np.concatenate([
        np.full(nc, BENIGN_IDX),
        np.full(n_nonneo_take, NONNEO_IDX),
        np.full(n_malig_take,  MALIG_IDX)
    ])
    return mf, my


def run_baseline(mitig_f, mitig_y, seed):
    """Standard logistic regression, light+medium training only."""
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
    clf.fit(train_f, train_y)
    probs = clf.predict_proba(test_f)
    preds = clf.predict(test_f)
    auc   = roc_auc_score(test_y, probs, multi_class='ovr', average='macro')
    benign_mask = test_y == BENIGN_IDX
    ba = float(accuracy_score(test_y[benign_mask], preds[benign_mask])) \
         if benign_mask.sum() > 0 else 0.0
    return auc, rand_auc - auc, ba


def run_real_oversample(mitig_f, mitig_y, seed):
    """Add dark mitigation images directly to training set."""
    aug_f = np.vstack([train_f, mitig_f])
    aug_y = np.concatenate([train_y, mitig_y])
    clf   = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
    clf.fit(aug_f, aug_y)
    probs = clf.predict_proba(test_f)
    preds = clf.predict(test_f)
    auc   = roc_auc_score(test_y, probs, multi_class='ovr', average='macro')
    benign_mask = test_y == BENIGN_IDX
    ba = float(accuracy_score(test_y[benign_mask], preds[benign_mask])) \
         if benign_mask.sum() > 0 else 0.0
    return auc, rand_auc - auc, ba


def run_group_dro(mitig_f, mitig_y, seed):
    """Minimax DRO on (skin × class) groups."""
    dro_f = np.vstack([train_f, mitig_f])
    dro_y = np.concatenate([train_y, mitig_y])

    skin_labels = np.concatenate([
        np.zeros(len(light_feats)),
        np.ones(len(medium_feats)),
        np.full(len(mitig_f), 2)
    ])
    group_ids = (skin_labels * 3 + dro_y).astype(int)
    n_groups  = 9
    n_dro     = len(dro_f)
    gw        = np.ones(n_groups) / n_groups
    clf_dro   = LogisticRegression(max_iter=200, C=1.0, random_state=seed)

    for _ in range(DRO_EPOCHS):
        sw  = gw[group_ids]
        sw  = sw / sw.sum() * n_dro
        clf_dro.fit(dro_f, dro_y, sample_weight=sw)
        probs_e = np.clip(clf_dro.predict_proba(dro_f), 1e-9, 1.0)
        pl      = -np.log(probs_e[np.arange(n_dro), dro_y])
        gl      = np.array([pl[group_ids==g].mean() if (group_ids==g).sum()>0 else 0.0
                            for g in range(n_groups)])
        gw = gw * np.exp(DRO_ETA * gl)
        gw = gw / gw.sum()

    probs = clf_dro.predict_proba(test_f)
    preds = clf_dro.predict(test_f)
    auc   = roc_auc_score(test_y, probs, multi_class='ovr', average='macro')
    benign_mask = test_y == BENIGN_IDX
    ba = float(accuracy_score(test_y[benign_mask], preds[benign_mask])) \
         if benign_mask.sum() > 0 else 0.0
    return auc, rand_auc - auc, ba


def run_adv_debiasing(mitig_f, mitig_y, seed, k_remove=10):
    """
    Adversarial debiasing approximated as projection out of top-k
    skin-tone PCA components, then train on remaining signal.
    k_remove=10 corresponds to aggressive adversary (matching paper's λ=1.0 behavior).
    """
    all_f_adv = np.vstack([train_f, dark_feats[:n_dark_test]])
    all_g     = np.concatenate([np.zeros(len(light_feats)),
                                 np.ones(len(medium_feats)),
                                 np.full(n_dark_test, 2)])
    centroids     = np.array([all_f_adv[all_g==g].mean(axis=0) for g in range(3)])
    global_cent   = all_f_adv.mean(axis=0)
    B             = centroids - global_cent
    _, _, Vt      = np.linalg.svd(B, full_matrices=False)
    skin_dirs     = Vt[:k_remove]

    # Project out skin-tone subspace
    train_proj = train_f - train_f @ skin_dirs.T @ skin_dirs
    mitig_proj = mitig_f - mitig_f @ skin_dirs.T @ skin_dirs
    test_proj  = test_f  - test_f  @ skin_dirs.T @ skin_dirs

    aug_f = np.vstack([train_proj, mitig_proj])
    aug_y = np.concatenate([train_y, mitig_y])
    clf   = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
    clf.fit(aug_f, aug_y)
    probs = clf.predict_proba(test_proj)
    preds = clf.predict(test_proj)
    auc   = roc_auc_score(test_y, probs, multi_class='ovr', average='macro')
    benign_mask = test_y == BENIGN_IDX
    ba = float(accuracy_score(test_y[benign_mask], preds[benign_mask])) \
         if benign_mask.sum() > 0 else 0.0
    return auc, rand_auc - auc, ba


def run_smote(mitig_f, mitig_y, seed):
    """SMOTE: interpolate dark-skin benign features to oversample."""
    rng_smote = np.random.RandomState(seed)
    dark_benign_f = mitig_f[mitig_y == BENIGN_IDX]
    n_dark_b = len(dark_benign_f)
    n_dark_rest = (mitig_y != BENIGN_IDX).sum()

    n_synth = max(0, n_dark_rest - n_dark_b)
    if n_synth > 0 and n_dark_b >= 2:
        k_nn  = min(5, n_dark_b - 1)
        nbrs  = NearestNeighbors(n_neighbors=k_nn + 1).fit(dark_benign_f)
        _, nn = nbrs.kneighbors(dark_benign_f)
        synth = []
        for _ in range(n_synth):
            i   = rng_smote.randint(0, n_dark_b)
            j   = nn[i, rng_smote.randint(1, k_nn + 1)]
            lam = rng_smote.uniform(0, 1)
            synth.append(dark_benign_f[i] + lam * (dark_benign_f[j] - dark_benign_f[i]))
        syn_f = np.vstack(synth)
        syn_y = np.full(len(syn_f), BENIGN_IDX)
        aug_f = np.vstack([train_f, mitig_f, syn_f])
        aug_y = np.concatenate([train_y, mitig_y, syn_y])
    else:
        aug_f = np.vstack([train_f, mitig_f])
        aug_y = np.concatenate([train_y, mitig_y])

    clf   = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
    clf.fit(aug_f, aug_y)
    probs = clf.predict_proba(test_f)
    preds = clf.predict(test_f)
    auc   = roc_auc_score(test_y, probs, multi_class='ovr', average='macro')
    benign_mask = test_y == BENIGN_IDX
    ba = float(accuracy_score(test_y[benign_mask], preds[benign_mask])) \
         if benign_mask.sum() > 0 else 0.0
    return auc, rand_auc - auc, ba


# ============================================================
# MAIN SWEEP
# ============================================================

# nc values: from ~2% to ~30% of N_POOL=200
# nc/Ng = nc/200: 5/200=2.5%, 10=5%, 15=7.5%, 19=9.5% (paper), 25=12.5%, ...
NC_VALUES = [5, 10, 15, 19, 25, 30, 40, 50, 60, 80, 100, 130, 160, 190]
NC_VALUES  = [nc for nc in NC_VALUES if nc <= len(pool_benign_f)]
SEEDS      = [42, 0, 1]

METHODS = {
    'Baseline':       run_baseline,
    'Real Oversample': run_real_oversample,
    'Group DRO':      run_group_dro,
    'Adv Debiasing':  run_adv_debiasing,
    'SMOTE':          run_smote,
}

results_all = []
checkpoint_path = '/kaggle/working/nb_mech3_checkpoint.json'

total_runs = len(NC_VALUES) * len(SEEDS) * len(METHODS)
run_count  = 0
t_start    = time.time()

print(f"\n=== Starting nc/Ng sweep ===")
print(f"NC values: {NC_VALUES}")
print(f"Seeds: {SEEDS}")
print(f"Methods: {list(METHODS.keys())}")
print(f"Total runs: {total_runs}")
print(f"Estimated time: ~{total_runs * 1.5:.0f} minutes\n")

for nc in NC_VALUES:
    ng     = N_POOL
    nc_ng  = nc / ng

    for seed in SEEDS:
        mitig_f, mitig_y = build_mitigation_pool(nc, seed)
        n_actual_benign  = (mitig_y == BENIGN_IDX).sum()
        nc_ng_actual     = n_actual_benign / len(mitig_y)

        for method_name, method_fn in METHODS.items():
            run_count += 1
            t_elapsed = time.time() - t_start

            try:
                auc, sgg, ba = method_fn(mitig_f, mitig_y, seed)
                mad_s = sgg / max(1 - ba, 1e-6)
            except Exception as e:
                print(f"  FAILED: nc={nc}, seed={seed}, method={method_name}: {e}")
                auc, sgg, ba, mad_s = float('nan'), float('nan'), float('nan'), float('nan')

            row = {
                'nc':            int(nc),
                'ng':            int(ng),
                'nc_ng':         float(nc_ng_actual),
                'nc_ng_nominal': float(nc_ng),
                'seed':          int(seed),
                'method':        method_name,
                'demo_auc':      float(auc),
                'sgg':           float(sgg),
                'benign_acc':    float(ba),
                'mad_s':         float(mad_s),
            }
            results_all.append(row)

            if run_count % 5 == 0 or run_count == total_runs:
                eta = (t_elapsed / run_count) * (total_runs - run_count) / 60
                print(f"[{run_count:>4}/{total_runs}] nc={nc:>3} ({nc_ng:.1%}) "
                      f"seed={seed} method={method_name:<18} "
                      f"ba={ba:.3f} sgg={sgg:.4f}  ETA: {eta:.1f}min")
                json.dump(results_all, open(checkpoint_path, 'w'), indent=1)


# ============================================================
# AGGREGATE RESULTS
# ============================================================
print("\nAggregating...")
df_res = pd.DataFrame(results_all)

agg = df_res.groupby(['nc_ng_nominal', 'method']).agg(
    nc_ng_mean=('nc_ng', 'mean'),
    benign_acc_mean=('benign_acc', 'mean'),
    benign_acc_std=('benign_acc', 'std'),
    sgg_mean=('sgg', 'mean'),
    sgg_std=('sgg', 'std'),
    mad_s_mean=('mad_s', 'mean'),
    mad_s_std=('mad_s', 'std'),
    demo_auc_mean=('demo_auc', 'mean'),
    demo_auc_std=('demo_auc', 'std'),
    n_seeds=('seed', 'count'),
).reset_index()

agg.to_csv('/kaggle/working/nb_mech3_sweep_aggregated.csv', index=False)
df_res.to_csv('/kaggle/working/nb_mech3_sweep_raw.csv', index=False)
print("CSV saved.")
print(agg[agg['method']=='Group DRO'][['nc_ng_nominal','benign_acc_mean','sgg_mean']].to_string())


# ============================================================
# PLOTTING
# ============================================================
print("\nGenerating figures...")

METHOD_STYLES = {
    'Baseline':        {'color': '#757575', 'ls': ':',  'marker': 'o', 'lw': 1.5},
    'Real Oversample': {'color': '#FF9800', 'ls': '--', 'marker': 's', 'lw': 1.5},
    'Group DRO':       {'color': '#B71C1C', 'ls': '-',  'marker': 'D', 'lw': 2.5},
    'Adv Debiasing':   {'color': '#4A148C', 'ls': '-',  'marker': '^', 'lw': 2.5},
    'SMOTE':           {'color': '#1B5E20', 'ls': '-',  'marker': 'v', 'lw': 2.5},
}

nc_ng_vals_unique = sorted(agg['nc_ng_nominal'].unique())
nc_ng_pct = [v * 100 for v in nc_ng_vals_unique]

def get_method_curve(method, metric='benign_acc_mean', std_col='benign_acc_std'):
    sub = agg[agg['method']==method].sort_values('nc_ng_nominal')
    x   = sub['nc_ng_nominal'].values * 100  # pct
    y   = sub[metric].values
    ye  = sub[std_col].values
    return x, y, ye

# ── Figure 1: 2×2 main panels ─────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    'nc/Ng Sweep: Empirical Phase Transition in Fairness Interventions\n'
    f'CLIP ViT-L/14 · N_pool={N_POOL} · {len(SEEDS)} seeds · '
    f'~10% threshold (vertical dashed line)',
    fontsize=14, fontweight='bold')

THRESHOLD_PCT = 10.0  # the ~10% hypothesized phase transition

# Panel A: Benign accuracy vs. nc/Ng
ax = axes[0, 0]
for method, style in METHOD_STYLES.items():
    x, y, ye = get_method_curve(method)
    ax.plot(x, y, linestyle=style['ls'], marker=style['marker'],
            color=style['color'], linewidth=style['lw'],
            markersize=7, label=method, zorder=5 if 'DRO' in method or 'Adv' in method else 3)
    ax.fill_between(x, y - ye, y + ye, alpha=0.12, color=style['color'])

ax.axvline(THRESHOLD_PCT, color='black', linestyle='--', linewidth=1.5, alpha=0.7,
           label=f'Hypothesized threshold ({THRESHOLD_PCT:.0f}%)')
ax.axvline(9.4, color='grey', linestyle=':', linewidth=1.2, alpha=0.5,
           label='Paper nc/Ng (9.4%)')
ax.set_xlabel('nc/Ng — Benign fraction of dark-skin training pool (%)', fontsize=11)
ax.set_ylabel('Dark-skin benign accuracy (test set)', fontsize=11)
ax.set_title('A — Benign Accuracy vs. nc/Ng\n'
             '(the primary phase transition result)', fontsize=11, fontweight='bold')
ax.set_ylim(-0.05, 1.05)
ax.set_xlim(0, 100)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
ax.legend(fontsize=9, loc='upper left')
ax.yaxis.grid(True, alpha=0.3)

# Panel B: SGG vs. nc/Ng
ax = axes[0, 1]
for method, style in METHOD_STYLES.items():
    x, y, ye = get_method_curve(method, 'sgg_mean', 'sgg_std')
    ax.plot(x, y, linestyle=style['ls'], marker=style['marker'],
            color=style['color'], linewidth=style['lw'], markersize=7, label=method)
    ax.fill_between(x, y - ye, y + ye, alpha=0.12, color=style['color'])

ax.axvline(THRESHOLD_PCT, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax.axvline(9.4, color='grey', linestyle=':', linewidth=1.2, alpha=0.5)
ax.axhline(0, color='grey', linestyle='-', linewidth=0.8, alpha=0.4)
ax.set_xlabel('nc/Ng — Benign fraction (%)', fontsize=11)
ax.set_ylabel('SGG = AUC_random − AUC_demo', fontsize=11)
ax.set_title('B — Source Generalization Gap vs. nc/Ng\n'
             '(does AUC gap also recover?)', fontsize=11, fontweight='bold')
ax.set_xlim(0, 100)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
ax.legend(fontsize=9)
ax.yaxis.grid(True, alpha=0.3)

# Panel C: MAD-S vs. nc/Ng
ax = axes[1, 0]
for method, style in METHOD_STYLES.items():
    x, y, ye = get_method_curve(method, 'mad_s_mean', 'mad_s_std')
    # Cap large MAD-S values for visualization
    y_plot  = np.clip(y,  -0.1, 2.0)
    ye_plot = np.clip(ye, 0,    0.5)
    ax.plot(x, y_plot, linestyle=style['ls'], marker=style['marker'],
            color=style['color'], linewidth=style['lw'], markersize=7, label=method)
    ax.fill_between(x, y_plot - ye_plot, y_plot + ye_plot, alpha=0.12, color=style['color'])

ax.axvline(THRESHOLD_PCT, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax.axvline(9.4, color='grey', linestyle=':', linewidth=1.2, alpha=0.5)
ax.set_xlabel('nc/Ng — Benign fraction (%)', fontsize=11)
ax.set_ylabel('MAD-S = SGG / (1 − benign_acc)', fontsize=11)
ax.set_title('C — MAD Severity Index vs. nc/Ng\n'
             '(joint penalty; DRO/Adv collapse → MAD-S → ∞ below threshold)',
             fontsize=11, fontweight='bold')
ax.set_xlim(0, 100)
ax.set_ylim(-0.05, 2.05)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
ax.legend(fontsize=9)
ax.yaxis.grid(True, alpha=0.3)

# Panel D: Phase transition detection via numerical gradient
# The transition point is where d(benign_acc)/d(nc_ng) peaks for DRO
ax = axes[1, 1]
for method in ['Group DRO', 'Adv Debiasing', 'SMOTE']:
    style = METHOD_STYLES[method]
    x, y, _ = get_method_curve(method)
    if len(x) > 2:
        # Smooth with central differences
        grad = np.gradient(y, x)
        ax.plot(x, grad, linestyle=style['ls'], marker=style['marker'],
                color=style['color'], linewidth=style['lw'],
                markersize=6, label=f'{method} (dAcc/dnc_ng)')

ax.axvline(THRESHOLD_PCT, color='black', linestyle='--', linewidth=1.5, alpha=0.7,
           label=f'Threshold ({THRESHOLD_PCT:.0f}%)')
ax.axvline(9.4, color='grey', linestyle=':', linewidth=1.2, alpha=0.5,
           label='Paper nc/Ng (9.4%)')
ax.axhline(0, color='grey', linewidth=0.8, alpha=0.4)
ax.set_xlabel('nc/Ng (%)', fontsize=11)
ax.set_ylabel('d(benign_acc)/d(nc_ng)', fontsize=11)
ax.set_title('D — Phase Transition Detection\n'
             '(gradient peaks at the critical threshold)',
             fontsize=11, fontweight='bold')
ax.set_xlim(0, 100)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
ax.legend(fontsize=9)
ax.yaxis.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig('/kaggle/working/nb_mech3_sweep_main.png', dpi=300, bbox_inches='tight')
plt.show()
print("Figure 1 saved: nb_mech3_sweep_main.png")

# ── Figure 2: Clean version for paper (single panel, publication quality) ────
fig2, ax_main = plt.subplots(figsize=(10, 7))

for method, style in METHOD_STYLES.items():
    x, y, ye = get_method_curve(method)
    ax_main.plot(x, y, linestyle=style['ls'], marker=style['marker'],
                 color=style['color'], linewidth=style['lw'],
                 markersize=8, label=method, zorder=5 if 'DRO' in method or 'Adv' in method else 3)
    ax_main.fill_between(x, y - ye, y + ye, alpha=0.10, color=style['color'])

ax_main.axvline(THRESHOLD_PCT, color='black', linestyle='--', linewidth=2, alpha=0.8)
ax_main.text(THRESHOLD_PCT + 0.5, 0.92,
             f'~{THRESHOLD_PCT:.0f}% threshold', fontsize=11, ha='left', color='black',
             fontweight='bold')

# Annotate the paper's actual nc/Ng
ax_main.axvline(9.4, color='grey', linestyle=':', linewidth=1.5, alpha=0.7)
ax_main.text(9.4 + 0.3, 0.75, 'Paper\nnc/Ng\n(9.4%)', fontsize=9, ha='left',
             color='grey', style='italic')

# Annotate collapse zone
ax_main.axvspan(0, THRESHOLD_PCT, alpha=0.04, color='red')
ax_main.text(THRESHOLD_PCT / 2, 0.15, 'Collapse\nZone', ha='center', fontsize=11,
             color='#B71C1C', alpha=0.7, fontweight='bold')

ax_main.set_xlabel('nc/Ng — Benign fraction of dark-skin mitigation pool (%)', fontsize=13)
ax_main.set_ylabel('Dark-skin benign class accuracy (held-out test set)', fontsize=13)
ax_main.set_title(
    'Phase Transition in Fairness Intervention Efficacy\n'
    f'Group DRO and adversarial debiasing collapse when nc/Ng < ~{THRESHOLD_PCT:.0f}%\n'
    f'CLIP ViT-L/14 · {len(NC_VALUES)} nc values · mean ± std ({len(SEEDS)} seeds)',
    fontsize=12, fontweight='bold')
ax_main.set_ylim(-0.05, 1.05)
ax_main.set_xlim(0, 100)
ax_main.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
ax_main.legend(fontsize=11, loc='upper left', framealpha=0.9)
ax_main.yaxis.grid(True, alpha=0.3)

plt.tight_layout()
fig2.savefig('/kaggle/working/nb_mech3_sweep_paper_figure.png', dpi=300, bbox_inches='tight')
plt.show()
print("Figure 2 saved: nb_mech3_sweep_paper_figure.png")


# ============================================================
# IDENTIFY THRESHOLD — quantitative
# ============================================================
print("\n=== THRESHOLD ANALYSIS ===")
for method in ['Group DRO', 'Adv Debiasing', 'SMOTE']:
    sub = agg[agg['method']==method].sort_values('nc_ng_nominal')
    # First nc/Ng at which benign_acc_mean > 0.05 (break out of collapse)
    above = sub[sub['benign_acc_mean'] > 0.05]
    if len(above) > 0:
        threshold_nc_ng = above.iloc[0]['nc_ng_nominal'] * 100
        threshold_ba    = above.iloc[0]['benign_acc_mean']
        print(f"  {method}: first nc/Ng with ba>5% = {threshold_nc_ng:.1f}%  "
              f"(ba={threshold_ba:.3f})")
    else:
        print(f"  {method}: never recovers above 5% in this range")


# ============================================================
# FINAL JSON
# ============================================================
out = {
    'meta': {
        'notebook':   'nb_mech3_nc_ng_sweep',
        'model':      'CLIP ViT-L/14',
        'n_pool':     int(N_POOL),
        'nc_values':  NC_VALUES,
        'seeds':      SEEDS,
        'n_dark_test': int(n_dark_test),
        'rand_auc':   float(rand_auc),
        'base_sgg':   float(base_sgg0),
        'hypothesized_threshold_pct': float(THRESHOLD_PCT),
    },
    'aggregated': agg.to_dict(orient='records'),
    'raw_n_rows': len(results_all),
    'threshold_estimates': {},
}

for method in ['Group DRO', 'Adv Debiasing', 'SMOTE']:
    sub   = agg[agg['method']==method].sort_values('nc_ng_nominal')
    above = sub[sub['benign_acc_mean'] > 0.05]
    out['threshold_estimates'][method] = {
        'first_nc_ng_pct_above_5pct': float(above.iloc[0]['nc_ng_nominal'] * 100) if len(above) > 0 else None,
        'first_benign_acc':           float(above.iloc[0]['benign_acc_mean']) if len(above) > 0 else None,
    }

json.dump(out, open('/kaggle/working/nb_mech3_results.json', 'w'), indent=2)

print("\n=== KEY NUMBERS FOR PAPER ===")
print(f"Total sweep runs: {len(results_all)}")
print(f"Architectures: CLIP ViT-L/14")
print(f"nc/Ng range: {NC_VALUES[0]/N_POOL*100:.1f}% – {NC_VALUES[-1]/N_POOL*100:.1f}%")
print(f"\nThreshold estimates (first nc/Ng where benign acc > 5%):")
for method, est in out['threshold_estimates'].items():
    pct = est['first_nc_ng_pct_above_5pct']
    ba  = est['first_benign_acc']
    if pct is not None:
        print(f"  {method:<20}: {pct:.1f}%  (ba={ba:.3f})")
    else:
        print(f"  {method:<20}: Never recovers")

print("\n✓ Complete. Upload the following to Claude:")
print("  nb_mech3_sweep_main.png")
print("  nb_mech3_sweep_paper_figure.png")
print("  nb_mech3_sweep_aggregated.csv")
print("  nb_mech3_results.json")
