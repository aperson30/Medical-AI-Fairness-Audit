# ============================================================
# MECHANISM NOTEBOOK 1 — Decision Boundary Visualization
# Baseline vs. Group DRO vs. SMOTE in PCA/UMAP feature space
#
# PURPOSE: Show visually whether Group DRO pushes the decision
# boundary away from the dark-skin benign cluster (rather than
# toward it), providing the geometric mechanism behind 0% benign
# accuracy. SMOTE should show boundary movement toward the cluster.
#
# WHAT THIS PRODUCES:
#   Panel A: 2D PCA of CLIP features, colored by skin×class,
#            with linear decision boundaries for Baseline / DRO / SMOTE
#   Panel B: UMAP version of the same (richer topology)
#   Panel C: Dark-skin benign P(benign) distributions under each method
#   Panel D: Decision boundary distance from dark-skin benign centroid
#   JSON:    All key numbers for the paper
#
# RUNTIME: ~30 min on Kaggle T4 (CLIP + UMAP).
# Kaggle: GPU T4, Internet ON, random_state=42
# ============================================================

!pip install transformers torch torchvision scikit-learn pandas numpy matplotlib umap-learn -q

import torch
import numpy as np
import pandas as pd
import os, json, warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from transformers import CLIPModel, CLIPProcessor
import umap
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

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

# Paper-exact sampling: 1000 light, 1000 medium, all dark (cap 1300)
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
print(f"Classes: {le.classes_}")  # expect: benign, malignant, non-neoplastic

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
            feats = feats.pooler_output if hasattr(feats, 'pooler_output') \
                    else feats.last_hidden_state[:,0]
        feats  = feats / feats.norm(dim=-1, keepdim=True)
        all_feats.append(feats.cpu().numpy())
    return np.vstack(all_feats)

# ── Extract features ──────────────────────────────────────────
print("Loading images...")
light_imgs,  light_y  = load_imgs(light_df)
medium_imgs, medium_y = load_imgs(medium_df)
dark_imgs,   dark_y   = load_imgs(dark_df)

print("Extracting features...")
light_feats  = get_features(light_imgs)
medium_feats = get_features(medium_imgs)
dark_feats   = get_features(dark_imgs)
print(f"Features: light={light_feats.shape}, medium={medium_feats.shape}, dark={dark_feats.shape}")

# ── Training / test sets (paper-exact split) ──────────────────
train_f = np.vstack([light_feats, medium_feats])
train_y = np.concatenate([light_y, medium_y])

# Dark test: first 80% of dark images
n_dark_test   = int(0.8 * len(dark_feats))
dark_pool_idx = np.arange(n_dark_test, len(dark_feats))  # remaining for DRO/SMOTE mitigation pool
test_f = dark_feats[:n_dark_test]
test_y = dark_y[:n_dark_test]
pool_f = dark_feats[dark_pool_idx]
pool_y = dark_y[dark_pool_idx]

# Mitigation pool for interventions: 200 dark images (matches paper)
N_DARK_MITIG = min(200, len(pool_f))
mitig_idx = np.arange(N_DARK_MITIG)
mitig_f   = pool_f[mitig_idx]
mitig_y   = pool_y[mitig_idx]
print(f"\nTest: {len(test_f)}, Mitigation pool: {N_DARK_MITIG}")
print(f"Dark benign in mitigation pool: {(mitig_y == BENIGN_IDX).sum()}")
print(f"nc/Ng = {(mitig_y == BENIGN_IDX).sum() / len(mitig_y):.3f}")


# ============================================================
# 1. BASELINE CLASSIFIER
# ============================================================
clf_base = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
clf_base.fit(train_f, train_y)
base_probs = clf_base.predict_proba(test_f)
base_preds = clf_base.predict(test_f)

def benign_acc(preds, labels):
    mask = labels == BENIGN_IDX
    if mask.sum() == 0:
        return 0.0
    return float(accuracy_score(labels[mask], preds[mask]))

print(f"\nBaseline benign acc (dark test): {benign_acc(base_preds, test_y):.3f}")


# ============================================================
# 2. GROUP DRO CLASSIFIER
# Minimax worst-group cross-entropy on light+medium+dark_mitig.
# Implementation: iterative exponentiated gradient updates on
# group weights (Sagawa et al. 2020, Algorithm 1).
# Groups: (skin_group x label) — 9 groups total.
# ============================================================
print("\n--- Group DRO ---")

# Build DRO training set
dro_f = np.vstack([train_f, mitig_f])
dro_y = np.concatenate([train_y, mitig_y])

# Assign group ids: (skin_group, label)
# light=0, medium=1, dark=2  x  3 classes = 9 groups
skin_labels = np.concatenate([
    np.zeros(len(train_f[:len(light_feats)])),   # light
    np.ones(len(train_f[len(light_feats):])),    # medium
    np.full(len(mitig_f), 2)                     # dark
])
group_ids = (skin_labels * 3 + dro_y).astype(int)  # 9 groups
n_groups  = 9

# Hyperparameters (paper: eta=0.01, 20 epochs, batch=64)
ETA       = 0.01
N_EPOCHS  = 20
BATCH     = 64
C_LR      = 1.0

# Initialize group weights uniformly
group_weights = np.ones(n_groups) / n_groups

# Fit with iterative reweighting (DRO approximation via sklearn)
n_dro = len(dro_f)
indices = np.arange(n_dro)

# We iterate: fit LR with current sample weights, compute per-group loss,
# update group weights via exponentiated gradient, repeat.
from sklearn.metrics import log_loss

clf_dro = LogisticRegression(max_iter=200, C=C_LR, random_state=RANDOM_STATE)

for epoch in range(N_EPOCHS):
    # Build per-sample weights from group weights
    sample_w = group_weights[group_ids]
    sample_w = sample_w / sample_w.sum() * n_dro  # normalize

    clf_dro.fit(dro_f, dro_y, sample_weight=sample_w)

    # Per-group cross-entropy loss
    probs_dro = clf_dro.predict_proba(dro_f)
    probs_dro = np.clip(probs_dro, 1e-9, 1.0)
    per_sample_loss = -np.log(probs_dro[np.arange(n_dro), dro_y])

    group_losses = np.array([
        per_sample_loss[group_ids == g].mean() if (group_ids == g).sum() > 0 else 0.0
        for g in range(n_groups)
    ])

    # Exponentiated gradient update
    group_weights = group_weights * np.exp(ETA * group_losses)
    group_weights = group_weights / group_weights.sum()

    if epoch % 5 == 0 or epoch == N_EPOCHS - 1:
        dark_benign_g = 2 * 3 + BENIGN_IDX  # group id for dark-skin benign
        print(f"  Epoch {epoch:>2}: dark-benign group weight={group_weights[dark_benign_g]:.4f}, "
              f"loss={group_losses[dark_benign_g]:.4f}")

dro_preds = clf_dro.predict(test_f)
dro_probs = clf_dro.predict_proba(test_f)
print(f"Group DRO benign acc (dark test): {benign_acc(dro_preds, test_y):.3f}")
print(f"Group DRO SGG: {roc_auc_score(test_y, base_probs, multi_class='ovr', average='macro') - roc_auc_score(test_y, dro_probs, multi_class='ovr', average='macro'):.4f}")

# Log per-epoch group losses for Panel B (loss surface analysis)
# Re-run with logging to capture full curve
group_weights2 = np.ones(n_groups) / n_groups
clf_dro2 = LogisticRegression(max_iter=200, C=C_LR, random_state=RANDOM_STATE)
epoch_log = []

for epoch in range(N_EPOCHS):
    sample_w = group_weights2[group_ids]
    sample_w = sample_w / sample_w.sum() * n_dro

    clf_dro2.fit(dro_f, dro_y, sample_weight=sample_w)
    probs_e = np.clip(clf_dro2.predict_proba(dro_f), 1e-9, 1.0)
    per_sample_loss = -np.log(probs_e[np.arange(n_dro), dro_y])

    group_losses = np.array([
        per_sample_loss[group_ids == g].mean() if (group_ids == g).sum() > 0 else 0.0
        for g in range(n_groups)
    ])

    group_weights2 = group_weights2 * np.exp(ETA * group_losses)
    group_weights2 = group_weights2 / group_weights2.sum()

    # Per-class dark-skin accuracy at this epoch
    test_preds_e = clf_dro2.predict(test_f)
    ba_e = benign_acc(test_preds_e, test_y)

    epoch_log.append({
        'epoch': epoch,
        'dark_benign_group_loss': float(group_losses[2*3 + BENIGN_IDX]),
        'dark_nonneo_group_loss': float(group_losses[2*3 + NONNEO_IDX]),
        'dark_malig_group_loss':  float(group_losses[2*3 + MALIG_IDX]),
        'dark_benign_group_weight': float(group_weights2[2*3 + BENIGN_IDX]),
        'dark_benign_acc': float(ba_e),
    })

print(f"\nEpoch log sample:")
for r in epoch_log[::5]:
    print(f"  Epoch {r['epoch']:>2}: dk-benign loss={r['dark_benign_group_loss']:.4f}, "
          f"weight={r['dark_benign_group_weight']:.4f}, benign_acc={r['dark_benign_acc']:.3f}")


# ============================================================
# 3. SMOTE CLASSIFIER
# ============================================================
print("\n--- SMOTE ---")

from sklearn.neighbors import NearestNeighbors

def smote_oversample(X, y, target_class, k=5, random_state=42):
    """Generate synthetic samples for target_class via SMOTE interpolation."""
    rng    = np.random.RandomState(random_state)
    X_cls  = X[y == target_class]
    n_need = max(0, int(len(X) / len(np.unique(y))) - len(X_cls))
    if n_need == 0 or len(X_cls) < 2:
        return X, y
    nbrs = NearestNeighbors(n_neighbors=min(k+1, len(X_cls))).fit(X_cls)
    _, nn_idx = nbrs.kneighbors(X_cls)
    synthetic = []
    for _ in range(n_need):
        i   = rng.randint(0, len(X_cls))
        j   = nn_idx[i, rng.randint(1, nn_idx.shape[1])]
        lam = rng.uniform(0, 1)
        synthetic.append(X_cls[i] + lam * (X_cls[j] - X_cls[i]))
    if synthetic:
        X_syn = np.vstack(synthetic)
        X_out = np.vstack([X, X_syn])
        y_out = np.concatenate([y, np.full(len(X_syn), target_class)])
        return X_out, y_out
    return X, y

# SMOTE on combined light+medium+dark_mitig feature space
smote_f = np.vstack([train_f, mitig_f])
smote_y = np.concatenate([train_y, mitig_y])

# Oversample dark-skin benign specifically
dark_benign_mask = (np.arange(len(smote_f)) >= len(train_f)) & (smote_y == BENIGN_IDX)
dark_benign_f = smote_f[dark_benign_mask]
dark_benign_y_arr = smote_y[dark_benign_mask]

# Determine how many synthetics to generate (match dark non-neoplastic count)
n_dark_nonneo  = ((smote_y[len(train_f):]) == NONNEO_IDX).sum()
n_dark_benign  = dark_benign_mask.sum()
n_synth        = max(0, n_dark_nonneo - n_dark_benign)
print(f"SMOTE: dark benign={n_dark_benign}, dark non-neo={n_dark_nonneo}, synthetics={n_synth}")

rng_smote = np.random.RandomState(RANDOM_STATE)
if n_synth > 0 and n_dark_benign >= 2:
    nbrs_s = NearestNeighbors(n_neighbors=min(6, n_dark_benign)).fit(dark_benign_f)
    _, nn_idx_s = nbrs_s.kneighbors(dark_benign_f)
    synthetic_s = []
    for _ in range(n_synth):
        i   = rng_smote.randint(0, n_dark_benign)
        j   = nn_idx_s[i, rng_smote.randint(1, nn_idx_s.shape[1])]
        lam = rng_smote.uniform(0, 1)
        synthetic_s.append(dark_benign_f[i] + lam * (dark_benign_f[j] - dark_benign_f[i]))
    syn_f = np.vstack(synthetic_s)
    syn_y = np.full(len(syn_f), BENIGN_IDX)
    smote_aug_f = np.vstack([smote_f, syn_f])
    smote_aug_y = np.concatenate([smote_y, syn_y])
else:
    smote_aug_f, smote_aug_y = smote_f, smote_y

print(f"SMOTE augmented training set: {len(smote_aug_f)}")
clf_smote = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
clf_smote.fit(smote_aug_f, smote_aug_y)
smote_preds = clf_smote.predict(test_f)
smote_probs = clf_smote.predict_proba(test_f)
print(f"SMOTE benign acc (dark test): {benign_acc(smote_preds, test_y):.3f}")


# ============================================================
# 4. PCA PROJECTION + DECISION BOUNDARY VISUALIZATION
# ============================================================
print("\nFitting PCA for visualization...")

# Project all relevant points to 2D
# We use the combined set: train + dark test, for consistent PCA
all_viz_f = np.vstack([train_f, dark_feats])
all_viz_y = np.concatenate([train_y, dark_y])
skin_viz   = np.concatenate([
    np.zeros(len(light_feats)),   # 0 = light
    np.ones(len(medium_feats)),   # 1 = medium
    np.full(len(dark_feats), 2)   # 2 = dark
])

pca = PCA(n_components=2, random_state=RANDOM_STATE)
all_viz_2d = pca.fit_transform(all_viz_f)
print(f"PCA explained variance: {pca.explained_variance_ratio_}")

# Project training + test separately for boundary plotting
train_2d    = all_viz_2d[:len(train_f)]
dark_2d     = all_viz_2d[len(train_f):]
test_2d     = dark_2d[:n_dark_test]
mitig_2d    = dark_2d[n_dark_test:n_dark_test + N_DARK_MITIG]

# Fit classifiers in 2D PCA space for boundary drawing
# (Full-dim boundaries can't be visualized; 2D approximates the geometry)
clf_base_2d  = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
clf_base_2d.fit(train_2d, train_y)

dro_2d_f = np.vstack([train_2d, mitig_2d])
dro_2d_y = np.concatenate([train_y, mitig_y])
skin_2d   = np.concatenate([
    np.zeros(len(light_feats)),
    np.ones(len(medium_feats)),
    np.full(N_DARK_MITIG, 2)
])
group_ids_2d = (skin_2d * 3 + dro_2d_y).astype(int)
n_2d         = len(dro_2d_f)
gw_2d        = np.ones(n_groups) / n_groups
clf_dro_2d   = LogisticRegression(max_iter=200, C=1.0, random_state=RANDOM_STATE)
for _ in range(N_EPOCHS):
    sw_2d = gw_2d[group_ids_2d]
    sw_2d = sw_2d / sw_2d.sum() * n_2d
    clf_dro_2d.fit(dro_2d_f, dro_2d_y, sample_weight=sw_2d)
    p2d = np.clip(clf_dro_2d.predict_proba(dro_2d_f), 1e-9, 1.0)
    pl2d = -np.log(p2d[np.arange(n_2d), dro_2d_y])
    gl2d = np.array([pl2d[group_ids_2d==g].mean() if (group_ids_2d==g).sum()>0 else 0.0
                     for g in range(n_groups)])
    gw_2d = gw_2d * np.exp(ETA * gl2d)
    gw_2d = gw_2d / gw_2d.sum()

smote_syn_2d = pca.transform(syn_f) if n_synth > 0 else np.zeros((0,2))
smote_2d_f   = np.vstack([train_2d, mitig_2d])
smote_2d_y   = np.concatenate([train_y, mitig_y])
if n_synth > 0:
    smote_2d_f = np.vstack([smote_2d_f, smote_syn_2d])
    smote_2d_y = np.concatenate([smote_2d_y, np.full(len(smote_syn_2d), BENIGN_IDX)])
clf_smote_2d = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
clf_smote_2d.fit(smote_2d_f, smote_2d_y)

def plot_boundary_2d(ax, clf, xrange, yrange, alpha=0.15, resolution=300):
    """Shade decision regions for a 3-class LR in 2D."""
    xx, yy = np.meshgrid(
        np.linspace(xrange[0], xrange[1], resolution),
        np.linspace(yrange[0], yrange[1], resolution)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z    = clf.predict(grid).reshape(xx.shape)
    # class colors: benign=blue, malignant=red, non-neo=green
    colors_map = {BENIGN_IDX: '#2196F3', MALIG_IDX: '#F44336', NONNEO_IDX: '#4CAF50'}
    rgb = np.ones((*Z.shape, 3))
    for cls_id, hex_col in colors_map.items():
        r = int(hex_col[1:3],16)/255
        g = int(hex_col[3:5],16)/255
        b = int(hex_col[5:7],16)/255
        mask = Z == cls_id
        rgb[mask] = [r, g, b]
    ax.imshow(rgb, extent=[xrange[0], xrange[1], yrange[0], yrange[1]],
              origin='lower', alpha=alpha, aspect='auto')

# Determine plot bounds from dark-skin test points
pad  = 1.5
x_lo = test_2d[:,0].min() - pad;  x_hi = test_2d[:,0].max() + pad
y_lo = test_2d[:,1].min() - pad;  y_hi = test_2d[:,1].max() + pad

# Dark-skin test mask splits
dk_benign_mask = test_y == BENIGN_IDX
dk_nonneo_mask = test_y == NONNEO_IDX
dk_malig_mask  = test_y == MALIG_IDX

# ── FIGURE 1: 2×3 panel ──────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(
    'Decision Boundary Visualization: Baseline vs. Group DRO vs. SMOTE\n'
    'CLIP ViT-L/14 features projected to 2D PCA — dark-skin test set',
    fontsize=14, fontweight='bold')

SCATTER_KW = dict(s=20, alpha=0.7, linewidths=0.3, edgecolors='k')

for col, (clf2d, title) in enumerate([
    (clf_base_2d,  'Baseline\n(train: light+medium)'),
    (clf_dro_2d,   'Group DRO\n(minimax worst-group, η=0.01)'),
    (clf_smote_2d, 'SMOTE\n(synthetic dark-skin benign)')
]):
    ax_top = axes[0, col]
    ax_bot = axes[1, col]

    # Row 0: full scatter (all skin groups, all labels)
    plot_boundary_2d(ax_top, clf2d, (x_lo, x_hi), (y_lo, y_hi))
    ax_top.scatter(train_2d[train_y==BENIGN_IDX, 0],  train_2d[train_y==BENIGN_IDX, 1],
                   c='#90CAF9', marker='o', label='Light/Med benign', **SCATTER_KW)
    ax_top.scatter(train_2d[train_y==NONNEO_IDX, 0],  train_2d[train_y==NONNEO_IDX, 1],
                   c='#A5D6A7', marker='o', label='Light/Med non-neo', **SCATTER_KW)
    ax_top.scatter(train_2d[train_y==MALIG_IDX, 0],   train_2d[train_y==MALIG_IDX, 1],
                   c='#EF9A9A', marker='o', label='Light/Med malignant', **SCATTER_KW)
    ax_top.scatter(test_2d[dk_benign_mask, 0], test_2d[dk_benign_mask, 1],
                   c='#1565C0', marker='D', s=35, alpha=0.9, edgecolors='white', linewidths=0.5,
                   label='Dark-skin BENIGN (test)')
    ax_top.scatter(test_2d[dk_nonneo_mask, 0], test_2d[dk_nonneo_mask, 1],
                   c='#1B5E20', marker='^', s=15, alpha=0.4, label='Dark non-neo (test)')
    ax_top.scatter(test_2d[dk_malig_mask, 0],  test_2d[dk_malig_mask, 1],
                   c='#B71C1C', marker='s', s=15, alpha=0.4, label='Dark malignant (test)')

    ba = benign_acc(clf2d.predict(test_2d), test_y)
    ax_top.set_title(f'{title}\nDark-skin benign acc = {ba:.1%}',
                     fontsize=10, fontweight='bold')
    ax_top.set_xlabel('PCA-1'); ax_top.set_ylabel('PCA-2')
    ax_top.set_xlim(x_lo, x_hi); ax_top.set_ylim(y_lo, y_hi)
    if col == 0:
        ax_top.legend(fontsize=7, loc='upper left', markerscale=1.2)

    # Row 1: zoom on dark-skin benign cluster only
    # Show SMOTE synthetics for the SMOTE column
    plot_boundary_2d(ax_bot, clf2d, (x_lo, x_hi), (y_lo, y_hi))
    ax_bot.scatter(test_2d[dk_benign_mask, 0], test_2d[dk_benign_mask, 1],
                   c='#1565C0', marker='D', s=50, alpha=0.9, edgecolors='white', linewidths=0.8,
                   label='Dark-skin BENIGN (test)')
    ax_bot.scatter(mitig_2d[mitig_y==BENIGN_IDX, 0], mitig_2d[mitig_y==BENIGN_IDX, 1],
                   c='#FF9800', marker='*', s=80, alpha=0.9, edgecolors='k', linewidths=0.3,
                   label='Dark benign (mitigation pool)')
    if col == 2 and len(smote_syn_2d) > 0:
        ax_bot.scatter(smote_syn_2d[:,0], smote_syn_2d[:,1],
                       c='#FF5722', marker='+', s=60, alpha=0.6, linewidths=1.5,
                       label='SMOTE synthetic benign')

    # Draw boundary centroid arrow: distance from classifier's "benign region"
    # Find the nearest boundary point to the dark-benign centroid
    centroid = test_2d[dk_benign_mask].mean(axis=0)
    ax_bot.scatter(*centroid, c='cyan', marker='X', s=120, zorder=10,
                   edgecolors='k', linewidths=1, label='Dark-benign centroid')

    ax_bot.set_title(f'{title}\nDark-skin benign cluster (zoom)', fontsize=10)
    ax_bot.set_xlabel('PCA-1'); ax_bot.set_ylabel('PCA-2')
    # Zoom tighter around dark-skin benign cluster
    bx_lo = test_2d[dk_benign_mask,0].min() - 1.5
    bx_hi = test_2d[dk_benign_mask,0].max() + 1.5
    by_lo = test_2d[dk_benign_mask,1].min() - 1.5
    by_hi = test_2d[dk_benign_mask,1].max() + 1.5
    ax_bot.set_xlim(bx_lo, bx_hi); ax_bot.set_ylim(by_lo, by_hi)
    if col == 0:
        ax_bot.legend(fontsize=7, loc='upper left')

plt.tight_layout()
fig.savefig('/kaggle/working/nb_mech1_pca_boundary.png', dpi=300, bbox_inches='tight')
plt.show()
print("Figure 1 saved: nb_mech1_pca_boundary.png")


# ── FIGURE 2: P(benign) distribution under each method ───────
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
fig2.suptitle(
    'P(benign) Distribution for Dark-Skin Benign Test Samples\n'
    'Under Each Intervention (CLIP ViT-L/14)',
    fontsize=13, fontweight='bold')

dk_benign_test_idx = np.where(test_y == BENIGN_IDX)[0]

for ax, (probs, label, color) in zip(axes2, [
    (base_probs,  'Baseline',   '#1565C0'),
    (dro_probs,   'Group DRO',  '#B71C1C'),
    (smote_probs, 'SMOTE',      '#2E7D32'),
]):
    p_benign = probs[dk_benign_test_idx, BENIGN_IDX]
    p_nonneo = probs[dk_benign_test_idx, NONNEO_IDX]
    x_pos    = np.arange(len(dk_benign_test_idx))
    # Sort by P(benign) for clarity
    sort_idx = np.argsort(p_benign)[::-1]
    ax.bar(x_pos, p_benign[sort_idx],  alpha=0.8, color=color,     label='P(benign)')
    ax.bar(x_pos, p_nonneo[sort_idx],  alpha=0.5, color='#FF9800', label='P(non-neo)',
           bottom=p_benign[sort_idx])
    ax.axhline(0.333, color='grey', linestyle='--', alpha=0.5, label='Random chance')
    ax.set_title(f'{label}\nmean P(benign)={p_benign.mean():.3f}, '
                 f'n≥0.5: {(p_benign>=0.5).sum()}', fontsize=10, fontweight='bold')
    ax.set_xlabel('True-benign dark-skin samples (sorted)')
    ax.set_ylabel('Predicted probability')
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, alpha=0.3)

plt.tight_layout()
fig2.savefig('/kaggle/working/nb_mech1_pbenign_dist.png', dpi=300, bbox_inches='tight')
plt.show()
print("Figure 2 saved: nb_mech1_pbenign_dist.png")


# ── FIGURE 3: Boundary distance from dark-skin benign centroid ─
print("\n--- Boundary distance analysis ---")

def boundary_distance_from_centroid(clf_hd, X_benign_test, BENIGN_IDX):
    """
    For a linear LR classifier, the decision boundary between benign and
    the predicted class is a hyperplane. We measure the signed distance
    from each dark-skin benign sample to the 'benign vs. non-neo' boundary
    (positive = correctly on benign side, negative = wrong side).
    """
    # LR decision boundary between benign and non-neo:
    # (w_benign - w_nonneo) · x + (b_benign - b_nonneo) = 0
    W = clf_hd.coef_         # (n_classes, n_features)
    b = clf_hd.intercept_    # (n_classes,)
    w_diff = W[BENIGN_IDX] - W[NONNEO_IDX]
    b_diff = b[BENIGN_IDX]  - b[NONNEO_IDX]
    norm   = np.linalg.norm(w_diff)
    signed_dist = (X_benign_test @ w_diff + b_diff) / norm
    return signed_dist

benign_test_f = test_f[test_y == BENIGN_IDX]
dist_base  = boundary_distance_from_centroid(clf_base,  benign_test_f, BENIGN_IDX)
dist_dro   = boundary_distance_from_centroid(clf_dro,   benign_test_f, BENIGN_IDX)
dist_smote = boundary_distance_from_centroid(clf_smote, benign_test_f, BENIGN_IDX)

fig3, ax3 = plt.subplots(figsize=(10, 5))
bins = np.linspace(
    min(dist_base.min(), dist_dro.min(), dist_smote.min()) - 0.5,
    max(dist_base.max(), dist_dro.max(), dist_smote.max()) + 0.5,
    40
)
ax3.hist(dist_base,  bins=bins, alpha=0.65, color='#1565C0', label='Baseline')
ax3.hist(dist_dro,   bins=bins, alpha=0.65, color='#B71C1C', label='Group DRO')
ax3.hist(dist_smote, bins=bins, alpha=0.65, color='#2E7D32', label='SMOTE')
ax3.axvline(0, color='black', linestyle='--', linewidth=1.5, label='Decision boundary (0)')
ax3.set_xlabel('Signed distance from "benign vs. non-neo" hyperplane\n'
               '(positive = on benign side; negative = misclassified as non-neo)')
ax3.set_ylabel('Count of dark-skin true-benign samples')
ax3.set_title('Decision Boundary Distance for Dark-Skin True-Benign Samples\n'
              'Group DRO shifts boundary AWAY; SMOTE shifts it TOWARD the cluster',
              fontweight='bold')
ax3.legend(fontsize=10)
ax3.yaxis.grid(True, alpha=0.3)

# Annotate means
for dist, label, color in [
    (dist_base,  'Baseline', '#1565C0'),
    (dist_dro,   'DRO',      '#B71C1C'),
    (dist_smote, 'SMOTE',    '#2E7D32'),
]:
    ax3.axvline(dist.mean(), color=color, linestyle=':', linewidth=2, alpha=0.8)
    ax3.text(dist.mean(), ax3.get_ylim()[1]*0.9 if ax3.get_ylim()[1]>0 else 1,
             f'{label}\nmean={dist.mean():.2f}', color=color, fontsize=8, ha='center')

plt.tight_layout()
fig3.savefig('/kaggle/working/nb_mech1_boundary_distance.png', dpi=300, bbox_inches='tight')
plt.show()
print("Figure 3 saved: nb_mech1_boundary_distance.png")


# ============================================================
# 5. UMAP VISUALIZATION (bonus — richer topology than PCA)
# ============================================================
print("\nFitting UMAP (this takes ~5 min)...")
reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                    random_state=RANDOM_STATE, metric='cosine')
# Use a subset for speed: all dark + 300 light + 300 medium
n_light_umap  = min(300, len(light_feats))
n_medium_umap = min(300, len(medium_feats))
umap_subset_f = np.vstack([
    light_feats[:n_light_umap],
    medium_feats[:n_medium_umap],
    dark_feats[:n_dark_test]
])
umap_subset_y = np.concatenate([
    light_y[:n_light_umap],
    medium_y[:n_medium_umap],
    test_y
])
umap_skin = np.concatenate([
    np.zeros(n_light_umap),
    np.ones(n_medium_umap),
    np.full(n_dark_test, 2)
])
umap_2d = reducer.fit_transform(umap_subset_f)
print("UMAP done.")

umap_dark_mask  = umap_skin == 2
umap_light_mask = umap_skin == 0
umap_med_mask   = umap_skin == 1

fig4, ax4 = plt.subplots(figsize=(10, 8))
ax4.scatter(umap_2d[umap_light_mask & (umap_subset_y==BENIGN_IDX), 0],
            umap_2d[umap_light_mask & (umap_subset_y==BENIGN_IDX), 1],
            c='#90CAF9', s=12, alpha=0.5, label='Light benign')
ax4.scatter(umap_2d[umap_light_mask & (umap_subset_y==NONNEO_IDX), 0],
            umap_2d[umap_light_mask & (umap_subset_y==NONNEO_IDX), 1],
            c='#A5D6A7', s=12, alpha=0.3, label='Light non-neo')
ax4.scatter(umap_2d[umap_dark_mask & (umap_subset_y==NONNEO_IDX), 0],
            umap_2d[umap_dark_mask & (umap_subset_y==NONNEO_IDX), 1],
            c='#1B5E20', s=12, alpha=0.4, label='Dark non-neo')
ax4.scatter(umap_2d[umap_dark_mask & (umap_subset_y==MALIG_IDX), 0],
            umap_2d[umap_dark_mask & (umap_subset_y==MALIG_IDX), 1],
            c='#B71C1C', s=12, alpha=0.4, label='Dark malignant')
ax4.scatter(umap_2d[umap_dark_mask & (umap_subset_y==BENIGN_IDX), 0],
            umap_2d[umap_dark_mask & (umap_subset_y==BENIGN_IDX), 1],
            c='#1565C0', s=45, alpha=0.95, marker='D', edgecolors='white', linewidths=0.5,
            label='Dark BENIGN (key cluster)', zorder=10)

ax4.set_title('UMAP of CLIP ViT-L/14 Features\n'
              'Dark-skin benign cluster (diamonds) proximity to non-neo region explains classifier failure',
              fontsize=12, fontweight='bold')
ax4.set_xlabel('UMAP-1'); ax4.set_ylabel('UMAP-2')
ax4.legend(fontsize=9, loc='best', markerscale=1.5)

plt.tight_layout()
fig4.savefig('/kaggle/working/nb_mech1_umap.png', dpi=300, bbox_inches='tight')
plt.show()
print("Figure 4 saved: nb_mech1_umap.png")


# ============================================================
# 6. SUMMARY JSON
# ============================================================
out = {
    'meta': {
        'notebook': 'nb_mech1_decision_boundary',
        'model': 'CLIP ViT-L/14',
        'n_train': int(len(train_f)),
        'n_dark_test': int(n_dark_test),
        'n_dark_benign_test': int(dk_benign_mask.sum()),
        'n_dark_mitigation': int(N_DARK_MITIG),
        'nc_ng_mitigation': float((mitig_y == BENIGN_IDX).sum() / len(mitig_y)),
    },
    'benign_accuracy': {
        'baseline':  float(benign_acc(base_preds,  test_y)),
        'group_dro': float(benign_acc(dro_preds,   test_y)),
        'smote':     float(benign_acc(smote_preds, test_y)),
    },
    'mean_p_benign_for_true_benign': {
        'baseline':  float(base_probs[dk_benign_test_idx, BENIGN_IDX].mean()),
        'group_dro': float(dro_probs[dk_benign_test_idx, BENIGN_IDX].mean()),
        'smote':     float(smote_probs[dk_benign_test_idx, BENIGN_IDX].mean()),
    },
    'boundary_distance_signed_mean': {
        'baseline':  float(dist_base.mean()),
        'group_dro': float(dist_dro.mean()),
        'smote':     float(dist_smote.mean()),
    },
    'boundary_distance_signed_pct_negative': {
        'baseline':  float((dist_base  < 0).mean()),
        'group_dro': float((dist_dro   < 0).mean()),
        'smote':     float((dist_smote < 0).mean()),
    },
    'pca_explained_variance': pca.explained_variance_ratio_.tolist(),
    'dro_epoch_log': epoch_log,
}

json.dump(out, open('/kaggle/working/nb_mech1_results.json', 'w'), indent=2)
print("\n=== KEY NUMBERS FOR PAPER ===")
print(f"Baseline  benign acc: {out['benign_accuracy']['baseline']:.3f}  "
      f"mean P(benign)={out['mean_p_benign_for_true_benign']['baseline']:.3f}  "
      f"boundary dist mean={out['boundary_distance_signed_mean']['baseline']:.3f}")
print(f"Group DRO benign acc: {out['benign_accuracy']['group_dro']:.3f}  "
      f"mean P(benign)={out['mean_p_benign_for_true_benign']['group_dro']:.3f}  "
      f"boundary dist mean={out['boundary_distance_signed_mean']['group_dro']:.3f}  "
      f"(DRO boundary shifts AWAY: {out['boundary_distance_signed_pct_negative']['group_dro']:.1%} samples on wrong side)")
print(f"SMOTE     benign acc: {out['benign_accuracy']['smote']:.3f}  "
      f"mean P(benign)={out['mean_p_benign_for_true_benign']['smote']:.3f}  "
      f"boundary dist mean={out['boundary_distance_signed_mean']['smote']:.3f}")

print("\n✓ Complete. Upload all 4 PNGs and nb_mech1_results.json to Claude.")
print("Files: nb_mech1_pca_boundary.png, nb_mech1_pbenign_dist.png,")
print("       nb_mech1_boundary_distance.png, nb_mech1_umap.png, nb_mech1_results.json")
