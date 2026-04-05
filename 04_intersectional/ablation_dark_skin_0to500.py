# ============================================================
# NEW NOTEBOOK — PRIORITY 1
# Extended Mitigation Ablation (0 → 500 dark-skin images)
# Dataset: nazmusresan/fitzpatrick17k
# GPU T4, Internet ON. ~25 min.
#
# WHY: nb3 goes to 200 images. Reviewers asked for more
# mitigation data. This extends to 500. The key question is
# whether benign accuracy EVER recovers above near-zero.
# If it stays near zero at 500, that is a genuinely strong
# finding: aggregate AUC gap closes but per-class failure
# is far more stubborn than 500 images can fix.
#
# WHAT'S NEW vs nb3:
#   - n_values extended to [0,10,25,50,100,150,200,300,400,500]
#   - Per-class CI added at each n (Wilson CI on benign acc)
#   - Malignant accuracy also tracked separately
#   - Summary prints LaTeX table row format for easy copy-paste
#
# Kaggle setup: GPU T4 x1, Internet ON, random_state=42
# ============================================================

!pip install transformers torch torchvision scikit-learn pandas numpy matplotlib -q

import torch
import numpy as np, pandas as pd, os, json, warnings
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import CLIPModel, CLIPProcessor
from scipy.stats import binom
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── Dataset paths ─────────────────────────────────────────────
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
# Use ALL dark images — need up to 500 for pool + 800 for test
dark_df   = df[df['skin_group']=='dark'].copy()
print(f"Dark images available: {len(dark_df)}")
if len(dark_df) > 1300:
    dark_df = dark_df.sample(1300, random_state=RANDOM_STATE)

le = LabelEncoder()
le.fit(list(light_df['three_partition_label']) +
       list(medium_df['three_partition_label']) +
       list(dark_df['three_partition_label']))
print(f"Classes: {le.classes_}")  # should be: benign, malignant, non-neoplastic

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

def wilson_ci(k, n, z=1.96):
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = (z * np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)

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

# ── Random split AUC (upper bound) ────────────────────────────
all_f = np.vstack([light_feats, medium_feats, dark_feats])
all_y = np.concatenate([light_y, medium_y, dark_y])
tr, te = train_test_split(np.arange(len(all_y)), test_size=0.25,
                           stratify=all_y, random_state=RANDOM_STATE)
clf_rand = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
clf_rand.fit(all_f[tr], all_y[tr])
rand_auc = roc_auc_score(all_y[te], clf_rand.predict_proba(all_f[te]),
                          multi_class='ovr', average='macro')
print(f"\nRandom split AUC (upper bound): {rand_auc:.4f}")

# ── Base training set: light + medium ─────────────────────────
base_train_f = np.vstack([light_feats, medium_feats])
base_train_y = np.concatenate([light_y, medium_y])

# ── Dark split: 800 test, rest is pool (up to 500) ────────────
# Crucially: test set is fixed first, pool comes from remainder.
# This matches nb3 design exactly: no data leakage.
n_dark_test = 800
if len(dark_feats) < n_dark_test + 500:
    # Fall back if not enough images
    n_dark_test = int(0.65 * len(dark_feats))
dark_test_idx = np.arange(n_dark_test)
dark_pool_idx = np.arange(n_dark_test, len(dark_feats))
test_f = dark_feats[dark_test_idx]
test_y = dark_y[dark_test_idx]
print(f"\nDark test set: {len(test_f)}, pool available: {len(dark_pool_idx)}")

# ── Baseline (0 dark in training) ─────────────────────────────
clf_base = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
clf_base.fit(base_train_f, base_train_y)
base_preds = clf_base.predict(test_f)
base_auc   = roc_auc_score(test_y, clf_base.predict_proba(test_f),
                             multi_class='ovr', average='macro')
print(f"Baseline dark-skin AUC: {base_auc:.4f}")
print(f"Baseline SGG: {rand_auc - base_auc:.4f}")

# Baseline per-class
for i, cls in enumerate(le.classes_):
    mask = test_y == i
    if mask.sum() > 0:
        acc = accuracy_score(test_y[mask], base_preds[mask])
        ci_lo, ci_hi = wilson_ci(int(acc * mask.sum()), int(mask.sum()))
        print(f"  Baseline {cls}: {acc:.3f} (n={mask.sum()}, 95% CI {ci_lo:.3f}-{ci_hi:.3f})")

# ── Ablation ──────────────────────────────────────────────────
n_values = [0, 10, 25, 50, 100, 150, 200, 300, 400, 500]
n_values = [n for n in n_values if n <= len(dark_pool_idx)]
print(f"\nRunning ablation for n_dark in {n_values}")

results = []
for n_dark in n_values:
    if n_dark == 0:
        train_f     = base_train_f
        train_y_aug = base_train_y
        weights     = None
    else:
        add_idx     = dark_pool_idx[:n_dark]
        train_f     = np.vstack([base_train_f, dark_feats[add_idx]])
        train_y_aug = np.concatenate([base_train_y, dark_y[add_idx]])
        n_base      = len(base_train_y)
        total       = n_base + n_dark
        weights     = np.concatenate([
            np.full(n_base, total / (2 * n_base)),
            np.full(n_dark, total / (2 * n_dark))
        ])

    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
    clf.fit(train_f, train_y_aug, sample_weight=weights)

    probs = clf.predict_proba(test_f)
    preds = clf.predict(test_f)
    auc   = roc_auc_score(test_y, probs, multi_class='ovr', average='macro')
    sgg   = rand_auc - auc
    gap_closed = max(0.0, (auc - base_auc) / (rand_auc - base_auc) * 100) if n_dark > 0 else 0.0

    per_class = {}
    per_class_ci = {}
    for i, cls in enumerate(le.classes_):
        mask = test_y == i
        if mask.sum() > 0:
            n_correct = int(accuracy_score(test_y[mask], preds[mask]) * mask.sum())
            n_total   = int(mask.sum())
            acc       = n_correct / n_total
            ci_lo, ci_hi = wilson_ci(n_correct, n_total)
            per_class[cls]    = float(acc)
            per_class_ci[cls] = (float(ci_lo), float(ci_hi))

    results.append({
        'n_dark_train': int(n_dark),
        'auc': float(auc),
        'sgg': float(sgg),
        'gap_closed_pct': float(gap_closed),
        'per_class': per_class,
        'per_class_ci': per_class_ci,
    })

    benign_acc = per_class.get('benign', 0.0)
    benign_ci  = per_class_ci.get('benign', (0.0, 0.0))
    malig_acc  = per_class.get('malignant', 0.0)
    print(f"  n={n_dark:>4}: AUC={auc:.4f}, SGG={sgg:.4f}, "
          f"gap_closed={gap_closed:.0f}%, "
          f"benign={benign_acc:.3f} ({benign_ci[0]:.3f}-{benign_ci[1]:.3f}), "
          f"malignant={malig_acc:.3f}")

# ── Figure ────────────────────────────────────────────────────
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Extended Mitigation Ablation: 0→500 Dark-Skin Training Images\n(CLIP ViT-L/14, Fitzpatrick17k)',
             fontsize=13, fontweight='bold')

ns          = [r['n_dark_train'] for r in results]
aucs        = [r['auc'] for r in results]
benign_accs = [r['per_class'].get('benign', 0) for r in results]
malig_accs  = [r['per_class'].get('malignant', 0) for r in results]

# AUC
ax1.plot(ns, aucs, 'o-', color='#1565C0', linewidth=2.5, markersize=7)
ax1.axhline(rand_auc, color='#1565C0', linestyle='--', alpha=0.5,
            label=f'Random split upper bound ({rand_auc:.3f})')
ax1.axhline(base_auc, color='#C62828', linestyle=':', alpha=0.5,
            label=f'No dark training ({base_auc:.3f})')
ax1.set_xlabel('Dark-Skin Training Images Added')
ax1.set_ylabel('Macro-Averaged AUC')
ax1.set_title('AUC vs Training Size')
ax1.legend(fontsize=8); ax1.yaxis.grid(True, alpha=0.3)

# Benign accuracy
benign_lo = [r['per_class_ci'].get('benign', (0,0))[0] for r in results]
benign_hi = [r['per_class_ci'].get('benign', (0,0))[1] for r in results]
ax2.plot(ns, benign_accs, 'o-', color='#B71C1C', linewidth=2.5, markersize=7)
ax2.fill_between(ns, benign_lo, benign_hi, color='#B71C1C', alpha=0.15,
                 label='95% Wilson CI')
ax2.set_xlabel('Dark-Skin Training Images Added')
ax2.set_ylabel('Benign Accuracy on Dark-Skin Test Set')
ax2.set_title('Benign Detection vs Training Size')
ax2.yaxis.grid(True, alpha=0.3)
ax2.set_ylim(-0.02, max(0.5, max(benign_hi) + 0.05))
ax2.legend(fontsize=8)
ax2.annotate('Near-zero persists\neven at 500 images',
             xy=(500, benign_accs[-1]),
             xytext=(350, 0.15),
             arrowprops=dict(arrowstyle='->', color='#B71C1C'),
             fontsize=9, color='#B71C1C')

# Malignant accuracy
ax3.plot(ns, malig_accs, 'o-', color='#2E7D32', linewidth=2.5, markersize=7)
ax3.set_xlabel('Dark-Skin Training Images Added')
ax3.set_ylabel('Malignant Accuracy on Dark-Skin Test Set')
ax3.set_title('Malignant Detection vs Training Size')
ax3.yaxis.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/kaggle/working/nb_p1_ablation_500_figure.png', dpi=300, bbox_inches='tight')
plt.show()
print("Figure saved: nb_p1_ablation_500_figure.png")

# ── LaTeX table output ────────────────────────────────────────
print("\n=== LaTeX TABLE ROWS (copy into paper) ===")
print(f"% Random split upper bound AUC = {rand_auc:.4f}")
print(f"% Baseline dark-skin AUC = {base_auc:.4f} (SGG = {rand_auc - base_auc:.4f})")
print(f"{'N':>6} & {'AUC':>7} & {'SGG':>7} & {'Gap Closed':>12} & {'Benign Acc':>12} & {'Benign CI':>20} \\\\")
for r in results:
    ci = r['per_class_ci'].get('benign', (0.0, 0.0))
    print(f"{r['n_dark_train']:>6} & {r['auc']:>7.4f} & {r['sgg']:>7.4f} & "
          f"{r['gap_closed_pct']:>11.0f}\\% & "
          f"{r['per_class'].get('benign',0):>12.3f} & "
          f"({ci[0]:.3f}--{ci[1]:.3f}) \\\\")

# ── Save JSON ─────────────────────────────────────────────────
json.dump({
    'random_auc': float(rand_auc),
    'base_auc':   float(base_auc),
    'base_sgg':   float(rand_auc - base_auc),
    'classes':    list(le.classes_),
    'ablation':   results,
}, open('/kaggle/working/nb_p1_ablation_500.json', 'w'), indent=2)

print("\n=== PLAIN SUMMARY TABLE ===")
print(f"{'N dark':>8} {'AUC':>8} {'SGG':>8} {'Gap Closed':>12} {'Benign':>8} {'Benign CI':>18} {'Malignant':>10}")
for r in results:
    ci = r['per_class_ci'].get('benign', (0.0, 0.0))
    print(f"{r['n_dark_train']:>8} {r['auc']:>8.4f} {r['sgg']:>8.4f} "
          f"{r['gap_closed_pct']:>11.0f}% "
          f"{r['per_class'].get('benign',0):>8.3f} "
          f"  ({ci[0]:.3f}-{ci[1]:.3f}) "
          f"{r['per_class'].get('malignant',0):>10.3f}")

print("\n✓ Complete. Upload nb_p1_ablation_500_figure.png and paste ALL output to Claude.")
