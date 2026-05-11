# ============================================================
# NOTEBOOK 3 — Training Set Size Ablation (FIXED)
# Dataset: nazmusresan/fitzpatrick17k
# GPU T4, Internet ON. ~20 min.
# ============================================================

!pip install transformers torch torchvision scikit-learn pandas numpy matplotlib -q

import torch
import numpy as np, pandas as pd, os, json, warnings
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import CLIPModel, CLIPProcessor
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

fitz_csv     = '/kaggle/input/datasets/nazmusresan/fitzpatrick17k/New folder/fitzpatrick17k (1).csv'
fitz_img_dir = '/kaggle/input/datasets/nazmusresan/fitzpatrick17k/New folder/background removed'

df = pd.read_csv(fitz_csv)
df = df[df['fitzpatrick_scale'] > 0]
image_files = {f.replace('.jpg','').replace('.png',''):
               os.path.join(fitz_img_dir, f)
               for f in os.listdir(fitz_img_dir)
               if f.endswith('.jpg') or f.endswith('.png')}
df['local_path'] = df['md5hash'].map(image_files)
df = df[df['local_path'].notna()].copy()
df['skin_group'] = df['fitzpatrick_scale'].apply(
    lambda x: 'light' if x <= 2 else 'medium' if x <= 4 else 'dark')

MAX = 1000
light_df  = df[df['skin_group']=='light'].sample(MAX, random_state=42)
medium_df = df[df['skin_group']=='medium'].sample(MAX, random_state=42)
dark_df   = df[df['skin_group']=='dark'].sample(min(MAX, len(df[df['skin_group']=='dark'])), random_state=42)

le = LabelEncoder()
le.fit(list(light_df['three_partition_label']) +
       list(medium_df['three_partition_label']) +
       list(dark_df['three_partition_label']))
print(f"Classes: {le.classes_}")

print("Loading CLIP...")
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
        except: pass
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
        feats = feats / feats.norm(dim=-1, keepdim=True)
        all_feats.append(feats.cpu().numpy())
    return np.vstack(all_feats)

print("Loading images...")
light_imgs, light_y   = load_imgs(light_df)
medium_imgs, medium_y = load_imgs(medium_df)
dark_imgs, dark_y     = load_imgs(dark_df)

print("Extracting features...")
light_feats  = get_features(light_imgs)
medium_feats = get_features(medium_imgs)
dark_feats   = get_features(dark_imgs)

base_train_f = np.vstack([light_feats, medium_feats])
base_train_y = np.concatenate([light_y, medium_y])

# Random split AUC (upper bound)
all_f = np.vstack([light_feats, medium_feats, dark_feats])
all_y = np.concatenate([light_y, medium_y, dark_y])
tr, te = train_test_split(np.arange(len(all_y)), test_size=0.25, stratify=all_y, random_state=42)
clf_rand = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
clf_rand.fit(all_f[tr], all_y[tr])
rand_auc = roc_auc_score(all_y[te], clf_rand.predict_proba(all_f[te]),
                          multi_class='ovr', average='macro')
print(f"\nRandom split AUC (upper bound): {rand_auc:.4f}")

# Baseline
clf_base = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
clf_base.fit(base_train_f, base_train_y)
base_auc = roc_auc_score(dark_y, clf_base.predict_proba(dark_feats),
                          multi_class='ovr', average='macro')
print(f"Baseline dark-skin AUC (0 dark train): {base_auc:.4f}")
print(f"Baseline SGG: {rand_auc - base_auc:.4f}")

# Test set: 80% of dark images
n_dark_test = int(0.8 * len(dark_feats))
dark_test_idx = np.arange(n_dark_test)
dark_pool_idx = np.arange(n_dark_test, len(dark_feats))
test_f = dark_feats[dark_test_idx]
test_y = dark_y[dark_test_idx]

n_values = [0, 10, 25, 50, 100, 150, 200, 300, 400, 500]
n_values = [n for n in n_values if n <= len(dark_pool_idx)]

print("\n=== ABLATION: Dark-Skin Training Size ===")
results_ablation = []
for n_dark in n_values:
    if n_dark == 0:
        train_f = base_train_f
        train_y_aug = base_train_y
        weights = None
    else:
        add_idx = dark_pool_idx[:n_dark]
        train_f = np.vstack([base_train_f, dark_feats[add_idx]])
        train_y_aug = np.concatenate([base_train_y, dark_y[add_idx]])
        n_base = len(base_train_y)
        total = n_base + n_dark
        weights = np.concatenate([
            np.full(n_base, total / (2 * n_base)),
            np.full(n_dark, total / (2 * n_dark))
        ])

    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(train_f, train_y_aug, sample_weight=weights)

    probs = clf.predict_proba(test_f)
    preds = clf.predict(test_f)
    auc = roc_auc_score(test_y, probs, multi_class='ovr', average='macro')
    sgg = rand_auc - auc
    gap_closed = max(0.0, (auc - base_auc) / (rand_auc - base_auc) * 100) if n_dark > 0 else 0.0

    per_class = {}
    for i, cls in enumerate(le.classes_):
        mask = test_y == i
        if mask.sum() > 0:
            per_class[cls] = float(accuracy_score(test_y[mask], preds[mask]))

    results_ablation.append({
        'n_dark_train': int(n_dark),
        'auc': float(auc),
        'sgg': float(sgg),
        'gap_closed_pct': float(gap_closed),
        'per_class': per_class
    })
    print(f"  n_dark={n_dark:>4}: AUC={auc:.4f}, SGG={sgg:.4f}, "
          f"gap_closed={gap_closed:.0f}%, "
          f"benign={per_class.get('benign',0):.3f}")

# Figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Training Set Size vs Dark-Skin Performance (CLIP ViT-L/14)',
             fontsize=13, fontweight='bold')

ns   = [r['n_dark_train'] for r in results_ablation]
aucs = [r['auc'] for r in results_ablation]
benign_accs = [r['per_class'].get('benign', 0) for r in results_ablation]

ax1.plot(ns, aucs, 'o-', color='#1565C0', linewidth=2.5, markersize=7)
ax1.axhline(rand_auc, color='#1565C0', linestyle='--', alpha=0.5,
            label=f'Random split upper bound ({rand_auc:.3f})')
ax1.axhline(base_auc, color='#C62828', linestyle=':', alpha=0.5,
            label=f'No dark-skin training ({base_auc:.3f})')
ax1.set_xlabel('Dark-Skin Training Images Added')
ax1.set_ylabel('Macro-Averaged AUC on Dark-Skin Test Set')
ax1.set_title('AUC vs Training Set Size')
ax1.legend(fontsize=9); ax1.yaxis.grid(True, alpha=0.3)

ax2.plot(ns, benign_accs, 'o-', color='#B71C1C', linewidth=2.5, markersize=7)
ax2.set_xlabel('Dark-Skin Training Images Added')
ax2.set_ylabel('Benign Accuracy on Dark-Skin Test Set')
ax2.set_title('Benign Detection vs Training Set Size')
ax2.yaxis.grid(True, alpha=0.3)
ax2.set_ylim(-0.02, max(0.5, max(benign_accs) + 0.05))

plt.tight_layout()
plt.savefig('/kaggle/working/nb3_ablation_figure.png', dpi=300, bbox_inches='tight')
plt.show()
print("Figure saved.")

json.dump({'random_auc': float(rand_auc), 'base_auc': float(base_auc),
           'ablation': results_ablation},
          open('/kaggle/working/nb3_ablation.json','w'), indent=2)

print("\n=== SUMMARY TABLE ===")
print(f"{'N dark':>8} {'AUC':>8} {'SGG':>8} {'Gap Closed':>12} {'Benign':>8}")
for r in results_ablation:
    print(f"{r['n_dark_train']:>8} {r['auc']:>8.4f} {r['sgg']:>8.4f} "
          f"{r['gap_closed_pct']:>11.0f}% {r['per_class'].get('benign',0):>8.3f}")

print("\n✓ Complete. Upload nb3_ablation_figure.png and paste output to Claude.")
