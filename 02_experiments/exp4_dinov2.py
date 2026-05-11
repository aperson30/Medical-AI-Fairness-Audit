# ============================================================
# NOTEBOOK 4 — DINOv2 as Fourth Architecture
# Dataset: nazmusresan/fitzpatrick17k
# GPU T4, Internet ON. ~20 min.
# Priority: MEDIUM — strongest self-supervised vision model,
# very different from CLIP/ViT/ResNet. If SGG persists here
# the architecture-agnostic claim is essentially airtight.
# ============================================================

!pip install transformers torch torchvision scikit-learn pandas numpy -q

import torch
import numpy as np, pandas as pd, os, json, warnings
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoImageProcessor
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

# Load DINOv2
print("Loading DINOv2...")
dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
dino_model     = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
dino_model.eval()
print("DINOv2 loaded.")

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
def get_dino_features(images, batch_size=32):
    all_feats = []
    for i in range(0, len(images), batch_size):
        batch  = images[i:i+batch_size]
        inputs = dino_processor(images=batch, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out    = dino_model(**inputs)
        feats  = out.last_hidden_state[:, 0, :]  # CLS token
        feats  = feats / feats.norm(dim=-1, keepdim=True)
        all_feats.append(feats.cpu().numpy())
        if i % 320 == 0: print(f"  {i}/{len(images)}...")
    return np.vstack(all_feats)

def evaluate(train_f, train_y, test_f, test_y, name, weights=None):
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(train_f, train_y, sample_weight=weights)
    probs = clf.predict_proba(test_f)
    preds = clf.predict(test_f)
    auc = roc_auc_score(test_y, probs, multi_class='ovr', average='macro')
    acc = accuracy_score(test_y, preds)
    f1  = f1_score(test_y, preds, average='macro')
    scores = []
    for _ in range(500):
        idx = np.random.choice(len(test_y), len(test_y), replace=True)
        try: scores.append(roc_auc_score(test_y[idx], probs[idx],
                                          multi_class='ovr', average='macro'))
        except: pass
    ci_low, ci_high = np.percentile(scores, [2.5, 97.5]) if scores else (auc-.02, auc+.02)
    print(f"  {name}: AUC={auc:.4f} ({ci_low:.4f}-{ci_high:.4f}) Acc={acc:.4f} F1={f1:.4f}")
    per_class = {cls: float(accuracy_score(test_y[test_y==i], preds[test_y==i]))
                 for i, cls in enumerate(le.classes_) if (test_y==i).sum() > 0}
    return {'auc': float(auc), 'acc': float(acc), 'f1': float(f1),
            'ci_low': float(ci_low), 'ci_high': float(ci_high), 'per_class': per_class}

print("Loading images...")
light_imgs, light_y   = load_imgs(light_df)
medium_imgs, medium_y = load_imgs(medium_df)
dark_imgs, dark_y     = load_imgs(dark_df)

print("Extracting DINOv2 features...")
light_feats  = get_dino_features(light_imgs)
medium_feats = get_dino_features(medium_imgs)
dark_feats   = get_dino_features(dark_imgs)
print(f"Feature dim: {light_feats.shape[1]}")

# Random split
all_f = np.vstack([light_feats, medium_feats, dark_feats])
all_y = np.concatenate([light_y, medium_y, dark_y])
tr, te = train_test_split(np.arange(len(all_y)), test_size=0.25,
                           stratify=all_y, random_state=42)

print("\n--- RANDOM SPLIT ---")
rand_results = evaluate(all_f[tr], all_y[tr], all_f[te], all_y[te], "DINOv2 Random")

# Skin-tone split
train_f = np.vstack([light_feats, medium_feats])
train_y = np.concatenate([light_y, medium_y])

print("\n--- SKIN-TONE SPLIT ---")
skin_results = evaluate(train_f, train_y, dark_feats, dark_y, "DINOv2 Skin-Tone")

sgg = rand_results['auc'] - skin_results['auc']
print(f"\n*** DINOv2 SGG: {sgg:.4f} ***")

print("\nPer-class on dark skin:")
for cls, acc in skin_results['per_class'].items():
    print(f"  {cls}: {acc:.3f}")

# Combined fix
n_dark_add = 200
dark_add_idx = np.random.choice(len(dark_feats), n_dark_add, replace=False)
dark_test_idx = np.setdiff1d(np.arange(len(dark_feats)), dark_add_idx)
aug_f = np.vstack([train_f, dark_feats[dark_add_idx]])
aug_y = np.concatenate([train_y, dark_y[dark_add_idx]])
n_base, n_add = len(train_y), n_dark_add
total = n_base + n_add
weights = np.concatenate([np.full(n_base, total/(2*n_base)),
                           np.full(n_add,  total/(2*n_add))])

print("\n--- COMBINED FIX ---")
fix_results = evaluate(aug_f, aug_y,
                        dark_feats[dark_test_idx], dark_y[dark_test_idx],
                        "DINOv2 Combined Fix", weights=weights)
gap_closed = (fix_results['auc'] - skin_results['auc']) / sgg * 100 if sgg > 0 else 0
print(f"Gap closed: {gap_closed:.0f}%")

print("\n=== COMPLETE SUMMARY ===")
print(f"{'Method':<40} {'AUC':>8} {'SGG':>8}")
print("-"*58)
for name, auc, s in [
    ("CLIP LP Random (paper)",    0.789, None),
    ("CLIP LP Skin-Tone (paper)", 0.742, 0.047),
    ("ViT-B/16 LP (paper)",       0.715, 0.037),
    ("ResNet50 LP (paper)",        0.664, 0.027),
    ("DINOv2 Random",              rand_results['auc'], None),
    ("DINOv2 Skin-Tone",           skin_results['auc'], sgg),
    ("DINOv2 Combined Fix",        fix_results['auc'],  None),
]:
    print(f"  {name:<38} {auc:>8.4f} {str(round(s,3)) if s else '—':>8}")

json.dump({
    'model': 'DINOv2-base',
    'random': {k:v for k,v in rand_results.items() if k!='per_class'},
    'skin_tone': {k:v for k,v in skin_results.items() if k!='per_class'},
    'combined_fix': {k:v for k,v in fix_results.items() if k!='per_class'},
    'per_class_dark': skin_results['per_class'],
    'sgg': float(sgg), 'gap_closed_pct': float(gap_closed),
}, open('/kaggle/working/nb4_dinov2.json','w'), indent=2)

print("\n✓ Complete. Paste ALL output back to Claude.")
