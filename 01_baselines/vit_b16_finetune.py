# ============================================================
# NOTEBOOK — ViT-B/16 Third Model Experiment
# Datasets: nazmusresan/fitzpatrick17k + nih-chest-xrays
# GPU T4 x1, Internet ON
# Expected runtime: ~20 minutes
# Adds ViT-B/16 as third architecture to all results tables
# After running, paste ALL output back to Claude
# ============================================================

!pip install transformers torch torchvision scikit-learn pandas numpy -q

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import ViTModel, ViTImageProcessor
from collections import Counter
import warnings, os, json
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# ── Load ViT-B/16 ─────────────────────────────────────────────
print("\nLoading ViT-B/16...")
vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device)
vit_model.eval()
print("ViT-B/16 loaded.")

# ── Feature extraction ────────────────────────────────────────
@torch.no_grad()
def get_vit_features(images, batch_size=32):
    all_feats = []
    valid = [img for img in images if img is not None]
    for i in range(0, len(valid), batch_size):
        batch = valid[i:i+batch_size]
        inputs = vit_processor(images=batch, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = vit_model(**inputs)
        # CLS token as image representation
        feats = outputs.last_hidden_state[:, 0, :]
        feats = feats / feats.norm(dim=-1, keepdim=True)
        all_feats.append(feats.cpu().numpy())
        if i % 320 == 0:
            print(f"  ViT: {i}/{len(valid)}...")
    return np.vstack(all_feats)

def evaluate(train_feats, train_labels, test_feats, test_labels, name, weights=None):
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(train_feats, train_labels, sample_weight=weights)
    probs = clf.predict_proba(test_feats)
    preds = clf.predict(test_feats)

    n_classes = len(np.unique(train_labels))
    if n_classes == 2:
        auc = roc_auc_score(test_labels, probs[:, 1])
    else:
        auc = roc_auc_score(test_labels, probs, multi_class='ovr', average='macro')

    acc = accuracy_score(test_labels, preds)
    f1 = f1_score(test_labels, preds, average='macro')

    scores = []
    for _ in range(500):
        idx = np.random.choice(len(test_labels), len(test_labels), replace=True)
        try:
            if n_classes == 2:
                scores.append(roc_auc_score(test_labels[idx], probs[idx, 1]))
            else:
                scores.append(roc_auc_score(test_labels[idx], probs[idx],
                                             multi_class='ovr', average='macro'))
        except: pass
    ci_low, ci_high = np.percentile(scores, [2.5, 97.5])
    print(f"{name} | AUC: {auc:.4f} ({ci_low:.4f}-{ci_high:.4f}) | Acc: {acc:.4f} | F1: {f1:.4f}")
    return {'auc': auc, 'acc': acc, 'f1': f1, 'ci_low': ci_low, 'ci_high': ci_high}, clf, probs

# ══════════════════════════════════════════════════════════════
# DATASET 1: FITZPATRICK17k
# ══════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("DATASET 1: FITZPATRICK17k — ViT-B/16")
print("="*55)

fitz_csv = '/kaggle/input/datasets/nazmusresan/fitzpatrick17k/New folder/fitzpatrick17k (1).csv'
fitz_img_dir = '/kaggle/input/datasets/nazmusresan/fitzpatrick17k/New folder/background removed'

df = pd.read_csv(fitz_csv)
df = df[df['fitzpatrick_scale'] > 0]

image_files = {}
for f in os.listdir(fitz_img_dir):
    if f.endswith('.jpg') or f.endswith('.png'):
        stem = f.replace('.jpg','').replace('.png','')
        image_files[stem] = os.path.join(fitz_img_dir, f)

df['local_path'] = df['md5hash'].map(image_files)
df = df[df['local_path'].notna()].copy()
df['skin_group'] = df['fitzpatrick_scale'].apply(
    lambda x: 'light' if x <= 2 else 'medium' if x <= 4 else 'dark'
)

MAX = 1000
light_df = df[df['skin_group']=='light'].sample(MAX, random_state=42)
medium_df = df[df['skin_group']=='medium'].sample(MAX, random_state=42)
dark_df = df[df['skin_group']=='dark'].sample(min(MAX, len(df[df['skin_group']=='dark'])), random_state=42)

def load_imgs(dataframe, label_col='three_partition_label'):
    imgs, lbls = [], []
    for _, row in dataframe.iterrows():
        try:
            img = Image.open(row['local_path']).convert('RGB').resize((224, 224))
            imgs.append(img)
            lbls.append(row[label_col])
        except: pass
    return imgs, lbls

print("Loading Fitzpatrick images...")
light_imgs, light_lbls = load_imgs(light_df)
medium_imgs, medium_lbls = load_imgs(medium_df)
dark_imgs, dark_lbls = load_imgs(dark_df)
print(f"Loaded: light={len(light_imgs)}, medium={len(medium_imgs)}, dark={len(dark_imgs)}")

print("\nExtracting ViT-B/16 features (Fitzpatrick)...")
light_feats = get_vit_features(light_imgs)
medium_feats = get_vit_features(medium_imgs)
dark_feats = get_vit_features(dark_imgs)
print(f"Feature shapes: {light_feats.shape}, {medium_feats.shape}, {dark_feats.shape}")

le = LabelEncoder()
all_lbls = light_lbls + medium_lbls + dark_lbls
le.fit(all_lbls)
light_y = le.transform(light_lbls)
medium_y = le.transform(medium_lbls)
dark_y = le.transform(dark_lbls)

# Random split
print("\n--- FITZPATRICK: RANDOM SPLIT ---")
all_feats = np.vstack([light_feats, medium_feats, dark_feats])
all_y = np.concatenate([light_y, medium_y, dark_y])
train_idx, test_idx = train_test_split(
    np.arange(len(all_y)), test_size=0.25, stratify=all_y, random_state=42)
fitz_random, _, _ = evaluate(all_feats[train_idx], all_y[train_idx],
                              all_feats[test_idx], all_y[test_idx], "ViT-B/16 Random")

# Skin-tone split
print("\n--- FITZPATRICK: SKIN-TONE SPLIT ---")
train_feats = np.vstack([light_feats, medium_feats])
train_y = np.concatenate([light_y, medium_y])
fitz_skin, clf_skin, skin_probs = evaluate(train_feats, train_y,
                                            dark_feats, dark_y, "ViT-B/16 Skin-Tone")

fitz_sgg = fitz_random['auc'] - fitz_skin['auc']
print(f"\n*** FITZPATRICK SGG (ViT-B/16): {fitz_sgg:.4f} ***")

# Per-class accuracy on dark skin
print("\n--- PER-CLASS ACCURACY ON DARK SKIN (ViT-B/16) ---")
skin_preds = clf_skin.predict(dark_feats)
for i, cls in enumerate(le.classes_):
    mask = dark_y == i
    if mask.sum() > 0:
        cls_acc = accuracy_score(dark_y[mask], skin_preds[mask])
        print(f"  {cls}: {cls_acc:.3f} (n={mask.sum()})")

# Mitigation — combined fix
print("\n--- FITZPATRICK: COMBINED FIX (ViT-B/16) ---")
n_dark_train = int(0.2 * len(dark_feats))
dark_train_idx = np.random.choice(len(dark_feats), n_dark_train, replace=False)
dark_test_idx = np.setdiff1d(np.arange(len(dark_feats)), dark_train_idx)

aug_train_feats = np.vstack([train_feats, dark_feats[dark_train_idx]])
aug_train_y = np.concatenate([train_y, dark_y[dark_train_idx]])
aug_groups = np.array(['light_medium']*len(train_y) + ['dark']*n_dark_train)
group_counts = Counter(aug_groups)
aug_total = len(aug_groups)
aug_weights = np.array([
    aug_total / (len(group_counts) * group_counts[g]) for g in aug_groups
])

fitz_combined, _, _ = evaluate(aug_train_feats, aug_train_y,
                                dark_feats[dark_test_idx], dark_y[dark_test_idx],
                                "ViT-B/16 Combined Fix", weights=aug_weights)

fitz_gap_closed = (fitz_combined['auc'] - fitz_skin['auc']) / fitz_sgg * 100 if fitz_sgg > 0 else 0
print(f"Gap closed: {fitz_gap_closed:.0f}%")

# ══════════════════════════════════════════════════════════════
# DATASET 2: NIH ChestX-ray14 — Age split
# ══════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("DATASET 2: NIH ChestX-ray14 — AGE SPLIT (ViT-B/16)")
print("="*55)

data_path = '/kaggle/input/datasets/organizations/nih-chest-xrays/data'
df_nih = pd.read_csv(os.path.join(data_path, 'Data_Entry_2017.csv'))
df_nih = df_nih[(df_nih['Patient Age'] > 0) & (df_nih['Patient Age'] < 120)]
df_nih['has_finding'] = (df_nih['Finding Labels'] != 'No Finding').astype(int)
df_nih['age_group'] = pd.cut(df_nih['Patient Age'],
    bins=[0, 40, 60, 120], labels=['young', 'middle', 'older'])

img_index = {}
for d in os.listdir(data_path):
    if d.startswith('images_'):
        img_dir = os.path.join(data_path, d, 'images')
        if os.path.exists(img_dir):
            for f in os.listdir(img_dir):
                img_index[f] = os.path.join(img_dir, f)

df_nih['image_path'] = df_nih['Image Index'].map(img_index)
df_nih = df_nih[df_nih['image_path'].notna()].copy()

N = 1000
young_df = df_nih[df_nih['age_group']=='young'].sample(N, random_state=42)
middle_df = df_nih[df_nih['age_group']=='middle'].sample(N, random_state=42)
older_df = df_nih[df_nih['age_group']=='older'].sample(min(N, len(df_nih[df_nih['age_group']=='older'])), random_state=42)

def load_nih_imgs(dataframe):
    imgs, lbls = [], []
    for _, row in dataframe.iterrows():
        try:
            img = Image.open(row['image_path']).convert('RGB')
            imgs.append(img)
            lbls.append(row['has_finding'])
        except: pass
    return imgs, np.array(lbls[:len(imgs)])

print("Loading NIH images...")
young_imgs, young_labels = load_nih_imgs(young_df)
middle_imgs, middle_labels = load_nih_imgs(middle_df)
older_imgs, older_labels = load_nih_imgs(older_df)
print(f"Loaded: young={len(young_imgs)}, middle={len(middle_imgs)}, older={len(older_imgs)}")

print("\nExtracting ViT-B/16 features (NIH)...")
young_feats = get_vit_features(young_imgs)
middle_feats = get_vit_features(middle_imgs)
older_feats = get_vit_features(older_imgs)

# Random split
print("\n--- NIH: RANDOM SPLIT ---")
all_nih_feats = np.vstack([young_feats, middle_feats, older_feats])
all_nih_labels = np.concatenate([young_labels, middle_labels, older_labels])
train_idx, test_idx = train_test_split(
    np.arange(len(all_nih_labels)), test_size=0.25,
    stratify=all_nih_labels, random_state=42)
nih_random, _, _ = evaluate(all_nih_feats[train_idx], all_nih_labels[train_idx],
                             all_nih_feats[test_idx], all_nih_labels[test_idx],
                             "ViT-B/16 Random")

# Age-aware split
print("\n--- NIH: AGE-AWARE SPLIT ---")
train_nih_feats = np.vstack([young_feats, middle_feats])
train_nih_labels = np.concatenate([young_labels, middle_labels])
nih_age, _, _ = evaluate(train_nih_feats, train_nih_labels,
                          older_feats, older_labels, "ViT-B/16 Age-Aware")

nih_sgg = nih_random['auc'] - nih_age['auc']
print(f"\n*** NIH AGE SGG (ViT-B/16): {nih_sgg:.4f} ***")

# ══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("COMPLETE ViT-B/16 RESULTS SUMMARY")
print("="*55)

print("\nFitzpatrick17k:")
print(f"  Random Split AUC:    {fitz_random['auc']:.4f} ({fitz_random['ci_low']:.4f}-{fitz_random['ci_high']:.4f})")
print(f"  Skin-Tone Split AUC: {fitz_skin['auc']:.4f} ({fitz_skin['ci_low']:.4f}-{fitz_skin['ci_high']:.4f})")
print(f"  Combined Fix AUC:    {fitz_combined['auc']:.4f} ({fitz_combined['ci_low']:.4f}-{fitz_combined['ci_high']:.4f})")
print(f"  SGG:                 {fitz_sgg:.4f}")
print(f"  Gap closed:          {fitz_gap_closed:.0f}%")

print("\nNIH ChestX-ray14 (Age):")
print(f"  Random Split AUC:    {nih_random['auc']:.4f} ({nih_random['ci_low']:.4f}-{nih_random['ci_high']:.4f})")
print(f"  Age-Aware Split AUC: {nih_age['auc']:.4f} ({nih_age['ci_low']:.4f}-{nih_age['ci_high']:.4f})")
print(f"  SGG:                 {nih_sgg:.4f}")

print("\nViT-B/16 feature dimension:", light_feats.shape[1])

results = {
    'model': 'ViT-B/16',
    'fitzpatrick': {
        'random': fitz_random,
        'skin_tone': fitz_skin,
        'combined_fix': fitz_combined,
        'sgg': fitz_sgg,
        'gap_closed_pct': fitz_gap_closed,
    },
    'nih_age': {
        'random': nih_random,
        'age_aware': nih_age,
        'sgg': nih_sgg,
    }
}

with open('/kaggle/working/vit_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nSaved to /kaggle/working/vit_results.json")
print("\n✓ Complete. Paste ALL output back to Claude.")
