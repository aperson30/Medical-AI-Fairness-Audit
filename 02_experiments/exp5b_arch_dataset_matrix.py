# ============================================================
# NOTEBOOK 5 — Complete Architecture × Dataset Matrix
# Datasets: nih-chest-xrays (add this dataset)
#            Fed-ISIC downloads automatically
# GPU T4, Internet ON. ~25 min.
# Priority: MEDIUM — completes the 4×3 architecture×dataset table
#
# Adds ViT-B/16 to Fed-ISIC2019 and NIH results.
# ============================================================

!pip install transformers torch torchvision scikit-learn datasets pandas numpy -q

import torch
import numpy as np, pandas as pd, os, json, warnings
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from transformers import ViTModel, ViTImageProcessor
from datasets import load_dataset
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

print("Loading ViT-B/16...")
vit_proc  = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device)
vit_model.eval()
print("ViT-B/16 loaded.")

@torch.no_grad()
def get_features(images, batch_size=32):
    all_feats = []
    for i in range(0, len(images), batch_size):
        batch  = images[i:i+batch_size]
        inputs = vit_proc(images=batch, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feats  = vit_model(**inputs).last_hidden_state[:, 0, :]
        feats  = feats / feats.norm(dim=-1, keepdim=True)
        all_feats.append(feats.cpu().numpy())
        if i % 320 == 0: print(f"  {i}/{len(images)}...")
    return np.vstack(all_feats)

def evaluate(train_f, train_y, test_f, test_y, name):
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(train_f, train_y)
    probs = clf.predict_proba(test_f)
    preds = clf.predict(test_f)
    n_cls = len(np.unique(np.concatenate([train_y, test_y])))
    if n_cls == 2:
        auc = roc_auc_score(test_y, probs[:,1])
    else:
        auc = roc_auc_score(test_y, probs, multi_class='ovr', average='macro')
    acc = accuracy_score(test_y, preds)
    f1  = f1_score(test_y, preds, average='macro', zero_division=0)
    scores = []
    for _ in range(500):
        idx = np.random.choice(len(test_y), len(test_y), replace=True)
        try:
            s = (roc_auc_score(test_y[idx], probs[idx,1]) if n_cls==2
                 else roc_auc_score(test_y[idx], probs[idx],
                                    multi_class='ovr', average='macro')
                 if len(np.unique(test_y[idx]))==n_cls else None)
            if s: scores.append(s)
        except: pass
    ci_low  = float(np.percentile(scores, 2.5))  if scores else float(auc-.02)
    ci_high = float(np.percentile(scores, 97.5)) if scores else float(auc+.02)
    print(f"  {name}: AUC={auc:.4f} ({ci_low:.4f}-{ci_high:.4f}) Acc={acc:.4f}")
    return {'auc': float(auc), 'acc': float(acc), 'f1': float(f1),
            'ci_low': ci_low, 'ci_high': ci_high}

# ══════════════════════════════════════════════════════════════
# PART 1: Fed-ISIC2019
# ══════════════════════════════════════════════════════════════
print("\n=== PART 1: Fed-ISIC2019 ===")
dataset = load_dataset("flwrlabs/fed-isic2019")
print(f"Splits: {list(dataset.keys())}")

def load_isic(split, n=1500):
    data = dataset[split]
    imgs, labels = [], []
    for idx in range(min(n, len(data))):
        try:
            item = data[idx]
            img  = item['image']
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.array(img, dtype=np.uint8))
            imgs.append(img.convert('RGB').resize((224,224)))
            labels.append(int(item['label']))
        except: pass
    return imgs, np.array(labels, dtype=np.int64)

train_isic_imgs, train_isic_y = load_isic('train', 1500)
test_isic_imgs,  test_isic_y  = load_isic('test',  500)
print(f"ISIC: train={len(train_isic_imgs)}, test={len(test_isic_imgs)}")
print(f"ISIC classes: {len(np.unique(train_isic_y))}")

print("Extracting ISIC features...")
train_isic_f = get_features(train_isic_imgs)
test_isic_f  = get_features(test_isic_imgs)

# Random split
all_isic_f = np.vstack([train_isic_f, test_isic_f])
all_isic_y = np.concatenate([train_isic_y, test_isic_y])
tr, te = train_test_split(np.arange(len(all_isic_y)), test_size=0.25,
                           stratify=all_isic_y, random_state=42)
isic_rand = evaluate(all_isic_f[tr], all_isic_y[tr],
                      all_isic_f[te], all_isic_y[te], "ViT-B/16 ISIC Random")
isic_inst = evaluate(train_isic_f, train_isic_y, test_isic_f, test_isic_y,
                      "ViT-B/16 ISIC Institution-like")
isic_sgg = isic_rand['auc'] - isic_inst['auc']
ci_overlap = bool(isic_inst['ci_low'] < isic_rand['ci_high'] and
                  isic_rand['ci_low'] < isic_inst['ci_high'])
print(f"ISIC SGG: {isic_sgg:.4f} ({'not significant' if ci_overlap else 'significant'})")

# ══════════════════════════════════════════════════════════════
# PART 2: NIH ChestX-ray14 — Sex split
# ══════════════════════════════════════════════════════════════
print("\n=== PART 2: NIH ChestX-ray14 — Sex Split ===")
nih_path = '/kaggle/input/datasets/organizations/nih-chest-xrays/data'
df_nih = pd.read_csv(os.path.join(nih_path, 'Data_Entry_2017.csv'))
df_nih = df_nih[(df_nih['Patient Age']>0) & (df_nih['Patient Age']<120)]
df_nih['has_finding'] = (df_nih['Finding Labels'] != 'No Finding').astype(int)

img_index = {}
for d in os.listdir(nih_path):
    if d.startswith('images_'):
        img_dir = os.path.join(nih_path, d, 'images')
        if os.path.exists(img_dir):
            for f in os.listdir(img_dir):
                img_index[f] = os.path.join(img_dir, f)
df_nih['image_path'] = df_nih['Image Index'].map(img_index)
df_nih = df_nih[df_nih['image_path'].notna()].copy()

N = 1000
male_df   = df_nih[df_nih['Patient Gender']=='M'].sample(N, random_state=42)
female_df = df_nih[df_nih['Patient Gender']=='F'].sample(N, random_state=42)

def load_nih(dataframe):
    imgs, lbls = [], []
    for _, row in dataframe.iterrows():
        try:
            img = Image.open(row['image_path']).convert('RGB')
            imgs.append(img)
            lbls.append(int(row['has_finding']))
        except: pass
    return imgs, np.array(lbls, dtype=np.int64)

print("Loading NIH images...")
male_imgs,   male_y   = load_nih(male_df)
female_imgs, female_y = load_nih(female_df)
print(f"Male: {len(male_imgs)}, Female: {len(female_imgs)}")

print("Extracting NIH features...")
male_feats   = get_features(male_imgs)
female_feats = get_features(female_imgs)

# Random split
all_nih_f = np.vstack([male_feats, female_feats])
all_nih_y = np.concatenate([male_y, female_y])
tr, te = train_test_split(np.arange(len(all_nih_y)), test_size=0.25,
                           stratify=all_nih_y, random_state=42)
nih_rand = evaluate(all_nih_f[tr], all_nih_y[tr], all_nih_f[te], all_nih_y[te],
                     "ViT-B/16 NIH Random")
# Sex split: train on male, test on female
nih_sex = evaluate(male_feats, male_y, female_feats, female_y,
                    "ViT-B/16 NIH Sex-Aware")
nih_sgg = nih_rand['auc'] - nih_sex['auc']
print(f"NIH Sex SGG: {nih_sgg:.4f}")

# ── Summary ───────────────────────────────────────────────────
print("\n=== COMPLETE SUMMARY ===")
print(f"\nFed-ISIC2019:")
print(f"  CLIP Random: 0.863 | CLIP Inst: 0.845 | SGG: 0.019")
print(f"  ResNet50 Rand: 0.852 | ResNet50 Inst: 0.850 | SGG: 0.002")
print(f"  ViT-B/16 Rand: {isic_rand['auc']:.4f} | ViT-B/16 Inst: {isic_inst['auc']:.4f} | SGG: {isic_sgg:.4f}")
print(f"\nNIH Sex Split:")
print(f"  ViT-B/16 Random: {nih_rand['auc']:.4f} | Sex-Aware: {nih_sex['auc']:.4f} | SGG: {nih_sgg:.4f}")

json.dump({
    'fed_isic': {'random': isic_rand, 'institution': isic_inst,
                  'sgg': float(isic_sgg), 'ci_overlap': ci_overlap},
    'nih_sex': {'random': nih_rand, 'sex_aware': nih_sex, 'sgg': float(nih_sgg)},
}, open('/kaggle/working/nb5_matrix.json','w'), indent=2)
print("\n✓ Complete. Paste ALL output back to Claude.")
