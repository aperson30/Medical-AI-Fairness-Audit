# ============================================================
# NOTEBOOK — Larger Dark-Skin Sample (ALL 2168 dark images)
# Datasets: nazmusresan/fitzpatrick17k
# GPU T4 x1, Internet ON
# Expected runtime: ~15 minutes
# Purpose: address n=97 reviewer concern
# After running, paste ALL output back to Claude
# ============================================================

!pip install transformers torch torchvision scikit-learn pandas numpy scipy -q

import torch
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import CLIPProcessor, CLIPModel
from scipy import stats
import warnings, os, json
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── Load data ─────────────────────────────────────────────────
fitz_csv = '/kaggle/input/datasets/nazmusresan/fitzpatrick17k/New folder/fitzpatrick17k (1).csv'
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

print(f"Total: {len(df)}")
print(f"Dark skin available: {len(df[df['skin_group']=='dark'])}")
print(df[df['skin_group']=='dark']['three_partition_label'].value_counts())

# Train: 1000 light + 1000 medium (same as before)
# Test: ALL 2168 dark-skin images (up from 800)
light_df = df[df['skin_group']=='light'].sample(1000, random_state=42)
medium_df = df[df['skin_group']=='medium'].sample(1000, random_state=42)
dark_df = df[df['skin_group']=='dark']  # ALL

print(f"\nTrain: {len(light_df)} light + {len(medium_df)} medium | Test: {len(dark_df)} dark (ALL)")

# ── Load CLIP ─────────────────────────────────────────────────
print("\nLoading CLIP...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
clip_model.eval()
print("CLIP loaded.")

# ── Feature extraction ────────────────────────────────────────
@torch.no_grad()
def get_features(dataframe, desc="", batch_size=32):
    imgs, lbls = [], []
    for _, row in dataframe.iterrows():
        try:
            img = Image.open(row['local_path']).convert('RGB').resize((224,224))
            imgs.append(img)
            lbls.append(row['three_partition_label'])
        except: pass
    all_feats = []
    for i in range(0, len(imgs), batch_size):
        batch = imgs[i:i+batch_size]
        inputs = clip_processor(images=batch, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feats = clip_model.get_image_features(**inputs)
        if not isinstance(feats, torch.Tensor):
            feats = feats.pooler_output if hasattr(feats, 'pooler_output') \
                    else feats.last_hidden_state[:,0]
        feats = feats / feats.norm(dim=-1, keepdim=True)
        all_feats.append(feats.cpu().numpy())
        if i % 320 == 0:
            print(f"  {desc}: {i}/{len(imgs)}...")
    return np.vstack(all_feats), lbls

print("\nExtracting features...")
light_feats, light_lbls = get_features(light_df, "light")
medium_feats, medium_lbls = get_features(medium_df, "medium")
dark_feats, dark_lbls = get_features(dark_df, "dark ALL")

le = LabelEncoder()
le.fit(light_lbls + medium_lbls + dark_lbls)
light_y = le.transform(light_lbls)
medium_y = le.transform(medium_lbls)
dark_y = le.transform(dark_lbls)
print(f"Classes: {le.classes_}")
print(f"Dark class counts: {dict(zip(le.classes_, np.bincount(dark_y)))}")

# ── Skin-tone split classifier ────────────────────────────────
train_feats = np.vstack([light_feats, medium_feats])
train_y = np.concatenate([light_y, medium_y])
clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
clf.fit(train_feats, train_y)

dark_preds = clf.predict(dark_feats)
dark_probs = clf.predict_proba(dark_feats)

# ── Per-class results ─────────────────────────────────────────
print("\n" + "="*55)
print("PER-CLASS ACCURACY — ALL DARK-SKIN IMAGES")
print("="*55)
benign_idx = list(le.classes_).index('benign')
mal_idx = list(le.classes_).index('malignant')

for i, cls in enumerate(le.classes_):
    mask = dark_y == i
    n = mask.sum()
    acc_cls = accuracy_score(dark_y[mask], dark_preds[mask])
    ci = stats.binom.interval(0.95, n, max(acc_cls, 1e-10))
    print(f"  {cls}: n={n}, acc={acc_cls:.3f} "
          f"(95% CI: {ci[0]/n:.3f}-{ci[1]/n:.3f})")

# ── AUC and SGG ──────────────────────────────────────────────
auc = roc_auc_score(dark_y, dark_probs, multi_class='ovr', average='macro')
scores = []
for _ in range(1000):
    idx = np.random.choice(len(dark_y), len(dark_y), replace=True)
    try:
        scores.append(roc_auc_score(dark_y[idx], dark_probs[idx],
                                     multi_class='ovr', average='macro'))
    except: pass
ci_low, ci_high = np.percentile(scores, [2.5, 97.5])
print(f"\nFull dark AUC: {auc:.4f} ({ci_low:.4f}-{ci_high:.4f})")

# Random split AUC
all_feats = np.vstack([light_feats, medium_feats, dark_feats])
all_y = np.concatenate([light_y, medium_y, dark_y])
tr, te = train_test_split(np.arange(len(all_y)), test_size=0.25,
                           stratify=all_y, random_state=42)
clf_rand = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
clf_rand.fit(all_feats[tr], all_y[tr])
rand_probs = clf_rand.predict_proba(all_feats[te])
rand_auc = roc_auc_score(all_y[te], rand_probs, multi_class='ovr', average='macro')
scores_rand = []
for _ in range(1000):
    idx = np.random.choice(len(te), len(te), replace=True)
    try:
        scores_rand.append(roc_auc_score(all_y[te[idx]], rand_probs[idx],
                                          multi_class='ovr', average='macro'))
    except: pass
rand_ci_low, rand_ci_high = np.percentile(scores_rand, [2.5, 97.5])

sgg = rand_auc - auc
print(f"Random AUC: {rand_auc:.4f} ({rand_ci_low:.4f}-{rand_ci_high:.4f})")
print(f"SGG: {sgg:.4f}")
print(f"CIs overlap: {ci_high > rand_ci_low}")

# Confusion matrix
print(f"\nConfusion matrix:")
print(f"Classes: {le.classes_}")
print(confusion_matrix(dark_y, dark_preds))

# ── Summary ───────────────────────────────────────────────────
print("\n" + "="*55)
print("SUMMARY — KEY NUMBERS FOR PAPER")
print("="*55)
benign_mask = dark_y == benign_idx
benign_acc = accuracy_score(dark_y[benign_mask], dark_preds[benign_mask])
benign_n = benign_mask.sum()
benign_ci = stats.binom.interval(0.95, benign_n, max(benign_acc, 1e-10))
mal_mask = dark_y == mal_idx
mal_acc = accuracy_score(dark_y[mal_mask], dark_preds[mal_mask])

print(f"Dark test n (total): {len(dark_y)}")
print(f"Benign: n={benign_n}, acc={benign_acc:.3f}, "
      f"CI ({benign_ci[0]/benign_n:.3f}-{benign_ci[1]/benign_n:.3f})")
print(f"Malignant: n={mal_mask.sum()}, acc={mal_acc:.3f}")
print(f"Dark AUC: {auc:.4f} ({ci_low:.4f}-{ci_high:.4f})")
print(f"Random AUC: {rand_auc:.4f} ({rand_ci_low:.4f}-{rand_ci_high:.4f})")
print(f"SGG: {sgg:.4f}")
print(f"\nOriginal (n=800): benign acc=0.000, SGG=0.047")
print(f"Full set (n={len(dark_y)}): benign acc={benign_acc:.3f}, SGG={sgg:.4f}")

results = {
    'n_dark': int(len(dark_y)),
    'n_benign': int(benign_n),
    'benign_acc': float(benign_acc),
    'benign_ci': [float(benign_ci[0]/benign_n), float(benign_ci[1]/benign_n)],
    'n_malignant': int(mal_mask.sum()),
    'malignant_acc': float(mal_acc),
    'dark_auc': float(auc),
    'dark_ci': [float(ci_low), float(ci_high)],
    'rand_auc': float(rand_auc),
    'rand_ci': [float(rand_ci_low), float(rand_ci_high)],
    'sgg': float(sgg),
}
with open('/kaggle/working/larger_sample_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved to /kaggle/working/larger_sample_results.json")
print("\n✓ Complete. Paste ALL output back to Claude.")
