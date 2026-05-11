# ============================================================
# NOTEBOOK 2 — Intersectional Analysis (Age × Skin Tone)
# Datasets: nazmusresan/fitzpatrick17k + nih-chest-xrays
# GPU T4, Internet ON. ~30 min.
# Priority: HIGH — new finding, no prior paper has this
#
# Tests whether dark-skin + older patients show LARGER gaps
# than either axis alone. If yes, it's a genuinely new result.
# ============================================================

!pip install transformers torch torchvision scikit-learn pandas numpy -q

import torch
import numpy as np, pandas as pd, os, json, warnings
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import CLIPModel, CLIPProcessor
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── LOAD CLIP ─────────────────────────────────────────────────
print("Loading CLIP...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
clip_model.eval()
print("CLIP loaded.")

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
        if i % 320 == 0: print(f"  {i}/{len(images)}...")
    return np.vstack(all_feats)

def load_img(path):
    try: return Image.open(path).convert('RGB').resize((224,224))
    except: return None

def evaluate(train_f, train_y, test_f, test_y, name):
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(train_f, train_y)
    probs = clf.predict_proba(test_f)
    preds = clf.predict(test_f)
    n_cls = len(np.unique(train_y))
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
            if n_cls == 2:
                scores.append(roc_auc_score(test_y[idx], probs[idx,1]))
            elif len(np.unique(test_y[idx])) == n_cls:
                scores.append(roc_auc_score(test_y[idx], probs[idx],
                                             multi_class='ovr', average='macro'))
        except: pass
    ci_low, ci_high = np.percentile(scores, [2.5, 97.5]) if scores else (auc-.02, auc+.02)
    print(f"  {name:<45} AUC={auc:.4f} ({ci_low:.4f}-{ci_high:.4f})")
    return {'auc': float(auc), 'ci_low': float(ci_low), 'ci_high': float(ci_high),
            'acc': float(acc), 'f1': float(f1), 'n_test': len(test_y)}

# ══════════════════════════════════════════════════════════════
# PART 1: NIH ChestX-ray14 — Age × Sex Intersectional
# (older women vs older men vs younger patients)
# ══════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("PART 1: NIH — Age × Sex Intersectional")
print("="*55)

nih_path = '/kaggle/input/datasets/organizations/nih-chest-xrays/data'
df_nih = pd.read_csv(os.path.join(nih_path, 'Data_Entry_2017.csv'))
df_nih = df_nih[(df_nih['Patient Age'] > 0) & (df_nih['Patient Age'] < 120)]
df_nih['has_finding'] = (df_nih['Finding Labels'] != 'No Finding').astype(int)
df_nih['age_group']   = pd.cut(df_nih['Patient Age'],
    bins=[0,40,60,120], labels=['young','middle','older'])

img_index = {}
for d in os.listdir(nih_path):
    if d.startswith('images_'):
        img_dir = os.path.join(nih_path, d, 'images')
        if os.path.exists(img_dir):
            for f in os.listdir(img_dir):
                img_index[f] = os.path.join(img_dir, f)

df_nih['image_path'] = df_nih['Image Index'].map(img_index)
df_nih = df_nih[df_nih['image_path'].notna()].copy()

print("Pathology prevalence by age × sex:")
for age in ['young','middle','older']:
    for sex in ['M','F']:
        sub = df_nih[(df_nih['age_group']==age) & (df_nih['Patient Gender']==sex)]
        print(f"  {age} {sex}: {sub['has_finding'].mean():.3f} (n={len(sub):,})")

N = 500
np.random.seed(42)
groups = {}
for age in ['young','middle','older']:
    for sex in ['M','F']:
        sub = df_nih[(df_nih['age_group']==age) & (df_nih['Patient Gender']==sex)]
        sample = sub.sample(min(N, len(sub)), random_state=42)
        imgs   = [load_img(p) for p in sample['image_path']]
        valid  = [(img, lbl) for img, lbl in
                  zip(imgs, sample['has_finding'].tolist()) if img is not None]
        if valid:
            groups[f"{age}_{sex}"] = {
                'imgs':   [v[0] for v in valid],
                'labels': np.array([v[1] for v in valid])
            }
            print(f"  Loaded {age}_{sex}: n={len(valid)}, "
                  f"prev={groups[f'{age}_{sex}']['labels'].mean():.3f}")

print("\nExtracting NIH features...")
for key in groups:
    groups[key]['feats'] = get_features(groups[key]['imgs'])
    print(f"  {key}: {groups[key]['feats'].shape}")

# Build training set: young + middle (both sexes)
train_feats  = np.vstack([groups[k]['feats']  for k in groups if 'young' in k or 'middle' in k])
train_labels = np.concatenate([groups[k]['labels'] for k in groups if 'young' in k or 'middle' in k])

# Random split baseline
all_feats  = np.vstack([groups[k]['feats']  for k in groups])
all_labels = np.concatenate([groups[k]['labels'] for k in groups])
tr, te = train_test_split(np.arange(len(all_labels)), test_size=0.25,
                           stratify=all_labels, random_state=42)

print("\n--- NIH RESULTS ---")
rand_nih = evaluate(all_feats[tr], all_labels[tr], all_feats[te], all_labels[te],
                     "NIH Random Split")

nih_results = {}
for key in ['older_M', 'older_F']:
    if key in groups:
        r = evaluate(train_feats, train_labels,
                      groups[key]['feats'], groups[key]['labels'],
                      f"NIH Age-Aware → {key}")
        nih_results[key] = r

# Intersectional: train on young+middle, test on older women specifically
if 'older_F' in groups and 'older_M' in groups:
    sgg_older_m = rand_nih['auc'] - nih_results.get('older_M', {}).get('auc', 0)
    sgg_older_f = rand_nih['auc'] - nih_results.get('older_F', {}).get('auc', 0)
    print(f"\n  SGG older men:   {sgg_older_m:.4f}")
    print(f"  SGG older women: {sgg_older_f:.4f}")
    print(f"  Intersectional amplification: {sgg_older_f - sgg_older_m:.4f}")

# ══════════════════════════════════════════════════════════════
# PART 2: Fitzpatrick — Skin Tone per-class SGG
# (how does malignant detection gap vary across Fitzpatrick I-VI?)
# ══════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("PART 2: Fitzpatrick — Granular Skin Tone Analysis")
print("Train on I-II, test on III, IV, V, VI separately")
print("="*55)

fitz_csv     = '/kaggle/input/datasets/nazmusresan/fitzpatrick17k/New folder/fitzpatrick17k (1).csv'
fitz_img_dir = '/kaggle/input/datasets/nazmusresan/fitzpatrick17k/New folder/background removed'

df_f = pd.read_csv(fitz_csv)
df_f = df_f[df_f['fitzpatrick_scale'] > 0]
img_files = {f.replace('.jpg','').replace('.png',''):
             os.path.join(fitz_img_dir, f)
             for f in os.listdir(fitz_img_dir)
             if f.endswith('.jpg') or f.endswith('.png')}
df_f['local_path'] = df_f['md5hash'].map(img_files)
df_f = df_f[df_f['local_path'].notna()].copy()

le_f = LabelEncoder()
le_f.fit(df_f['three_partition_label'].dropna())
print(f"Fitzpatrick classes: {le_f.classes_}")

# Sample per Fitzpatrick level
fitz_groups = {}
for level in [1,2,3,4,5,6]:
    sub = df_f[df_f['fitzpatrick_scale']==level]
    sample = sub.sample(min(300, len(sub)), random_state=42)
    imgs, lbls = [], []
    for _, row in sample.iterrows():
        try:
            img = Image.open(row['local_path']).convert('RGB').resize((224,224))
            imgs.append(img)
            lbls.append(le_f.transform([row['three_partition_label']])[0])
        except: pass
    if imgs:
        fitz_groups[level] = {'imgs': imgs, 'labels': np.array(lbls)}
        print(f"  Fitzpatrick {level}: n={len(imgs)}, "
              f"dist={dict(zip(le_f.classes_, np.bincount(np.array(lbls), minlength=3)))}")

print("\nExtracting Fitzpatrick features...")
for level in fitz_groups:
    fitz_groups[level]['feats'] = get_features(fitz_groups[level]['imgs'])

# Train on Fitzpatrick I (lightest) only
train_f = fitz_groups[1]['feats']
train_y = fitz_groups[1]['labels']

# Random split baseline
all_f = np.vstack([fitz_groups[k]['feats']  for k in sorted(fitz_groups)])
all_y = np.concatenate([fitz_groups[k]['labels'] for k in sorted(fitz_groups)])
tr, te = train_test_split(np.arange(len(all_y)), test_size=0.25,
                           stratify=all_y, random_state=42)
rand_fitz = evaluate(all_f[tr], all_y[tr], all_f[te], all_y[te], "Fitz Random")

print("\n--- FITZPATRICK GRANULAR RESULTS ---")
fitz_sgg = {}
for level in sorted(fitz_groups):
    if level == 1: continue
    r = evaluate(train_f, train_y,
                  fitz_groups[level]['feats'], fitz_groups[level]['labels'],
                  f"Train I → Test Fitz-{level}")
    fitz_sgg[level] = rand_fitz['auc'] - r['auc']
    print(f"    SGG Fitz-{level}: {fitz_sgg[level]:.4f}")

print("\nMonotonic degradation across Fitzpatrick scale:")
for level in sorted(fitz_sgg):
    bar = "█" * int(fitz_sgg[level] * 200)
    print(f"  Fitz-{level}: {fitz_sgg[level]:.4f} {bar}")

# ── Save ──────────────────────────────────────────────────────
results = {
    'nih_intersectional': {k: v for k, v in nih_results.items()},
    'nih_random_auc': rand_nih['auc'],
    'fitzpatrick_granular_sgg': {str(k): float(v) for k, v in fitz_sgg.items()},
    'fitzpatrick_random_auc': rand_fitz['auc'],
}
json.dump(results, open('/kaggle/working/nb2_intersectional.json','w'), indent=2)
print("\n✓ Complete. Paste ALL output back to Claude.")
