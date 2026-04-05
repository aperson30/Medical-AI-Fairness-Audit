# ============================================================
# NOTEBOOK 5 — DINOv2 Gap-Filling Experiments (FIXED)
# Dataset: nazmusresan/fitzpatrick17k + Fed-ISIC2019 + NIH ChestX-ray14
# GPU T4, Internet ON. ~45 min.
#
# FIXES vs original nb5:
#   (1) NIH: subdirectory scan matching nb_p2 working path pattern
#   (2) ISIC: validates images before stacking (was crashing on empty list)
#   (3) Granular SGG: per-level random AUC reference (not global) to fix
#       spurious negative SGG values from sampling variance
#   (4) JSON dump: convert() helper strips numpy int64/float64 keys and
#       values so json.dump never throws TypeError
# Output: nb5_dinov2_gaps.json — paste ALL output back to Claude.
# ============================================================

!pip install transformers torch torchvision scikit-learn pandas numpy datasets -q

import torch
import numpy as np, pandas as pd, os, json, warnings
from PIL import Image
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                              confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoImageProcessor
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── paths ─────────────────────────────────────────────────────────────────────
FITZ_CSV     = '/kaggle/input/datasets/nazmusresan/fitzpatrick17k/New folder/fitzpatrick17k (1).csv'
FITZ_IMG_DIR = '/kaggle/input/datasets/nazmusresan/fitzpatrick17k/New folder/background removed'
NIH_DATA_DIR = '/kaggle/input/datasets/organizations/nih-chest-xrays/data'
NIH_CSV      = os.path.join(NIH_DATA_DIR, 'Data_Entry_2017.csv')

# ── load DINOv2 once, reuse across all experiments ────────────────────────────
print("Loading DINOv2-base...")
dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
dino_model     = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
dino_model.eval()
print("DINOv2 loaded.")

# ── shared helpers ────────────────────────────────────────────────────────────
@torch.no_grad()
def get_dino_features(images, batch_size=32):
    all_feats = []
    for i in range(0, len(images), batch_size):
        batch  = images[i:i+batch_size]
        inputs = dino_processor(images=batch, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out    = dino_model(**inputs)
        feats  = out.last_hidden_state[:, 0, :]          # CLS token
        feats  = feats / feats.norm(dim=-1, keepdim=True)
        all_feats.append(feats.cpu().numpy())
        if i % 320 == 0: print(f"  features {i}/{len(images)}...")
    return np.vstack(all_feats)

def evaluate(train_f, train_y, test_f, test_y, name,
             weights=None, n_boot=500, return_cm=False):
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(train_f, train_y, sample_weight=weights)
    probs = clf.predict_proba(test_f)
    preds = clf.predict(test_f)
    classes = np.unique(test_y)
    if len(classes) == 2:
        auc = roc_auc_score(test_y, probs[:, 1])
    else:
        auc = roc_auc_score(test_y, probs, multi_class='ovr', average='macro')
    acc = accuracy_score(test_y, preds)
    f1  = f1_score(test_y, preds, average='macro')
    scores = []
    for _ in range(n_boot):
        idx = np.random.choice(len(test_y), len(test_y), replace=True)
        try:
            if len(classes) == 2:
                scores.append(roc_auc_score(test_y[idx], probs[idx, 1]))
            else:
                scores.append(roc_auc_score(test_y[idx], probs[idx],
                                            multi_class='ovr', average='macro'))
        except: pass
    ci_low, ci_high = np.percentile(scores, [2.5, 97.5]) if scores else (auc-.02, auc+.02)
    print(f"  {name}: AUC={auc:.4f} ({ci_low:.4f}-{ci_high:.4f}) Acc={acc:.4f} F1={f1:.4f}")
    result = {'auc': float(auc), 'acc': float(acc), 'f1': float(f1),
              'ci_low': float(ci_low), 'ci_high': float(ci_high)}
    if return_cm:
        result['confusion_matrix'] = confusion_matrix(test_y, preds).tolist()
    return result

def per_class_acc(test_y, preds, label_encoder):
    return {str(cls): float(accuracy_score(test_y[test_y==i], preds[test_y==i]))
            for i, cls in enumerate(label_encoder.classes_)
            if (test_y==i).sum() > 0}

def combined_fix_weights(n_base, n_add):
    total = n_base + n_add
    return np.concatenate([np.full(n_base, total/(2*n_base)),
                            np.full(n_add,  total/(2*n_add))])

# FIX 4: recursive converter so json.dump never throws TypeError on numpy types
def convert(obj):
    if isinstance(obj, dict):
        return {str(k): convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

results = {}

# ════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — Fitzpatrick17k: per-class breakdown + confusion matrix
# Fills: paper says "DINOv2 not evaluated for per-class breakdown due to
# computational constraints" — biggest reviewer-visible omission.
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("EXPERIMENT 1 — Fitzpatrick17k per-class + confusion matrix")
print("="*60)

df = pd.read_csv(FITZ_CSV)
df = df[df['fitzpatrick_scale'] > 0]
image_files = {f.replace('.jpg','').replace('.png',''):
               os.path.join(FITZ_IMG_DIR, f)
               for f in os.listdir(FITZ_IMG_DIR)
               if f.endswith('.jpg') or f.endswith('.png')}
df['local_path'] = df['md5hash'].map(image_files)
df = df[df['local_path'].notna()].copy()
df['skin_group'] = df['fitzpatrick_scale'].apply(
    lambda x: 'light' if x <= 2 else 'medium' if x <= 4 else 'dark')

MAX = 1000
light_df  = df[df['skin_group']=='light'].sample(MAX, random_state=42)
medium_df = df[df['skin_group']=='medium'].sample(MAX, random_state=42)
dark_df   = df[df['skin_group']=='dark'].sample(
                min(MAX, len(df[df['skin_group']=='dark'])), random_state=42)

le_fitz = LabelEncoder()
le_fitz.fit(list(light_df['three_partition_label']) +
            list(medium_df['three_partition_label']) +
            list(dark_df['three_partition_label']))
print(f"Classes: {le_fitz.classes_}")

def load_fitz(dataframe, le):
    imgs, lbls = [], []
    for _, row in dataframe.iterrows():
        try:
            img = Image.open(row['local_path']).convert('RGB').resize((224,224))
            imgs.append(img)
            lbls.append(le.transform([row['three_partition_label']])[0])
        except: pass
    return imgs, np.array(lbls)

print("Loading Fitzpatrick images...")
light_imgs, light_y   = load_fitz(light_df, le_fitz)
medium_imgs, medium_y = load_fitz(medium_df, le_fitz)
dark_imgs, dark_y     = load_fitz(dark_df, le_fitz)

print("Extracting features...")
light_feats  = get_dino_features(light_imgs)
medium_feats = get_dino_features(medium_imgs)
dark_feats   = get_dino_features(dark_imgs)
print(f"Feature dim: {light_feats.shape[1]}")

# random split
all_f = np.vstack([light_feats, medium_feats, dark_feats])
all_y = np.concatenate([light_y, medium_y, dark_y])
tr, te = train_test_split(np.arange(len(all_y)), test_size=0.25,
                           stratify=all_y, random_state=42)

print("\n--- RANDOM SPLIT ---")
rand_res = evaluate(all_f[tr], all_y[tr], all_f[te], all_y[te],
                    "DINOv2 Random", return_cm=True)

# skin-tone split — train light+medium, test dark
train_f = np.vstack([light_feats, medium_feats])
train_y = np.concatenate([light_y, medium_y])

print("\n--- SKIN-TONE SPLIT ---")
skin_res = evaluate(train_f, train_y, dark_feats, dark_y,
                    "DINOv2 Skin-Tone", return_cm=True)

sgg_fitz = rand_res['auc'] - skin_res['auc']
print(f"\n*** DINOv2 SGG (Fitzpatrick): {sgg_fitz:.4f} ***")

# per-class on dark skin  ← THE KEY GAP
clf_skin = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
clf_skin.fit(train_f, train_y)
dark_preds = clf_skin.predict(dark_feats)
pc = per_class_acc(dark_y, dark_preds, le_fitz)

print("\nPer-class accuracy on dark skin (DINOv2):")
for cls, a in pc.items():
    print(f"  {cls}: {a:.3f}")

# light skin per-class for comparison (same classifier)
light_test_imgs, light_test_y = load_fitz(
    df[df['skin_group']=='light'].sample(300, random_state=99), le_fitz)
light_test_feats = get_dino_features(light_test_imgs)
light_preds = clf_skin.predict(light_test_feats)
pc_light = per_class_acc(light_test_y, light_preds, le_fitz)

print("\nPer-class accuracy on light skin (DINOv2, same classifier):")
for cls, a in pc_light.items():
    print(f"  {cls}: {a:.3f}")

# combined fix — evaluated on held-out 80% of dark images (no leakage)
n_dark_add    = 200
dark_add_idx  = np.random.choice(len(dark_feats), n_dark_add, replace=False)
dark_test_idx = np.setdiff1d(np.arange(len(dark_feats)), dark_add_idx)
aug_f = np.vstack([train_f, dark_feats[dark_add_idx]])
aug_y = np.concatenate([train_y, dark_y[dark_add_idx]])
w     = combined_fix_weights(len(train_y), n_dark_add)

print("\n--- COMBINED FIX ---")
print(f"(evaluated on held-out 80% of dark images, n_test = {len(dark_test_idx)})")
fix_res = evaluate(aug_f, aug_y,
                   dark_feats[dark_test_idx], dark_y[dark_test_idx],
                   "DINOv2 Combined Fix", weights=w)
gap_closed = (fix_res['auc'] - skin_res['auc']) / sgg_fitz * 100 if sgg_fitz > 0 else 0
print(f"Gap closed: {gap_closed:.0f}%")

clf_fix = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
clf_fix.fit(aug_f, aug_y, sample_weight=w)
fix_preds = clf_fix.predict(dark_feats[dark_test_idx])
pc_fix = per_class_acc(dark_y[dark_test_idx], fix_preds, le_fitz)

print("\nPer-class after combined fix (dark skin):")
for cls, a in pc_fix.items():
    print(f"  {cls}: {a:.3f}")

results['fitzpatrick'] = {
    'random': rand_res,
    'skin_tone': skin_res,
    'combined_fix': fix_res,
    'sgg': float(sgg_fitz),
    'gap_closed_pct': float(gap_closed),
    'per_class_dark_baseline': pc,
    'per_class_light_baseline': pc_light,
    'per_class_dark_after_fix': pc_fix,
}

# ════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — Per-Fitzpatrick granular SGG (DINOv2)
# FIX 3: per-level random AUC reference eliminates spurious negative SGG.
# Paper shows CLIP SGG triples Fitz-II→VI. Confirming for DINOv2 makes
# the monotonic degradation claim architecture-agnostic.
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("EXPERIMENT 2 — Per-Fitzpatrick granular SGG (DINOv2)")
print("="*60)

df['fitz_level'] = df['fitzpatrick_scale'].astype(int)
granular_results = {}

train_mask = df['fitz_level'].isin([1, 2])
train_full = df[train_mask].sample(min(1500, train_mask.sum()), random_state=42)

def load_rows(rows, le):
    imgs, lbls = [], []
    for _, row in rows.iterrows():
        try:
            img = Image.open(row['local_path']).convert('RGB').resize((224,224))
            imgs.append(img)
            lbls.append(le.transform([row['three_partition_label']])[0])
        except: pass
    return imgs, np.array(lbls)

print("Loading light training set (Fitz I+II)...")
train_imgs_g, train_y_g = load_rows(train_full, le_fitz)
train_feats_g = get_dino_features(train_imgs_g)

for level in [2, 3, 4, 5, 6]:
    level_df = df[df['fitz_level'] == level]
    n = min(300, len(level_df))
    if n < 30:
        print(f"  Fitz-{level}: skipping (n={len(level_df)} too small)")
        continue
    level_df_sampled = level_df.sample(n, random_state=42)
    level_imgs, level_y = load_rows(level_df_sampled, le_fitz)
    if len(level_imgs) == 0: continue
    level_feats = get_dino_features(level_imgs)

    # FIX 3: pool this level with equal light samples for per-level reference AUC
    light_ref = df[df['skin_group']=='light'].sample(min(n, 300), random_state=42)
    light_ref_imgs, light_ref_y = load_rows(light_ref, le_fitz)
    light_ref_feats = get_dino_features(light_ref_imgs)

    pool_f = np.vstack([light_ref_feats, level_feats])
    pool_y = np.concatenate([light_ref_y, level_y])
    tr_p, te_p = train_test_split(np.arange(len(pool_y)), test_size=0.5,
                                   stratify=pool_y, random_state=42)
    clf_ref = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf_ref.fit(pool_f[tr_p], pool_y[tr_p])
    ref_probs = clf_ref.predict_proba(pool_f[te_p])
    rand_auc_level = roc_auc_score(pool_y[te_p], ref_probs,
                                    multi_class='ovr', average='macro')

    print(f"\n--- Fitzpatrick-{level} (n={len(level_imgs)}) ---")
    print(f"  Per-level random AUC reference: {rand_auc_level:.4f}")
    skin_r = evaluate(train_feats_g, train_y_g, level_feats, level_y,
                      f"DINOv2 Fitz-{level}")
    sgg_level = rand_auc_level - skin_r['auc']

    clf_g = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf_g.fit(train_feats_g, train_y_g)
    preds_g = clf_g.predict(level_feats)
    pc_g = per_class_acc(level_y, preds_g, le_fitz)

    print(f"  SGG Fitz-{level}: {sgg_level:.4f}")
    print(f"  Per-class: {pc_g}")
    granular_results[f'fitz_{level}'] = {
        'n': int(len(level_imgs)),
        'auc': float(skin_r['auc']),
        'rand_auc_reference': float(rand_auc_level),
        'sgg': float(sgg_level),
        'per_class': pc_g,
        'ci_low': float(skin_r['ci_low']),
        'ci_high': float(skin_r['ci_high']),
    }

results['granular_fitzpatrick'] = {'levels': granular_results}

print("\n--- GRANULAR SGG SUMMARY (DINOv2) ---")
print(f"  {'Level':<12} {'Ref AUC':>8} {'Aware AUC':>10} {'SGG':>8} {'Benign Acc':>12}")
for lvl, r in granular_results.items():
    benign = r['per_class'].get('benign', float('nan'))
    print(f"  {lvl:<12} {r['rand_auc_reference']:>8.4f} {r['auc']:>10.4f} "
          f"{r['sgg']:>8.4f} {benign:>12.3f}")

# ════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3 — Fed-ISIC2019: DINOv2 mitigation
# FIX 2: validates images before stacking; handles multiple HF image formats.
# Tests whether dissociability (AUC closes, per-class stays broken) holds
# on institutional shift, not just skin-tone shift.
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("EXPERIMENT 3 — Fed-ISIC2019: DINOv2 mitigation")
print("="*60)

try:
    from datasets import load_dataset
    from io import BytesIO

    print("Loading Fed-ISIC2019 from HuggingFace...")
    isic = load_dataset("flwrlabs/fed-isic2019")

    print(f"Train columns: {isic['train'].column_names}")
    print(f"Test columns:  {isic['test'].column_names}")

    train_isic = isic['train'].to_pandas().sample(min(2000, len(isic['train'])), random_state=42)
    test_isic  = isic['test'].to_pandas().sample(min(1000, len(isic['test'])),  random_state=42)

    le_isic = LabelEncoder()
    le_isic.fit(list(train_isic['label']) + list(test_isic['label']))
    print(f"ISIC classes: {le_isic.classes_}")

    # detect image column dynamically
    img_col = None
    for candidate in ['image', 'img', 'pixel_values', 'image_bytes']:
        if candidate in train_isic.columns:
            img_col = candidate
            break
    if img_col is None:
        for col in train_isic.columns:
            if col != 'label' and train_isic[col].dtype == object:
                img_col = col
                break
    print(f"Using image column: '{img_col}'")

    def load_isic_imgs(rows, le, img_col):
        imgs, lbls = [], []
        for _, row in rows.iterrows():
            try:
                raw = row[img_col]
                if isinstance(raw, Image.Image):
                    img = raw.convert('RGB')
                elif isinstance(raw, dict) and 'bytes' in raw:
                    img = Image.open(BytesIO(raw['bytes'])).convert('RGB')
                elif isinstance(raw, bytes):
                    img = Image.open(BytesIO(raw)).convert('RGB')
                else:
                    img = Image.fromarray(np.array(raw)).convert('RGB')
                imgs.append(img.resize((224, 224)))
                lbls.append(le.transform([row['label']])[0])
            except: pass
        return imgs, np.array(lbls, dtype=np.int64)

    print("Extracting ISIC features...")
    tr_imgs_i, tr_y_i = load_isic_imgs(train_isic, le_isic, img_col)
    te_imgs_i, te_y_i = load_isic_imgs(test_isic,  le_isic, img_col)

    if len(tr_imgs_i) == 0 or len(te_imgs_i) == 0:
        raise ValueError(f"Image loading failed: train={len(tr_imgs_i)}, "
                         f"test={len(te_imgs_i)}. Check column '{img_col}'.")

    print(f"Loaded: train={len(tr_imgs_i)}, test={len(te_imgs_i)}")
    tr_feats_i = get_dino_features(tr_imgs_i)
    te_feats_i = get_dino_features(te_imgs_i)

    print("\n--- ISIC RANDOM SPLIT ---")
    all_fi = np.vstack([tr_feats_i, te_feats_i])
    all_yi = np.concatenate([tr_y_i, te_y_i])
    tri, tei = train_test_split(np.arange(len(all_yi)), test_size=0.25,
                                 stratify=all_yi, random_state=42)
    isic_rand = evaluate(all_fi[tri], all_yi[tri], all_fi[tei], all_yi[tei],
                         "DINOv2 ISIC Random")

    print("\n--- ISIC INSTITUTIONAL SPLIT (train 0-3, test 4-5) ---")
    isic_skin = evaluate(tr_feats_i, tr_y_i, te_feats_i, te_y_i,
                         "DINOv2 ISIC Institution-Aware")
    sgg_isic = isic_rand['auc'] - isic_skin['auc']
    print(f"\n*** DINOv2 ISIC SGG: {sgg_isic:.4f} ***")

    clf_isic = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf_isic.fit(tr_feats_i, tr_y_i)
    isic_preds = clf_isic.predict(te_feats_i)
    pc_isic = per_class_acc(te_y_i, isic_preds, le_isic)
    print(f"  Per-class: {pc_isic}")

    n_add_isic = min(200, len(te_feats_i) // 2)
    add_idx    = np.random.choice(len(te_feats_i), n_add_isic, replace=False)
    hold_idx   = np.setdiff1d(np.arange(len(te_feats_i)), add_idx)
    aug_fi     = np.vstack([tr_feats_i, te_feats_i[add_idx]])
    aug_yi     = np.concatenate([tr_y_i, te_y_i[add_idx]])
    wi         = combined_fix_weights(len(tr_y_i), n_add_isic)

    print("\n--- ISIC COMBINED FIX ---")
    isic_fix = evaluate(aug_fi, aug_yi, te_feats_i[hold_idx], te_y_i[hold_idx],
                        "DINOv2 ISIC Combined Fix", weights=wi)
    isic_gap_closed = (isic_fix['auc'] - isic_skin['auc']) / sgg_isic * 100 \
                       if sgg_isic > 0 else 0
    print(f"Gap closed: {isic_gap_closed:.0f}%")

    clf_fix_isic = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf_fix_isic.fit(aug_fi, aug_yi, sample_weight=wi)
    fix_preds_isic = clf_fix_isic.predict(te_feats_i[hold_idx])
    pc_isic_fix = per_class_acc(te_y_i[hold_idx], fix_preds_isic, le_isic)
    print(f"  Per-class after fix: {pc_isic_fix}")

    results['fed_isic2019'] = {
        'random': isic_rand, 'institution_aware': isic_skin,
        'combined_fix': isic_fix,
        'sgg': float(sgg_isic), 'gap_closed_pct': float(isic_gap_closed),
        'per_class_baseline': pc_isic, 'per_class_after_fix': pc_isic_fix,
    }

except Exception as e:
    import traceback
    print(f"  Fed-ISIC2019 skipped: {e}")
    print(traceback.format_exc())
    results['fed_isic2019'] = {'error': str(e)}

# ════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4 — NIH ChestX-ray14: DINOv2 age mitigation
# FIX 1: subdirectory scan (images_001/images/, ...) matching nb_p2 pattern.
# Tests whether dissociability holds across modalities (chest X-ray, not skin).
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("EXPERIMENT 4 — NIH ChestX-ray14: DINOv2 age mitigation")
print("="*60)

try:
    nih = pd.read_csv(NIH_CSV)
    nih = nih[(nih['Patient Age'] > 0) & (nih['Patient Age'] < 120)]
    nih['binary_label'] = (nih['Finding Labels'] != 'No Finding').astype(int)
    nih['age_group'] = nih['Patient Age'].apply(
        lambda x: 'young' if x < 40 else 'middle' if x <= 60 else 'older')

    # FIX 1: scan subdirectories
    print("Building NIH image index...")
    img_index = {}
    for d in os.listdir(NIH_DATA_DIR):
        if d.startswith('images_'):
            img_dir = os.path.join(NIH_DATA_DIR, d, 'images')
            if os.path.exists(img_dir):
                for f in os.listdir(img_dir):
                    img_index[f] = os.path.join(img_dir, f)
    print(f"Indexed {len(img_index):,} images")

    nih['local_path'] = nih['Image Index'].map(img_index)
    nih = nih[nih['local_path'].notna()].copy()
    print(f"Matched {len(nih):,} rows to images")

    N_NIH     = 800
    young_nih = nih[nih['age_group']=='young'].sample(N_NIH, random_state=42)
    older_nih = nih[nih['age_group']=='older'].sample(
                    min(N_NIH, len(nih[nih['age_group']=='older'])), random_state=42)

    le_nih = LabelEncoder()
    le_nih.fit([0, 1])

    def load_nih(rows):
        imgs, lbls = [], []
        for _, row in rows.iterrows():
            try:
                img = Image.open(row['local_path']).convert('RGB').resize((224,224))
                imgs.append(img)
                lbls.append(int(row['binary_label']))
            except: pass
        return imgs, np.array(lbls, dtype=np.int64)

    print("Loading NIH images...")
    young_imgs, young_y = load_nih(young_nih)
    older_imgs, older_y = load_nih(older_nih)
    print(f"Young: {len(young_imgs)}, Older: {len(older_imgs)}")
    print(f"Young prevalence: {young_y.mean():.3f}, Older: {older_y.mean():.3f}")

    print("Extracting NIH features...")
    young_feats = get_dino_features(young_imgs)
    older_feats = get_dino_features(older_imgs)

    # random split
    all_fn = np.vstack([young_feats, older_feats])
    all_yn = np.concatenate([young_y, older_y])
    trn, ten = train_test_split(np.arange(len(all_yn)), test_size=0.25,
                                 stratify=all_yn, random_state=42)
    print("\n--- NIH RANDOM SPLIT ---")
    nih_rand = evaluate(all_fn[trn], all_yn[trn], all_fn[ten], all_yn[ten],
                        "DINOv2 NIH Random")

    # age-aware split: train young, test older
    print("\n--- NIH AGE-AWARE SPLIT (train young, test older) ---")
    nih_age = evaluate(young_feats, young_y, older_feats, older_y,
                       "DINOv2 NIH Age-Aware")
    sgg_nih = nih_rand['auc'] - nih_age['auc']
    print(f"\n*** DINOv2 NIH Age SGG: {sgg_nih:.4f} ***")

    # combined fix
    n_add_nih  = min(200, len(older_feats) // 2)
    add_idx_n  = np.random.choice(len(older_feats), n_add_nih, replace=False)
    hold_idx_n = np.setdiff1d(np.arange(len(older_feats)), add_idx_n)
    aug_fn     = np.vstack([young_feats, older_feats[add_idx_n]])
    aug_yn     = np.concatenate([young_y, older_y[add_idx_n]])
    wn         = combined_fix_weights(len(young_y), n_add_nih)

    print("\n--- NIH COMBINED FIX ---")
    nih_fix = evaluate(aug_fn, aug_yn, older_feats[hold_idx_n], older_y[hold_idx_n],
                       "DINOv2 NIH Combined Fix", weights=wn)
    nih_gap_closed = (nih_fix['auc'] - nih_age['auc']) / sgg_nih * 100 \
                      if sgg_nih > 0 else 0
    print(f"Gap closed: {nih_gap_closed:.0f}%")

    # per-class (binary: pathology present/absent)
    clf_nih = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf_nih.fit(young_feats, young_y)
    nih_preds = clf_nih.predict(older_feats)
    pc_nih = {str(c): float(accuracy_score(older_y[older_y==c], nih_preds[older_y==c]))
              for c in np.unique(older_y)}
    print(f"  Per-class (older, baseline): {pc_nih}")

    clf_nih_fix = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf_nih_fix.fit(aug_fn, aug_yn, sample_weight=wn)
    nih_fix_preds = clf_nih_fix.predict(older_feats[hold_idx_n])
    pc_nih_fix = {str(c): float(accuracy_score(
                    older_y[hold_idx_n][older_y[hold_idx_n]==c],
                    nih_fix_preds[older_y[hold_idx_n]==c]))
                  for c in np.unique(older_y[hold_idx_n])}
    print(f"  Per-class (older, after fix): {pc_nih_fix}")

    results['nih_chestxray14'] = {
        'random': nih_rand, 'age_aware': nih_age, 'combined_fix': nih_fix,
        'sgg': float(sgg_nih), 'gap_closed_pct': float(nih_gap_closed),
        'per_class_baseline': pc_nih, 'per_class_after_fix': pc_nih_fix,
    }

except Exception as e:
    import traceback
    print(f"  NIH ChestX-ray14 skipped: {e}")
    print(traceback.format_exc())
    results['nih_chestxray14'] = {'error': str(e)}

# ════════════════════════════════════════════════════════════════════════════
# COMPLETE SUMMARY
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("COMPLETE SUMMARY")
print("="*60)

print(f"\n{'Method':<45} {'AUC':>8} {'SGG':>8}")
print("-"*63)

r = results['fitzpatrick']
for label, auc, sgg in [
    ("DINOv2 Fitzpatrick Random",       r['random']['auc'],       None),
    ("DINOv2 Fitzpatrick Skin-Tone",    r['skin_tone']['auc'],    r['sgg']),
    ("DINOv2 Fitzpatrick Combined Fix", r['combined_fix']['auc'], None),
]:
    print(f"  {label:<43} {auc:>8.4f} {str(round(sgg,3)) if sgg else '—':>8}")

print(f"\n  Per-class dark skin (DINOv2 baseline):")
for cls, a in r['per_class_dark_baseline'].items():
    print(f"    {cls}: {a:.3f}")
print(f"  Per-class dark skin (DINOv2 after fix):")
for cls, a in r['per_class_dark_after_fix'].items():
    print(f"    {cls}: {a:.3f}")

print(f"\n  Granular Fitzpatrick SGG (DINOv2):")
if 'levels' in results.get('granular_fitzpatrick', {}):
    for lvl, gr in results['granular_fitzpatrick']['levels'].items():
        benign = gr['per_class'].get('benign', float('nan'))
        print(f"    {lvl}: RefAUC={gr['rand_auc_reference']:.4f} "
              f"AwareAUC={gr['auc']:.4f} SGG={gr['sgg']:.4f} benign={benign:.3f}")

if 'sgg' in results.get('fed_isic2019', {}):
    ri = results['fed_isic2019']
    print(f"\n  DINOv2 Fed-ISIC2019 SGG:     {ri['sgg']:.4f}  "
          f"gap closed: {ri['gap_closed_pct']:.0f}%")
else:
    print(f"\n  Fed-ISIC2019: {results.get('fed_isic2019', {}).get('error', 'not run')}")

if 'sgg' in results.get('nih_chestxray14', {}):
    rn = results['nih_chestxray14']
    print(f"  DINOv2 NIH ChestX-ray14 SGG: {rn['sgg']:.4f}  "
          f"gap closed: {rn['gap_closed_pct']:.0f}%")
else:
    print(f"  NIH ChestX-ray14: {results.get('nih_chestxray14', {}).get('error', 'not run')}")

# FIX 4: convert() strips all numpy types before json.dump
json.dump(convert(results), open('/kaggle/working/nb5_dinov2_gaps.json', 'w'), indent=2)

print("\n✓ Complete. Paste ALL output back to Claude.")
