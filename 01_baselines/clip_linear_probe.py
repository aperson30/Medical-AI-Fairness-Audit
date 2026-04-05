# ============================================================
# NOTEBOOK — CLIP Linear Probe (Paper's Primary Baseline)
# Dataset: nazmusresan/fitzpatrick17k
# GPU T4, Internet ON. ~20 min.
# This replicates Table 1 "CLIP ViT-L/14 Random/Skin-Tone" rows
# ============================================================

!pip install transformers torch torchvision scikit-learn pandas numpy -q

import torch
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from transformers import CLIPModel, CLIPProcessor
import warnings, os, json
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── LOAD DATA ────────────────────────────────────────────────
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

MAX = 800
light_df  = df[df['skin_group']=='light'].sample(MAX, random_state=42)
medium_df = df[df['skin_group']=='medium'].sample(MAX, random_state=42)
dark_df   = df[df['skin_group']=='dark'].sample(
    min(MAX, len(df[df['skin_group']=='dark'])), random_state=42)

le = LabelEncoder()
le.fit(list(light_df['three_partition_label']) +
       list(medium_df['three_partition_label']) +
       list(dark_df['three_partition_label']))
print(f"Classes: {le.classes_}")

# ── LOAD CLIP ─────────────────────────────────────────────────
print("Loading CLIP...")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
clip_model.eval()
print("CLIP loaded.")

# ── EXTRACT FEATURES ──────────────────────────────────────────
@torch.no_grad()
def get_clip_features(dataframe, batch_size=32):
    all_feats, all_labels = [], []
    images, labels = [], []
    
    for _, row in dataframe.iterrows():
        try:
            img = Image.open(row['local_path']).convert('RGB')
            images.append(img)
            labels.append(le.transform([row['three_partition_label']])[0])
        except: pass
    
    print(f"  Processing {len(images)} images...")
    for i in range(0, len(images), batch_size):
        batch_imgs = images[i:i+batch_size]
        inputs = processor(images=batch_imgs, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Extract vision features
        vision_out = clip_model.vision_model(**inputs)
        features = clip_model.visual_projection(vision_out.pooler_output)
        features = features / features.norm(dim=-1, keepdim=True)
        
        all_feats.append(features.cpu().numpy())
        if i % 320 == 0:
            print(f"    {i}/{len(images)}...")
    
    return np.vstack(all_feats), np.array(labels, dtype=np.int64)

# ── EVALUATION ────────────────────────────────────────────────
def evaluate(train_f, train_y, test_f, test_y, name):
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(train_f, train_y)
    probs = clf.predict_proba(test_f)
    preds = clf.predict(test_f)
    
    auc = roc_auc_score(test_y, probs, multi_class='ovr', average='macro')
    acc = accuracy_score(test_y, preds)
    f1  = f1_score(test_y, preds, average='macro')
    
    # Bootstrap CI
    scores = []
    for _ in range(500):
        idx = np.random.choice(len(test_y), len(test_y), replace=True)
        try:
            scores.append(roc_auc_score(test_y[idx], probs[idx],
                                       multi_class='ovr', average='macro'))
        except: pass
    ci_low, ci_high = np.percentile(scores, [2.5, 97.5])
    
    print(f"  {name}: AUC={auc:.4f} ({ci_low:.4f}-{ci_high:.4f}) Acc={acc:.4f} F1={f1:.4f}")
    
    # Per-class accuracy
    per_class = {}
    for i, cls in enumerate(le.classes_):
        mask = test_y == i
        if mask.sum() > 0:
            per_class[cls] = float(accuracy_score(test_y[mask], preds[mask]))
    
    return {'auc': float(auc), 'acc': float(acc), 'f1': float(f1),
            'ci_low': float(ci_low), 'ci_high': float(ci_high),
            'per_class': per_class}

# ── EXPERIMENT 1: RANDOM SPLIT ────────────────────────────────
print("\n=== EXPERIMENT 1: RANDOM SPLIT ===")
all_df = pd.concat([light_df, medium_df, dark_df]).reset_index(drop=True)

print("Extracting features...")
all_feats, all_labels = get_clip_features(all_df)

tr_idx, te_idx = train_test_split(np.arange(len(all_labels)), test_size=0.25,
                                   stratify=all_labels, random_state=42)

rand_results = evaluate(all_feats[tr_idx], all_labels[tr_idx],
                       all_feats[te_idx], all_labels[te_idx],
                       "CLIP LP Random")

# ── EXPERIMENT 2: SKIN-TONE SPLIT ─────────────────────────────
print("\n=== EXPERIMENT 2: SKIN-TONE SPLIT ===")
train_skin_df = pd.concat([light_df, medium_df]).reset_index(drop=True)
test_skin_df  = dark_df.reset_index(drop=True)

print("Extracting features...")
train_feats, train_labels = get_clip_features(train_skin_df)
test_feats, test_labels = get_clip_features(test_skin_df)

skin_results = evaluate(train_feats, train_labels, test_feats, test_labels,
                       "CLIP LP Skin-Tone")

print("\nPer-class accuracy on dark skin:")
for cls, acc in skin_results['per_class'].items():
    print(f"  {cls}: {acc:.3f}")

sgg = rand_results['auc'] - skin_results['auc']
print(f"\n*** CLIP LINEAR PROBE SGG: {sgg:.4f} ***")
print(f"Paper reports: 0.047")
print(f"Match: {'YES ✓' if abs(sgg - 0.047) < 0.005 else 'NO - investigate'}")

# ── SAVE RESULTS ──────────────────────────────────────────────
results = {
    'model': 'CLIP_linear_probe',
    'random': {k:v for k,v in rand_results.items() if k!='per_class'},
    'skin_tone': {k:v for k,v in skin_results.items() if k!='per_class'},
    'per_class_dark': skin_results['per_class'],
    'sgg': float(sgg),
}
with open('/kaggle/working/clip_linear_probe_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n=== SUMMARY ===")
print(f"{'Method':<30} {'AUC':>8} {'CI':>20} {'SGG':>8}")
print("-"*68)
print(f"{'CLIP LP Random':<30} {rand_results['auc']:>8.4f} ({rand_results['ci_low']:.3f}-{rand_results['ci_high']:.3f})")
print(f"{'CLIP LP Skin-Tone':<30} {skin_results['auc']:>8.4f} ({skin_results['ci_low']:.3f}-{skin_results['ci_high']:.3f}) {sgg:>8.4f}")
print(f"{'Paper CLIP LP Random':<30} {'0.789':>8} {'(0.759-0.819)':>20}")
print(f"{'Paper CLIP LP Skin-Tone':<30} {'0.742':>8} {'(0.708-0.775)':>20} {'0.047':>8}")

print("\n✓ Complete. This is the paper's primary baseline.")
