# ============================================================
# OVERNIGHT NOTEBOOK — Fitzpatrick17k Skin Tone Experiments
# Accounts 1 and 2
# Datasets: nazmusresan/fitzpatrick17k
# GPU T4 x1, Internet ON
# Expected runtime: 4-6 hours
# Do NOT stop this notebook — let it run overnight
# ============================================================

# ── CELL 1: Install ──────────────────────────────────────────
!pip install transformers torch torchvision pandas numpy scikit-learn Pillow -q

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import CLIPProcessor, CLIPModel
import torchvision.models as models
import torchvision.transforms as transforms
from collections import Counter
import warnings
import os
import json
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# ── CELL 2: Find Fitzpatrick data path ───────────────────────
print("\nSearching for Fitzpatrick17k data...")

fitz_img_dir = None
fitz_csv = None

for root, dirs, files in os.walk('/kaggle/input'):
    for f in files:
        if f.endswith('.csv') and 'fitzpatrick' in f.lower():
            fitz_csv = os.path.join(root, f)
            print(f"Found CSV: {fitz_csv}")
    for d in dirs:
        if 'background' in d.lower() or 'removed' in d.lower():
            candidate = os.path.join(root, d)
            sample_files = [x for x in os.listdir(candidate) if x.endswith('.jpg') or x.endswith('.png')]
            if len(sample_files) > 100:
                fitz_img_dir = candidate
                print(f"Found image dir: {fitz_img_dir} ({len(sample_files)} images)")

if fitz_img_dir is None:
    # Fallback: find any dir with lots of images
    for root, dirs, files in os.walk('/kaggle/input'):
        imgs = [f for f in files if f.endswith('.jpg') or f.endswith('.png')]
        if len(imgs) > 1000:
            fitz_img_dir = root
            print(f"Fallback image dir: {fitz_img_dir} ({len(imgs)} images)")
            break

print(f"\nImage dir: {fitz_img_dir}")
print(f"CSV path: {fitz_csv}")

# ── CELL 3: Load metadata and match to images ─────────────────
print("\nLoading Fitzpatrick17k metadata...")

df = pd.read_csv(fitz_csv)
print(f"CSV shape: {df.shape}")
print(f"CSV columns: {df.columns.tolist()}")
print(f"First 3 rows:\n{df.head(3)}")

# Find fitzpatrick scale column
fitz_col = next((c for c in df.columns if 'fitzpatrick' in c.lower() and 'scale' in c.lower()), None)
if fitz_col is None:
    fitz_col = next((c for c in df.columns if 'fitzpatrick' in c.lower()), None)
print(f"\nFitzpatrick column: {fitz_col}")
print(f"Fitzpatrick values: {df[fitz_col].value_counts().sort_index()}")

# Find label column
label_col = next((c for c in ['three_partition_label', 'label', 'condition', 'disease'] 
                  if c in df.columns), None)
print(f"Label column: {label_col}")
print(f"Label distribution:\n{df[label_col].value_counts()}")

# Find image identifier column
id_col = next((c for c in ['md5hash', 'image_id', 'filename', 'image'] 
               if c in df.columns), None)
print(f"Image ID column: {id_col}")

# List actual image files
image_files = {}
for f in os.listdir(fitz_img_dir):
    if f.endswith('.jpg') or f.endswith('.png'):
        stem = f.replace('.jpg', '').replace('.png', '')
        image_files[stem] = os.path.join(fitz_img_dir, f)

print(f"\nTotal images on disk: {len(image_files)}")
print(f"Sample image names: {list(image_files.keys())[:5]}")

# ── CELL 4: Match metadata to local images ────────────────────
print("\nMatching metadata to local images...")

# Try to match by md5hash or filename
df['local_path'] = None

if id_col == 'md5hash':
    df['local_path'] = df[id_col].map(image_files)
else:
    # Try matching by condition name in filename
    def find_image(row):
        if id_col and row[id_col] in image_files:
            return image_files[row[id_col]]
        # Try condition-based matching
        condition = str(row.get(label_col, '')).lower().replace(' ', '_')
        matches = [p for k, p in image_files.items() if condition in k.lower()]
        return matches[0] if matches else None
    df['local_path'] = df.apply(find_image, axis=1)

matched = df['local_path'].notna().sum()
print(f"Matched {matched}/{len(df)} images to local files")

# If matching by md5hash fails, match by label-based filename
if matched < 100:
    print("Low match count — trying label-based filename matching...")
    # Images named like "melanoma17.jpg" — match to melanoma label
    def match_by_label(row):
        label_clean = str(row.get(label_col, '')).lower().replace(' ', '_').replace('-', '_')
        for k, p in image_files.items():
            if label_clean[:8] in k.lower():
                return p
        return None
    df['local_path'] = df.apply(match_by_label, axis=1)
    matched = df['local_path'].notna().sum()
    print(f"After label matching: {matched}/{len(df)} images")

# Filter to matched rows
df_matched = df[df['local_path'].notna()].copy()
df_matched = df_matched[df_matched[fitz_col] > 0]  # remove unknown skin tones

print(f"\nFinal matched dataset: {len(df_matched)} images")
print(f"Skin tone distribution:\n{df_matched[fitz_col].value_counts().sort_index()}")

# ── CELL 5: Assign skin tone groups ──────────────────────────
df_matched['skin_group'] = df_matched[fitz_col].apply(
    lambda x: 'light' if x <= 2 else 'medium' if x <= 4 else 'dark'
)

print(f"\nSkin group distribution:")
print(df_matched['skin_group'].value_counts())
print(f"\nLabel distribution by skin group:")
print(df_matched.groupby('skin_group')[label_col].value_counts())

# ── CELL 6: Load models ───────────────────────────────────────
print("\nLoading CLIP ViT-L/14...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
clip_model.eval()
print("CLIP loaded.")

print("Loading ResNet50...")
resnet = models.resnet50(pretrained=True)
resnet_features = nn.Sequential(*list(resnet.children())[:-1]).to(device)
resnet_features.eval()

resnet_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
print("ResNet50 loaded.")

# ── CELL 7: Feature extraction ────────────────────────────────
def load_images_from_paths(paths, max_size=224):
    images = []
    failed = 0
    for p in paths:
        try:
            img = Image.open(p).convert('RGB').resize((max_size, max_size))
            images.append(img)
        except:
            images.append(None)
            failed += 1
    return images, failed

@torch.no_grad()
def extract_clip_features(images, batch_size=32):
    all_feats = []
    valid_images = [img for img in images if img is not None]
    for i in range(0, len(valid_images), batch_size):
        batch = valid_images[i:i+batch_size]
        inputs = clip_processor(images=batch, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feats = clip_model.get_image_features(**inputs)
        if not isinstance(feats, torch.Tensor):
            feats = feats.pooler_output if hasattr(feats, 'pooler_output') else feats.last_hidden_state[:, 0]
        feats = feats / feats.norm(dim=-1, keepdim=True)
        all_feats.append(feats.cpu().numpy())
        if i % 320 == 0:
            print(f"  CLIP: {i}/{len(valid_images)} done...")
    return np.vstack(all_feats)

@torch.no_grad()
def extract_resnet_features(images, batch_size=32):
    all_feats = []
    valid_images = [img for img in images if img is not None]
    for i in range(0, len(valid_images), batch_size):
        batch = valid_images[i:i+batch_size]
        tensors = torch.stack([resnet_transform(img) for img in batch]).to(device)
        feats = resnet_features(tensors).squeeze(-1).squeeze(-1)
        all_feats.append(feats.cpu().numpy())
        if i % 320 == 0:
            print(f"  ResNet: {i}/{len(valid_images)} done...")
    return np.vstack(all_feats)

def evaluate(train_feats, train_labels, test_feats, test_labels):
    le = LabelEncoder()
    train_y = le.fit_transform(train_labels)
    test_y = le.transform(test_labels)
    
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(train_feats, train_y)
    
    probs = clf.predict_proba(test_feats)
    preds = clf.predict(test_feats)
    
    results = {
        'auc': roc_auc_score(test_y, probs, multi_class='ovr', average='macro') if len(le.classes_) > 2 else roc_auc_score(test_y, probs[:, 1]),
        'accuracy': accuracy_score(test_y, preds),
        'f1': f1_score(test_y, preds, average='macro'),
        'n_train': len(train_y),
        'n_test': len(test_y),
        'classes': le.classes_.tolist()
    }
    return results, clf, le

def bootstrap_auc(y_true, y_probs, n=1000):
    le = LabelEncoder()
    y_enc = le.fit_transform(y_true)
    scores = []
    for _ in range(n):
        idx = np.random.choice(len(y_enc), len(y_enc), replace=True)
        try:
            s = roc_auc_score(y_enc[idx], y_probs[idx], multi_class='ovr', average='macro')
            scores.append(s)
        except:
            pass
    return np.mean(scores), np.percentile(scores, 2.5), np.percentile(scores, 97.5)

# ── CELL 8: Load all images by skin group ────────────────────
print("\n" + "="*60)
print("LOADING IMAGES BY SKIN GROUP")
print("="*60)

# Use all available images, cap at reasonable sizes
light_df = df_matched[df_matched['skin_group'] == 'light']
medium_df = df_matched[df_matched['skin_group'] == 'medium']
dark_df = df_matched[df_matched['skin_group'] == 'dark']

print(f"Available: light={len(light_df)}, medium={len(medium_df)}, dark={len(dark_df)}")

# Cap sizes for memory management
MAX_PER_GROUP = 1000
light_df = light_df.sample(min(MAX_PER_GROUP, len(light_df)), random_state=42)
medium_df = medium_df.sample(min(MAX_PER_GROUP, len(medium_df)), random_state=42)
dark_df = dark_df.sample(min(MAX_PER_GROUP, len(dark_df)), random_state=42)

print(f"Using: light={len(light_df)}, medium={len(medium_df)}, dark={len(dark_df)}")

print("\nLoading light skin images...")
light_images, light_failed = load_images_from_paths(light_df['local_path'].tolist())
print(f"  Loaded: {len(light_images)-light_failed}, Failed: {light_failed}")

print("Loading medium skin images...")
medium_images, medium_failed = load_images_from_paths(medium_df['local_path'].tolist())
print(f"  Loaded: {len(medium_images)-medium_failed}, Failed: {medium_failed}")

print("Loading dark skin images...")
dark_images, dark_failed = load_images_from_paths(dark_df['local_path'].tolist())
print(f"  Loaded: {len(dark_images)-dark_failed}, Failed: {dark_failed}")

# Filter out failed
light_valid = [(img, lbl) for img, lbl in zip(light_images, light_df[label_col].tolist()) if img is not None]
medium_valid = [(img, lbl) for img, lbl in zip(medium_images, medium_df[label_col].tolist()) if img is not None]
dark_valid = [(img, lbl) for img, lbl in zip(dark_images, dark_df[label_col].tolist()) if img is not None]

light_imgs, light_lbls = zip(*light_valid) if light_valid else ([], [])
medium_imgs, medium_lbls = zip(*medium_valid) if medium_valid else ([], [])
dark_imgs, dark_lbls = zip(*dark_valid) if dark_valid else ([], [])

print(f"\nFinal usable: light={len(light_imgs)}, medium={len(medium_imgs)}, dark={len(dark_imgs)}")

# ── CELL 9: Extract all features ─────────────────────────────
print("\n" + "="*60)
print("EXTRACTING FEATURES")
print("="*60)

print("\nCLIP features — light skin...")
light_clip = extract_clip_features(list(light_imgs))
print("CLIP features — medium skin...")
medium_clip = extract_clip_features(list(medium_imgs))
print("CLIP features — dark skin...")
dark_clip = extract_clip_features(list(dark_imgs))

print("\nResNet features — light skin...")
light_resnet = extract_resnet_features(list(light_imgs))
print("ResNet features — medium skin...")
medium_resnet = extract_resnet_features(list(medium_imgs))
print("ResNet features — dark skin...")
dark_resnet = extract_resnet_features(list(dark_imgs))

print(f"\nCLIP feature shapes: light={light_clip.shape}, medium={medium_clip.shape}, dark={dark_clip.shape}")
print(f"ResNet feature shapes: light={light_resnet.shape}, medium={medium_resnet.shape}, dark={dark_resnet.shape}")

# ── CELL 10: EXPERIMENT 1 — Random split ─────────────────────
print("\n" + "="*60)
print("EXPERIMENT 1: RANDOM SPLIT (all skin tones mixed)")
print("="*60)

all_clip = np.vstack([light_clip, medium_clip, dark_clip])
all_resnet = np.vstack([light_resnet, medium_resnet, dark_resnet])
all_labels = list(light_lbls) + list(medium_lbls) + list(dark_lbls)

print(f"Total images: {len(all_labels)}")
print(f"Label distribution: {Counter(all_labels)}")

le_global = LabelEncoder()
all_y = le_global.fit_transform(all_labels)

train_idx, test_idx = train_test_split(
    np.arange(len(all_y)), test_size=0.25,
    stratify=all_y, random_state=42
)

results_random = {}
for model_name, feats in [('CLIP', all_clip), ('ResNet50', all_resnet)]:
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(feats[train_idx], all_y[train_idx])
    probs = clf.predict_proba(feats[test_idx])
    preds = clf.predict(feats[test_idx])
    
    auc = roc_auc_score(all_y[test_idx], probs, multi_class='ovr', average='macro')
    acc = accuracy_score(all_y[test_idx], preds)
    f1 = f1_score(all_y[test_idx], preds, average='macro')
    
    mean_auc, ci_low, ci_high = bootstrap_auc(
        [all_labels[i] for i in test_idx], probs
    )
    
    results_random[model_name] = {'auc': auc, 'acc': acc, 'f1': f1, 
                                   'ci_low': ci_low, 'ci_high': ci_high}
    print(f"{model_name} | AUC: {auc:.4f} ({ci_low:.4f}-{ci_high:.4f}) | Acc: {acc:.4f} | F1: {f1:.4f}")

# ── CELL 11: EXPERIMENT 2 — Skin-tone aware split ────────────
print("\n" + "="*60)
print("EXPERIMENT 2: SKIN-TONE AWARE SPLIT")
print("Train: light (I-II) + medium (III-IV)")
print("Test:  dark (V-VI)")
print("="*60)

train_clip = np.vstack([light_clip, medium_clip])
train_resnet = np.vstack([light_resnet, medium_resnet])
train_labels = list(light_lbls) + list(medium_lbls)

test_clip = dark_clip
test_resnet = dark_resnet
test_labels = list(dark_lbls)

print(f"Train: {len(train_labels)} images")
print(f"Test: {len(test_labels)} images (dark skin only)")
print(f"Train label dist: {Counter(train_labels)}")
print(f"Test label dist: {Counter(test_labels)}")

le_skin = LabelEncoder()
le_skin.fit(train_labels + test_labels)
train_y = le_skin.transform(train_labels)
test_y = le_skin.transform(test_labels)

results_skin = {}
skin_clfs = {}
for model_name, train_feats, test_feats in [
    ('CLIP', train_clip, test_clip),
    ('ResNet50', train_resnet, test_resnet)
]:
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(train_feats, train_y)
    probs = clf.predict_proba(test_feats)
    preds = clf.predict(test_feats)
    
    try:
        auc = roc_auc_score(test_y, probs, multi_class='ovr', average='macro')
        acc = accuracy_score(test_y, preds)
        f1 = f1_score(test_y, preds, average='macro')
        mean_auc, ci_low, ci_high = bootstrap_auc(test_labels, probs)
        
        results_skin[model_name] = {'auc': auc, 'acc': acc, 'f1': f1,
                                     'ci_low': ci_low, 'ci_high': ci_high}
        skin_clfs[model_name] = (clf, probs, preds)
        print(f"{model_name} | AUC: {auc:.4f} ({ci_low:.4f}-{ci_high:.4f}) | Acc: {acc:.4f} | F1: {f1:.4f}")
    except Exception as e:
        print(f"{model_name} error: {e}")

# ── CELL 12: Per-class analysis ───────────────────────────────
print("\n" + "="*60)
print("PER-CLASS BREAKDOWN ON DARK SKIN TEST SET")
print("="*60)

for model_name, (clf, probs, preds) in skin_clfs.items():
    print(f"\n{model_name}:")
    for i, cls in enumerate(le_skin.classes_):
        mask = test_y == i
        if mask.sum() > 0:
            cls_acc = accuracy_score(test_y[mask], preds[mask])
            cls_n = mask.sum()
            print(f"  {cls}: acc={cls_acc:.3f} (n={cls_n})")

# ── CELL 13: Compute SGG ──────────────────────────────────────
print("\n" + "="*60)
print("SOURCE GENERALIZATION GAP (SGG)")
print("="*60)

sgg_results = {}
for model_name in ['CLIP', 'ResNet50']:
    if model_name in results_random and model_name in results_skin:
        sgg = results_random[model_name]['auc'] - results_skin[model_name]['auc']
        sgg_results[model_name] = sgg
        print(f"{model_name}:")
        print(f"  Random split AUC:    {results_random[model_name]['auc']:.4f}")
        print(f"  Skin-aware split AUC: {results_skin[model_name]['auc']:.4f}")
        print(f"  SGG:                  {sgg:.4f}")

# ── CELL 14: DACC Implementation ─────────────────────────────
print("\n" + "="*60)
print("APPLYING DACC — Demographic-Aware Contrastive Calibration")
print("="*60)

import torch.nn.functional as F

class GroupTemperatureScaling(nn.Module):
    """
    Simplified DACC: group-specific temperature scaling.
    One temperature parameter per skin tone group.
    Learned on a held-out validation set.
    """
    def __init__(self, groups):
        super().__init__()
        self.temperatures = nn.ParameterDict({
            str(g): nn.Parameter(torch.ones(1) * 1.5)
            for g in groups
        })
    
    def forward(self, logits, groups):
        calibrated = torch.zeros_like(logits)
        for i, g in enumerate(groups):
            T = torch.clamp(self.temperatures[str(g)], 0.05, 10.0)
            calibrated[i] = logits[i] / T
        return calibrated

# Split light+medium into train and val for DACC
val_size = int(0.2 * len(train_labels))
val_clip = train_clip[-val_size:]
val_resnet = train_resnet[-val_size:]
val_labels = train_labels[-val_size:]
val_groups = ['light'] * (val_size // 2) + ['medium'] * (val_size - val_size // 2)

dacc_results = {}

for model_name, train_feats, test_feats, val_feats in [
    ('CLIP', train_clip, test_clip, val_clip),
    ('ResNet50', train_resnet, test_resnet, val_resnet)
]:
    print(f"\nTraining DACC for {model_name}...")
    
    # Get logits from trained classifier
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(train_feats[:-val_size], train_y[:-val_size])
    
    val_logits = torch.FloatTensor(clf.decision_function(val_feats))
    val_y_t = torch.LongTensor(le_skin.transform(val_labels))
    
    # Train group temperature scaling
    dacc = GroupTemperatureScaling(['light', 'medium', 'dark'])
    optimizer = torch.optim.Adam(dacc.parameters(), lr=0.01)
    
    for epoch in range(100):
        optimizer.zero_grad()
        calibrated = dacc(val_logits, val_groups)
        loss = F.cross_entropy(calibrated, val_y_t)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: loss={loss.item():.4f}")
    
    # Print learned temperatures
    print(f"  Learned temperatures:")
    for g in ['light', 'medium', 'dark']:
        T = torch.clamp(dacc.temperatures[str(g)], 0.05, 10.0).item()
        print(f"    {g}: T={T:.4f}")
    
    # Evaluate DACC on dark skin test set
    test_logits = torch.FloatTensor(clf.decision_function(test_feats))
    test_groups_list = ['dark'] * len(test_labels)
    
    with torch.no_grad():
        calibrated_test = dacc(test_logits, test_groups_list)
        dacc_probs = F.softmax(calibrated_test, dim=-1).numpy()
        dacc_preds = calibrated_test.argmax(dim=-1).numpy()
    
    try:
        auc_dacc = roc_auc_score(test_y, dacc_probs, multi_class='ovr', average='macro')
        acc_dacc = accuracy_score(test_y, dacc_preds)
        f1_dacc = f1_score(test_y, dacc_preds, average='macro')
        mean_auc, ci_low, ci_high = bootstrap_auc(test_labels, dacc_probs)
        
        dacc_results[model_name] = {'auc': auc_dacc, 'acc': acc_dacc, 'f1': f1_dacc,
                                     'ci_low': ci_low, 'ci_high': ci_high}
        print(f"  DACC | AUC: {auc_dacc:.4f} ({ci_low:.4f}-{ci_high:.4f}) | Acc: {acc_dacc:.4f} | F1: {f1_dacc:.4f}")
    except Exception as e:
        print(f"  DACC evaluation error: {e}")

# ── CELL 15: Final results table ──────────────────────────────
print("\n" + "="*60)
print("FITZPATRICK17k — COMPLETE RESULTS TABLE")
print("="*60)
print(f"{'Method':<35} {'AUC':>8} {'95% CI':>15} {'Acc':>8} {'F1':>8}")
print("-"*80)

for model_name in ['CLIP', 'ResNet50']:
    if model_name in results_random:
        r = results_random[model_name]
        print(f"{model_name} Random Split{'':<20} {r['auc']:>8.4f} ({r['ci_low']:.4f}-{r['ci_high']:.4f}) {r['acc']:>8.4f} {r['f1']:>8.4f}")
    if model_name in results_skin:
        r = results_skin[model_name]
        print(f"{model_name} Skin-Tone Split{'':<17} {r['auc']:>8.4f} ({r['ci_low']:.4f}-{r['ci_high']:.4f}) {r['acc']:>8.4f} {r['f1']:>8.4f}")
    if model_name in dacc_results:
        r = dacc_results[model_name]
        print(f"{model_name} Skin-Tone + DACC{'':<16} {r['auc']:>8.4f} ({r['ci_low']:.4f}-{r['ci_high']:.4f}) {r['acc']:>8.4f} {r['f1']:>8.4f}")
    if model_name in sgg_results:
        print(f"  → SGG ({model_name}): {sgg_results[model_name]:.4f}")
    print()

# ── CELL 16: Save all results ─────────────────────────────────
results_to_save = {
    'dataset': 'fitzpatrick17k',
    'random_split': results_random,
    'skin_tone_split': results_skin,
    'dacc': dacc_results,
    'sgg': sgg_results,
    'n_light': len(light_imgs),
    'n_medium': len(medium_imgs),
    'n_dark': len(dark_imgs),
    'label_classes': le_skin.classes_.tolist()
}

with open('/kaggle/working/fitzpatrick_results.json', 'w') as f:
    json.dump(results_to_save, f, indent=2)

print("\nResults saved to /kaggle/working/fitzpatrick_results.json")
print("\n✓ Overnight notebook complete. Paste ALL output back to Claude.")
