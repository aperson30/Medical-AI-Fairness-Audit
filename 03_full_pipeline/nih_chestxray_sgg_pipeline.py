# ============================================================
# OVERNIGHT NOTEBOOK — NIH ChestX-ray14 Age & Sex Experiments
# Accounts 3, 4, 5, 6
# Datasets: nih-chest-xrays (already added)
# GPU T4 x1, Internet ON
# Expected runtime: 6-8 hours
# Do NOT stop this notebook — let it run overnight
# ============================================================

# ── CELL 1: Install ──────────────────────────────────────────
!pip install transformers torch torchvision pandas numpy scikit-learn Pillow -q

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from transformers import CLIPProcessor, CLIPModel
import torchvision.models as models
import torchvision.transforms as transforms
from collections import Counter
import warnings
import os
import json
import random
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# ── CELL 2: Load NIH metadata ────────────────────────────────
print("\nLoading NIH ChestX-ray14 metadata...")

data_path = '/kaggle/input/datasets/organizations/nih-chest-xrays/data'
csv_path = os.path.join(data_path, 'Data_Entry_2017.csv')

df = pd.read_csv(csv_path)
print(f"Loaded: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Clean age — remove obviously wrong values
df = df[df['Patient Age'] < 120]
df = df[df['Patient Age'] > 0]

# Binary task: pneumonia detection (clinically important, clear positive/negative)
# Also do No Finding vs Any Finding (simpler, cleaner)
df['has_finding'] = (df['Finding Labels'] != 'No Finding').astype(int)
df['label_str'] = df['has_finding'].map({0: 'no_finding', 1: 'finding'})

# Age groups
df['age_group'] = pd.cut(df['Patient Age'],
                          bins=[0, 40, 60, 120],
                          labels=['young (<40)', 'middle (40-60)', 'older (>60)'])

print(f"\nTotal patients after cleaning: {len(df)}")
print(f"\nAge group distribution:")
print(df['age_group'].value_counts())
print(f"\nSex distribution:")
print(df['Patient Gender'].value_counts())
print(f"\nFinding distribution:")
print(df['label_str'].value_counts())
print(f"\nFinding rate by age group:")
print(df.groupby('age_group')['has_finding'].mean().round(3))
print(f"\nFinding rate by sex:")
print(df.groupby('Patient Gender')['has_finding'].mean().round(3))

# ── CELL 3: Find image paths ─────────────────────────────────
print("\nBuilding image path index...")

image_dirs = [d for d in os.listdir(data_path) if d.startswith('images_')]
print(f"Image directories: {image_dirs}")

# Build index: filename -> full path
img_index = {}
for img_dir in image_dirs:
    dir_path = os.path.join(data_path, img_dir, 'images')
    if os.path.exists(dir_path):
        for f in os.listdir(dir_path):
            if f.endswith('.png') or f.endswith('.jpg'):
                img_index[f] = os.path.join(dir_path, f)

print(f"Total images indexed: {len(img_index)}")

# Match to metadata
df['image_path'] = df['Image Index'].map(img_index)
matched = df['image_path'].notna().sum()
print(f"Matched {matched}/{len(df)} images to paths")

df = df[df['image_path'].notna()].copy()
print(f"Working dataset: {len(df)} images")

# ── CELL 4: Load models ───────────────────────────────────────
print("\nLoading models...")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
clip_model.eval()
print("CLIP loaded.")

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

# ── CELL 5: Feature extraction functions ─────────────────────
def load_image(path):
    try:
        return Image.open(path).convert('RGB')
    except:
        return None

@torch.no_grad()
def extract_clip_features(images, batch_size=64):
    all_feats = []
    valid = [img for img in images if img is not None]
    for i in range(0, len(valid), batch_size):
        batch = valid[i:i+batch_size]
        inputs = clip_processor(images=batch, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feats = clip_model.get_image_features(**inputs)
        if not isinstance(feats, torch.Tensor):
            feats = feats.pooler_output if hasattr(feats, 'pooler_output') else feats.last_hidden_state[:, 0]
        feats = feats / feats.norm(dim=-1, keepdim=True)
        all_feats.append(feats.cpu().numpy())
        if i % 640 == 0:
            print(f"  CLIP: {i}/{len(valid)}...")
    return np.vstack(all_feats)

@torch.no_grad()
def extract_resnet_features(images, batch_size=64):
    all_feats = []
    valid = [img for img in images if img is not None]
    for i in range(0, len(valid), batch_size):
        batch = valid[i:i+batch_size]
        tensors = torch.stack([resnet_transform(img) for img in batch]).to(device)
        feats = resnet_features(tensors).squeeze(-1).squeeze(-1)
        all_feats.append(feats.cpu().numpy())
        if i % 640 == 0:
            print(f"  ResNet: {i}/{len(valid)}...")
    return np.vstack(all_feats)

def run_experiment(train_feats, train_labels, test_feats, test_labels, name=""):
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(train_feats, train_labels)
    probs = clf.predict_proba(test_feats)
    preds = clf.predict(test_feats)
    
    n_classes = len(np.unique(train_labels))
    if n_classes == 2:
        auc = roc_auc_score(test_labels, probs[:, 1])
    else:
        auc = roc_auc_score(test_labels, probs, multi_class='ovr', average='macro')
    
    acc = accuracy_score(test_labels, preds)
    f1 = f1_score(test_labels, preds, average='macro')
    
    # Bootstrap CI
    scores = []
    for _ in range(500):
        idx = np.random.choice(len(test_labels), len(test_labels), replace=True)
        try:
            if n_classes == 2:
                s = roc_auc_score(test_labels[idx], probs[idx, 1])
            else:
                s = roc_auc_score(test_labels[idx], probs[idx], multi_class='ovr', average='macro')
            scores.append(s)
        except:
            pass
    ci_low, ci_high = np.percentile(scores, [2.5, 97.5])
    
    print(f"{name} | AUC: {auc:.4f} ({ci_low:.4f}-{ci_high:.4f}) | Acc: {acc:.4f} | F1: {f1:.4f} | n_test={len(test_labels)}")
    return {'auc': auc, 'acc': acc, 'f1': f1, 'ci_low': ci_low, 'ci_high': ci_high, 
            'n_train': len(train_labels), 'n_test': len(test_labels)}, clf, probs

# ── CELL 6: Sample dataset ───────────────────────────────────
print("\n" + "="*60)
print("SAMPLING DATASET")
print("="*60)

# Use 3000 images total for manageable overnight run
# Stratified by age group and sex and label
N_TOTAL = 3000
N_PER_GROUP = N_TOTAL // 3

young_df = df[df['age_group'] == 'young (<40)'].sample(min(N_PER_GROUP, len(df[df['age_group']=='young (<40)'])), random_state=42)
middle_df = df[df['age_group'] == 'middle (40-60)'].sample(min(N_PER_GROUP, len(df[df['age_group']=='middle (40-60)'])), random_state=42)
older_df = df[df['age_group'] == 'older (>60)'].sample(min(N_PER_GROUP, len(df[df['age_group']=='older (>60)'])), random_state=42)

male_df = df[df['Patient Gender'] == 'M'].sample(min(N_PER_GROUP*2, len(df[df['Patient Gender']=='M'])), random_state=42)
female_df = df[df['Patient Gender'] == 'F'].sample(min(N_PER_GROUP*2, len(df[df['Patient Gender']=='F'])), random_state=42)

print(f"Young: {len(young_df)}, Middle: {len(middle_df)}, Older: {len(older_df)}")
print(f"Male: {len(male_df)}, Female: {len(female_df)}")

# ── CELL 7: Load and extract features — age groups ───────────
print("\n" + "="*60)
print("LOADING IMAGES — AGE GROUPS")
print("="*60)

print("Loading young images...")
young_imgs = [load_image(p) for p in young_df['image_path']]
young_imgs = [img for img in young_imgs if img is not None]
young_labels = np.array(young_df['has_finding'].tolist()[:len(young_imgs)])

print("Loading middle-age images...")
middle_imgs = [load_image(p) for p in middle_df['image_path']]
middle_imgs = [img for img in middle_imgs if img is not None]
middle_labels = np.array(middle_df['has_finding'].tolist()[:len(middle_imgs)])

print("Loading older images...")
older_imgs = [load_image(p) for p in older_df['image_path']]
older_imgs = [img for img in older_imgs if img is not None]
older_labels = np.array(older_df['has_finding'].tolist()[:len(older_imgs)])

print(f"Loaded: young={len(young_imgs)}, middle={len(middle_imgs)}, older={len(older_imgs)}")

print("\nExtracting CLIP features — age groups...")
young_clip = extract_clip_features(young_imgs)
middle_clip = extract_clip_features(middle_imgs)
older_clip = extract_clip_features(older_imgs)

print("\nExtracting ResNet features — age groups...")
young_resnet = extract_resnet_features(young_imgs)
middle_resnet = extract_resnet_features(middle_imgs)
older_resnet = extract_resnet_features(older_imgs)

print(f"\nFeature shapes: young={young_clip.shape}, middle={middle_clip.shape}, older={older_clip.shape}")

# ── CELL 8: Load and extract features — sex ──────────────────
print("\n" + "="*60)
print("LOADING IMAGES — SEX")
print("="*60)

print("Loading male images...")
male_imgs = [load_image(p) for p in male_df['image_path']]
male_imgs = [img for img in male_imgs if img is not None]
male_labels = np.array(male_df['has_finding'].tolist()[:len(male_imgs)])

print("Loading female images...")
female_imgs = [load_image(p) for p in female_df['image_path']]
female_imgs = [img for img in female_imgs if img is not None]
female_labels = np.array(female_df['has_finding'].tolist()[:len(female_imgs)])

print(f"Loaded: male={len(male_imgs)}, female={len(female_imgs)}")

print("\nExtracting CLIP features — sex...")
male_clip = extract_clip_features(male_imgs)
female_clip = extract_clip_features(female_imgs)

print("\nExtracting ResNet features — sex...")
male_resnet = extract_resnet_features(male_imgs)
female_resnet = extract_resnet_features(female_imgs)

print(f"\nFeature shapes: male={male_clip.shape}, female={female_clip.shape}")

# ── CELL 9: AGE EXPERIMENT — Random split ────────────────────
print("\n" + "="*60)
print("AGE EXPERIMENT 1: RANDOM SPLIT")
print("="*60)

all_age_clip = np.vstack([young_clip, middle_clip, older_clip])
all_age_resnet = np.vstack([young_resnet, middle_resnet, older_resnet])
all_age_labels = np.concatenate([young_labels, middle_labels, older_labels])

train_idx, test_idx = train_test_split(
    np.arange(len(all_age_labels)),
    test_size=0.25, stratify=all_age_labels, random_state=42
)

age_random_results = {}
for model_name, feats in [('CLIP', all_age_clip), ('ResNet50', all_age_resnet)]:
    result, clf, probs = run_experiment(
        feats[train_idx], all_age_labels[train_idx],
        feats[test_idx], all_age_labels[test_idx],
        name=f"{model_name} Random"
    )
    age_random_results[model_name] = result

# ── CELL 10: AGE EXPERIMENT — Age-aware split ────────────────
print("\n" + "="*60)
print("AGE EXPERIMENT 2: AGE-AWARE SPLIT")
print("Train: young + middle | Test: older")
print("="*60)

age_aware_results = {}
age_aware_clfs = {}
for model_name, y_clip, m_clip, o_clip, y_resnet, m_resnet, o_resnet in [
    ('CLIP', young_clip, middle_clip, older_clip, None, None, None),
    ('ResNet50', young_resnet, middle_resnet, older_resnet, None, None, None)
]:
    if model_name == 'CLIP':
        train_feats = np.vstack([young_clip, middle_clip])
        test_feats = older_clip
    else:
        train_feats = np.vstack([young_resnet, middle_resnet])
        test_feats = older_resnet
    
    train_labels = np.concatenate([young_labels, middle_labels])
    test_labels = older_labels
    
    result, clf, probs = run_experiment(
        train_feats, train_labels, test_feats, test_labels,
        name=f"{model_name} Age-Aware"
    )
    age_aware_results[model_name] = result
    age_aware_clfs[model_name] = (clf, probs, test_labels)

# ── CELL 11: SEX EXPERIMENT — Random split ───────────────────
print("\n" + "="*60)
print("SEX EXPERIMENT 1: RANDOM SPLIT")
print("="*60)

all_sex_clip = np.vstack([male_clip, female_clip])
all_sex_resnet = np.vstack([male_resnet, female_resnet])
all_sex_labels = np.concatenate([male_labels, female_labels])

train_idx_s, test_idx_s = train_test_split(
    np.arange(len(all_sex_labels)),
    test_size=0.25, stratify=all_sex_labels, random_state=42
)

sex_random_results = {}
for model_name, feats in [('CLIP', all_sex_clip), ('ResNet50', all_sex_resnet)]:
    result, clf, probs = run_experiment(
        feats[train_idx_s], all_sex_labels[train_idx_s],
        feats[test_idx_s], all_sex_labels[test_idx_s],
        name=f"{model_name} Random"
    )
    sex_random_results[model_name] = result

# ── CELL 12: SEX EXPERIMENT — Sex-aware split ────────────────
print("\n" + "="*60)
print("SEX EXPERIMENT 2: SEX-AWARE SPLIT")
print("Train: male | Test: female")
print("="*60)

sex_aware_results = {}
for model_name, train_feats, test_feats in [
    ('CLIP', male_clip, female_clip),
    ('ResNet50', male_resnet, female_resnet)
]:
    result, clf, probs = run_experiment(
        train_feats, male_labels, test_feats, female_labels,
        name=f"{model_name} Sex-Aware"
    )
    sex_aware_results[model_name] = result

# ── CELL 13: DACC on NIH ─────────────────────────────────────
print("\n" + "="*60)
print("APPLYING DACC — NIH ChestX-ray14")
print("="*60)

class GroupTemperatureScaling(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.temperatures = nn.ParameterDict({
            str(g).replace('<', 'lt').replace('>', 'gt').replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_'): 
            nn.Parameter(torch.ones(1) * 1.5)
            for g in groups
        })
    
    def get_temp(self, g):
        key = str(g).replace('<', 'lt').replace('>', 'gt').replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        return torch.clamp(self.temperatures[key], 0.05, 10.0)
    
    def forward(self, logits, groups):
        calibrated = torch.zeros_like(logits)
        for i, g in enumerate(groups):
            T = self.get_temp(g)
            calibrated[i] = logits[i] / T
        return calibrated

dacc_age_results = {}
dacc_sex_results = {}

# DACC for age experiment
print("\nDACC for age experiment...")
for model_name in ['CLIP', 'ResNet50']:
    if model_name == 'CLIP':
        train_feats = np.vstack([young_clip, middle_clip])
        test_feats = older_clip
    else:
        train_feats = np.vstack([young_resnet, middle_resnet])
        test_feats = older_resnet
    
    train_labels_age = np.concatenate([young_labels, middle_labels])
    train_groups = ['young'] * len(young_labels) + ['middle'] * len(middle_labels)
    
    # Use last 20% of train as validation for DACC
    val_n = int(0.2 * len(train_feats))
    val_feats = train_feats[-val_n:]
    val_labels_t = train_labels_age[-val_n:]
    val_groups_t = train_groups[-val_n:]
    
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(train_feats[:-val_n], train_labels_age[:-val_n])
    
    val_logits = torch.FloatTensor(clf.decision_function(val_feats))
    if val_logits.dim() == 1:
        val_logits = val_logits.unsqueeze(1)
        val_logits = torch.cat([-val_logits, val_logits], dim=1)
    val_y_t = torch.LongTensor(val_labels_t)
    
    dacc = GroupTemperatureScaling(['young', 'middle', 'older'])
    optimizer = torch.optim.Adam(dacc.parameters(), lr=0.01)
    
    for epoch in range(100):
        optimizer.zero_grad()
        calibrated = dacc(val_logits, val_groups_t)
        loss = F.cross_entropy(calibrated, val_y_t)
        loss.backward()
        optimizer.step()
    
    test_logits = torch.FloatTensor(clf.decision_function(test_feats))
    if test_logits.dim() == 1:
        test_logits = test_logits.unsqueeze(1)
        test_logits = torch.cat([-test_logits, test_logits], dim=1)
    test_groups_list = ['older'] * len(older_labels)
    
    with torch.no_grad():
        calibrated_test = dacc(test_logits, test_groups_list)
        dacc_probs = F.softmax(calibrated_test, dim=-1).numpy()
        dacc_preds = calibrated_test.argmax(dim=-1).numpy()
    
    try:
        auc = roc_auc_score(older_labels, dacc_probs[:, 1])
        acc = accuracy_score(older_labels, dacc_preds)
        f1 = f1_score(older_labels, dacc_preds, average='macro')
        scores = []
        for _ in range(500):
            idx = np.random.choice(len(older_labels), len(older_labels), replace=True)
            try:
                scores.append(roc_auc_score(older_labels[idx], dacc_probs[idx, 1]))
            except:
                pass
        ci_low, ci_high = np.percentile(scores, [2.5, 97.5])
        dacc_age_results[model_name] = {'auc': auc, 'acc': acc, 'f1': f1, 'ci_low': ci_low, 'ci_high': ci_high}
        print(f"{model_name} DACC Age | AUC: {auc:.4f} ({ci_low:.4f}-{ci_high:.4f}) | Acc: {acc:.4f} | F1: {f1:.4f}")
        
        temps = {g: dacc.get_temp(g).item() for g in ['young', 'middle', 'older']}
        print(f"  Temperatures: {temps}")
    except Exception as e:
        print(f"{model_name} DACC age error: {e}")

# DACC for sex experiment  
print("\nDACC for sex experiment...")
for model_name in ['CLIP', 'ResNet50']:
    if model_name == 'CLIP':
        train_feats = male_clip
        test_feats = female_clip
    else:
        train_feats = male_resnet
        test_feats = female_resnet
    
    val_n = int(0.2 * len(train_feats))
    val_feats = train_feats[-val_n:]
    val_labels_t = male_labels[-val_n:]
    val_groups_t = ['male'] * val_n
    
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(train_feats[:-val_n], male_labels[:-val_n])
    
    val_logits = torch.FloatTensor(clf.decision_function(val_feats))
    if val_logits.dim() == 1:
        val_logits = val_logits.unsqueeze(1)
        val_logits = torch.cat([-val_logits, val_logits], dim=1)
    val_y_t = torch.LongTensor(val_labels_t)
    
    dacc = GroupTemperatureScaling(['male', 'female'])
    optimizer = torch.optim.Adam(dacc.parameters(), lr=0.01)
    
    for epoch in range(100):
        optimizer.zero_grad()
        calibrated = dacc(val_logits, val_groups_t)
        loss = F.cross_entropy(calibrated, val_y_t)
        loss.backward()
        optimizer.step()
    
    test_logits = torch.FloatTensor(clf.decision_function(test_feats))
    if test_logits.dim() == 1:
        test_logits = test_logits.unsqueeze(1)
        test_logits = torch.cat([-test_logits, test_logits], dim=1)
    test_groups_list = ['female'] * len(female_labels)
    
    with torch.no_grad():
        calibrated_test = dacc(test_logits, test_groups_list)
        dacc_probs = F.softmax(calibrated_test, dim=-1).numpy()
        dacc_preds = calibrated_test.argmax(dim=-1).numpy()
    
    try:
        auc = roc_auc_score(female_labels, dacc_probs[:, 1])
        acc = accuracy_score(female_labels, dacc_preds)
        f1 = f1_score(female_labels, dacc_preds, average='macro')
        scores = []
        for _ in range(500):
            idx = np.random.choice(len(female_labels), len(female_labels), replace=True)
            try:
                scores.append(roc_auc_score(female_labels[idx], dacc_probs[idx, 1]))
            except:
                pass
        ci_low, ci_high = np.percentile(scores, [2.5, 97.5])
        dacc_sex_results[model_name] = {'auc': auc, 'acc': acc, 'f1': f1, 'ci_low': ci_low, 'ci_high': ci_high}
        print(f"{model_name} DACC Sex | AUC: {auc:.4f} ({ci_low:.4f}-{ci_high:.4f}) | Acc: {acc:.4f} | F1: {f1:.4f}")
        
        temps = {g: dacc.get_temp(g).item() for g in ['male', 'female']}
        print(f"  Temperatures: {temps}")
    except Exception as e:
        print(f"{model_name} DACC sex error: {e}")

# ── CELL 14: Final results table ──────────────────────────────
print("\n" + "="*60)
print("NIH ChestX-ray14 — COMPLETE RESULTS TABLE")
print("="*60)

print("\n--- AGE GENERALIZATION ---")
print(f"{'Method':<35} {'AUC':>8} {'95% CI':>15} {'Acc':>8} {'F1':>8}")
print("-"*80)
for model_name in ['CLIP', 'ResNet50']:
    if model_name in age_random_results:
        r = age_random_results[model_name]
        print(f"{model_name} Random Split{'':<20} {r['auc']:>8.4f} ({r['ci_low']:.4f}-{r['ci_high']:.4f}) {r['acc']:>8.4f} {r['f1']:>8.4f}")
    if model_name in age_aware_results:
        r = age_aware_results[model_name]
        print(f"{model_name} Age-Aware Split{'':<17} {r['auc']:>8.4f} ({r['ci_low']:.4f}-{r['ci_high']:.4f}) {r['acc']:>8.4f} {r['f1']:>8.4f}")
    if model_name in dacc_age_results:
        r = dacc_age_results[model_name]
        print(f"{model_name} Age-Aware + DACC{'':<16} {r['auc']:>8.4f} ({r['ci_low']:.4f}-{r['ci_high']:.4f}) {r['acc']:>8.4f} {r['f1']:>8.4f}")
    if model_name in age_random_results and model_name in age_aware_results:
        sgg = age_random_results[model_name]['auc'] - age_aware_results[model_name]['auc']
        print(f"  → Age SGG ({model_name}): {sgg:.4f}")

print("\n--- SEX GENERALIZATION ---")
print(f"{'Method':<35} {'AUC':>8} {'95% CI':>15} {'Acc':>8} {'F1':>8}")
print("-"*80)
for model_name in ['CLIP', 'ResNet50']:
    if model_name in sex_random_results:
        r = sex_random_results[model_name]
        print(f"{model_name} Random Split{'':<20} {r['auc']:>8.4f} ({r['ci_low']:.4f}-{r['ci_high']:.4f}) {r['acc']:>8.4f} {r['f1']:>8.4f}")
    if model_name in sex_aware_results:
        r = sex_aware_results[model_name]
        print(f"{model_name} Sex-Aware Split{'':<17} {r['auc']:>8.4f} ({r['ci_low']:.4f}-{r['ci_high']:.4f}) {r['acc']:>8.4f} {r['f1']:>8.4f}")
    if model_name in dacc_sex_results:
        r = dacc_sex_results[model_name]
        print(f"{model_name} Sex-Aware + DACC{'':<16} {r['auc']:>8.4f} ({r['ci_low']:.4f}-{r['ci_high']:.4f}) {r['acc']:>8.4f} {r['f1']:>8.4f}")
    if model_name in sex_random_results and model_name in sex_aware_results:
        sgg = sex_random_results[model_name]['auc'] - sex_aware_results[model_name]['auc']
        print(f"  → Sex SGG ({model_name}): {sgg:.4f}")

# ── CELL 15: Save results ────────────────────────────────────
results_to_save = {
    'dataset': 'nih_chestxray14',
    'age_random': age_random_results,
    'age_aware': age_aware_results,
    'dacc_age': dacc_age_results,
    'sex_random': sex_random_results,
    'sex_aware': sex_aware_results,
    'dacc_sex': dacc_sex_results,
    'n_young': len(young_imgs),
    'n_middle': len(middle_imgs),
    'n_older': len(older_imgs),
    'n_male': len(male_imgs),
    'n_female': len(female_imgs),
}

with open('/kaggle/working/nih_results.json', 'w') as f:
    json.dump(results_to_save, f, indent=2)

print("\nResults saved to /kaggle/working/nih_results.json")
print("\n✓ Overnight notebook complete. Paste ALL output back to Claude.")
