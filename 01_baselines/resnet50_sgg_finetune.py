# ============================================================
# NOTEBOOK — ResNet50 Fine-Tuning on Fitzpatrick17k
# Dataset: nazmusresan/fitzpatrick17k
# GPU T4 x1, Internet ON
# Expected runtime: ~25 minutes
#
# Tests whether SGG persists under ResNet50 fine-tuning.
# We already showed CLIP fine-tuned SGG = 0.040 (vs 0.047 linear probe).
# If ResNet50 also shows persistent SGG under fine-tuning, the
# fine-tuning-robust claim covers all three architectures.
#
# After running, paste ALL output back to Claude
# ============================================================

!pip install torch torchvision scikit-learn pandas numpy -q

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings, os, json
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

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

MAX = 1000
light_df  = df[df['skin_group']=='light'].sample(MAX, random_state=42)
medium_df = df[df['skin_group']=='medium'].sample(MAX, random_state=42)
dark_df   = df[df['skin_group']=='dark'].sample(
    min(MAX, len(df[df['skin_group']=='dark'])), random_state=42)

le = LabelEncoder()
all_labels = (list(light_df['three_partition_label']) +
              list(medium_df['three_partition_label']) +
              list(dark_df['three_partition_label']))
le.fit(all_labels)
print(f"Classes: {le.classes_}")

# ── Dataset ───────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize(256), transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
test_transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class SkinDataset(Dataset):
    def __init__(self, dataframe, label_encoder, transform):
        self.transform = transform
        self.images, self.labels = [], []
        for _, row in dataframe.iterrows():
            try:
                img = Image.open(row['local_path']).convert('RGB')
                self.images.append(img)
                self.labels.append(label_encoder.transform([row['three_partition_label']])[0])
            except: pass
        print(f"  Loaded {len(self.images)} images")

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        return self.transform(self.images[idx]), torch.tensor(self.labels[idx], dtype=torch.long)

# ── Model ─────────────────────────────────────────────────────
class ResNet50FineTuned(nn.Module):
    def __init__(self, num_classes=3, dropout=0.3):
        super().__init__()
        backbone = models.resnet50(pretrained=True)
        # Freeze all but last 2 blocks + FC
        for name, param in backbone.named_parameters():
            if not any(x in name for x in ['layer4', 'fc']):
                param.requires_grad = False
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feats = self.backbone(x).squeeze(-1).squeeze(-1)
        return self.classifier(feats)

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, name):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            probs = torch.softmax(model(imgs.to(device)), dim=-1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    probs = np.vstack(all_probs)
    labels = np.concatenate(all_labels)
    preds = probs.argmax(axis=1)
    auc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average='macro')
    scores = []
    for _ in range(500):
        idx = np.random.choice(len(labels), len(labels), replace=True)
        try: scores.append(roc_auc_score(labels[idx], probs[idx],
                                          multi_class='ovr', average='macro'))
        except: pass
    ci_low, ci_high = np.percentile(scores, [2.5, 97.5])
    print(f"  {name}: AUC={auc:.4f} ({ci_low:.4f}-{ci_high:.4f}) Acc={acc:.4f} F1={f1:.4f}")

    # Per-class on dark skin
    per_class = {}
    for i, cls in enumerate(le.classes_):
        mask = labels == i
        if mask.sum() > 0:
            per_class[cls] = accuracy_score(labels[mask], preds[mask])

    return {'auc': auc, 'acc': acc, 'f1': f1,
            'ci_low': ci_low, 'ci_high': ci_high,
            'per_class': per_class}, probs, labels

# ══════════════════════════════════════════════════════════════
# EXPERIMENT 1: RANDOM SPLIT
# ══════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("EXPERIMENT 1: RANDOM SPLIT — Fine-tuned ResNet50")
print("="*55)

all_df = pd.concat([light_df, medium_df, dark_df]).reset_index(drop=True)
tr_idx, te_idx = train_test_split(
    np.arange(len(all_df)), test_size=0.25,
    stratify=all_df['three_partition_label'].values, random_state=42)

print("Loading datasets...")
train_rand = SkinDataset(all_df.iloc[tr_idx], le, train_transform)
test_rand  = SkinDataset(all_df.iloc[te_idx], le, test_transform)
train_rand_dl = DataLoader(train_rand, batch_size=32, shuffle=True,  num_workers=2)
test_rand_dl  = DataLoader(test_rand,  batch_size=64, shuffle=False, num_workers=2)

model_rand = ResNet50FineTuned(num_classes=3).to(device)
optimizer  = optim.AdamW(
    filter(lambda p: p.requires_grad, model_rand.parameters()),
    lr=1e-4, weight_decay=0.01)
criterion  = nn.CrossEntropyLoss()

print("Fine-tuning (random split)...")
for epoch in range(5):
    loss = train_epoch(model_rand, train_rand_dl, optimizer, criterion)
    if epoch % 2 == 0 or epoch == 4:
        print(f"  Epoch {epoch+1}/5, Loss: {loss:.4f}")

rand_results, _, _ = evaluate(model_rand, test_rand_dl, "ResNet50 Fine-tuned Random")

# ══════════════════════════════════════════════════════════════
# EXPERIMENT 2: SKIN-TONE SPLIT
# ══════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("EXPERIMENT 2: SKIN-TONE SPLIT — Fine-tuned ResNet50")
print("Train: light + medium | Test: dark")
print("="*55)

train_skin_df = pd.concat([light_df, medium_df]).reset_index(drop=True)
test_skin_df  = dark_df.reset_index(drop=True)

print("Loading datasets...")
train_skin = SkinDataset(train_skin_df, le, train_transform)
test_skin  = SkinDataset(test_skin_df,  le, test_transform)
train_skin_dl = DataLoader(train_skin, batch_size=32, shuffle=True,  num_workers=2)
test_skin_dl  = DataLoader(test_skin,  batch_size=64, shuffle=False, num_workers=2)

model_skin = ResNet50FineTuned(num_classes=3).to(device)
optimizer2 = optim.AdamW(
    filter(lambda p: p.requires_grad, model_skin.parameters()),
    lr=1e-4, weight_decay=0.01)

print("Fine-tuning (skin-tone split)...")
for epoch in range(5):
    loss = train_epoch(model_skin, train_skin_dl, optimizer2, criterion)
    if epoch % 2 == 0 or epoch == 4:
        print(f"  Epoch {epoch+1}/5, Loss: {loss:.4f}")

skin_results, skin_probs, skin_labels = evaluate(
    model_skin, test_skin_dl, "ResNet50 Fine-tuned Skin-Tone")

print("\nPer-class accuracy on dark skin (ResNet50 fine-tuned):")
for cls, acc in skin_results['per_class'].items():
    print(f"  {cls}: {acc:.3f}")

sgg_ft = rand_results['auc'] - skin_results['auc']
print(f"\n*** RESNET50 FINE-TUNED SGG: {sgg_ft:.4f} ***")
print(f"ResNet50 linear probe SGG was: 0.027")
print(f"SGG {'PERSISTS' if sgg_ft > 0.01 else 'REDUCED/ELIMINATED'} under fine-tuning")

# ── Summary ───────────────────────────────────────────────────
print("\n" + "="*55)
print("COMPLETE SUMMARY")
print("="*55)
print(f"\n{'Method':<45} {'AUC':>8} {'SGG':>8}")
print("-"*63)
print(f"{'ResNet50 LP Random (from paper)':<45} {'0.691':>8} {'—':>8}")
print(f"{'ResNet50 LP Skin-Tone (from paper)':<45} {'0.664':>8} {'0.027':>8}")
print(f"{'ResNet50 FT Random':<45} {rand_results['auc']:>8.4f} {'—':>8}")
print(f"{'ResNet50 FT Skin-Tone':<45} {skin_results['auc']:>8.4f} {sgg_ft:>8.4f}")
print(f"\nCLIP comparison:")
print(f"{'CLIP LP SGG':<45} {'0.047':>8}")
print(f"{'CLIP FT SGG':<45} {'0.040':>8}")
print(f"{'ResNet50 LP SGG':<45} {'0.027':>8}")
print(f"{'ResNet50 FT SGG':<45} {sgg_ft:>8.4f}")

results = {
    'model': 'ResNet50_finetuned',
    'random_split': rand_results,
    'skin_tone_split': skin_results,
    'sgg': sgg_ft,
    'linear_probe_sgg': 0.027,
}
for k in ['random_split', 'skin_tone_split']:
    results[k].pop('per_class', None)

with open('/kaggle/working/resnet50_finetune_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nSaved to /kaggle/working/resnet50_finetune_results.json")
print("\n✓ Complete. Paste ALL output back to Claude.")
