# ============================================================
# NOTEBOOK — Fine-Tuned CLIP on Fitzpatrick17k
# Datasets: nazmusresan/fitzpatrick17k
# GPU T4 x1, Internet ON
# Expected runtime: ~30-40 minutes
# Tests whether SGG persists under fine-tuning (not just linear probe)
# After running, paste ALL output back to Claude
# ============================================================

!pip install transformers torch torchvision scikit-learn pandas numpy -q

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import CLIPModel, CLIPProcessor
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
light_df = df[df['skin_group']=='light'].sample(MAX, random_state=42)
medium_df = df[df['skin_group']=='medium'].sample(MAX, random_state=42)
dark_df = df[df['skin_group']=='dark'].sample(
    min(MAX, len(df[df['skin_group']=='dark'])), random_state=42)

le = LabelEncoder()
all_labels = list(light_df['three_partition_label']) + \
             list(medium_df['three_partition_label']) + \
             list(dark_df['three_partition_label'])
le.fit(all_labels)
print(f"Classes: {le.classes_}")

# ── Dataset class ─────────────────────────────────────────────
class SkinDataset(Dataset):
    def __init__(self, dataframe, processor, label_encoder):
        self.df = dataframe.reset_index(drop=True)
        self.processor = processor
        self.le = label_encoder
        self.valid_indices = []
        self.images = []
        self.labels = []
        for i, row in self.df.iterrows():
            try:
                img = Image.open(row['local_path']).convert('RGB')
                self.images.append(img)
                self.labels.append(self.le.transform([row['three_partition_label']])[0])
                self.valid_indices.append(i)
            except: pass
        print(f"  Loaded {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        inputs = self.processor(images=img, return_tensors="pt", padding=True)
        pixel_values = inputs['pixel_values'].squeeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return pixel_values, label

# ── Fine-tuning model ─────────────────────────────────────────
class CLIPFineTuned(nn.Module):
    def __init__(self, clip_model, num_classes=3, dropout=0.3):
        super().__init__()
        self.vision_model = clip_model.vision_model
        self.visual_projection = clip_model.visual_projection
        hidden_size = clip_model.config.projection_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, pixel_values):
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        pooled_output = vision_outputs.pooler_output
        image_features = self.visual_projection(pooled_output)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = self.classifier(image_features)
        return logits

def evaluate_model(model, dataloader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for pixel_values, labels in dataloader:
            pixel_values = pixel_values.to(device)
            logits = model(pixel_values)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)
    preds = all_probs.argmax(axis=1)
    auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds, average='macro')
    # Bootstrap CI
    scores = []
    for _ in range(500):
        idx = np.random.choice(len(all_labels), len(all_labels), replace=True)
        try:
            scores.append(roc_auc_score(all_labels[idx], all_probs[idx],
                                         multi_class='ovr', average='macro'))
        except: pass
    ci_low, ci_high = np.percentile(scores, [2.5, 97.5])
    return {'auc': auc, 'acc': acc, 'f1': f1,
            'ci_low': ci_low, 'ci_high': ci_high,
            'probs': all_probs, 'labels': all_labels, 'preds': preds}

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for pixel_values, labels in dataloader:
        pixel_values, labels = pixel_values.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(pixel_values)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# ── Load CLIP ─────────────────────────────────────────────────
print("\nLoading CLIP...")
clip_base = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
print("CLIP loaded.")

# ══════════════════════════════════════════════════════════════
# EXPERIMENT 1: RANDOM SPLIT (Fine-tuned)
# ══════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("EXPERIMENT 1: RANDOM SPLIT — Fine-tuned CLIP")
print("="*55)

all_df = pd.concat([light_df, medium_df, dark_df]).reset_index(drop=True)
train_idx, test_idx = train_test_split(
    np.arange(len(all_df)), test_size=0.25,
    stratify=all_df['three_partition_label'].values, random_state=42)

train_rand_df = all_df.iloc[train_idx]
test_rand_df = all_df.iloc[test_idx]

print(f"Train: {len(train_rand_df)}, Test: {len(test_rand_df)}")
print("Loading datasets...")
train_rand_ds = SkinDataset(train_rand_df, processor, le)
test_rand_ds = SkinDataset(test_rand_df, processor, le)

train_rand_dl = DataLoader(train_rand_ds, batch_size=16, shuffle=True, num_workers=2)
test_rand_dl = DataLoader(test_rand_ds, batch_size=32, shuffle=False, num_workers=2)

# Fine-tune
model_rand = CLIPFineTuned(clip_base, num_classes=3).to(device)
# Freeze vision backbone, only train classifier + projection
for param in model_rand.vision_model.parameters():
    param.requires_grad = False
optimizer = optim.AdamW(
    list(model_rand.visual_projection.parameters()) +
    list(model_rand.classifier.parameters()),
    lr=1e-4, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

print("Fine-tuning (random split)...")
for epoch in range(5):
    loss = train_epoch(model_rand, train_rand_dl, optimizer, criterion, device)
    if (epoch+1) % 2 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1}/5, Loss: {loss:.4f}")

rand_results = evaluate_model(model_rand, test_rand_dl, device)
print(f"\nRandom Split — AUC: {rand_results['auc']:.4f} "
      f"({rand_results['ci_low']:.4f}-{rand_results['ci_high']:.4f}) "
      f"| Acc: {rand_results['acc']:.4f} | F1: {rand_results['f1']:.4f}")

# ══════════════════════════════════════════════════════════════
# EXPERIMENT 2: SKIN-TONE SPLIT (Fine-tuned)
# ══════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("EXPERIMENT 2: SKIN-TONE SPLIT — Fine-tuned CLIP")
print("Train: light + medium | Test: dark")
print("="*55)

train_skin_df = pd.concat([light_df, medium_df]).reset_index(drop=True)
test_skin_df = dark_df.reset_index(drop=True)

print(f"Train: {len(train_skin_df)}, Test: {len(test_skin_df)}")
print("Loading datasets...")
train_skin_ds = SkinDataset(train_skin_df, processor, le)
test_skin_ds = SkinDataset(test_skin_df, processor, le)

train_skin_dl = DataLoader(train_skin_ds, batch_size=16, shuffle=True, num_workers=2)
test_skin_dl = DataLoader(test_skin_ds, batch_size=32, shuffle=False, num_workers=2)

# Reload fresh model
clip_base2 = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
model_skin = CLIPFineTuned(clip_base2, num_classes=3).to(device)
for param in model_skin.vision_model.parameters():
    param.requires_grad = False
optimizer2 = optim.AdamW(
    list(model_skin.visual_projection.parameters()) +
    list(model_skin.classifier.parameters()),
    lr=1e-4, weight_decay=0.01)

print("Fine-tuning (skin-tone split)...")
for epoch in range(5):
    loss = train_epoch(model_skin, train_skin_dl, optimizer2, criterion, device)
    if (epoch+1) % 2 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1}/5, Loss: {loss:.4f}")

skin_results = evaluate_model(model_skin, test_skin_dl, device)
print(f"\nSkin-Tone Split — AUC: {skin_results['auc']:.4f} "
      f"({skin_results['ci_low']:.4f}-{skin_results['ci_high']:.4f}) "
      f"| Acc: {skin_results['acc']:.4f} | F1: {skin_results['f1']:.4f}")

# Per-class accuracy
print("\nPer-class accuracy on dark skin (fine-tuned):")
for i, cls in enumerate(le.classes_):
    mask = skin_results['labels'] == i
    if mask.sum() > 0:
        acc = accuracy_score(skin_results['labels'][mask], skin_results['preds'][mask])
        print(f"  {cls}: {acc:.3f} (n={mask.sum()})")

# SGG
sgg_ft = rand_results['auc'] - skin_results['auc']
print(f"\n*** FINE-TUNED SGG: {sgg_ft:.4f} ***")
print(f"Linear probe SGG was: 0.0473")
print(f"SGG {'PERSISTS' if sgg_ft > 0.01 else 'REDUCED'} under fine-tuning")

# ── Summary ───────────────────────────────────────────────────
print("\n" + "="*55)
print("COMPLETE SUMMARY — KEY NUMBERS FOR PAPER")
print("="*55)
print(f"{'Method':<40} {'AUC':>8} {'SGG':>8}")
print("-"*58)
print(f"{'Linear probe Random Split':<40} {'0.789':>8} {'—':>8}")
print(f"{'Linear probe Skin-Tone Split':<40} {'0.742':>8} {'0.047':>8}")
print(f"{'Fine-tuned Random Split':<40} {rand_results['auc']:>8.4f} {'—':>8}")
print(f"{'Fine-tuned Skin-Tone Split':<40} {skin_results['auc']:>8.4f} {sgg_ft:>8.4f}")

results = {
    'random_split': rand_results,
    'skin_tone_split': skin_results,
    'sgg': sgg_ft,
    'linear_probe_sgg': 0.0473,
    'classes': list(le.classes_),
}
# Remove non-serializable arrays
for k in ['random_split', 'skin_tone_split']:
    for kk in ['probs', 'labels', 'preds']:
        results[k].pop(kk, None)

with open('/kaggle/working/finetune_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nSaved to /kaggle/working/finetune_results.json")
print("\n✓ Complete. Paste ALL output back to Claude.")
