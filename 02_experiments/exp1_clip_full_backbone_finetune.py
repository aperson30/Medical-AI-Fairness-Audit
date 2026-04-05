# ============================================================
# NOTEBOOK 1 — Full Backbone CLIP Fine-Tuning
# Dataset: nazmusresan/fitzpatrick17k
# GPU T4, Internet ON. ~60 min.
# Priority: HIGHEST — closes linear probe criticism completely
# ============================================================

!pip install transformers torch torchvision scikit-learn pandas numpy -q

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np, pandas as pd, os, json, warnings
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import CLIPModel, CLIPProcessor
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

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
dark_df   = df[df['skin_group']=='dark'].sample(min(MAX, len(df[df['skin_group']=='dark'])), random_state=42)

le = LabelEncoder()
le.fit(list(light_df['three_partition_label']) +
       list(medium_df['three_partition_label']) +
       list(dark_df['three_partition_label']))
print(f"Classes: {le.classes_}")

class SkinDataset(Dataset):
    def __init__(self, dataframe, processor, label_encoder):
        self.processor = processor
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
        inputs = self.processor(images=self.images[idx], return_tensors="pt", padding=True)
        return inputs['pixel_values'].squeeze(0), torch.tensor(self.labels[idx], dtype=torch.long)

class CLIPFullFT(nn.Module):
    def __init__(self, clip_model, num_classes=3, dropout=0.3):
        super().__init__()
        self.vision_model = clip_model.vision_model
        self.visual_projection = clip_model.visual_projection
        hidden = clip_model.config.projection_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden, 256),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, num_classes))
        for p in self.parameters(): p.requires_grad = True

    def forward(self, pixel_values):
        out  = self.vision_model(pixel_values=pixel_values)
        proj = self.visual_projection(out.pooler_output)
        proj = proj / proj.norm(dim=-1, keepdim=True)
        return self.classifier(proj)

def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            loss = criterion(model(imgs), labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
        total += loss.item()
    return total / len(loader)

@torch.no_grad()
def evaluate_model(model, loader, name):
    model.eval()
    all_logits, all_labels = [], []
    for imgs, labels in loader:
        with torch.cuda.amp.autocast():
            logits = model(imgs.to(device))
        all_logits.append(logits.cpu().float())
        all_labels.append(labels)
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs  = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs  = probs / probs.sum(axis=1, keepdims=True)
    preds  = probs.argmax(axis=1)
    auc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average='macro')
    scores = []
    for _ in range(500):
        idx = np.random.choice(len(labels), len(labels), replace=True)
        try: scores.append(roc_auc_score(labels[idx], probs[idx], multi_class='ovr', average='macro'))
        except: pass
    ci_low, ci_high = np.percentile(scores, [2.5, 97.5])
    print(f"  {name}: AUC={auc:.4f} ({ci_low:.4f}-{ci_high:.4f}) Acc={acc:.4f} F1={f1:.4f}")
    per_class = {cls: float(accuracy_score(labels[labels==i], preds[labels==i]))
                 for i, cls in enumerate(le.classes_) if (labels==i).sum() > 0}
    return {'auc': float(auc), 'acc': float(acc), 'f1': float(f1),
            'ci_low': float(ci_low), 'ci_high': float(ci_high), 'per_class': per_class}

print("Loading processor...")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
criterion = nn.CrossEntropyLoss()

def run_experiment(train_df, test_df, exp_name):
    print(f"\nLoading datasets for {exp_name}...")
    train_ds = SkinDataset(train_df, processor, le)
    test_ds  = SkinDataset(test_df,  processor, le)
    train_dl = DataLoader(train_ds, batch_size=8,  shuffle=True,  num_workers=2)
    test_dl  = DataLoader(test_ds,  batch_size=16, shuffle=False, num_workers=2)
    clip     = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    model    = CLIPFullFT(clip).to(device)
    opt      = optim.AdamW([
        {'params': model.vision_model.parameters(),      'lr': 1e-6},
        {'params': model.visual_projection.parameters(), 'lr': 1e-5},
        {'params': model.classifier.parameters(),        'lr': 1e-4},
    ], weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(3):
        loss = train_epoch(model, train_dl, opt, criterion, scaler)
        print(f"  Epoch {epoch+1}/3 Loss={loss:.4f}")
        torch.cuda.empty_cache()
    results = evaluate_model(model, test_dl, exp_name)
    del model, clip; torch.cuda.empty_cache()
    return results

# Random split
all_df = pd.concat([light_df, medium_df, dark_df]).reset_index(drop=True)
tr, te = train_test_split(np.arange(len(all_df)), test_size=0.25,
                           stratify=all_df['three_partition_label'].values, random_state=42)
print("\n=== EXPERIMENT 1: RANDOM SPLIT ===")
rand_results = run_experiment(all_df.iloc[tr], all_df.iloc[te], "CLIP Full FT Random")

# Skin-tone split
print("\n=== EXPERIMENT 2: SKIN-TONE SPLIT ===")
skin_results = run_experiment(
    pd.concat([light_df, medium_df]).reset_index(drop=True),
    dark_df.reset_index(drop=True), "CLIP Full FT Skin-Tone")

print("\nPer-class on dark skin:")
for cls, acc in skin_results['per_class'].items():
    print(f"  {cls}: {acc:.3f}")

sgg = rand_results['auc'] - skin_results['auc']
print(f"\n*** FULL FT CLIP SGG: {sgg:.4f} ***")
print(f"Head-only FT SGG: 0.040 | Linear probe SGG: 0.047")
print(f"SGG {'PERSISTS' if sgg > 0.01 else 'REDUCED'} under full fine-tuning")

print("\n=== SUMMARY ===")
for name, auc, sgg_val in [
    ("CLIP LP Random",        0.789, None),
    ("CLIP LP Skin-Tone",     0.742, 0.047),
    ("CLIP Head FT Skin-Tone",0.742, 0.040),
    ("CLIP Full FT Random",   rand_results['auc'], None),
    ("CLIP Full FT Skin-Tone",skin_results['auc'], sgg),
]:
    print(f"  {name:<30} AUC={auc:.4f}  SGG={sgg_val if sgg_val else '—'}")

json.dump({'random': {k:v for k,v in rand_results.items() if k!='per_class'},
           'skin_tone': {k:v for k,v in skin_results.items() if k!='per_class'},
           'per_class_dark': skin_results['per_class'],
           'sgg': float(sgg)},
          open('/kaggle/working/nb1_clip_full_ft.json','w'), indent=2)
print("\n✓ Complete. Paste ALL output back to Claude.")
