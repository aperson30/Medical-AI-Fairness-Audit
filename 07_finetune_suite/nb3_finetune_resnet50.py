# ============================================================
# NOTEBOOK 3 — Fine-Tune ResNet-50 (single T4)
# Dataset: nazmusresan/fitzpatrick17k
# GPU T4 x1, Internet ON
# Expected runtime: ~1.5-2 hours total (smallest of the four)
#
# Mirrors nb_finetune_v3_single_gpu.py exactly, but with ResNet-50 backbone
# (torchvision pretrained on ImageNet). Same bug fix, seeds, protocol.
# ============================================================

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

!pip install torch torchvision scikit-learn pandas numpy imbalanced-learn Pillow -q

import gc
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as tv_models
import torchvision.transforms as transforms
from PIL import Image
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

warnings.filterwarnings('ignore')

n_gpus = torch.cuda.device_count()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device} | GPUs: {n_gpus}")

CFG = dict(
    FITZ_CSV          = '/kaggle/input/datasets/nazmusresan/fitzpatrick17k/New folder/fitzpatrick17k (1).csv',
    IMG_DIR           = '/kaggle/input/datasets/nazmusresan/fitzpatrick17k/New folder/background removed',
    RESULTS_DIR       = 'results_ft_resnet50',
    RAW_PREDS_DIR     = 'results_ft_resnet50/raw_preds',
    CLASS_LABELS      = ['non-neoplastic', 'benign', 'malignant'],
    SEEDS             = [42, 0, 1, 7, 99],
    GDRO_ETA          = 0.01,
    REAL_OVERSAMPLE_N = 200,
    FT_LAST_N_BLOCKS  = 2,           # ResNet has 4 layer groups; unfreeze last 2
    BATCH_SIZE        = 32,           # ResNet-50 is small, big batch OK
    GRAD_ACCUM        = 1,
    EVAL_BATCH_SIZE   = 64,
    EPOCHS            = 5,
    LR_HEAD           = 1e-4,
    LR_BACKBONE       = 1e-5,
    WEIGHT_DECAY      = 0.01,
    DROPOUT           = 0.3,
    MIXED_PRECISION   = True,
    SMOTE_HEAD_EPOCHS = 20,
    N_BOOTSTRAP       = 1000,
)

for d in [CFG['RESULTS_DIR'], CFG['RAW_PREDS_DIR']]:
    os.makedirs(d, exist_ok=True)
print(f"Effective batch: {CFG['BATCH_SIZE'] * CFG['GRAD_ACCUM']}")

def make_autocast():
    return torch.amp.autocast('cuda') if torch.cuda.is_available() else torch.amp.autocast('cpu')
def make_scaler():
    return torch.amp.GradScaler('cuda') if CFG['MIXED_PRECISION'] and torch.cuda.is_available() else None

# ── Load data ─────────────────────────────────────────────────
print('\nLoading metadata...')
df = pd.read_csv(CFG['FITZ_CSV'])
df = df[df['fitzpatrick_scale'].notna() & (df['fitzpatrick_scale'] > 0)]
df = df[df['three_partition_label'].isin(CFG['CLASS_LABELS'])]
image_files = {f.replace('.jpg','').replace('.png',''): os.path.join(CFG['IMG_DIR'], f)
               for f in os.listdir(CFG['IMG_DIR']) if f.endswith('.jpg') or f.endswith('.png')}
df['local_path'] = df['md5hash'].map(image_files)
df = df[df['local_path'].notna()].copy()
class_map = {name: i for i, name in enumerate(CFG['CLASS_LABELS'])}
df['target'] = df['three_partition_label'].map(class_map)
df['fitzpatrick_scale'] = df['fitzpatrick_scale'].astype(int)
light_df = df[df['fitzpatrick_scale'].isin([1, 2])].copy()
dark_df  = df[df['fitzpatrick_scale'].isin([5, 6])].copy()
print(f'Light: {len(light_df)} | Dark: {len(dark_df)} '
      f'(benign: {(dark_df["target"]==1).sum()})')
assert len(dark_df) == 2168 and (dark_df['target'] == 1).sum() == 203

# ── ResNet50 transform pipeline (ImageNet normalization) ──
resnet_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class SkinDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.transform = transform
        self.images, self.labels, self.fitz = [], [], []
        for _, row in dataframe.reset_index(drop=True).iterrows():
            try:
                img = Image.open(row['local_path']).convert('RGB')
                self.images.append(img)
                self.labels.append(int(row['target']))
                self.fitz.append(int(row['fitzpatrick_scale']))
            except Exception: pass
        print(f'  Loaded {len(self.images)} images')
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        return self.transform(self.images[idx]), torch.tensor(self.labels[idx], dtype=torch.long)

class GroupedDataset(Dataset):
    def __init__(self, base_ds, group_labels):
        self.base = base_ds
        self.groups = np.asarray(group_labels)
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        px, lbl = self.base[idx]
        return px, lbl, torch.tensor(self.groups[idx], dtype=torch.long)

# ── Model: ResNet-50 ──────────────────────────────────────────
class ResNet50FineTuned(nn.Module):
    def __init__(self, num_classes=3, dropout=0.3, ft_last_n=2):
        super().__init__()
        backbone = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
        # Replace fc with identity, attach our own classifier
        in_features = backbone.fc.in_features  # 2048
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Freeze everything, then unfreeze the last ft_last_n layer groups
        for p in self.backbone.parameters():
            p.requires_grad = False
        layers_to_unfreeze = []
        if ft_last_n >= 1: layers_to_unfreeze.append(self.backbone.layer4)
        if ft_last_n >= 2: layers_to_unfreeze.append(self.backbone.layer3)
        if ft_last_n >= 3: layers_to_unfreeze.append(self.backbone.layer2)
        if ft_last_n >= 4: layers_to_unfreeze.append(self.backbone.layer1)
        for layer in layers_to_unfreeze:
            for p in layer.parameters():
                p.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, pixel_values):
        feats = self.backbone(pixel_values)
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
        return self.classifier(feats)

    def get_features(self, pixel_values):
        feats = self.backbone(pixel_values)
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
        return feats

def make_param_groups(model):
    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and 'classifier' not in n]
    head_params     = [p for n, p in model.named_parameters()
                       if p.requires_grad and 'classifier' in n]
    return [{'params': backbone_params, 'lr': CFG['LR_BACKBONE']},
            {'params': head_params,     'lr': CFG['LR_HEAD']}]

# ── Helpers (same as nb2) ─────────────────────────────────────
def wilson_ci(k, n, z=1.96):
    if n == 0: return (float('nan'), float('nan'))
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * (p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5 / denom
    return max(0.0, center - margin), min(1.0, center + margin)

def bootstrap_auc_ci(y_true, y_score, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        try: aucs.append(roc_auc_score(y_true[idx], y_score[idx], multi_class='ovr'))
        except ValueError: continue
    return tuple(np.percentile(aucs, [2.5, 97.5])) if aucs else (float('nan'), float('nan'))

def evaluate_full(y_true, y_proba, y_pred):
    classes_present = np.unique(y_true)
    if len(classes_present) < y_proba.shape[1]:
        per_class_aucs = []
        for c in classes_present:
            yb = (y_true == c).astype(int)
            try: per_class_aucs.append(roc_auc_score(yb, y_proba[:, c]))
            except ValueError: pass
        auc = float(np.mean(per_class_aucs)) if per_class_aucs else float('nan')
    else:
        auc = roc_auc_score(y_true, y_proba, multi_class='ovr')
    res = {'auc': auc}
    for c, name in [(0,'non_neo'),(1,'benign'),(2,'malignant')]:
        m = (y_true == c)
        res[f'acc_{name}_dark'] = float((y_pred[m] == c).mean()) if m.sum() > 0 else float('nan')
    return res

@torch.no_grad()
def predict(model, dataloader):
    model.eval()
    all_probs, all_labels = [], []
    for px, lbl in dataloader:
        px = px.to(device)
        if CFG['MIXED_PRECISION']:
            with make_autocast(): logits = model(px)
        else: logits = model(px)
        all_probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        all_labels.append(lbl.numpy())
    return np.vstack(all_probs), np.concatenate(all_labels)

@torch.no_grad()
def extract_features(model, dataloader):
    model.eval()
    all_feats, all_labels = [], []
    for px, lbl in dataloader:
        px = px.to(device)
        if CFG['MIXED_PRECISION']:
            with make_autocast(): feats = model.get_features(px)
        else: feats = model.get_features(px)
        all_feats.append(feats.float().cpu().numpy())
        all_labels.append(lbl.numpy())
    return np.vstack(all_feats), np.concatenate(all_labels)

def _flush(scaler, opt):
    if scaler is not None:
        scaler.step(opt); scaler.update()
    else: opt.step()
    opt.zero_grad()

def train_baseline(model, dataloader, n_epochs):
    optimizer = optim.AdamW(make_param_groups(model), weight_decay=CFG['WEIGHT_DECAY'])
    criterion = nn.CrossEntropyLoss()
    scaler = make_scaler()
    for epoch in range(n_epochs):
        model.train()
        total_loss, n_batches, pending = 0.0, 0, 0
        optimizer.zero_grad()
        for px, lbl in dataloader:
            px, lbl = px.to(device), lbl.to(device)
            if CFG['MIXED_PRECISION']:
                with make_autocast():
                    loss = criterion(model(px), lbl) / CFG['GRAD_ACCUM']
                scaler.scale(loss).backward()
            else:
                loss = criterion(model(px), lbl) / CFG['GRAD_ACCUM']
                loss.backward()
            pending += 1
            if pending == CFG['GRAD_ACCUM']:
                _flush(scaler, optimizer); pending = 0
            total_loss += loss.item() * CFG['GRAD_ACCUM']
            n_batches += 1
        if pending > 0: _flush(scaler, optimizer)
        print(f'    Epoch {epoch+1}/{n_epochs}  loss={total_loss/n_batches:.4f}')

def train_group_dro(model, dataset, group_labels, n_epochs, n_groups, batch_size, seed):
    print(f'    GDRO group counts: {np.bincount(group_labels, minlength=n_groups).tolist()}')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=2, pin_memory=True)
    optimizer = optim.AdamW(make_param_groups(model), weight_decay=CFG['WEIGHT_DECAY'])
    criterion = nn.CrossEntropyLoss(reduction='none')
    scaler = make_scaler()
    group_weights = torch.ones(n_groups, device=device) / n_groups
    weight_log = []
    for epoch in range(n_epochs):
        model.train()
        total_loss, n_batches, pending = 0.0, 0, 0
        optimizer.zero_grad()
        for px, lbl, grp in dataloader:
            px, lbl, grp = px.to(device), lbl.to(device), grp.to(device)
            if CFG['MIXED_PRECISION']:
                with make_autocast():
                    logits = model(px)
                    losses = criterion(logits, lbl)
                    group_losses = torch.zeros(n_groups, device=device)
                    for g in range(n_groups):
                        m = (grp == g)
                        if m.sum() > 0: group_losses[g] = losses[m].mean()
                gl = group_losses.detach().float()
                group_weights = group_weights * torch.exp(CFG['GDRO_ETA'] * gl)
                group_weights = group_weights / group_weights.sum()
                if not torch.isfinite(group_weights).all():
                    group_weights = torch.ones(n_groups, device=device) / n_groups
                weighted_loss = (group_weights * group_losses.float()).sum() / CFG['GRAD_ACCUM']
                scaler.scale(weighted_loss).backward()
            else:
                logits = model(px)
                losses = criterion(logits, lbl)
                group_losses = torch.zeros(n_groups, device=device)
                for g in range(n_groups):
                    m = (grp == g)
                    if m.sum() > 0: group_losses[g] = losses[m].mean()
                group_weights = group_weights * torch.exp(CFG['GDRO_ETA'] * group_losses.detach())
                group_weights = group_weights / group_weights.sum()
                if not torch.isfinite(group_weights).all():
                    group_weights = torch.ones(n_groups, device=device) / n_groups
                weighted_loss = (group_weights * group_losses).sum() / CFG['GRAD_ACCUM']
                weighted_loss.backward()
            pending += 1
            if pending == CFG['GRAD_ACCUM']:
                _flush(scaler, optimizer); pending = 0
            total_loss += weighted_loss.item() * CFG['GRAD_ACCUM']
            n_batches += 1
        if pending > 0: _flush(scaler, optimizer)
        weight_log.append(group_weights.cpu().numpy().tolist())
        print(f'    Epoch {epoch+1}/{n_epochs}  loss={total_loss/n_batches:.4f}  '
              f'group_w=[{",".join(f"{w:.3f}" for w in group_weights.cpu().numpy())}]')
    return weight_log

# ── Build datasets ────────────────────────────────────────────
print('\nBuilding datasets...')
light_ds = SkinDataset(light_df, resnet_transform)
dark_ds  = SkinDataset(dark_df, resnet_transform)

def make_pool_split(seed, n_pool=CFG['REAL_OVERSAMPLE_N']):
    rng = np.random.default_rng(seed)
    dark_labels = np.array(dark_ds.labels)
    pool_idx = []
    for cls in [0, 1, 2]:
        cls_idx = np.where(dark_labels == cls)[0]
        n_take = min(int(round(n_pool * len(cls_idx) / len(dark_labels))), len(cls_idx))
        pool_idx.extend(rng.choice(cls_idx, n_take, replace=False).tolist())
    pool_idx = np.array(sorted(pool_idx))
    test_idx = np.array([i for i in range(len(dark_ds)) if i not in set(pool_idx)])
    return pool_idx, test_idx

# ══════════════════════════════════════════════════════════════
INTERVENTIONS = ['baseline', 'group_dro', 'smote']
all_results = []
all_weight_logs = {}
t_start = time.time()
dl_batch = CFG['BATCH_SIZE']
eval_dl = CFG['EVAL_BATCH_SIZE']

full_dark_loader = DataLoader(dark_ds, batch_size=eval_dl, shuffle=False,
                              num_workers=2, pin_memory=True)

for seed in CFG['SEEDS']:
    print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)

    pool_idx, test_idx = make_pool_split(seed)
    test_subset = Subset(dark_ds, test_idx.tolist())
    pool_subset = Subset(dark_ds, pool_idx.tolist())
    intv_test_loader = DataLoader(test_subset, batch_size=eval_dl, shuffle=False,
                                  num_workers=2, pin_memory=True)

    for intervention in INTERVENTIONS:
        print(f'\n── {intervention.upper()} (seed {seed}) ──')
        t0 = time.time()
        model = ResNet50FineTuned(num_classes=3, dropout=CFG['DROPOUT'],
                                  ft_last_n=CFG['FT_LAST_N_BLOCKS']).to(device)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'  Trainable params: {n_train/1e6:.1f}M')

        try:
            if intervention == 'baseline':
                train_dl = DataLoader(light_ds, batch_size=dl_batch, shuffle=True,
                                      num_workers=2, pin_memory=True)
                train_baseline(model, train_dl, CFG['EPOCHS'])
                eval_loader = full_dark_loader

            elif intervention == 'group_dro':
                combined = ConcatDataset([light_ds, pool_subset])
                group_labels = np.array([0]*len(light_ds) + [1]*len(pool_subset))
                grouped = GroupedDataset(combined, group_labels)
                weight_log = train_group_dro(model, grouped, group_labels,
                                             CFG['EPOCHS'], 2, dl_batch, seed)
                all_weight_logs[seed] = weight_log
                eval_loader = intv_test_loader

            elif intervention == 'smote':
                train_dl = DataLoader(light_ds, batch_size=dl_batch, shuffle=True,
                                      num_workers=2, pin_memory=True)
                train_baseline(model, train_dl, CFG['EPOCHS'])
                print('    Extracting features...')
                light_feats, light_lbls = extract_features(model,
                    DataLoader(light_ds, batch_size=eval_dl, shuffle=False,
                               num_workers=2, pin_memory=True))
                dark_feats, dark_lbls = extract_features(model,
                    DataLoader(pool_subset, batch_size=eval_dl, shuffle=False,
                               num_workers=2, pin_memory=True))
                combo_feats = np.vstack([light_feats, dark_feats])
                combo_lbls = np.concatenate([light_lbls, dark_lbls])
                print(f'    Pre-SMOTE: {np.bincount(combo_lbls)}')
                try:
                    k = max(1, min(5, int(np.bincount(combo_lbls).min()) - 1))
                    sf, sy = SMOTE(random_state=seed, k_neighbors=k).fit_resample(combo_feats, combo_lbls)
                    norms = np.linalg.norm(sf, axis=1, keepdims=True)
                    sf = sf / np.maximum(norms, 1e-8)
                    print(f'    Post-SMOTE: {np.bincount(sy)}')
                except Exception as e:
                    print(f'    SMOTE failed ({e})')
                    sf, sy = combo_feats, combo_lbls

                for p in model.backbone.parameters(): p.requires_grad = False
                X = torch.tensor(sf, dtype=torch.float32).to(device)
                Y = torch.tensor(sy, dtype=torch.long).to(device)
                head_opt = optim.AdamW(model.classifier.parameters(),
                                       lr=CFG['LR_HEAD'], weight_decay=CFG['WEIGHT_DECAY'])
                criterion = nn.CrossEntropyLoss()
                model.classifier.train()
                for ep in range(CFG['SMOTE_HEAD_EPOCHS']):
                    perm = torch.randperm(len(X))
                    ep_loss = 0.0
                    for i in range(0, len(X), 64):
                        idx = perm[i:i+64]
                        head_opt.zero_grad()
                        loss = criterion(model.classifier(X[idx]), Y[idx])
                        loss.backward(); head_opt.step()
                        ep_loss += loss.item()
                    if (ep+1) % 5 == 0:
                        print(f'      Head epoch {ep+1}/{CFG["SMOTE_HEAD_EPOCHS"]}  loss={ep_loss:.3f}')
                del X, Y, sf, sy, combo_feats, combo_lbls, light_feats, light_lbls, dark_feats, dark_lbls
                gc.collect(); torch.cuda.empty_cache()
                eval_loader = intv_test_loader

            print('  Evaluating...')
            proba, labels = predict(model, eval_loader)
            preds = proba.argmax(axis=1)
            res = evaluate_full(labels, proba, preds)
            n_b = int((labels == 1).sum())
            k_b = int(((preds == 1) & (labels == 1)).sum())
            w_lo, w_hi = wilson_ci(k_b, n_b)
            b_lo, b_hi = bootstrap_auc_ci(labels, proba, CFG['N_BOOTSTRAP'], seed)
            cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
            print(f'  demo_auc={res["auc"]:.4f}  benign={res["acc_benign_dark"]:.4f} '
                  f'({k_b}/{n_b})  non_neo={res["acc_non_neo_dark"]:.4f}  '
                  f'malig={res["acc_malignant_dark"]:.4f}')
            print(f'  CM: non-neo {cm[0]} | benign {cm[1]} | malig {cm[2]}')

            raw_dir = os.path.join(CFG['RAW_PREDS_DIR'], intervention, f'seed{seed}')
            os.makedirs(raw_dir, exist_ok=True)
            np.save(f'{raw_dir}/y_true.npy', labels)
            np.save(f'{raw_dir}/y_pred.npy', preds)
            np.save(f'{raw_dir}/y_proba.npy', proba)

            all_results.append({
                'seed': seed, 'intervention': intervention,
                'demo_auc': res['auc'], 'demo_ci_lo': b_lo, 'demo_ci_hi': b_hi,
                'acc_non_neo_dark': res['acc_non_neo_dark'],
                'acc_benign_dark': res['acc_benign_dark'],
                'acc_malignant_dark': res['acc_malignant_dark'],
                'benign_wilson_lo': w_lo, 'benign_wilson_hi': w_hi,
                'n_dark_benign': n_b, 'n_dark_total': len(labels),
            })
            pd.DataFrame(all_results).to_csv(
                os.path.join(CFG['RESULTS_DIR'], 'ft_resnet50_results.csv'), index=False)
            print(f'  ✓ Done in {(time.time()-t0)/60:.1f} min '
                  f'(total: {(time.time()-t_start)/60:.1f} min)')
        except Exception as e:
            print(f'  ✗ FAILED: {e}')
            import traceback; traceback.print_exc()
        finally:
            del model
            gc.collect(); torch.cuda.empty_cache()

import json
with open(os.path.join(CFG['RESULTS_DIR'], 'gdro_weight_trajectories.json'), 'w') as f:
    json.dump({f'seed{k}': v for k, v in all_weight_logs.items()}, f, indent=2)

print(f"\n{'='*60}\nSUMMARY (ResNet-50 fine-tune)\n{'='*60}")
df_results = pd.DataFrame(all_results)
df_results.to_csv(os.path.join(CFG['RESULTS_DIR'], 'ft_resnet50_results.csv'), index=False)
summary = df_results.groupby('intervention').agg(
    demo_auc_mean=('demo_auc','mean'), demo_auc_std=('demo_auc','std'),
    benign_acc_mean=('acc_benign_dark','mean'), benign_acc_std=('acc_benign_dark','std'),
    malig_acc_mean=('acc_malignant_dark','mean'), malig_acc_std=('acc_malignant_dark','std'),
    non_neo_acc_mean=('acc_non_neo_dark','mean'), non_neo_acc_std=('acc_non_neo_dark','std'),
).round(4)
print(summary.to_string())

total = time.time() - t_start
print(f'\n✓ ALL DONE in {total/60:.1f} min ({total/3600:.2f} h)')
print('\nPaste this entire output back to Claude.')
