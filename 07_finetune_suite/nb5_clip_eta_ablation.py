# ============================================================
# NOTEBOOK 5 — CLIP Fine-Tune Group DRO η-Ablation (single T4)
# Dataset: nazmusresan/fitzpatrick17k
# GPU T4 x1, Internet ON
# Expected runtime: ~3-4 hours (4 etas × 2 seeds × ~12 min/run)
#
# PURPOSE
# -------
# Test whether ANY value of GDRO_ETA recovers benign accuracy on dark skin
# under CLIP ViT-L/14 fine-tuning. Your main fine-tune run already showed
# eta=0.01 (Sagawa et al. recommended) fails — DRO benign acc ≈ baseline.
#
# This notebook sweeps eta ∈ {0.001, 0.01, 0.1, 1.0} on 2 seeds each.
# Two possible findings, both publishable:
#
#   - NO eta works → DRO is unrescuable on this task structure regardless
#                    of hyperparameter. Strongest possible negative claim.
#
#   - SOME eta (likely 0.1 or 1.0) works → DRO is salvageable but ONLY at
#                    hyperparameters far outside the recommended range, and
#                    the standard practice (eta=0.01) fails.
#
# Both turn into solid systematic-ablation contributions for the new paper.
#
# Only DRO is run here (no baseline, no SMOTE) since those don't depend on
# eta. Use the existing nb_finetune_v3 baseline numbers for comparison.
# ============================================================

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

!pip install transformers torch torchvision scikit-learn pandas numpy Pillow -q

import gc
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from transformers import CLIPModel, CLIPProcessor

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

CFG = dict(
    FITZ_CSV          = '/kaggle/input/datasets/nazmusresan/fitzpatrick17k/New folder/fitzpatrick17k (1).csv',
    IMG_DIR           = '/kaggle/input/datasets/nazmusresan/fitzpatrick17k/New folder/background removed',
    RESULTS_DIR       = 'results_eta_ablation',
    CLASS_LABELS      = ['non-neoplastic', 'benign', 'malignant'],

    # Sweep
    SEEDS             = [42, 0],            # 2 seeds per eta to manage compute
    GDRO_ETAS         = [0.001, 0.01, 0.1, 1.0],
    REAL_OVERSAMPLE_N = 200,

    # Same fine-tuning hyperparameters as nb_finetune_v3
    FT_LAST_N_BLOCKS  = 4,
    BATCH_SIZE        = 8,
    GRAD_ACCUM        = 4,
    EVAL_BATCH_SIZE   = 16,
    EPOCHS            = 5,
    LR_HEAD           = 1e-4,
    LR_BACKBONE       = 1e-5,
    WEIGHT_DECAY      = 0.01,
    DROPOUT           = 0.3,
    MIXED_PRECISION   = True,
    N_BOOTSTRAP       = 1000,
)

os.makedirs(CFG['RESULTS_DIR'], exist_ok=True)

def make_autocast():
    return torch.amp.autocast('cuda') if torch.cuda.is_available() else torch.amp.autocast('cpu')
def make_scaler():
    return torch.amp.GradScaler('cuda') if CFG['MIXED_PRECISION'] and torch.cuda.is_available() else None

# ── Load data ─────────────────────────────────────────────────
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

class SkinDataset(Dataset):
    def __init__(self, dataframe, processor):
        self.processor = processor
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
        inputs = self.processor(images=self.images[idx], return_tensors='pt')
        return inputs['pixel_values'].squeeze(0), torch.tensor(self.labels[idx], dtype=torch.long)

class GroupedDataset(Dataset):
    def __init__(self, base_ds, group_labels):
        self.base = base_ds
        self.groups = np.asarray(group_labels)
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        px, lbl = self.base[idx]
        return px, lbl, torch.tensor(self.groups[idx], dtype=torch.long)

class CLIPFineTuned(nn.Module):
    def __init__(self, clip_model, num_classes=3, dropout=0.3, ft_last_n=4):
        super().__init__()
        self.vision_model = clip_model.vision_model
        self.visual_projection = clip_model.visual_projection
        hidden_size = clip_model.config.projection_dim
        for p in self.vision_model.parameters(): p.requires_grad = False
        for p in self.visual_projection.parameters(): p.requires_grad = True
        n_layers = len(self.vision_model.encoder.layers)
        for i in range(n_layers - ft_last_n, n_layers):
            for p in self.vision_model.encoder.layers[i].parameters():
                p.requires_grad = True
        for p in self.vision_model.post_layernorm.parameters():
            p.requires_grad = True
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_size, 256),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, num_classes),
        )
    def forward(self, pixel_values):
        out = self.vision_model(pixel_values=pixel_values)
        feats = self.visual_projection(out.pooler_output)
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
        return self.classifier(feats)

def make_param_groups(model):
    backbone = [p for n, p in model.named_parameters()
                if p.requires_grad and ('vision_model' in n or 'visual_projection' in n)]
    head     = [p for n, p in model.named_parameters()
                if p.requires_grad and 'classifier' in n]
    return [{'params': backbone, 'lr': CFG['LR_BACKBONE']},
            {'params': head,     'lr': CFG['LR_HEAD']}]

# ── Helpers ───────────────────────────────────────────────────
def wilson_ci(k, n, z=1.96):
    if n == 0: return (float('nan'), float('nan'))
    p = k/n; denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    margin = z * (p*(1-p)/n + z**2/(4*n**2))**0.5 / denom
    return max(0.0, center-margin), min(1.0, center+margin)

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
    probs, lbls = [], []
    for px, lbl in dataloader:
        px = px.to(device)
        if CFG['MIXED_PRECISION']:
            with make_autocast(): logits = model(px)
        else: logits = model(px)
        probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        lbls.append(lbl.numpy())
    return np.vstack(probs), np.concatenate(lbls)

def _flush(scaler, opt):
    if scaler is not None:
        scaler.step(opt); scaler.update()
    else: opt.step()
    opt.zero_grad()

def train_group_dro(model, dataset, group_labels, n_epochs, n_groups,
                    batch_size, seed, eta):
    """Same as nb_finetune_v3, but takes eta as a parameter."""
    print(f'    GDRO eta={eta}  group counts: {np.bincount(group_labels, minlength=n_groups).tolist()}')
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
                group_weights = group_weights * torch.exp(eta * gl)
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
                group_weights = group_weights * torch.exp(eta * group_losses.detach())
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
print('\nLoading CLIP processor...')
processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
print('Building datasets...')
light_ds = SkinDataset(light_df, processor)
dark_ds  = SkinDataset(dark_df, processor)

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
# MAIN SWEEP
# ══════════════════════════════════════════════════════════════
all_results = []
all_weight_logs = {}
t_start = time.time()
dl_batch = CFG['BATCH_SIZE']
eval_dl = CFG['EVAL_BATCH_SIZE']

print(f"\nSweeping eta ∈ {CFG['GDRO_ETAS']} × seeds ∈ {CFG['SEEDS']} "
      f"= {len(CFG['GDRO_ETAS']) * len(CFG['SEEDS'])} runs")

for eta in CFG['GDRO_ETAS']:
    print(f"\n{'='*60}\nETA = {eta}\n{'='*60}")

    for seed in CFG['SEEDS']:
        print(f"\n── seed {seed}, eta {eta} ──")
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
        t0 = time.time()

        pool_idx, test_idx = make_pool_split(seed)
        test_subset = Subset(dark_ds, test_idx.tolist())
        pool_subset = Subset(dark_ds, pool_idx.tolist())
        intv_test_loader = DataLoader(test_subset, batch_size=eval_dl, shuffle=False,
                                      num_workers=2, pin_memory=True)

        clip_base = CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
        model = CLIPFineTuned(clip_base, num_classes=3,
                              dropout=CFG['DROPOUT'],
                              ft_last_n=CFG['FT_LAST_N_BLOCKS']).to(device)

        try:
            combined = ConcatDataset([light_ds, pool_subset])
            group_labels = np.array([0]*len(light_ds) + [1]*len(pool_subset))
            grouped = GroupedDataset(combined, group_labels)
            weight_log = train_group_dro(model, grouped, group_labels,
                                          CFG['EPOCHS'], 2, dl_batch, seed, eta)
            all_weight_logs[(eta, seed)] = weight_log

            print('  Evaluating...')
            proba, labels = predict(model, intv_test_loader)
            preds = proba.argmax(axis=1)
            res = evaluate_full(labels, proba, preds)
            n_b = int((labels == 1).sum())
            k_b = int(((preds == 1) & (labels == 1)).sum())
            w_lo, w_hi = wilson_ci(k_b, n_b)
            b_lo, b_hi = bootstrap_auc_ci(labels, proba, CFG['N_BOOTSTRAP'], seed)
            print(f'  eta={eta} seed={seed}: '
                  f'demo_auc={res["auc"]:.4f}  '
                  f'benign={res["acc_benign_dark"]:.4f} ({k_b}/{n_b})  '
                  f'non_neo={res["acc_non_neo_dark"]:.4f}  '
                  f'malig={res["acc_malignant_dark"]:.4f}')

            all_results.append({
                'eta': eta, 'seed': seed,
                'demo_auc': res['auc'], 'demo_ci_lo': b_lo, 'demo_ci_hi': b_hi,
                'acc_non_neo_dark': res['acc_non_neo_dark'],
                'acc_benign_dark': res['acc_benign_dark'],
                'acc_malignant_dark': res['acc_malignant_dark'],
                'benign_wilson_lo': w_lo, 'benign_wilson_hi': w_hi,
                'n_dark_benign': n_b, 'n_dark_total': len(labels),
                'final_dark_weight': weight_log[-1][1] if len(weight_log[-1]) > 1 else None,
            })
            pd.DataFrame(all_results).to_csv(
                os.path.join(CFG['RESULTS_DIR'], 'eta_ablation_results.csv'), index=False)
            print(f'  ✓ Done in {(time.time()-t0)/60:.1f} min '
                  f'(total: {(time.time()-t_start)/60:.1f} min)')
        except Exception as e:
            print(f'  ✗ FAILED: {e}')
            import traceback; traceback.print_exc()
        finally:
            del model, clip_base
            gc.collect(); torch.cuda.empty_cache()

# ── Save weight trajectories ──────────────────────────────────
import json
serializable = {f'eta{eta}_seed{seed}': v for (eta, seed), v in all_weight_logs.items()}
with open(os.path.join(CFG['RESULTS_DIR'], 'eta_weight_trajectories.json'), 'w') as f:
    json.dump(serializable, f, indent=2)

print(f"\n{'='*60}\nETA SWEEP SUMMARY\n{'='*60}")
df_results = pd.DataFrame(all_results)
df_results.to_csv(os.path.join(CFG['RESULTS_DIR'], 'eta_ablation_results.csv'), index=False)
summary = df_results.groupby('eta').agg(
    benign_acc_mean=('acc_benign_dark','mean'), benign_acc_std=('acc_benign_dark','std'),
    demo_auc_mean=('demo_auc','mean'), demo_auc_std=('demo_auc','std'),
    final_dark_weight_mean=('final_dark_weight','mean'),
).round(4)
print(summary.to_string())

print('\n── Interpretation ──')
print('Reference: nb_finetune_v3 baseline (eta-independent) benign ≈ 0.326 ± 0.050')
print()
print('If ALL etas have benign ≈ baseline (~0.30) → DRO is unrescuable. Strongest claim.')
print('If SOME eta (likely 0.1 or 1.0) has benign noticeably > baseline → DRO is salvageable')
print('   but only outside the recommended range. Reframe as "default settings fail".')
print('If final_dark_weight ≈ 0 across all etas → weight collapse is universal.')
print('If final_dark_weight ≠ 0 for large etas → those etas keep dark in the loss.')

total = time.time() - t_start
print(f'\n✓ ALL DONE in {total/60:.1f} min ({total/3600:.2f} h)')
print('\nPaste this entire output back to Claude.')
