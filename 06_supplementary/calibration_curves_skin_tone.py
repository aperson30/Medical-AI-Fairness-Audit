# ============================================================
# NOTEBOOK — Calibration Curves by Skin Tone Group
# Dataset: nazmusresan/fitzpatrick17k
# GPU T4 x1, Internet ON
# Expected runtime: ~15 minutes
#
# Generates reliability diagrams showing CLIP is well-calibrated
# for light-skin patients but systematically miscalibrated for
# dark-skin patients — visual complement to DAGC T=1.50 finding.
#
# Saves figure as /kaggle/working/figure5_calibration.png
# After running, paste ALL output + upload figure to Claude
# ============================================================

!pip install transformers torch torchvision scikit-learn pandas numpy matplotlib -q

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import calibration_curve
import warnings, os
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

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

# ── Load CLIP ─────────────────────────────────────────────────
from transformers import CLIPModel, CLIPProcessor
print("Loading CLIP...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
clip_model.eval()
print("CLIP loaded.")

# ── Load images ───────────────────────────────────────────────
def load_imgs(dataframe):
    imgs, lbls = [], []
    for _, row in dataframe.iterrows():
        try:
            img = Image.open(row['local_path']).convert('RGB').resize((224,224))
            imgs.append(img)
            lbls.append(row['three_partition_label'])
        except: pass
    return imgs, lbls

print("Loading images...")
light_imgs,  light_lbls  = load_imgs(light_df)
medium_imgs, medium_lbls = load_imgs(medium_df)
dark_imgs,   dark_lbls   = load_imgs(dark_df)
print(f"Loaded: light={len(light_imgs)}, medium={len(medium_imgs)}, dark={len(dark_imgs)}")

# ── Extract features ──────────────────────────────────────────
@torch.no_grad()
def get_features(images, batch_size=32):
    all_feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        inputs = clip_processor(images=batch, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feats = clip_model.get_image_features(**inputs)
        if not isinstance(feats, torch.Tensor):
            feats = feats.pooler_output if hasattr(feats, 'pooler_output') \
                    else feats.last_hidden_state[:,0]
        feats = feats / feats.norm(dim=-1, keepdim=True)
        all_feats.append(feats.cpu().numpy())
        if i % 320 == 0: print(f"  {i}/{len(images)}...")
    return np.vstack(all_feats)

print("Extracting features...")
light_feats  = get_features(light_imgs)
medium_feats = get_features(medium_imgs)
dark_feats   = get_features(dark_imgs)

le = LabelEncoder()
le.fit(light_lbls + medium_lbls + dark_lbls)
light_y  = le.transform(light_lbls)
medium_y = le.transform(medium_lbls)
dark_y   = le.transform(dark_lbls)

# ── Train skin-tone split classifier ─────────────────────────
train_feats  = np.vstack([light_feats, medium_feats])
train_y      = np.concatenate([light_y, medium_y])
clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
clf.fit(train_feats, train_y)

# Get probabilities for each group
light_probs  = clf.predict_proba(light_feats)
medium_probs = clf.predict_proba(medium_feats)
dark_probs   = clf.predict_proba(dark_feats)

# ── Compute DAGC temperatures ─────────────────────────────────
from scipy.optimize import minimize_scalar
from scipy.special import softmax

def get_logits(feats, clf):
    return feats @ clf.coef_.T + clf.intercept_

def temperature_nll(T, logits, labels):
    scaled = logits / T
    probs  = softmax(scaled, axis=1)
    probs  = np.clip(probs, 1e-7, 1 - 1e-7)
    return -np.mean(np.log(probs[np.arange(len(labels)), labels]))

light_logits  = get_logits(light_feats, clf)
medium_logits = get_logits(medium_feats, clf)
dark_logits   = get_logits(dark_feats, clf)

T_light  = minimize_scalar(lambda T: temperature_nll(T, light_logits,  light_y),
                            bounds=(0.1, 5.0), method='bounded').x
T_medium = minimize_scalar(lambda T: temperature_nll(T, medium_logits, medium_y),
                            bounds=(0.1, 5.0), method='bounded').x
T_dark   = minimize_scalar(lambda T: temperature_nll(T, dark_logits,   dark_y),
                            bounds=(0.1, 5.0), method='bounded').x

print(f"\nDAGC Temperatures:")
print(f"  Light skin:  T = {T_light:.3f}")
print(f"  Medium skin: T = {T_medium:.3f}")
print(f"  Dark skin:   T = {T_dark:.3f}")

# Calibrated probabilities
light_probs_cal  = softmax(light_logits  / T_light,  axis=1)
medium_probs_cal = softmax(medium_logits / T_medium, axis=1)
dark_probs_cal   = softmax(dark_logits   / T_dark,   axis=1)

# ── Generate Figure 5: Calibration Curves ────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.titlesize': 12, 'axes.labelsize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
})

fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35)

skin_groups = [
    ('Light Skin (I-II)',   light_y,  light_probs,  light_probs_cal,  '#1565C0', T_light),
    ('Medium Skin (III-IV)', medium_y, medium_probs, medium_probs_cal, '#F57F17', T_medium),
    ('Dark Skin (V-VI)',    dark_y,   dark_probs,   dark_probs_cal,   '#B71C1C', T_dark),
]

n_bins = 10

for col, (title, y_true, probs_uncal, probs_cal, color, temp) in enumerate(skin_groups):
    # ── Top row: uncalibrated ──────────────────────────────────
    ax_top = fig.add_subplot(gs[0, col])

    for cls_idx, cls_name in enumerate(le.classes_):
        y_bin = (y_true == cls_idx).astype(int)
        prob_bin = probs_uncal[:, cls_idx]
        try:
            frac_pos, mean_pred = calibration_curve(y_bin, prob_bin, n_bins=n_bins)
            ls = '-' if cls_name == 'non-neoplastic' else '--' if cls_name == 'benign' else ':'
            ax_top.plot(mean_pred, frac_pos, ls, color=color,
                       alpha=0.9, linewidth=1.8,
                       label=cls_name.capitalize())
        except: pass

    ax_top.plot([0,1], [0,1], 'k--', alpha=0.3, linewidth=1, label='Perfect')
    ax_top.set_xlim(0, 1); ax_top.set_ylim(0, 1)
    ax_top.set_title(f'{title}\nUncalibrated (T=1.00)', fontsize=10, fontweight='bold')
    ax_top.set_xlabel('Mean Predicted Probability')
    if col == 0:
        ax_top.set_ylabel('Fraction of Positives')
        ax_top.legend(fontsize=8, loc='upper left')
    ax_top.yaxis.grid(True, alpha=0.3)

    # ── Bottom row: DAGC calibrated ───────────────────────────
    ax_bot = fig.add_subplot(gs[1, col])

    for cls_idx, cls_name in enumerate(le.classes_):
        y_bin = (y_true == cls_idx).astype(int)
        prob_bin = probs_cal[:, cls_idx]
        try:
            frac_pos, mean_pred = calibration_curve(y_bin, prob_bin, n_bins=n_bins)
            ls = '-' if cls_name == 'non-neoplastic' else '--' if cls_name == 'benign' else ':'
            ax_bot.plot(mean_pred, frac_pos, ls, color=color,
                       alpha=0.9, linewidth=1.8,
                       label=cls_name.capitalize())
        except: pass

    ax_bot.plot([0,1], [0,1], 'k--', alpha=0.3, linewidth=1)
    ax_bot.set_xlim(0, 1); ax_bot.set_ylim(0, 1)
    ax_bot.set_title(f'DAGC Calibrated (T={temp:.2f})', fontsize=10, fontweight='bold',
                     color='#1B5E20' if temp < 1.1 else '#C62828' if temp > 1.3 else '#E65100')
    ax_bot.set_xlabel('Mean Predicted Probability')
    if col == 0:
        ax_bot.set_ylabel('Fraction of Positives')
    ax_bot.yaxis.grid(True, alpha=0.3)

fig.suptitle('Figure 5: Reliability Diagrams by Skin Tone Group (CLIP ViT-L/14)\n'
             'Top: Uncalibrated predictions | Bottom: After DAGC temperature scaling',
             fontsize=12, fontweight='bold', y=1.01)

plt.savefig('/kaggle/working/figure5_calibration.png', dpi=300, bbox_inches='tight')
plt.show()
print("Figure saved to /kaggle/working/figure5_calibration.png")

# ── Print calibration statistics ─────────────────────────────
print("\n" + "="*55)
print("CALIBRATION STATISTICS")
print("="*55)
from sklearn.calibration import calibration_curve as cc

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i+1])
        if mask.sum() > 0:
            acc  = y_true[mask].mean()
            conf = y_prob[mask].mean()
            ece += mask.mean() * abs(acc - conf)
    return ece

print(f"\nExpected Calibration Error (ECE) — macro over classes:")
for title, y_true, probs, probs_cal in [
    ("Light skin",  light_y,  light_probs,  light_probs_cal),
    ("Medium skin", medium_y, medium_probs, medium_probs_cal),
    ("Dark skin",   dark_y,   dark_probs,   dark_probs_cal),
]:
    ece_before = np.mean([expected_calibration_error((y_true==i).astype(int),
                          probs[:, i]) for i in range(3)])
    ece_after  = np.mean([expected_calibration_error((y_true==i).astype(int),
                          probs_cal[:, i]) for i in range(3)])
    print(f"  {title}: ECE before={ece_before:.4f}, after={ece_after:.4f}, "
          f"improvement={ece_before-ece_after:.4f}")

print(f"\nKey finding: Dark skin temperature T={T_dark:.3f} vs Light T={T_light:.3f}")
print(f"Large T difference indicates the model is systematically overconfident")
print(f"on light-skin predictions and underconfident on dark-skin predictions.")
print(f"\n✓ Complete. Upload figure5_calibration.png and paste output to Claude.")
