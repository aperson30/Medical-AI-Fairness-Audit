# ============================================================
# NEW NOTEBOOK — PRIORITY 2
# Complete Architecture × Dataset Matrix (finish nb5)
# Dataset: nih-chest-xrays
# GPU T4, Internet ON. ~20 min.
#
# WHY: nb5 was never fully run. This completes the
# ViT-B/16 NIH sex split result, finishing the full
# architecture × dataset matrix. Low-hanging fruit:
# the notebook exists, just needs a clean full run.
#
# WHAT'S NEW vs nb5:
#   - Focused only on NIH sex split (drop ISIC, already done)
#   - Adds ResNet50 to the sex split too so all 3 architectures
#     are reported (CLIP + ResNet50 already in paper, just ViT missing)
#   - Adds bootstrap CIs (1000 samples, matching all other tables)
#   - Clean summary output matching paper table format
#
# Expected results (from handoff doc):
#   CLIP SGG=0.012 (not sig), ResNet50 SGG=0.008 (not sig)
#   ViT-B/16: unknown — this run will complete it
#
# Kaggle setup: GPU T4 x1, Internet ON, random_state=42
# Kaggle dataset needed: organizations/nih-chest-xrays
# ============================================================

!pip install transformers torch torchvision scikit-learn pandas numpy -q

import torch
import numpy as np, pandas as pd, os, json, warnings
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from transformers import (
    ViTModel, ViTImageProcessor,
    CLIPModel, CLIPProcessor,
    ResNetModel, AutoFeatureExtractor,
)
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
N_BOOTSTRAP  = 1000
np.random.seed(RANDOM_STATE)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── Dataset path ──────────────────────────────────────────────
nih_path = '/kaggle/input/datasets/organizations/nih-chest-xrays/data'

# ── Load NIH metadata ─────────────────────────────────────────
df_nih = pd.read_csv(os.path.join(nih_path, 'Data_Entry_2017.csv'))
df_nih = df_nih[(df_nih['Patient Age'] > 0) & (df_nih['Patient Age'] < 120)]
df_nih['has_finding'] = (df_nih['Finding Labels'] != 'No Finding').astype(int)
df_nih['age_group']   = pd.cut(df_nih['Patient Age'],
    bins=[0, 40, 60, 120], labels=['young', 'middle', 'older'])

# Build image index
print("Building NIH image index...")
img_index = {}
for d in os.listdir(nih_path):
    if d.startswith('images_'):
        img_dir = os.path.join(nih_path, d, 'images')
        if os.path.exists(img_dir):
            for f in os.listdir(img_dir):
                img_index[f] = os.path.join(img_dir, f)
print(f"Indexed {len(img_index):,} images")

df_nih['image_path'] = df_nih['Image Index'].map(img_index)
df_nih = df_nih[df_nih['image_path'].notna()].copy()

# Print prevalence by sex
print("\nPathology prevalence by sex:")
for sex in ['M', 'F']:
    sub = df_nih[df_nih['Patient Gender'] == sex]
    print(f"  {sex}: {sub['has_finding'].mean():.3f} (n={len(sub):,})")

# ── Sample equal groups ───────────────────────────────────────
N = 1000
male_df   = df_nih[df_nih['Patient Gender'] == 'M'].sample(N, random_state=RANDOM_STATE)
female_df = df_nih[df_nih['Patient Gender'] == 'F'].sample(N, random_state=RANDOM_STATE)

def load_nih(dataframe):
    imgs, lbls = [], []
    for _, row in dataframe.iterrows():
        try:
            img = Image.open(row['image_path']).convert('RGB').resize((224, 224))
            imgs.append(img)
            lbls.append(int(row['has_finding']))
        except:
            pass
    return imgs, np.array(lbls, dtype=np.int64)

print("\nLoading NIH images...")
male_imgs,   male_y   = load_nih(male_df)
female_imgs, female_y = load_nih(female_df)
print(f"Male: {len(male_imgs)}, Female: {len(female_imgs)}")
print(f"Male prevalence: {male_y.mean():.3f}, Female: {female_y.mean():.3f}")

# ── Feature extractor ─────────────────────────────────────────
def evaluate_sgg(model_name, train_f, train_y, test_f, test_y,
                 all_f, all_y, label=''):
    """
    Compute random split AUC and demographic-aware AUC.
    Returns dict with both AUC values, SGG, CIs.
    """
    # Random split
    tr, te = train_test_split(np.arange(len(all_y)), test_size=0.25,
                               stratify=all_y, random_state=RANDOM_STATE)
    clf_r = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
    clf_r.fit(all_f[tr], all_y[tr])
    rand_probs = clf_r.predict_proba(all_f[te])
    rand_auc   = roc_auc_score(all_y[te], rand_probs[:, 1])

    # Bootstrap CI for random AUC
    rand_scores = []
    for _ in range(N_BOOTSTRAP):
        idx = np.random.choice(len(te), len(te), replace=True)
        try:
            rand_scores.append(roc_auc_score(all_y[te][idx], rand_probs[idx, 1]))
        except:
            pass
    rand_ci = (float(np.percentile(rand_scores, 2.5)),
               float(np.percentile(rand_scores, 97.5))) if rand_scores else (0.0, 0.0)

    # Sex-aware split
    clf_s = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
    clf_s.fit(train_f, train_y)
    sex_probs = clf_s.predict_proba(test_f)
    sex_auc   = roc_auc_score(test_y, sex_probs[:, 1])

    # Bootstrap CI for sex AUC
    sex_scores = []
    for _ in range(N_BOOTSTRAP):
        idx = np.random.choice(len(test_y), len(test_y), replace=True)
        try:
            sex_scores.append(roc_auc_score(test_y[idx], sex_probs[idx, 1]))
        except:
            pass
    sex_ci = (float(np.percentile(sex_scores, 2.5)),
              float(np.percentile(sex_scores, 97.5))) if sex_scores else (0.0, 0.0)

    sgg = rand_auc - sex_auc
    sgg_norm = sgg / rand_auc if rand_auc > 0 else 0.0
    ci_overlap = bool(sex_ci[0] < rand_ci[1] and rand_ci[0] < sex_ci[1])
    significant = not ci_overlap

    print(f"\n  {model_name} {label}")
    print(f"    Random:   AUC={rand_auc:.4f} ({rand_ci[0]:.4f}-{rand_ci[1]:.4f})")
    print(f"    Sex-Aware:AUC={sex_auc:.4f} ({sex_ci[0]:.4f}-{sex_ci[1]:.4f})")
    print(f"    SGG={sgg:.4f}, SGGnorm={sgg_norm:.1%}, significant={significant}")

    return {
        'rand_auc':  float(rand_auc),
        'rand_ci':   rand_ci,
        'sex_auc':   float(sex_auc),
        'sex_ci':    sex_ci,
        'sgg':       float(sgg),
        'sgg_norm':  float(sgg_norm),
        'significant': significant,
    }

# ══════════════════════════════════════════════════════════════
# ARCHITECTURE 1: ViT-B/16 (the missing one)
# ══════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("ARCHITECTURE 1: ViT-B/16 (primary target)")
print("="*55)

vit_proc  = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device)
vit_model.eval()
print("ViT-B/16 loaded.")

@torch.no_grad()
def get_vit_features(images, batch_size=32):
    all_feats = []
    for i in range(0, len(images), batch_size):
        batch  = images[i:i+batch_size]
        inputs = vit_proc(images=batch, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feats  = vit_model(**inputs).last_hidden_state[:, 0, :]
        feats  = feats / feats.norm(dim=-1, keepdim=True)
        all_feats.append(feats.cpu().numpy())
        if i % 320 == 0:
            print(f"  ViT features: {i}/{len(images)}...")
    return np.vstack(all_feats)

print("Extracting ViT features...")
male_vit   = get_vit_features(male_imgs)
female_vit = get_vit_features(female_imgs)
all_vit    = np.vstack([male_vit, female_vit])
all_y_sex  = np.concatenate([male_y, female_y])

# Train on male (majority in dataset), test on female
vit_results = evaluate_sgg(
    "ViT-B/16", male_vit, male_y, female_vit, female_y,
    all_vit, all_y_sex, label="NIH Sex Split"
)

# Free memory
del vit_model
torch.cuda.empty_cache()

# ══════════════════════════════════════════════════════════════
# ARCHITECTURE 2: CLIP (verify / confirm from paper)
# ══════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("ARCHITECTURE 2: CLIP ViT-L/14 (confirm paper result)")
print("="*55)

clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
clip_model.eval()
print("CLIP loaded.")

@torch.no_grad()
def get_clip_features(images, batch_size=32):
    all_feats = []
    for i in range(0, len(images), batch_size):
        batch  = images[i:i+batch_size]
        inputs = clip_proc(images=batch, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feats  = clip_model.get_image_features(**inputs)
        if not isinstance(feats, torch.Tensor):
            feats = feats.pooler_output if hasattr(feats, 'pooler_output') \
                    else feats.last_hidden_state[:,0]
        feats  = feats / feats.norm(dim=-1, keepdim=True)
        all_feats.append(feats.cpu().numpy())
        if i % 320 == 0:
            print(f"  CLIP features: {i}/{len(images)}...")
    return np.vstack(all_feats)

print("Extracting CLIP features...")
male_clip  = get_clip_features(male_imgs)
female_clip = get_clip_features(female_imgs)
all_clip   = np.vstack([male_clip, female_clip])

clip_results = evaluate_sgg(
    "CLIP ViT-L/14", male_clip, male_y, female_clip, female_y,
    all_clip, all_y_sex, label="NIH Sex Split"
)

del clip_model
torch.cuda.empty_cache()

# ══════════════════════════════════════════════════════════════
# ARCHITECTURE 3: ResNet50 (confirm paper result)
# ══════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("ARCHITECTURE 3: ResNet50 (confirm paper result)")
print("="*55)

import torchvision.models as tvm
import torchvision.transforms as tvt

resnet = tvm.resnet50(pretrained=True)
resnet.fc = torch.nn.Identity()
resnet = resnet.to(device)
resnet.eval()
print("ResNet50 loaded.")

resnet_transform = tvt.Compose([
    tvt.Resize((224, 224)),
    tvt.ToTensor(),
    tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@torch.no_grad()
def get_resnet_features(images, batch_size=32):
    all_feats = []
    for i in range(0, len(images), batch_size):
        batch_tensors = torch.stack([
            resnet_transform(img) for img in images[i:i+batch_size]
        ]).to(device)
        feats = resnet(batch_tensors)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        all_feats.append(feats.cpu().numpy())
        if i % 320 == 0:
            print(f"  ResNet features: {i}/{len(images)}...")
    return np.vstack(all_feats)

print("Extracting ResNet50 features...")
male_res   = get_resnet_features(male_imgs)
female_res = get_resnet_features(female_imgs)
all_res    = np.vstack([male_res, female_res])

res_results = evaluate_sgg(
    "ResNet50", male_res, male_y, female_res, female_y,
    all_res, all_y_sex, label="NIH Sex Split"
)

del resnet
torch.cuda.empty_cache()

# ── Summary ───────────────────────────────────────────────────
all_results = {
    'vit_b16':  vit_results,
    'clip':     clip_results,
    'resnet50': res_results,
}

print("\n" + "="*55)
print("COMPLETE SUMMARY — NIH ChestX-ray14 Sex Split")
print("="*55)
print(f"\n{'Method':<25} {'Rand AUC':>10} {'Sex AUC':>10} {'SGG':>8} {'SGGnorm':>10} {'Sig?':>8}")
print("-"*75)
for arch, name in [('clip','CLIP ViT-L/14'), ('vit_b16','ViT-B/16'), ('resnet50','ResNet50')]:
    r = all_results[arch]
    print(f"{name + ' Random':<25} {r['rand_auc']:>10.4f} {'---':>10} {'---':>8} {'---':>10} {'---':>8}")
    sig_str = '* sig' if r['significant'] else '  n.s.'
    print(f"{name + ' Sex-Aware':<25} {'---':>10} {r['sex_auc']:>10.4f} {r['sgg']:>8.4f} {r['sgg_norm']:>9.1%} {sig_str:>8}")

print("\n=== LaTeX TABLE ROWS ===")
print("% NIH ChestX-ray14 sex split results")
for arch, name in [('clip','CLIP ViT-L/14'), ('vit_b16','ViT-B/16'), ('resnet50','ResNet50')]:
    r = all_results[arch]
    sig = '*' if r['significant'] else '\\dag'
    print(f"{name} Random & {r['rand_auc']:.3f} & ({r['rand_ci'][0]:.3f}--{r['rand_ci'][1]:.3f}) & --- & --- \\\\")
    print(f"{name} Sex-Aware & {r['sex_auc']:.3f} & ({r['sex_ci'][0]:.3f}--{r['sex_ci'][1]:.3f}) & {r['sgg']:.3f}{sig} & {r['sgg_norm']:.1%} \\\\")

json.dump(all_results, open('/kaggle/working/nb_p2_nih_sex_matrix.json', 'w'), indent=2)
print("\n✓ Complete. Paste ALL output back to Claude.")
