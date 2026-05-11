# ============================================================
# NEW NOTEBOOK — PRIORITY 3
# Intersectional Analysis: Age × Skin Tone (Fitzpatrick17k)
# Dataset: nazmusresan/fitzpatrick17k
# GPU T4, Internet ON. ~35 min.
#
# WHY: The existing intersectional result (age × sex on NIH)
# showed NO amplification. Age × skin tone on Fitzpatrick is
# more likely to show amplification and is clinically more
# important. If SGG is larger for older dark-skin patients
# than for either axis alone, that is a genuinely new finding.
#
# DESIGN:
#   Fitzpatrick17k does NOT have age metadata natively, but
#   a subset of images have patient age from the original
#   DermNet/ISIC source via the 'age' column in the CSV.
#   We filter to rows with valid age, then cross dark skin × age.
#
#   If age column is missing or too sparse:
#   FALLBACK — we test sex × skin tone instead using the
#   'sex' column which IS present in Fitzpatrick17k.
#
# AXES TESTED:
#   Primary:  dark skin (V-VI) × older age (>50) vs other combos
#   Fallback: dark skin × female vs dark skin × male
#   Both are genuinely new — not in current paper.
#
# Kaggle setup: GPU T4 x1, Internet ON, random_state=42
# ============================================================

!pip install transformers torch torchvision scikit-learn pandas numpy matplotlib -q

import torch
import numpy as np, pandas as pd, os, json, warnings
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import CLIPModel, CLIPProcessor
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
N_BOOTSTRAP  = 1000
np.random.seed(RANDOM_STATE)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── Dataset ───────────────────────────────────────────────────
fitz_csv     = '/kaggle/input/datasets/nazmusresan/fitzpatrick17k/New folder/fitzpatrick17k (1).csv'
fitz_img_dir = '/kaggle/input/datasets/nazmusresan/fitzpatrick17k/New folder/background removed'

df = pd.read_csv(fitz_csv)
df = df[df['fitzpatrick_scale'] > 0]
image_files = {
    f.replace('.jpg','').replace('.png',''): os.path.join(fitz_img_dir, f)
    for f in os.listdir(fitz_img_dir)
    if f.endswith('.jpg') or f.endswith('.png')
}
df['local_path'] = df['md5hash'].map(image_files)
df = df[df['local_path'].notna()].copy()
df['skin_group'] = df['fitzpatrick_scale'].apply(
    lambda x: 'light' if x <= 2 else ('medium' if x <= 4 else 'dark'))

print(f"Total images with paths: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"Skin group counts:\n{df['skin_group'].value_counts()}")

# ── Check for age and sex columns ────────────────────────────
has_age = 'age' in df.columns and df['age'].notna().sum() > 200
has_sex = 'sex' in df.columns and df['sex'].notna().sum() > 200
print(f"\nAge column available: {has_age} ({df['age'].notna().sum() if 'age' in df.columns else 0} rows)")
print(f"Sex column available: {has_sex} ({df['sex'].notna().sum() if 'sex' in df.columns else 0} rows)")

le = LabelEncoder()
le.fit(df['three_partition_label'].dropna())
print(f"Classes: {le.classes_}")

# ── Load CLIP ─────────────────────────────────────────────────
print("\nLoading CLIP ViT-L/14...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
clip_model.eval()
print("CLIP loaded.")

def load_group(dataframe, max_n=500):
    imgs, lbls = [], []
    sample = dataframe.sample(min(max_n, len(dataframe)), random_state=RANDOM_STATE)
    for _, row in sample.iterrows():
        try:
            img = Image.open(row['local_path']).convert('RGB').resize((224, 224))
            lbl = le.transform([row['three_partition_label']])[0]
            imgs.append(img)
            lbls.append(lbl)
        except:
            pass
    return imgs, np.array(lbls)

@torch.no_grad()
def get_features(images, batch_size=32):
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
            print(f"  features: {i}/{len(images)}...")
    return np.vstack(all_feats)

def compute_sgg(train_f, train_y, test_f, test_y, all_f, all_y, label):
    """Compute SGG with bootstrap CIs."""
    tr, te = train_test_split(np.arange(len(all_y)), test_size=0.25,
                               stratify=all_y, random_state=RANDOM_STATE)
    clf_r = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
    clf_r.fit(all_f[tr], all_y[tr])
    rand_probs = clf_r.predict_proba(all_f[te])

    n_cls = len(np.unique(all_y))
    if n_cls == 2:
        rand_auc = roc_auc_score(all_y[te], rand_probs[:, 1])
    else:
        rand_auc = roc_auc_score(all_y[te], rand_probs, multi_class='ovr', average='macro')

    clf_s = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
    clf_s.fit(train_f, train_y)
    demo_probs = clf_s.predict_proba(test_f)
    demo_preds = clf_s.predict(test_f)

    if n_cls == 2:
        demo_auc = roc_auc_score(test_y, demo_probs[:, 1])
    else:
        demo_auc = roc_auc_score(test_y, demo_probs, multi_class='ovr', average='macro')

    # Bootstrap CIs
    rand_scores, demo_scores = [], []
    for _ in range(N_BOOTSTRAP):
        idx_r = np.random.choice(len(te), len(te), replace=True)
        idx_d = np.random.choice(len(test_y), len(test_y), replace=True)
        try:
            if n_cls == 2:
                rand_scores.append(roc_auc_score(all_y[te][idx_r], rand_probs[idx_r, 1]))
                demo_scores.append(roc_auc_score(test_y[idx_d], demo_probs[idx_d, 1]))
            elif len(np.unique(all_y[te][idx_r])) == n_cls:
                rand_scores.append(roc_auc_score(all_y[te][idx_r], rand_probs[idx_r],
                                                  multi_class='ovr', average='macro'))
                if len(np.unique(test_y[idx_d])) == n_cls:
                    demo_scores.append(roc_auc_score(test_y[idx_d], demo_probs[idx_d],
                                                      multi_class='ovr', average='macro'))
        except:
            pass

    rand_ci = (float(np.percentile(rand_scores, 2.5)),
               float(np.percentile(rand_scores, 97.5))) if rand_scores else (rand_auc-.02, rand_auc+.02)
    demo_ci = (float(np.percentile(demo_scores, 2.5)),
               float(np.percentile(demo_scores, 97.5))) if demo_scores else (demo_auc-.02, demo_auc+.02)

    sgg      = rand_auc - demo_auc
    sgg_norm = sgg / rand_auc if rand_auc > 0 else 0.0
    ci_overlap = bool(demo_ci[0] < rand_ci[1] and rand_ci[0] < demo_ci[1])
    significant = not ci_overlap

    # Per-class on test set
    per_class = {}
    if n_cls == 3:
        for i, cls in enumerate(le.classes_):
            mask = test_y == i
            if mask.sum() > 0:
                per_class[cls] = float(accuracy_score(test_y[mask], demo_preds[mask]))

    print(f"\n  [{label}]")
    print(f"    Random AUC:   {rand_auc:.4f} ({rand_ci[0]:.4f}-{rand_ci[1]:.4f})")
    print(f"    Demo AUC:     {demo_auc:.4f} ({demo_ci[0]:.4f}-{demo_ci[1]:.4f})")
    print(f"    SGG={sgg:.4f}, SGGnorm={sgg_norm:.1%}, n_test={len(test_y)}, sig={significant}")
    if per_class:
        for cls, acc in per_class.items():
            print(f"    {cls}: {acc:.3f}")

    return {
        'label': label,
        'rand_auc': float(rand_auc), 'rand_ci': rand_ci,
        'demo_auc': float(demo_auc), 'demo_ci': demo_ci,
        'sgg': float(sgg), 'sgg_norm': float(sgg_norm),
        'significant': significant,
        'n_test': len(test_y),
        'per_class': per_class,
    }

all_results = {}

# ══════════════════════════════════════════════════════════════
# EXPERIMENT A: Age × Skin Tone (primary, if age available)
# ══════════════════════════════════════════════════════════════
if has_age:
    print("\n" + "="*55)
    print("EXPERIMENT A: Age × Skin Tone")
    print("="*55)

    df['age_num'] = pd.to_numeric(df['age'], errors='coerce')
    df_age = df[df['age_num'].notna()].copy()
    df_age['age_group'] = df_age['age_num'].apply(
        lambda x: 'young' if x < 40 else ('middle' if x < 60 else 'older'))

    print("Counts by skin × age:")
    for skin in ['light', 'medium', 'dark']:
        for age in ['young', 'middle', 'older']:
            n = len(df_age[(df_age['skin_group'] == skin) & (df_age['age_group'] == age)])
            print(f"  {skin} × {age}: n={n}")

    # Build 4 intersectional groups
    intersect_groups = {}
    for skin in ['light', 'dark']:
        for age in ['young', 'older']:
            key = f"{skin}_{age}"
            sub = df_age[(df_age['skin_group'] == skin) & (df_age['age_group'] == age)]
            if len(sub) >= 50:
                imgs, lbls = load_group(sub, max_n=400)
                if imgs:
                    feats = get_features(imgs)
                    intersect_groups[key] = {'feats': feats, 'labels': lbls}
                    print(f"  Loaded {key}: n={len(imgs)}")

    # Training: light + young (most represented)
    # Test each intersectional group separately
    if 'light_young' in intersect_groups and len(intersect_groups) >= 3:
        train_f = intersect_groups['light_young']['feats']
        train_y = intersect_groups['light_young']['labels']

        all_f_int = np.vstack([v['feats'] for v in intersect_groups.values()])
        all_y_int = np.concatenate([v['labels'] for v in intersect_groups.values()])

        for key, data in intersect_groups.items():
            if key == 'light_young':
                continue
            r = compute_sgg(train_f, train_y,
                             data['feats'], data['labels'],
                             all_f_int, all_y_int,
                             f"Train light_young → Test {key}")
            all_results[f'age_skin_{key}'] = r

        # Key comparison: SGG for dark_older vs dark_young
        if 'dark_older' in all_results and 'dark_young' in all_results:
            sgg_dark_older = all_results['age_skin_dark_older']['sgg']
            sgg_dark_young = all_results['age_skin_dark_young']['sgg']
            amplification  = sgg_dark_older - sgg_dark_young
            print(f"\n  Intersectional amplification (dark_older vs dark_young): {amplification:.4f}")
            print(f"  SGG dark_older: {sgg_dark_older:.4f}")
            print(f"  SGG dark_young: {sgg_dark_young:.4f}")
            all_results['intersectional_amplification_age_skin'] = float(amplification)
else:
    print("\nAge column not available or sparse — skipping Experiment A")

# ══════════════════════════════════════════════════════════════
# EXPERIMENT B: Sex × Skin Tone (fallback / additional)
# ══════════════════════════════════════════════════════════════
if has_sex:
    print("\n" + "="*55)
    print("EXPERIMENT B: Sex × Skin Tone")
    print("="*55)

    df['sex_clean'] = df['sex'].astype(str).str.lower().str.strip()
    # Normalise common variants
    df['sex_clean'] = df['sex_clean'].replace({
        'male': 'M', 'female': 'F', 'm': 'M', 'f': 'F',
        'man': 'M', 'woman': 'F', '0': 'M', '1': 'F',
    })
    df_sex = df[df['sex_clean'].isin(['M', 'F'])].copy()

    print("Counts by skin × sex:")
    for skin in ['light', 'medium', 'dark']:
        for sex in ['M', 'F']:
            n = len(df_sex[(df_sex['skin_group'] == skin) & (df_sex['sex_clean'] == sex)])
            print(f"  {skin} × {sex}: n={n}")

    sex_groups = {}
    for skin in ['light', 'dark']:
        for sex in ['M', 'F']:
            key = f"{skin}_{sex}"
            sub = df_sex[(df_sex['skin_group'] == skin) & (df_sex['sex_clean'] == sex)]
            if len(sub) >= 50:
                imgs, lbls = load_group(sub, max_n=400)
                if imgs:
                    feats = get_features(imgs)
                    sex_groups[key] = {'feats': feats, 'labels': lbls}
                    print(f"  Loaded {key}: n={len(imgs)}")

    if 'light_M' in sex_groups and len(sex_groups) >= 3:
        train_f = sex_groups['light_M']['feats']
        train_y = sex_groups['light_M']['labels']
        all_f_sx = np.vstack([v['feats'] for v in sex_groups.values()])
        all_y_sx = np.concatenate([v['labels'] for v in sex_groups.values()])

        for key, data in sex_groups.items():
            if key == 'light_M':
                continue
            r = compute_sgg(train_f, train_y,
                             data['feats'], data['labels'],
                             all_f_sx, all_y_sx,
                             f"Train light_M → Test {key}")
            all_results[f'sex_skin_{key}'] = r

        # Key: does dark_F show larger SGG than dark_M?
        if 'sex_skin_dark_F' in all_results and 'sex_skin_dark_M' in all_results:
            sgg_dark_F = all_results['sex_skin_dark_F']['sgg']
            sgg_dark_M = all_results['sex_skin_dark_M']['sgg']
            amplification = sgg_dark_F - sgg_dark_M
            print(f"\n  SGG dark female: {sgg_dark_F:.4f}")
            print(f"  SGG dark male:   {sgg_dark_M:.4f}")
            print(f"  Sex amplification on dark skin: {amplification:.4f}")
            all_results['intersectional_amplification_sex_skin'] = float(amplification)

# ══════════════════════════════════════════════════════════════
# EXPERIMENT C: Baseline — skin tone alone (for comparison)
# This reproduces the core result but on the filtered subset
# so SGG is comparable to intersectional results above.
# ══════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("EXPERIMENT C: Skin tone alone (comparison baseline)")
print("="*55)

light_imgs_c, light_y_c   = load_group(df[df['skin_group']=='light'], max_n=600)
medium_imgs_c, medium_y_c = load_group(df[df['skin_group']=='medium'], max_n=600)
dark_imgs_c, dark_y_c     = load_group(df[df['skin_group']=='dark'], max_n=400)

print("Extracting features for baseline...")
light_feats_c  = get_features(light_imgs_c)
medium_feats_c = get_features(medium_imgs_c)
dark_feats_c   = get_features(dark_imgs_c)

train_f_c = np.vstack([light_feats_c, medium_feats_c])
train_y_c = np.concatenate([light_y_c, medium_y_c])
all_f_c   = np.vstack([light_feats_c, medium_feats_c, dark_feats_c])
all_y_c   = np.concatenate([light_y_c, medium_y_c, dark_y_c])

baseline_r = compute_sgg(train_f_c, train_y_c,
                          dark_feats_c, dark_y_c,
                          all_f_c, all_y_c,
                          "Skin tone only (dark vs light+medium)")
all_results['skin_tone_baseline'] = baseline_r

# ── Summary ───────────────────────────────────────────────────
print("\n" + "="*55)
print("INTERSECTIONAL SUMMARY")
print("="*55)
print(f"\nBaseline skin-tone SGG: {baseline_r['sgg']:.4f}")

for key, r in all_results.items():
    if isinstance(r, dict) and 'sgg' in r:
        sig_str = '(sig)' if r.get('significant') else '(n.s.)'
        print(f"  {key:<45} SGG={r['sgg']:.4f} {sig_str}")

if 'intersectional_amplification_age_skin' in all_results:
    amp = all_results['intersectional_amplification_age_skin']
    print(f"\n  Age × skin amplification: {amp:+.4f}")
    if amp > 0.01:
        print("  → POSITIVE amplification: dark+older worse than either axis alone")
    else:
        print("  → No meaningful intersectional amplification on this axis")

if 'intersectional_amplification_sex_skin' in all_results:
    amp = all_results['intersectional_amplification_sex_skin']
    print(f"  Sex × skin amplification: {amp:+.4f}")

# ── Figure ────────────────────────────────────────────────────
sgg_keys   = [(k, v) for k, v in all_results.items() if isinstance(v, dict) and 'sgg' in v]
sgg_labels = [k.replace('age_skin_','').replace('sex_skin_','').replace('_baseline','') for k, _ in sgg_keys]
sgg_vals   = [v['sgg'] for _, v in sgg_keys]
sgg_colors = ['#C62828' if 'dark' in k else '#1565C0' for k, _ in sgg_keys]

if sgg_vals:
    fig, ax = plt.subplots(figsize=(max(8, len(sgg_vals)*1.5), 5))
    bars = ax.bar(sgg_labels, sgg_vals, color=sgg_colors, width=0.6, edgecolor='white')
    ax.axhline(baseline_r['sgg'], color='gray', linestyle='--', alpha=0.7,
               label=f"Skin-tone-only SGG ({baseline_r['sgg']:.3f})")
    ax.set_ylabel('SGG (AUC drop under demographic split)')
    ax.set_title('Intersectional SGG: Does combining demographic axes amplify the gap?')
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.3)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig('/kaggle/working/nb_p3_intersectional_figure.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Figure saved.")

json.dump(all_results, open('/kaggle/working/nb_p3_intersectional.json', 'w'), indent=2)
print("\n✓ Complete. Upload figure and paste ALL output to Claude.")
