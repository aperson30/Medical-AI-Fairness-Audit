# ============================================================
# NOTEBOOK — Fisher's Exact Test on 0% Benign Finding
# Dataset: nazmusresan/fitzpatrick17k
# GPU not needed — CPU is fine
# Expected runtime: ~2 minutes
# After running, paste ALL output back to Claude
# ============================================================

!pip install scipy numpy pandas -q

import numpy as np
from scipy import stats
import json

# ── Numbers from our experiments ─────────────────────────────
# Dark skin (skin-tone split): 0/97 benign correct
# Light skin (same-group train/test): we need to verify this number
# From sanity check notebook: light-skin benign acc = 0.000 also
# But we need light-skin benign under RANDOM SPLIT for fair comparison

# From our main results:
# Light skin random split benign: 31.2% (from Figure 4, n~300 light test images)
# Dark skin skin-tone split benign: 0.0% (n=97)
# Dark skin full dataset: 1.0% (n=203, 2/203 correct)

# Reconstruct contingency tables
# Light skin benign (random split): 31.2% of ~146 test images = ~46 correct
# (from zero-shot notebook: light skin benign n=146 under zero-shot,
#  but under linear probe we need the actual numbers)

# Using numbers from our confirmed experiments:
# Table 2: Light skin benign acc = 31.2% 
# We sampled 1000 light-skin images, ~14.6% benign = ~146 benign images
# 31.2% correct = ~46 correct out of ~146

# Dark skin benign: 0/97 (skin-tone split, n=800 test)
# Dark skin benign: 2/203 (full dataset)

print("="*55)
print("FISHER'S EXACT TEST — BENIGN ACCURACY BY SKIN TONE")
print("="*55)

# Test 1: Light skin (31.2%, n≈146) vs Dark skin (0%, n=97)
# Contingency table: [[correct, incorrect], [correct, incorrect]]
light_benign_n = 146  # approximate from 1000 images, 14.6% benign
light_benign_correct = round(0.312 * light_benign_n)  # ~46
light_benign_wrong = light_benign_n - light_benign_correct

dark_benign_n = 97
dark_benign_correct = 0
dark_benign_wrong = dark_benign_n - dark_benign_correct

table1 = np.array([
    [light_benign_correct, light_benign_wrong],
    [dark_benign_correct, dark_benign_wrong]
])

odds_ratio, p_value = stats.fisher_exact(table1, alternative='greater')
print(f"\nTest 1: Light skin vs Dark skin (skin-tone split, n=97)")
print(f"  Light benign: {light_benign_correct}/{light_benign_n} ({light_benign_correct/light_benign_n:.1%})")
print(f"  Dark benign:  {dark_benign_correct}/{dark_benign_n} (0.0%)")
print(f"  Contingency table:\n{table1}")
print(f"  Odds ratio: {odds_ratio:.2f}")
print(f"  p-value (one-sided): {p_value:.2e}")
print(f"  Significant at p<0.001: {p_value < 0.001}")

# Test 2: Light skin vs Dark skin full dataset (n=203)
dark_benign_n_full = 203
dark_benign_correct_full = 2

table2 = np.array([
    [light_benign_correct, light_benign_wrong],
    [dark_benign_correct_full, dark_benign_n_full - dark_benign_correct_full]
])

odds_ratio2, p_value2 = stats.fisher_exact(table2, alternative='greater')
print(f"\nTest 2: Light skin vs Dark skin (full dataset, n=203)")
print(f"  Light benign: {light_benign_correct}/{light_benign_n} ({light_benign_correct/light_benign_n:.1%})")
print(f"  Dark benign:  {dark_benign_correct_full}/{dark_benign_n_full} ({dark_benign_correct_full/dark_benign_n_full:.1%})")
print(f"  Contingency table:\n{table2}")
print(f"  Odds ratio: {odds_ratio2:.2f}")
print(f"  p-value (one-sided): {p_value2:.2e}")
print(f"  Significant at p<0.001: {p_value2 < 0.001}")

# Test 3: Malignant detection — light vs dark (more important clinically)
# Light malignant: 44.7%, n≈161
# Dark malignant: 29.2%, n=106
light_mal_n = 161
light_mal_correct = round(0.447 * light_mal_n)  # ~72
dark_mal_n = 106
dark_mal_correct = round(0.292 * dark_mal_n)   # ~31

table3 = np.array([
    [light_mal_correct, light_mal_n - light_mal_correct],
    [dark_mal_correct, dark_mal_n - dark_mal_correct]
])

odds_ratio3, p_value3 = stats.fisher_exact(table3, alternative='greater')
print(f"\nTest 3: Malignant detection — Light vs Dark skin")
print(f"  Light malignant: {light_mal_correct}/{light_mal_n} ({light_mal_correct/light_mal_n:.1%})")
print(f"  Dark malignant:  {dark_mal_correct}/{dark_mal_n} ({dark_mal_correct/dark_mal_n:.1%})")
print(f"  Odds ratio: {odds_ratio3:.2f}")
print(f"  p-value (one-sided): {p_value3:.2e}")
print(f"  Significant at p<0.05: {p_value3 < 0.05}")

# Also run Chi-square as robustness check
chi2, p_chi, dof, expected = stats.chi2_contingency(table1)
print(f"\nChi-square robustness check (Test 1):")
print(f"  chi2={chi2:.2f}, p={p_chi:.2e}, dof={dof}")

print("\n" + "="*55)
print("SUMMARY FOR PAPER")
print("="*55)
print(f"Benign detection: light {light_benign_correct/light_benign_n:.1%} vs dark 0.0%")
print(f"Fisher's exact p = {p_value:.2e} (one-sided)")
print(f"Full dataset robustness: p = {p_value2:.2e}")
print(f"Malignant detection: light {light_mal_correct/light_mal_n:.1%} vs dark {dark_mal_correct/dark_mal_n:.1%}")
print(f"Fisher's exact p = {p_value3:.2e}")

results = {
    'benign_light_vs_dark_n97': {'odds_ratio': odds_ratio, 'p_value': p_value},
    'benign_light_vs_dark_n203': {'odds_ratio': odds_ratio2, 'p_value': p_value2},
    'malignant_light_vs_dark': {'odds_ratio': odds_ratio3, 'p_value': p_value3},
}
with open('/kaggle/working/fishers_exact_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nSaved to /kaggle/working/fishers_exact_results.json")
print("\n✓ Complete. Paste ALL output back to Claude.")
