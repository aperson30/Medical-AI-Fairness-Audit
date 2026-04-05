import pandas as pd
import numpy as np
import os
from scipy.stats import spearmanr

base_path = 'paper_export_final/results/00_baseline_results.csv'
intv_path = 'paper_export_final/results/04_intervention_matrix.csv'
theory_path = 'paper_export_final/results/05_mad_theory.csv'

df_base = pd.read_csv(base_path)
df_intv = pd.read_csv(intv_path)

print("Patching Gap Closed % calculations...")

for i, row in df_intv.iterrows():
    m = row['model']
    s = row['seed']
    base_match = df_base[(df_base['model'] == m) & (df_base['seed'] == s)].iloc[0]
    true_baseline_sgg = base_match['sgg']
    
    sgg = row['sgg']
    # Dynamic gap closing using the actual baseline for that specific model/seed!
    gap_pct = max(0, (true_baseline_sgg - sgg) / true_baseline_sgg * 100)
    
    # Re-eval MAD flag: gap closed > 20, benign acc < 0.05
    acc = row['acc_benign_dark']
    mad_flag = (gap_pct > 20.0) and (acc < 0.05 if not np.isnan(acc) else True)
    
    df_intv.at[i, 'gap_closed_pct'] = gap_pct
    df_intv.at[i, 'mad_flag'] = mad_flag

for i, row in df_base.iterrows():
    df_base.at[i, 'gap_closed_pct'] = 0.0

df_base.to_csv(base_path, index=False)
df_intv.to_csv(intv_path, index=False)

print("Patching Theory data...")
nc_ng = 97 / 2168
theory_rows = []
for df_src, label in [(df_base, "baseline"), (df_intv, "intervention")]:
    for _, row in df_src.iterrows():
        gap = row.get('gap_closed_pct', np.nan)
        acc = row.get('acc_benign_dark', np.nan)
        denom = 1.0 - acc
        sev = (gap / 100.0) / denom if denom > 0 else 0.0
        theory_rows.append({
            'dataset': 'fitzpatrick17k',
            'model': row['model'],
            'intervention': row.get('intervention', label),
            'seed': row['seed'],
            'nc_ng': nc_ng,
            'gap_closed_pct': gap,
            'benign_acc': acc,
            'mad_severity': sev,
        })

df_theory_raw = pd.DataFrame(theory_rows)
df_theory_raw = df_theory_raw[df_theory_raw['nc_ng'].notna() & df_theory_raw['mad_severity'].notna()]
rho, p_val = spearmanr(df_theory_raw['nc_ng'], df_theory_raw['mad_severity'])
df_theory_agg = df_theory_raw.groupby(['model', 'intervention', 'nc_ng']).agg(
    mad_severity_mean=('mad_severity', 'mean'),
    mad_severity_std=('mad_severity', 'std'),
    gap_closed_mean=('gap_closed_pct', 'mean'),
    benign_acc_mean=('benign_acc', 'mean'),
).reset_index()

theory_save = df_theory_agg.copy()
theory_save['dataset'] = 'fitzpatrick17k'
theory_save = pd.concat([theory_save, pd.DataFrame([{
    'dataset': 'statistics', 'model': 'all', 'intervention': 'spearman_rho',
    'nc_ng': rho, 'mad_severity_mean': p_val,
    'mad_severity_std': np.nan, 'gap_closed_mean': np.nan, 'benign_acc_mean': np.nan,
}])], ignore_index=True)

theory_save.to_csv(theory_path, index=False)
print("Data fully patched and ready for analysis!")
