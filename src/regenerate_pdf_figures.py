#!/usr/bin/env python3
"""
Regenerate the two comparison figures as vector PDF (not PNG), reading from the
existing result CSVs so no model retraining is needed:
  - model_comparison.pdf   (per-model Test R2/MSE/MAE bars, original dataset, multi-seed means)
  - model_correlation.pdf  (prediction-correlation heatmap, with axis labels de-overlapped)

Style matches src/run_figures.py.
"""

import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

BASE, LABEL, TICK, LEGEND, ANNOT = 18, 18, 17, 17, 16
plt.rcParams.update({
    'font.size': BASE, 'axes.labelsize': LABEL, 'xtick.labelsize': TICK,
    'ytick.labelsize': TICK, 'legend.fontsize': LEGEND,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,   # embed TrueType for vector PDF
})
OUT = 'manuscript/figure'
os.makedirs(OUT, exist_ok=True)

# ────────────────────────────────────────────────────────────────
# Fig: Model Comparison  (original dataset, multi-seed means)
# ────────────────────────────────────────────────────────────────
df_res = pd.read_csv('results/summary/model_results.csv')
df_res = df_res.sort_values('Test_R2', ascending=True)

name_map = {
    'Support Vector Regression': 'SVR', 'TabNet': 'TabNet',
    'Explainable Model': 'Explainable\nModel', 'Extra Trees': 'Extra Trees',
    'Neural Network': 'Neural Net', 'Random Forest': 'Random Forest',
    'Gradient Boosting': 'Grad. Boost.', 'K-Nearest Neighbors': 'KNN',
    'KAN Model': 'KAN', 'Elastic Net': 'Elastic Net',
    'Lasso Regression': 'Lasso', 'Ridge Regression': 'Ridge',
    'Linear Regression': 'Linear Reg.', 'Decision Tree': 'Dec. Tree',
}
df_res['ShortName'] = df_res['Model'].map(name_map).fillna(df_res['Model'])

colors = []
for _, r in df_res.iterrows():
    if r['Model'] == 'Support Vector Regression': colors.append('#2196F3')
    elif r['Model'] == 'Explainable Model': colors.append('#FF9800')
    elif r['Model'] == 'TabNet': colors.append('#4CAF50')
    else: colors.append('#9E9E9E')

fig, axes = plt.subplots(1, 3, figsize=(16, 6.5))
metrics_list = [('Test_R2', 'Test R²'), ('Test_MSE', 'Test MSE'), ('Test_MAE', 'Test MAE')]

for ax, (col, xlabel) in zip(axes, metrics_list):
    vals = df_res[col].values
    vmin, vmax = vals.min(), vals.max()
    span = vmax - vmin if vmax != vmin else 1
    bars = ax.barh(df_res['ShortName'], vals, color=colors, edgecolor='white', linewidth=0.5, height=0.65)
    ax.set_xlabel(xlabel, fontsize=LABEL)
    ax.axvline(0, color='black', linewidth=0.7)
    ax.set_xlim(vmin - span * 0.05, vmax + span * 0.18)
    for bar, v in zip(bars, vals):
        xpos = span * 0.03 if v < 0 else v + span * 0.015
        ax.text(xpos, bar.get_y() + bar.get_height()/2, f'{v:.3f}', va='center', ha='left', fontsize=ANNOT)

patch_svr = mpatches.Patch(color='#2196F3', label='Best Test R² (SVR)')
patch_exp = mpatches.Patch(color='#FF9800', label='Explainable Model (Ours)')
patch_tab = mpatches.Patch(color='#4CAF50', label='TabNet')
fig.legend(handles=[patch_svr, patch_exp, patch_tab], fontsize=LEGEND-1, loc='lower center',
           ncol=3, bbox_to_anchor=(0.5, -0.06), frameon=False)
plt.tight_layout(pad=1.8)
plt.subplots_adjust(bottom=0.18)
plt.savefig(f'{OUT}/model_comparison.pdf', bbox_inches='tight')
plt.close()
print("  ✅ model_comparison.pdf")

# ────────────────────────────────────────────────────────────────
# Fig: Model Correlation Heatmap  (axis labels de-overlapped)
# ────────────────────────────────────────────────────────────────
corr_files = sorted(glob.glob('results/summary/model_correlation_*.csv'))
if not corr_files:
    raise SystemExit("No correlation CSV found")
corr_df = pd.read_csv(corr_files[0], index_col=0)

# Single-line short labels (the previous \n-wrapped labels were overlapping on the rotated x-axis)
short = {
    'Linear Regression': 'Linear', 'Ridge Regression': 'Ridge', 'Lasso Regression': 'Lasso',
    'Elastic Net': 'ElasticNet', 'Random Forest': 'RF', 'Extra Trees': 'ExtraTrees',
    'Gradient Boosting': 'GradBoost', 'Support Vector Regression': 'SVR',
    'K-Nearest Neighbors': 'KNN', 'Decision Tree': 'DecTree', 'Neural Network': 'NeuralNet',
    'Explainable Model': 'Explainable', 'KAN Model': 'KAN', 'TabNet': 'TabNet',
}
corr_df.index = [short.get(x, x) for x in corr_df.index]
corr_df.columns = [short.get(x, x) for x in corr_df.columns]

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_df, dtype=bool), k=1)
sns.heatmap(corr_df, ax=ax, annot=True, fmt='.2f', cmap='RdYlGn', vmin=-0.2, vmax=1.0,
            linewidths=0.4, linecolor='white', annot_kws={'size': 10}, mask=mask, square=True,
            cbar_kws={'shrink': 0.7, 'label': 'Pearson r', 'pad': 0.02})
ax.set_xticklabels(ax.get_xticklabels(), fontsize=TICK-2, rotation=45, ha='right', rotation_mode='anchor')
ax.set_yticklabels(ax.get_yticklabels(), fontsize=TICK-2, rotation=0)
ax.tick_params(axis='both', length=0)
plt.tight_layout(pad=1.2)
plt.savefig(f'{OUT}/model_correlation.pdf', bbox_inches='tight')
plt.close()
print("  ✅ model_correlation.pdf")
