#!/usr/bin/env python3
"""
Generate paper figures and run ablation study:
1. Train Explainable Model and extract attention weights
2. Run ablation study (feature removal experiments)
3. Generate all paper figures (model comparison, training curves, attention weights, correlation heatmap)
"""

import os, sys, json, glob, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from models import ExplainableDBPsModel, ExplainableModelPredictor, PhysicsInformedFeatureExtractor

SEED = 42
TEST_SIZE = 0.3
np.random.seed(SEED)
torch.manual_seed(SEED)


# ══════════════════════════════════════════════════════════════
# Load original dataset
# ══════════════════════════════════════════════════════════════
df = pd.read_csv('data/data.csv')
if 'Sample' in df.columns:
    df = df.drop('Sample', axis=1)

target_names = df.columns[:5].tolist()
feature_names_raw = df.columns[5:].tolist()
# Normalize feature names for the model
feature_names_clean = [PhysicsInformedFeatureExtractor._normalize_feature_name(n) for n in feature_names_raw]

X = df.iloc[:, 5:].values
y = df.iloc[:, :5].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_s = scaler_X.fit_transform(X_train)
X_test_s = scaler_X.transform(X_test)
y_train_s = scaler_y.fit_transform(y_train)
y_test_s = scaler_y.transform(y_test)

print(f"Original dataset: {X.shape[0]} samples, {X.shape[1]} features, {y.shape[1]} targets")
print(f"Features (clean): {feature_names_clean}")
print(f"Targets: {target_names}")


# ══════════════════════════════════════════════════════════════
# 1. Train model & extract attention weights
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  1. Training model & extracting attention weights")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ExplainableDBPsModel(
    input_dim=X_train_s.shape[1], num_targets=y_train_s.shape[1],
    feature_names=feature_names_raw, x_mean=scaler_X.mean_, x_std=scaler_X.scale_
).to(device)

from models import HierarchicalConsistencyLoss
hierarchy_loss_fn = HierarchicalConsistencyLoss(
    target_names, y_mean=scaler_y.mean_, y_std=scaler_y.scale_, lambda_h=0.1
).to(device)

X_tr = torch.FloatTensor(X_train_s).to(device)
y_tr = torch.FloatTensor(y_train_s).to(device)
X_te = torch.FloatTensor(X_test_s).to(device)
y_te = torch.FloatTensor(y_test_s).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
criterion = torch.nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5)

# Training history
train_losses, test_losses, train_r2s, test_r2s, lrs = [], [], [], [], []

model.train()
for epoch in range(300):
    optimizer.zero_grad()
    pred = model(X_tr)
    mse_loss = criterion(pred, y_tr)
    physics_loss = hierarchy_loss_fn(pred)
    loss = mse_loss + physics_loss
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    lrs.append(optimizer.param_groups[0]['lr'])

    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            te_pred = model(X_te)
            te_loss = criterion(te_pred, y_te)
            test_losses.append(te_loss.item())
            tr_r2 = r2_score(y_train_s, model(X_tr).cpu().numpy())
            te_r2 = r2_score(y_test_s, te_pred.cpu().numpy())
            train_r2s.append(tr_r2)
            test_r2s.append(te_r2)
        scheduler.step(te_loss)
        model.train()
        if epoch % 50 == 0:
            print(f"  Epoch {epoch}: Train Loss={loss.item():.4f}, Test R²={te_r2:.4f}")

# Extract attention weights
model.eval()
with torch.no_grad():
    _, attn_weights, _ = model(X_tr, return_attention=True)
    mean_weights = attn_weights.mean(dim=0).cpu().numpy()

group_names = model.feature_extractor.active_groups
GROUP_DISPLAY = {
    'environmental': 'Environmental\nConditions',
    'organic_matter': 'Organic\nMatter',
    'disinfectant': 'Disinfectant',
    'nitrogen_species': 'Nitrogen\nCompounds',
    'halides': 'Halides'
}

print(f"\nAttention weights (new model):")
for g, w in zip(group_names, mean_weights):
    print(f"  {g}: {w*100:.1f}%")

# Final metrics
with torch.no_grad():
    tr_pred = model(X_tr).cpu().numpy()
    te_pred = model(X_te).cpu().numpy()
final_r2 = r2_score(y_test_s, te_pred)
print(f"\nFinal Test R²: {final_r2:.4f}")

# Save training history
os.makedirs('results/figures/training_curves', exist_ok=True)
history = {
    'epochs': 300, 'train_losses': train_losses, 'test_losses': test_losses,
    'train_r2_scores': train_r2s, 'test_r2_scores': test_r2s, 'learning_rates': lrs
}
with open('results/figures/training_curves/Explainable_Model_training_history_updated.json', 'w') as f:
    json.dump(history, f)


# ══════════════════════════════════════════════════════════════
# 2. Ablation study with new model
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  2. Running ablation study with new model")
print("="*60)

def train_ablation_model(X_tr_s, X_te_s, y_tr_s, y_te_s, feat_names, tgt_names,
                         x_mean, x_std, y_mean, y_std, epochs=300):
    """Train ExplainableDBPsModel for ablation and return metrics."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mdl = ExplainableDBPsModel(
        input_dim=X_tr_s.shape[1], num_targets=y_tr_s.shape[1],
        feature_names=feat_names, x_mean=x_mean, x_std=x_std
    ).to(dev)

    h_loss = HierarchicalConsistencyLoss(tgt_names, y_mean=y_mean, y_std=y_std, lambda_h=0.1).to(dev)

    xt = torch.FloatTensor(X_tr_s).to(dev)
    yt = torch.FloatTensor(y_tr_s).to(dev)
    xe = torch.FloatTensor(X_te_s).to(dev)

    opt = torch.optim.Adam(mdl.parameters(), lr=0.001, weight_decay=1e-5)
    crit = torch.nn.MSELoss()
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=50, factor=0.5)

    mdl.train()
    for ep in range(epochs):
        opt.zero_grad()
        p = mdl(xt)
        loss = crit(p, yt) + h_loss(p)
        loss.backward()
        opt.step()
        if ep % 10 == 0:
            mdl.eval()
            with torch.no_grad():
                te_p = mdl(xe)
                te_l = crit(te_p, torch.FloatTensor(y_te_s).to(dev))
            sched.step(te_l)
            mdl.train()

    mdl.eval()
    with torch.no_grad():
        tr_p = mdl(xt).cpu().numpy()
        te_p = mdl(xe).cpu().numpy()

    return {
        'Train R²': r2_score(y_tr_s, tr_p),
        'Test R²': r2_score(y_te_s, te_p),
        'Train MSE': mean_squared_error(y_tr_s, tr_p),
        'Test MSE': mean_squared_error(y_te_s, te_p),
        'Train MAE': mean_absolute_error(y_tr_s, tr_p),
        'Test MAE': mean_absolute_error(y_te_s, te_p),
    }


# Baseline (all features)
print("  Training baseline (all features)...")
baseline = train_ablation_model(
    X_train_s, X_test_s, y_train_s, y_test_s,
    feature_names_raw, target_names,
    scaler_X.mean_, scaler_X.scale_, scaler_y.mean_, scaler_y.scale_
)
print(f"  Baseline: Train R²={baseline['Train R²']:.4f}, Test R²={baseline['Test R²']:.4f}")

# Feature removal experiments
ablation_results = [{'Removed_Feature': 'None (Baseline)', **baseline, 'Delta_R2': 0.0}]

for i, (fname_raw, fname_clean) in enumerate(zip(feature_names_raw, feature_names_clean)):
    print(f"  Removing feature {i}: {fname_clean}...")

    # Remove feature from data
    X_tr_abl = np.delete(X_train_s, i, axis=1)
    X_te_abl = np.delete(X_test_s, i, axis=1)
    # Remove from feature names, scaler params
    fnames_remaining = [n for j, n in enumerate(feature_names_raw) if j != i]
    x_mean_remaining = np.delete(scaler_X.mean_, i)
    x_std_remaining = np.delete(scaler_X.scale_, i)

    metrics = train_ablation_model(
        X_tr_abl, X_te_abl, y_train_s, y_test_s,
        fnames_remaining, target_names,
        x_mean_remaining, x_std_remaining, scaler_y.mean_, scaler_y.scale_
    )
    delta = metrics['Test R²'] - baseline['Test R²']
    print(f"    Test R²={metrics['Test R²']:.4f}, ΔR²={delta:+.4f}")

    ablation_results.append({
        'Removed_Feature': fname_clean, **metrics, 'Delta_R2': delta
    })

# Save ablation results
os.makedirs('results/summary', exist_ok=True)
abl_df = pd.DataFrame(ablation_results)
abl_df.to_csv('results/summary/ablation_results.csv', index=False)
print("\n  Ablation results saved.")


# ══════════════════════════════════════════════════════════════
# 3. Copy multi-seed results for figure generation
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  3. Updating results files for figure generation")
print("="*60)

# Copy the new multi-seed results to where generate_paper_figures expects
import shutil
if os.path.exists('results_multiseed/model_results.csv'):
    shutil.copy('results_multiseed/model_results.csv', 'results/summary/model_results.csv')
    print("  Copied multiseed results to results/summary/model_results.csv")


# ══════════════════════════════════════════════════════════════
# 4. Generate all paper figures
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  4. Generating paper figures")
print("="*60)

# Global font settings
BASE, LABEL, TICK, LEGEND, ANNOT = 18, 18, 17, 17, 16
plt.rcParams.update({
    'font.size': BASE, 'axes.labelsize': LABEL, 'xtick.labelsize': TICK,
    'ytick.labelsize': TICK, 'legend.fontsize': LEGEND,
    'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 150,
})
OUT = 'manuscript/figure'
os.makedirs(OUT, exist_ok=True)

# ── Fig: Model Comparison ──
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
plt.savefig(f'{OUT}/model_comparison.png', bbox_inches='tight')
plt.close()
print("  ✅ model_comparison.png")

# ── Fig: Training Curves ──
n_epochs = history['epochs']
train_loss = history['train_losses']
test_loss_hist = history['test_losses']
train_r2_hist = history['train_r2_scores']
test_r2_hist = history['test_r2_scores']

train_epochs = list(range(1, n_epochs + 1))
interval = n_epochs // len(test_loss_hist)
test_epochs = list(range(interval, n_epochs + 1, interval))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
ax.plot(train_epochs, train_loss, label='Train Loss', color='#2196F3', linewidth=1.5)
ax.plot(test_epochs, test_loss_hist, label='Test Loss', color='#F44336', linewidth=1.5, linestyle='--')
ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss'); ax.legend()
ax.text(0.04, 0.96, '(A)', transform=ax.transAxes, fontsize=BASE+1, fontweight='bold', va='top')

ax = axes[1]
ax.plot(test_epochs, train_r2_hist, label='Train R²', color='#2196F3', linewidth=1.5)
ax.plot(test_epochs, test_r2_hist, label='Test R²', color='#F44336', linewidth=1.5, linestyle='--')
ax.set_xlabel('Epoch'); ax.set_ylabel('R²'); ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
ax.legend()
ax.text(0.04, 0.96, '(B)', transform=ax.transAxes, fontsize=BASE+1, fontweight='bold', va='top')

plt.tight_layout(pad=1.5)
plt.savefig(f'{OUT}/training_curve.png', bbox_inches='tight')
plt.close()
print("  ✅ training_curve.png")

# ── Fig: Attention Weights (with new values) ──
groups_display = [GROUP_DISPLAY[g] for g in group_names]
weights_pct = (mean_weights * 100).tolist()

# Chemical annotations per group (in canonical order)
ann_map = {
    'environmental': ('Temp, pH', 'reaction kinetics control'),
    'organic_matter': ('DOC, UVA254', 'primary organic precursors'),
    'disinfectant': ('Cl₂', 'narrow operational range'),
    'nitrogen_species': ('NH₄⁺-N, NO₂⁻-N', 'competitive Cl₂ consumers'),
    'halides': ('Br⁻', 'brominated DBPs pathway'),
}
ann_lines = [ann_map[g] for g in group_names]

# Sort by weight descending for display
sorted_idx = np.argsort(weights_pct)[::-1]
groups_sorted = [groups_display[i] for i in sorted_idx]
weights_sorted = [weights_pct[i] for i in sorted_idx]
ann_sorted = [ann_lines[i] for i in sorted_idx]
colors_att_all = ['#1976D2', '#388E3C', '#F57C00', '#7B1FA2', '#D32F2F']
colors_sorted = [colors_att_all[i] for i in sorted_idx]

fig, (ax_bar, ax_txt) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={'width_ratios': [3, 2]})

ypos = list(range(len(groups_sorted) - 1, -1, -1))
bars = ax_bar.barh(ypos, weights_sorted, color=colors_sorted, edgecolor='white', linewidth=0.5, height=0.6)
ax_bar.set_yticks(ypos)
ax_bar.set_yticklabels(groups_sorted, fontsize=TICK)
ax_bar.set_xlabel('Attention Weight (%)', fontsize=LABEL)
ax_bar.set_xlim(0, max(weights_sorted) + 10)
for bar, w in zip(bars, weights_sorted):
    ax_bar.text(w + 0.8, bar.get_y() + bar.get_height()/2, f'{w:.1f}%', va='center', ha='left',
                fontsize=BASE, fontweight='bold')

ax_txt.axis('off')
row_h = 1.0 / len(groups_sorted)
for i, (var, role) in enumerate(ann_sorted):
    yy = 1.0 - (i + 0.5) * row_h
    ax_txt.text(0.05, yy, var, transform=ax_txt.transAxes, va='center', ha='left',
                fontsize=BASE, fontweight='bold', color=colors_sorted[i])
    ax_txt.text(0.05, yy - row_h * 0.38, role, transform=ax_txt.transAxes, va='center', ha='left',
                fontsize=TICK-1, color='#555555', style='italic')

plt.tight_layout(pad=1.8)
plt.savefig(f'{OUT}/attention_weights.png', bbox_inches='tight')
plt.close()
print("  ✅ attention_weights.png")

# ── Fig: Model Correlation Heatmap ──
# Check if correlation file exists
corr_files = glob.glob('results/summary/model_correlation_*.csv')
if corr_files:
    corr_df = pd.read_csv(corr_files[0], index_col=0)
    short = {
        'Linear Regression': 'Linear', 'Ridge Regression': 'Ridge', 'Lasso Regression': 'Lasso',
        'Elastic Net': 'Elastic\nNet', 'Random Forest': 'Rand.\nForest', 'Extra Trees': 'Extra\nTrees',
        'Gradient Boosting': 'Grad.\nBoost', 'Support Vector Regression': 'SVR',
        'K-Nearest Neighbors': 'KNN', 'Decision Tree': 'Dec.\nTree', 'Neural Network': 'Neural\nNet',
        'Explainable Model': 'Explainable\nModel', 'KAN Model': 'KAN', 'TabNet': 'TabNet',
    }
    corr_df.index = [short.get(x, x) for x in corr_df.index]
    corr_df.columns = [short.get(x, x) for x in corr_df.columns]

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr_df, dtype=bool), k=1)
    sns.heatmap(corr_df, ax=ax, annot=True, fmt='.2f', cmap='RdYlGn', vmin=-0.2, vmax=1.0,
                linewidths=0.4, linecolor='white', annot_kws={'size': 9}, mask=mask,
                cbar_kws={'shrink': 0.8, 'label': 'Pearson r'})
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=TICK, rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=TICK, rotation=0)
    plt.tight_layout(pad=1.5)
    plt.savefig(f'{OUT}/model_correlation.png', bbox_inches='tight')
    plt.close()
    print("  ✅ model_correlation.png")
else:
    print("  ⚠️ No correlation CSV found, skipping heatmap")


print("\n" + "="*60)
print("  All done!")
print("="*60)

# Print summary of attention weights for paper
print("\nNew attention weights for paper:")
for g, w in zip(groups_sorted, weights_sorted):
    print(f"  {g.replace(chr(10), ' ')}: {w:.1f}%")

print("\nAblation results for paper:")
print(f"  Baseline Test R²: {baseline['Test R²']:.3f}")
for row in ablation_results[1:]:
    print(f"  Remove {row['Removed_Feature']:<10s}: Test R²={row['Test R²']:.3f}, ΔR²={row['Delta_R2']:+.3f}")
