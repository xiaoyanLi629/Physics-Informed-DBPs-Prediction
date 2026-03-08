"""
生成论文用图：无 title，字号统一增大3pt，保存到 manuscript/figure/
包含：
  1. model_comparison.png       - 模型性能对比（横向条形）
  2. training_curve.png         - Explainable Model 训练曲线
  3. attention_weights.png      - 注意力权重（新增）
  4. model_correlation.png      - 模型预测相关性热图
"""

import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# 切换到项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

# ── 全局字体设置（放大，无 title）────────────────────────────
BASE   = 14   # 正文字号
LABEL  = 14   # 轴标签
TICK   = 13   # 刻度
LEGEND = 13   # 图例
ANNOT  = 12   # 数值标注
plt.rcParams.update({
    'font.size':          BASE,
    'axes.labelsize':     LABEL,
    'xtick.labelsize':    TICK,
    'ytick.labelsize':    TICK,
    'legend.fontsize':    LEGEND,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'figure.dpi':         150,
})

OUT = 'manuscript/figure'
os.makedirs(OUT, exist_ok=True)

# ════════════════════════════════════════════════════════════
# 1. 模型性能对比图
# ════════════════════════════════════════════════════════════
df = pd.read_csv('results/summary/model_results.csv')
df = df.sort_values('Test_R2', ascending=True)

# 短名映射
name_map = {
    'Support Vector Regression': 'SVR',
    'TabNet': 'TabNet',
    'Explainable Model': 'Explainable\nModel',
    'Extra Trees': 'Extra Trees',
    'Neural Network': 'Neural Net',
    'Random Forest': 'Random Forest',
    'Gradient Boosting': 'Grad. Boost.',
    'K-Nearest Neighbors': 'KNN',
    'KAN Model': 'KAN',
    'Elastic Net': 'Elastic Net',
    'Lasso Regression': 'Lasso',
    'Ridge Regression': 'Ridge',
    'Linear Regression': 'Linear Reg.',
    'Decision Tree': 'Dec. Tree',
}
df['ShortName'] = df['Model'].map(name_map).fillna(df['Model'])

# 颜色：top-3 高亮
colors = []
for i, r in df.iterrows():
    if r['Model'] == 'Support Vector Regression':
        colors.append('#2196F3')   # 蓝 - 最佳R²
    elif r['Model'] == 'Explainable Model':
        colors.append('#FF9800')   # 橙 - 最佳MAE
    elif r['Model'] == 'TabNet':
        colors.append('#4CAF50')   # 绿
    else:
        colors.append('#9E9E9E')   # 灰

# 图更高，模型名字够大，每个 panel 给数值标注留出右侧空间
fig, axes = plt.subplots(1, 3, figsize=(16, 6.5))
metrics = [
    ('Test_R2',  'Test R²'),
    ('Test_MSE', 'Test MSE'),
    ('Test_MAE', 'Test MAE'),
]

for ax, (col, xlabel) in zip(axes, metrics):
    vals = df[col].values
    vmin, vmax = vals.min(), vals.max()
    span = vmax - vmin if vmax != vmin else 1
    bars = ax.barh(df['ShortName'], vals, color=colors,
                   edgecolor='white', linewidth=0.5, height=0.65)
    ax.set_xlabel(xlabel, fontsize=LABEL)
    ax.axvline(0, color='black', linewidth=0.7)
    ax.tick_params(axis='y', labelsize=TICK)
    ax.tick_params(axis='x', labelsize=TICK)
    # 右侧留空间放数值
    ax.set_xlim(vmin - span * 0.05, vmax + span * 0.18)
    for bar, v in zip(bars, vals):
        ax.text(v + span * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f'{v:.3f}', va='center', ha='left', fontsize=ANNOT)

# 图例放在第一个 panel 内右下
patch_svr = mpatches.Patch(color='#2196F3', label='Best Test R² (SVR)')
patch_exp = mpatches.Patch(color='#FF9800', label='Best Test MAE (Explainable Model)')
patch_tab = mpatches.Patch(color='#4CAF50', label='TabNet')
axes[0].legend(handles=[patch_svr, patch_exp, patch_tab],
               fontsize=LEGEND - 1, loc='lower right')

plt.tight_layout(pad=1.8)
plt.savefig(f'{OUT}/model_comparison.png', bbox_inches='tight')
plt.close()
print('✅ model_comparison.png')


# ════════════════════════════════════════════════════════════
# 2. Explainable Model 训练曲线（2-panel）
# ════════════════════════════════════════════════════════════
import glob
json_files = glob.glob('results/figures/training_curves/Explainable_Model_training_history_*.json')
with open(json_files[0]) as f:
    hist = json.load(f)

n_epochs    = hist['epochs']
train_loss  = hist['train_losses']
test_loss   = hist['test_losses']
train_r2    = hist['train_r2_scores']
test_r2     = hist['test_r2_scores']
lr          = hist['learning_rates']

# train 每 epoch 记录，test 每10 epoch 记录
train_epochs = list(range(1, n_epochs + 1))
interval     = n_epochs // len(test_loss)
test_epochs  = list(range(interval, n_epochs + 1, interval))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Panel A: Loss
ax = axes[0]
ax.plot(train_epochs, train_loss, label='Train Loss', color='#2196F3', linewidth=1.5)
ax.plot(test_epochs,  test_loss,  label='Test Loss',  color='#F44336', linewidth=1.5, linestyle='--')
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.legend()
ax.text(0.04, 0.96, '(A)', transform=ax.transAxes,
        fontsize=BASE+1, fontweight='bold', va='top')

# Panel B: R²
ax = axes[1]
ax.plot(test_epochs, train_r2, label='Train R²', color='#2196F3', linewidth=1.5)
ax.plot(test_epochs, test_r2,  label='Test R²',  color='#F44336', linewidth=1.5, linestyle='--')
ax.set_xlabel('Epoch')
ax.set_ylabel('R²')
ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
ax.legend()
ax.text(0.04, 0.96, '(B)', transform=ax.transAxes,
        fontsize=BASE+1, fontweight='bold', va='top')

plt.tight_layout(pad=1.5)
plt.savefig(f'{OUT}/training_curve.png', bbox_inches='tight')
plt.close()
print('✅ training_curve.png')


# ════════════════════════════════════════════════════════════
# 3. 注意力权重图（两列布局：左侧条形 + 右侧说明文字）
# ════════════════════════════════════════════════════════════
groups  = ['Organic\nMatter', 'Nitrogen\nCompounds', 'Environmental\nConditions',
           'Halides', 'Disinfectant']
weights = [44.0, 32.6, 14.7, 5.9, 2.8]
colors_att  = ['#1976D2', '#388E3C', '#F57C00', '#7B1FA2', '#D32F2F']
# 右侧说明：变量名 + 作用
ann_lines = [
    ('DOC, UVA254',        'primary organic precursors'),
    ('NH₄⁺-N, NO₂⁻-N',   'competitive Cl₂ consumers'),
    ('Temp, pH',           'reaction kinetics control'),
    ('Br⁻',               'brominated DBPs pathway'),
    ('Cl₂',               'narrow operational range'),
]

fig, (ax_bar, ax_txt) = plt.subplots(1, 2, figsize=(13, 5),
                                      gridspec_kw={'width_ratios': [3, 2]})

# ── 左：条形图 ──
ypos = list(range(len(groups) - 1, -1, -1))
bars = ax_bar.barh(ypos, weights, color=colors_att,
                   edgecolor='white', linewidth=0.5, height=0.6)
ax_bar.set_yticks(ypos)
ax_bar.set_yticklabels(groups, fontsize=TICK)
ax_bar.set_xlabel('Attention Weight (%)', fontsize=LABEL)
ax_bar.set_xlim(0, 55)
ax_bar.tick_params(axis='x', labelsize=TICK)
for bar, w in zip(bars, weights):
    ax_bar.text(w + 0.8, bar.get_y() + bar.get_height() / 2,
                f'{w:.1f}%', va='center', ha='left',
                fontsize=BASE, fontweight='bold')

# ── 右：纯文本注解（无轴） ──
ax_txt.axis('off')
row_h = 1.0 / len(groups)
for i, (var, role) in enumerate(ann_lines):
    y = 1.0 - (i + 0.5) * row_h
    ax_txt.text(0.05, y, var,  transform=ax_txt.transAxes,
                va='center', ha='left', fontsize=BASE,
                fontweight='bold', color=colors_att[i])
    ax_txt.text(0.05, y - row_h * 0.38, role, transform=ax_txt.transAxes,
                va='center', ha='left', fontsize=TICK - 1,
                color='#555555', style='italic')

plt.tight_layout(pad=1.8)
plt.savefig(f'{OUT}/attention_weights.png', bbox_inches='tight')
plt.close()
print('✅ attention_weights.png')


# ════════════════════════════════════════════════════════════
# 4. 模型预测相关性热图
# ════════════════════════════════════════════════════════════
corr_df = pd.read_csv('results/summary/model_correlation_20260308_101152.csv', index_col=0)

short = {
    'Linear Regression': 'Linear',
    'Ridge Regression': 'Ridge',
    'Lasso Regression': 'Lasso',
    'Elastic Net': 'Elastic\nNet',
    'Random Forest': 'Rand.\nForest',
    'Extra Trees': 'Extra\nTrees',
    'Gradient Boosting': 'Grad.\nBoost',
    'Support Vector Regression': 'SVR',
    'K-Nearest Neighbors': 'KNN',
    'Decision Tree': 'Dec.\nTree',
    'Neural Network': 'Neural\nNet',
    'Explainable Model': 'Explainable\nModel',
    'KAN Model': 'KAN',
    'TabNet': 'TabNet',
}
corr_df.index   = [short.get(x, x) for x in corr_df.index]
corr_df.columns = [short.get(x, x) for x in corr_df.columns]

fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr_df, dtype=bool), k=1)
sns.heatmap(corr_df, ax=ax, annot=True, fmt='.2f', cmap='RdYlGn',
            vmin=-0.2, vmax=1.0, linewidths=0.4, linecolor='white',
            annot_kws={'size': 9}, mask=mask,
            cbar_kws={'shrink': 0.8, 'label': 'Pearson r'})
ax.set_xticklabels(ax.get_xticklabels(), fontsize=TICK, rotation=45, ha='right')
ax.set_yticklabels(ax.get_yticklabels(), fontsize=TICK, rotation=0)
plt.tight_layout(pad=1.5)
plt.savefig(f'{OUT}/model_correlation.png', bbox_inches='tight')
plt.close()
print('✅ model_correlation.png')

print('\n所有图片已保存至 manuscript/figure/')
