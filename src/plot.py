import matplotlib.pyplot as plt
import numpy as np
import matplotlib
# matplotlib.use("Qt5Agg")
# ----------- 解决中文字体在 Ubuntu 上显示问题 -----------
matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']  
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 14

# ------------------ 数据（请替换） ------------------
models = ["FinalMLP", "DCN", "FM"]

mindctr_auc = [0.6307, 0.6167, 0.6107]
movielens_auc = [0.7815, 0.7820, 0.7733]

mindctr_logloss = [0.1661, 0.1635, 0.1605]
movielens_logloss = [0.5671, 0.5568, 0.5711]

# ------------------ 颜色：浅蓝 - 蓝 - 深蓝 ------------------
colors = ["#8ecae6", "#219ebc", "#023047"]

# 大画布
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

x = np.arange(len(models))
width = 0.5

# --- 图 1 ---
bars = axs[0, 0].bar(x, mindctr_auc, width, color=colors)
axs[0, 0].set_xticks(x)
axs[0, 0].set_xticklabels(models)
axs[0, 0].set_ylabel("AUC", fontsize=16)
axs[0, 0].set_title("MINDCTR", fontsize=18)
axs[0, 0].bar_label(bars, fmt='%.3f')
axs[0, 0].set_ylim(0, max(mindctr_auc) * 1.15)

# --- 图 2 ---
bars = axs[0, 1].bar(x, movielens_auc, width, color=colors)
axs[0, 1].set_xticks(x)
axs[0, 1].set_xticklabels(models)
axs[0, 1].set_ylabel("AUC", fontsize=16)
axs[0, 1].set_title("MovieLens_1mimpCTRAll", fontsize=18)
axs[0, 1].bar_label(bars, fmt='%.3f')
axs[0, 1].set_ylim(0, max(movielens_auc) * 1.15)

# --- 图 3 ---
bars = axs[1, 0].bar(x, mindctr_logloss, width, color=colors)
axs[1, 0].set_xticks(x)
axs[1, 0].set_xticklabels(models)
axs[1, 0].set_ylabel("Logloss", fontsize=16)
axs[1, 0].set_title("MINDCTR", fontsize=18)
axs[1, 0].bar_label(bars, fmt='%.3f')
axs[1, 0].set_ylim(0, max(mindctr_logloss) * 1.15)

# --- 图 4 ---
bars = axs[1, 1].bar(x, movielens_logloss, width, color=colors)
axs[1, 1].set_xticks(x)
axs[1, 1].set_xticklabels(models)
axs[1, 1].set_ylabel("Logloss", fontsize=16)
axs[1, 1].set_title("MovieLens_1mimpCTRAll", fontsize=18)
axs[1, 1].bar_label(bars, fmt='%.3f')
axs[1, 1].set_ylim(0, max(movielens_logloss) * 1.15)

plt.tight_layout()
plt.show()
