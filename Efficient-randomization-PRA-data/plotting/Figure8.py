import os as _os
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_ROOT))
_os.chdir(_ROOT)
(_ROOT / "Figures").mkdir(exist_ok=True)

folder_data = "./Data_for_Figure"
folder_save = "./Figures"

import pandas as pd
import numpy as np
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator

mpl.rcParams["font.family"] = 'Arial'
mpl.rcParams["mathtext.fontset"] = 'stix'

G = 45.0
TAU_S = 1.0 / G
TIME_TO_T_OVER_TAUS = 1.0 / TAU_S

fontsize=8
fontsize_tick = 7
fontsize_legend = 6
fontsize_text = fontsize_legend

mpl.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,

    "font.size": fontsize,
    "axes.labelsize": fontsize,
    "xtick.labelsize": fontsize_tick,
    "ytick.labelsize": fontsize_tick,
    "legend.fontsize": fontsize_legend,

    "xtick.direction": "in",
    "ytick.direction": "in",
})

colorR = 'blue'
colorC = 'red'

idx_list = [6, 46]

excel_path_dist = folder_data + '/Figure8/data_Fig_1b.xlsx'
df_b = pd.read_excel(excel_path_dist)
N_start = 86; N_end = 90
df_b_use = df_b[(df_b["iteration"] >= N_start) & (df_b["iteration"] <= N_end)]

df_b_C = df_b_use[df_b_use["idx"] == idx_list[0]].copy()
df_b_C["t_rel"] = df_b_C["time_s"] - 0.05 * (N_start - 1)
df_b_C["t_over_taus"] = df_b_C["t_rel"] * TIME_TO_T_OVER_TAUS
df_b_C["log_d"] = (df_b_C["log10_d"] + 6) * np.log(10)

grouped = [group for _, group in df_b_C.groupby("iteration")]
data_with_gaps = []
for group in grouped:
    data_with_gaps.append(group)
    nan_row = pd.DataFrame(np.nan, index=[0], columns=group.columns)
    data_with_gaps.append(nan_row)

df_b_plot_C = pd.concat(data_with_gaps, ignore_index=True)

df_b_R = df_b_use[df_b_use["idx"] == idx_list[1]].copy()
df_b_R["t_rel"] = df_b_R["time_s"] - 0.05 * (N_start - 1)
df_b_R["t_over_taus"] = df_b_R["t_rel"] * TIME_TO_T_OVER_TAUS
df_b_R["log_d"] = (df_b_R["log10_d"] + 6) * np.log(10)

grouped = [group for _, group in df_b_R.groupby("iteration")]
data_with_gaps = []
for group in grouped:
    data_with_gaps.append(group)
    nan_row = pd.DataFrame(np.nan, index=[0], columns=group.columns)
    data_with_gaps.append(nan_row)

df_b_plot_R = pd.concat(data_with_gaps, ignore_index=True)

excel_path = folder_data + '/Figure8/data_Fig_1c.xlsx'
df_c = pd.read_excel(excel_path)

sub_C = df_c[df_c["idx"] == idx_list[0]].copy()
iteration_C = sub_C["iteration"].to_numpy()
cum_LE_C = sub_C["lambda_cumavg_Hz"].to_numpy()
lambda_iter_C = sub_C["lambda_Hz"].to_numpy()

sub_R = df_c[df_c["idx"] == idx_list[1]].copy()
iteration_R = sub_R["iteration"].to_numpy()
cum_LE_R = sub_R["lambda_cumavg_Hz"].to_numpy()
lambda_iter_R = sub_R["lambda_Hz"].to_numpy()

N_iteration = 2000

ITER_TO_T_OVER_TAUS = 9.0 / 4.0
t_over_taus_C = iteration_C * ITER_TO_T_OVER_TAUS
t_over_taus_R = iteration_R * ITER_TO_T_OVER_TAUS
T_OVER_TAUS_MAX = N_iteration * ITER_TO_T_OVER_TAUS

lambda_iter_C_valid = lambda_iter_C[N_iteration // 2:]
lambda_iter_R_valid = lambda_iter_R[N_iteration // 2:]
lambda_iter_C_valid = lambda_iter_C_valid[np.isfinite(lambda_iter_C_valid)]
lambda_iter_R_valid = lambda_iter_R_valid[np.isfinite(lambda_iter_R_valid)]

N_C = len(lambda_iter_C_valid)
N_R = len(lambda_iter_R_valid)

avg_LE_C = np.mean(lambda_iter_C_valid)
avg_LE_R = np.mean(lambda_iter_R_valid)

se_LE_C = np.std(lambda_iter_C_valid, ddof=1) / np.sqrt(N_C) if N_C > 1 else np.nan
se_LE_R = np.std(lambda_iter_R_valid, ddof=1) / np.sqrt(N_R) if N_R > 1 else np.nan

figure_size = (86/25.4,38/25.4)

l = [72,72]
lb = [13,17,1]

fig, axes = plt.subplot_mosaic([
    ['blank0']*(sum(l)+sum(lb)),
    ['blank1']*lb[0]+['b']*l[0]+['blank2']*lb[1]+['c']*l[1]+['blank3']*lb[2],
    ['blank4']*(sum(l)+sum(lb))],
    constrained_layout=False, figsize=figure_size, gridspec_kw={'height_ratios':[0.01,1,0.205], 'hspace':0,'wspace':0})
plt.subplots_adjust(left=0, top=1, bottom=0, right=1)

ax = axes['blank0']; ax.axis('off')
ax = axes['blank1']; ax.axis('off')
ax = axes['blank2']; ax.axis('off')
ax = axes['blank3']; ax.axis('off')
ax = axes['blank4']; ax.axis('off')

subplotposXa = -0.182
subplotposYa = 1
subplotposXb = -0.2
subplotposYb = 1.0

ax = axes['b']

y0 = 0
ax.axhline(y=y0, color='k', linestyle='--', linewidth=0.8)

ax.plot(df_b_C["t_over_taus"], df_b_C["log_d"], color=colorC, linewidth=0.6, linestyle=':', label='chaotic')
ax.plot(df_b_R["t_over_taus"], df_b_R["log_d"], color=colorR, linewidth=0.6, linestyle=':', label='regular')

ax.plot(df_b_plot_C["t_over_taus"], df_b_plot_C["log_d"], color=colorC, linewidth=0.8, alpha=1, label='chaotic')
ax.plot(df_b_plot_R["t_over_taus"], df_b_plot_R["log_d"], color=colorR, linewidth=0.8, alpha=1, label='regular')

ax.set_xlabel(r'$t/\tau_s$', labelpad=1)
ax.set_ylabel(r'$\ln [d(t)/d(0)]$', labelpad=2)

x_min_a = min(df_b_plot_C["t_over_taus"].min(), df_b_plot_R["t_over_taus"].min())
x_max_a = max(df_b_plot_C["t_over_taus"].max(), df_b_plot_R["t_over_taus"].max())
ax.set_xlim(x_min_a - 0.5, x_max_a + 0.5)

ax.set_ylim(-1.5, 5.5)
ax.xaxis.set_major_locator(MultipleLocator(4))
ax.xaxis.set_minor_locator(MultipleLocator(2))
ax.yaxis.set_minor_locator(MultipleLocator(1))

ax.grid(which='both', linewidth=0.15, alpha=0.4)

xmin, xmax = ax.get_xlim(); xlen = xmax-xmin;
ymin, ymax = ax.get_ylim(); ylen = ymax-ymin

ax.text(xmin+xlen*subplotposXa,ymin+ylen*subplotposYa,'(a)',fontsize=fontsize,ha='left',va='top')

ax = axes['c']

ax.plot(t_over_taus_C, lambda_iter_C/G, color=colorC, alpha=0.15, linewidth=0.7, zorder=1)
ax.plot(t_over_taus_R, lambda_iter_R/G, color=colorR, alpha=0.15, linewidth=0.7, zorder=1)

ax.plot(t_over_taus_C, cum_LE_C/G, color=colorC, linewidth=1., label='chaotic', zorder=3)
ax.plot(t_over_taus_R, cum_LE_R/G, color=colorR, linewidth=1., label='regular', zorder=3)

x_text_C = T_OVER_TAUS_MAX - 80 * ITER_TO_T_OVER_TAUS
y_text_C = cum_LE_C[N_iteration - 1] / G + 4.3 / G

x_text_R = T_OVER_TAUS_MAX - 80 * ITER_TO_T_OVER_TAUS
y_text_R = cum_LE_R[N_iteration - 1] / G + 4.3 / G

avg_R_scaled = (avg_LE_R / G) * 1e3
se_R_scaled = (se_LE_R / G) * 1e3

ax.text(
    x_text_C, y_text_C,
    rf'$\lambda \tau_s = {avg_LE_C / G:.2f} \pm {se_LE_C / G:.2f}$',
    color="black", fontsize=fontsize_text,
    ha='right', va='bottom',
    bbox=dict(facecolor='none', edgecolor='none', alpha=0.8, boxstyle='round,pad=0.15')
)

ax.text(
    x_text_R, y_text_R,
    rf'$\lambda \tau_s = ({avg_R_scaled:.0f} \pm {se_R_scaled:.0f}) \times 10^{{-3}}$',
    color="black", fontsize=fontsize_text,
    ha='right', va='bottom',
    bbox=dict(facecolor='none', edgecolor='none', alpha=0.8, boxstyle='round,pad=0.15')
)

ax.set_xlabel(r'$t/\tau_s$', labelpad=1)
ax.set_ylabel(r'$\lambda \tau_s$', labelpad=-1)
ax.set_xlim(0, T_OVER_TAUS_MAX)

ax.set_ylim(-1.2, 3.8)

ax.xaxis.set_major_locator(MultipleLocator(2000))
ax.xaxis.set_minor_locator(MultipleLocator(1000))
ax.yaxis.set_major_locator(MultipleLocator(1.0))

ax.grid(which='both', linewidth=0.15, alpha=0.4)

xmin, xmax = ax.get_xlim(); xlen = xmax-xmin;
ymin, ymax = ax.get_ylim(); ylen = ymax-ymin

ax.text(xmin+xlen*subplotposXb,ymin+ylen*subplotposYb,'(b)',fontsize=fontsize,ha='left',va='top')

os.makedirs(folder_save, exist_ok=True)
plt.savefig(folder_save + '/Figure8.pdf',dpi=400)
plt.savefig(folder_save + '/Figure8.png',dpi=400)
plt.show()
