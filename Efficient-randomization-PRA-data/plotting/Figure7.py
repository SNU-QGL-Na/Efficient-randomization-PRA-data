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

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator

mpl.rcParams["font.family"] = 'Arial'
mpl.rcParams["mathtext.fontset"] = 'stix'

fontsize=8
fontsize_tick = 7
fontsize_legend = 6
subplotposXa = -0.17
subplotposYa = 1
subplotposXbc = -0.145
subplotposYbc = 1.0
markersize1 = 3
markerstar = 10

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

col = 'purple'

D_real = 800
Es = 45
Emax_plot = 1.65; Emin_plot = 0.0

file_path = folder_data + "/Figure7/shared_LE_summary.xlsx"
df = pd.read_excel(file_path).sort_values("D")

D_all   = df["D"].values
LE_all = df["Average_LE"].values; LE_err_all = df["Std_LE"].values

E_all = df["spin_energy"].values
Eb = E_all[D_all == D_real]
LEb = LE_all[D_all == D_real]
LEb_err = LE_err_all[D_all == D_real]

highlight_idx = 6
idx_all = df["idx"].values
mask_b = np.isclose(D_all, D_real)
idxb = idx_all[mask_b]
idx_circle = np.where(idxb == highlight_idx)[0]

idx_b_circle = idx_circle[0]

excel_path = folder_data + "/Figure7/data_Fig_5b.xlsx"
df = pd.read_excel(excel_path)

time = df["t"].values
theta_m_state0 = df["m_state0"].values

excel_path = folder_data + '/Figure7/data_Fig_5c.xlsx'
df = pd.read_excel(excel_path)

Lyapunov_exp_arr = df["Lyapunov_exp"].to_numpy()

Cumulative_Lyapunov_exp = []
for i in range(len(Lyapunov_exp_arr)):
    Cumulative_Lyapunov_exp.append(
        np.sum(Lyapunov_exp_arr[(i+1)//2 : i+1]) /
        (i + 1 - (i + 1)//2)
    )
Cumulative_Lyapunov_exp = np.array(Cumulative_Lyapunov_exp)

avg_val = np.mean(Lyapunov_exp_arr[(i+1)//2 : i+1])

figure_size = (178/25.4,50/25.4)

l = [144,144,144]
lb = [24,33,30,4]

fig, axes = plt.subplot_mosaic([
    ['blank0']*(sum(l)+sum(lb)),
    ['blank01']*(lb[0]+l[0]+lb[1]+l[1]+lb[2])+['c']*l[2]+['blank6']*lb[3],
    ['blank1']*lb[0]+['a']*l[0]+['blank2']*lb[1]+['b']*l[1]+['blank3']*lb[2]+['c']*l[2]     +['blank6']*lb[3],
    ['blank1']*lb[0]+['a']*l[0]+['blank2']*lb[1]+['b']*l[1]+['blank3']*lb[2]+['blank5']*l[2]+['blank6']*lb[3],
    ['blank1']*lb[0]+['a']*l[0]+['blank2']*lb[1]+['b']*l[1]+['blank3']*lb[2]+['d']*l[2]     +['blank6']*lb[3],
    ['blank02']*(lb[0]+l[0]+lb[1]+l[1]+lb[2])+['d']*l[2]+['blank6']*lb[3],
    ['blank7']*(sum(l)+sum(lb))],
    constrained_layout=False, figsize=figure_size, gridspec_kw={'height_ratios':[0.03,0.23,0.77,0.06,0.77,0.23,0.315], 'hspace':0,'wspace':0})
plt.subplots_adjust(left=0, top=1, bottom=0, right=1)

ax = axes['blank0']; ax.axis('off');
ax = axes['blank01']; ax.axis('off');
ax = axes['blank02']; ax.axis('off');
ax = axes['blank1']; ax.axis('off');
ax = axes['blank2']; ax.axis('off');
ax = axes['blank3']; ax.axis('off');

ax = axes['blank5']; ax.axis('off');
ax = axes['blank6']; ax.axis('off');
ax = axes['blank7']; ax.axis('off');

ax = axes['a']

ax.scatter(D_all/Es, LE_all/Es, color='black', s=5, linewidths=0, alpha=0.1, zorder = 10)
ax.vlines(D_real/Es, -0.3, 1.8, color = 'mediumseagreen', linewidth = 1.2, linestyles='--', alpha = 1)

ax.set_xlabel(r"$\hbar D_z/\epsilon_s$", labelpad = 0.1)
ax.set_ylabel(r"$\lambda\tau_s$")
ax.set_xlim(-1.5, 20)
ax.set_xticks([0,5,10,15,20], ['0','5','10','15','20'])
ax.set_ylim(-0.3, 1.8)
ax.set_yticks([1.5,1,0.5,0],['1.5','1.0','0.5','0'])
ax.tick_params(axis='y', which='major', pad = 2)
ax.yaxis.set_minor_locator(MultipleLocator(0.5))
ax.xaxis.set_minor_locator(MultipleLocator(1))

xmin, xmax = ax.get_xlim(); xlen = xmax/xmin;
ymin, ymax = ax.get_ylim(); ylen = ymax-ymin

ax.grid(which='both',linewidth=0.15,alpha=0.4)

xmin, xmax = ax.get_xlim(); xlen = xmax-xmin;
ymin, ymax = ax.get_ylim(); ylen = ymax-ymin

ax.text(xmin+xlen*subplotposXa,ymin+ylen*subplotposYa,'(a)',fontsize=fontsize,ha='left',va='top')

ax = axes['b']

ax.scatter(Eb, LEb/Es, s=markersize1, color='darkgreen', zorder=1)
ax.errorbar(Eb, LEb/Es, yerr=LEb_err/Es, linestyle="none", ecolor='darkgreen', elinewidth=0.4, capsize=1.0, capthick=0.4, zorder=3)

ax.scatter(Eb[idx_b_circle], LEb[idx_b_circle]/Es, s=markerstar, color='hotpink', edgecolors = 'purple', linewidth = 0.7, marker = 'D', zorder=5)

ax.set_xlabel(r"$E/\epsilon_s$", labelpad = 1)
ax.set_ylabel(r"$\lambda\tau_s$")
ax.set_xlim(Emin_plot, Emax_plot)
ax.set_ylim(-0.15, 1.1)
ax.set_yticks([0,0.5,1.0],['0','0.5','1.0'])
ax.set_xticks([0, 0.4, 0.8,1.2,1.6],['0','0.4','0.8','1.2','1.6'])
ax.tick_params(axis='y', which='major', pad = 2)
ax.xaxis.set_minor_locator(MultipleLocator(0.2))
ax.yaxis.set_minor_locator(MultipleLocator(0.5))

ax.grid(which='both',linewidth=0.15,alpha=0.4)

xmin, xmax = ax.get_xlim(); xlen = xmax-xmin;
ymin, ymax = ax.get_ylim(); ylen = ymax-ymin;
ax.text(xmin+xlen*subplotposXa,ymin+ylen*subplotposYa,'(b)',fontsize=fontsize,ha='left',va='top')
ax.text(1.1,1.05, rf'$\hbar D_z / \epsilon_s = {800/Es:.1f}$',fontsize=fontsize_tick,ha='left',va='top')

ax = axes['c']

ax.plot(time/(1/Es), theta_m_state0, color=col, linewidth=0.3)
ax.set_xlim(0, 10000)
ax.set_ylim(-1.10, 1.10)
ax.set_ylabel(r'$m$', labelpad = -2)
ax.set_xticks([0,4000,8000],['','',''])
ax.xaxis.set_minor_locator(MultipleLocator(2000))
ax.yaxis.set_minor_locator(MultipleLocator(0.5))

xmin, xmax = ax.get_xlim(); xlen = xmax-xmin;
ymin, ymax = ax.get_ylim(); ylen = ymax-ymin;
ax.text(xmin+xlen*subplotposXbc,ymin+ylen*subplotposYbc,'(c)',fontsize=fontsize,ha='left',va='top')

ax.grid(which='both',linewidth=0.15,alpha=0.4)

ax = axes['d']

ax.plot(np.arange(1,5001)*250/5000/(1/Es), Lyapunov_exp_arr/Es, color=col, alpha=0.15, linewidth=0.7, zorder=1, label='iteration-wise')
ax.plot(np.arange(1,5001)*250/5000/(1/Es), Cumulative_Lyapunov_exp/Es, color=col, linewidth=1.5, label='cumulative', zorder=3)

ax.set_xlabel(r"$t/\tau_s$", labelpad = 1)
ax.set_ylabel(r'$\lambda\tau_s$', labelpad = 2)
ax.set_xlim(0, 10000)
ax.set_ylim(-1, 3.2)
ax.legend(loc='upper right', frameon=False, fontsize=6.5)

ax.set_xticks([0,4000,8000],['0','4000','8000'])
ax.xaxis.set_minor_locator(MultipleLocator(2000))
ax.yaxis.set_minor_locator(MultipleLocator(1))

xmin, xmax = ax.get_xlim(); xlen = xmax-xmin;
ymin, ymax = ax.get_ylim(); ylen = ymax-ymin;
ax.text(xmin+xlen*subplotposXbc,ymin+ylen*subplotposYbc,'(d)',fontsize=fontsize,ha='left',va='top')

ax.grid(which='both',linewidth=0.15,alpha=0.4)

plt.savefig(folder_save+'/Figure7.pdf',dpi=400)
plt.savefig(folder_save+'/Figure7.png',dpi=400)
plt.savefig(folder_save+'/Figure7.eps',dpi=400)
plt.show()
