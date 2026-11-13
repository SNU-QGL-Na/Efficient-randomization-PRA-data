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
subplotposXab = -0.119
subplotposYab = 1.0
markersize = 5

mpl.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,

    "font.size": fontsize,
    "axes.labelsize": fontsize,
    "xtick.labelsize": fontsize_tick,
    "ytick.labelsize": fontsize_tick,
    "legend.fontsize": fontsize_tick,

    "xtick.direction": "in",
    "ytick.direction": "in",
})

colorCF = 'orange'
colorRF = 'lightskyblue'
colorCE = 'red'
colorRE = 'blue'

Es = 45
Dmax_data = 200
Dmin = 0.15; Dmax = 300

fileC = folder_data + "/Figure3/data_chaotic_Fig3.txt"
fileR = folder_data + "/Figure3/data_regular_Fig3.txt"

dataC = np.loadtxt(fileC)
dataR = np.loadtxt(fileR)
D_list = dataC[:,0]
S_C_all = dataC[:,1]; S_R_all = dataR[:,1]

mask_lin = D_list <= Dmax_data
D_linb   = D_list[mask_lin]/Es
S_C_lin  = S_C_all[mask_lin]
S_R_lin  = S_R_all[mask_lin]

file_path = folder_data + "/Figure3/data_Fig_3c.xlsx"
df = pd.read_excel(file_path).sort_values("D")

D_all   = df["D"].values/Es
LE_C_all = df["avg_LE_zdrive_1"].values/Es; LE_C_err_all = df["cum_LE_std_1"].values/Es
LE_R_all = df["avg_LE_zdrive_2"].values/Es; LE_R_err_all = df["cum_LE_std_2"].values/Es

mask_log = (D_all > 0) & (D_all <= Dmax_data/Es)
D_log    = D_all[mask_log]
LE_C_log = LE_C_all[mask_log]; LE_C_err_log = LE_C_err_all[mask_log]
LE_R_log = LE_R_all[mask_log]; LE_R_err_log = LE_R_err_all[mask_log]

figure_size = (86/25.4,86/25.4)

l = [144]
lb = [17,1]
fig, axes = plt.subplot_mosaic([
    ['blank0']*(sum(l)+sum(lb)),
    ['blank1']*lb[0]+['a']*l[0]+['blank2']*lb[1],
    ['blank3']*(sum(l)+sum(lb)),
    ['blank4']*lb[0]+['b']*l[0]+['blank5']*lb[1],
    ['blank6']*(sum(l)+sum(lb))],
    constrained_layout=False, figsize=figure_size, gridspec_kw={'height_ratios':[0.032,1,0.06,1,0.25], 'hspace':0,'wspace':0})
plt.subplots_adjust(left=0, top=1, bottom=0, right=1)

ax = axes['blank0']; ax.axis('off');
ax = axes['blank1']; ax.axis('off');
ax = axes['blank2']; ax.axis('off');
ax = axes['blank3']; ax.axis('off');
ax = axes['blank4']; ax.axis('off');
ax = axes['blank5']; ax.axis('off');
ax = axes['blank6']; ax.axis('off');

ax = axes['a']

ax.errorbar(D_log, LE_C_log, yerr=LE_C_err_log, linestyle="none", ecolor=colorCE, elinewidth=0.7, capsize=1.0, capthick=0.4, zorder=3)
ax.errorbar(D_log, LE_R_log, yerr=LE_R_err_log, linestyle="none", ecolor=colorRE, elinewidth=0.7, capsize=1.0, capthick=0.4, zorder=5)
ax.scatter(D_log, LE_R_log, marker='s', s = markersize, color=colorRF, edgecolors = colorRE, linewidth = 0.7, label="Regular", zorder=6)
ax.scatter(D_log, LE_C_log, marker='D', s = markersize, color=colorCF, edgecolors = colorCE, linewidth = 0.7,  label="Chaotic", zorder=7)

ax.set_xscale("log")
ax.set_xlabel(''); ax.set_xticklabels([])
ax.set_ylabel(r"$\lambda\tau_s$")
ax.set_xlim(Dmin/Es, Dmax/Es)
ax.set_ylim(-0.2, 1.5)
ax.yaxis.set_minor_locator(MultipleLocator(0.5))
ax.set_yticks([0, 0.5, 1,1.5])

xmin, xmax = ax.get_xlim(); xlen = xmax/xmin;
ymin, ymax = ax.get_ylim(); ylen = ymax-ymin

ax.text(xmin*xlen**subplotposXab,ymin+ylen*subplotposYab,'(a)',fontsize=fontsize,ha='left',va='top')

ax.legend(loc=(0.75, 0.12), frameon=False, handletextpad=0.1, labelspacing=1)

ax.grid(which='both',linewidth=0.15,alpha=0.4)

ax = axes['b']

ax.hlines(1, Dmin/Es, Dmax/Es, color = 'gray', linewidth = 1, linestyles='dashed', zorder=0)
ax.scatter(D_linb, np.exp(S_R_lin), marker='s', s = markersize, color=colorRF, edgecolors = colorRE, linewidth = 0.7, label="regular", zorder=1)
ax.scatter(D_linb, np.exp(S_C_lin), marker='D', s = markersize, color=colorCF, edgecolors = colorCE, linewidth = 0.7,  label="chaotic", zorder=2)

ax.set_xlabel(r"$\hbar D_z/\epsilon_s$")
ax.set_ylabel(r"$\mathcal{V}$")
ax.set_xlim(Dmin/Es, Dmax/Es)
ax.set_xscale("log")
ax.set_ylim(-0.1, 1.15)
ax.set_yticks([0, 0.5, 1],['0','0.5','1.0'])
ax.yaxis.set_minor_locator(MultipleLocator(0.25))
ax.set_xticks([0.01,0.1,1])

xmin, xmax = ax.get_xlim(); xlen = xmax/xmin;
ymin, ymax = ax.get_ylim(); ylen = ymax-ymin;
ax.text(xmin*xlen**subplotposXab,ymin+ylen*subplotposYab,'(b)',fontsize=fontsize,ha='left',va='top')

ax.grid(which='both',linewidth=0.15,alpha=0.4)

plt.savefig(folder_save+'/Figure3.pdf',dpi=400)
plt.savefig(folder_save+'/Figure3.png',dpi=400)
plt.show()
