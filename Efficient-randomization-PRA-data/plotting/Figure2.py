import os as _os
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_ROOT))
_os.chdir(_ROOT)
(_ROOT / "Figures").mkdir(exist_ok=True)

folder_data = "./Data_for_Figure/Figure2"
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
subplotposXab = -0.098
subplotposYab = 1.0
markersize1 = 2.5
markersize2 = 3.
markerstar = 10.

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

cmap = plt.get_cmap('viridis')

low_color  = cmap(0.)
high_color = cmap(0.8)

colorR = low_color
colorC = high_color

colorCF = 'orange'
colorRF = 'lightskyblue'
colorCE = 'red'
colorRE = 'blue'

E_crit = 0.79
Es = 45
LE_crit = 0.1

df = pd.read_excel(folder_data + "/data_Fig_2b.xlsx")
idx_b = df["idx"].values
x_b    = df["energy"].values
y_b    = df["Average LE"].values/Es
yerr_b = df["Std LE"].values/Es

mask_regular = (y_b < LE_crit)
mask_chaotic = ~mask_regular

x_Cb = x_b[idx_b == 6]
x_Rb = x_b[idx_b == 46]
y_Cb = y_b[idx_b == 6]
y_Rb = y_b[idx_b == 46]

file_path = folder_data + "/data_Fig_2a.txt"
SC = []; EC = []; SR = []; ER = []

data = np.loadtxt(file_path)
E = data[:,1]; S = data[:,2]; I = data[:,0];

maskC = (E >= E_crit) | (S >= -1.5)
maskC = (y_b > LE_crit)
maskR = ~maskC

EC = E[maskC].tolist(); ER = E[maskR].tolist()
SC = S[maskC].tolist(); SR = S[maskR].tolist()

x_Cc = E[I == 6]
x_Rc = E[I == 46]
y_Cc = S[I == 6]
y_Rc = S[I == 46]

figure_size = (86/25.4,86/25.4)

l = [144]
lb = [14,1]
fig, axes = plt.subplot_mosaic([
    ['blank0']*(sum(l)+sum(lb)),
    ['blank1']*lb[0]+['a']*l[0]+['blank2']*lb[1],
    ['blank3']*(sum(l)+sum(lb)),
    ['blank4']*lb[0]+['b']*l[0]+['blank5']*lb[1],
    ['blank6']*(sum(l)+sum(lb))],
    constrained_layout=False, figsize=figure_size, gridspec_kw={'height_ratios':[0.03,1,0.06,1,0.19], 'hspace':0,'wspace':0})
plt.subplots_adjust(left=0, top=1, bottom=0, right=1)

ax = axes['blank0']; ax.axis('off');
ax = axes['blank1']; ax.axis('off');
ax = axes['blank2']; ax.axis('off');
ax = axes['blank3']; ax.axis('off');
ax = axes['blank4']; ax.axis('off');
ax = axes['blank5']; ax.axis('off');
ax = axes['blank6']; ax.axis('off');

ax = axes['a']
Emax_plot = 1.65; Emin_plot = 0.0

ax.errorbar(x_b[mask_regular], y_b[mask_regular], yerr=yerr_b[mask_regular], fmt="o", ms=markersize2, alpha=1,zorder=1,
            mfc=colorR, mec='none', mew=0.5, ecolor=colorR, elinewidth=0.5, capsize=0.5, capthick=0.5, label=r"Regular, $\lambda\tau_s<0.1$")
ax.errorbar(x_b[mask_chaotic], y_b[mask_chaotic], yerr=yerr_b[mask_chaotic], fmt="o", ms=markersize2, alpha=1,zorder=1,
            mfc=colorC, mec='none', mew=0.5, ecolor=colorC, elinewidth=0.5, capsize=0.5, capthick=0.5, label=r"Chaotic, $\lambda\tau_s>0.1$")
ax.scatter(x_Cb,y_Cb, marker='D', s = markerstar, color=colorCF, edgecolors = colorCE, linewidth = 0.7, zorder=7)
ax.scatter(x_Rb,y_Rb, marker='s', s = markerstar, color=colorRF, edgecolors = colorRE, linewidth = 0.7, zorder=6)

ax.vlines(E_crit, -0.4, 2,color = 'gray', linewidth = 1, linestyles='dashed', zorder=0)
ax.hlines(0.1, Emin_plot, Emax_plot,color = colorR, linewidth = 0.5, linestyles='--', zorder=0)
ax.text(0.04,0.12, r'$\lambda\tau_s = 0.1$', ha = 'left', va = 'bottom', fontsize = fontsize_tick)

ax.set_ylabel(r"$\lambda\tau_s$", labelpad= 2)
ax.set_xlim(Emin_plot, Emax_plot)
ax.set_ylim(-0.3, 1.8)
ax.set_yticks([1.5,1,0.5,0],['1.5','1.0','0.5','0'])
ax.set_xticks([0, 0.4, 0.8,1.2,1.6])
ax.set_xticklabels([])
ax.tick_params(axis='y', which='major', pad = 2)
ax.xaxis.set_minor_locator(MultipleLocator(0.2))
ax.yaxis.set_minor_locator(MultipleLocator(0.5))

ax.legend(loc=(0.02, 0.68), frameon=False, handletextpad=0.1, labelspacing=1)

ax.grid(which='both',linewidth=0.15,alpha=0.4)

xmin, xmax = ax.get_xlim(); xlen = xmax-xmin;
ymin, ymax = ax.get_ylim(); ylen = ymax-ymin;
ax.text(xmin+xlen*subplotposXab,ymin+ylen*subplotposYab,'(a)',fontsize=fontsize,ha='left',va='top')

ax = axes['b']

ax.scatter(EC, np.exp(SC), s=markersize1, color=colorC, zorder=1)
ax.scatter(ER,np.exp(SR), s=markersize1, color=colorR, zorder=1)
ax.scatter(x_Cc,np.exp(y_Cc), marker='D', s = markerstar, color=colorCF, edgecolors = colorCE, linewidth = 0.7, zorder=7)
ax.scatter(x_Rc,np.exp(y_Rc), marker='s', s = markerstar, color=colorRF, edgecolors = colorRE, linewidth = 0.7, zorder=6)

ax.vlines(E_crit, -0.1,1.15,color = 'gray', linewidth = 1, linestyles='dashed', zorder=0)

ax.set_xlabel(r"$E/\epsilon_s$", labelpad = 1)
ax.set_ylabel(r"$\mathcal{V}$", labelpad = 2)

ax.set_xlim(Emin_plot, Emax_plot)
ax.set_ylim(-0.03, 0.35)
ax.set_xticks([0, 0.4, 0.8,1.2,1.6],['0','0.4','0.8','1.2','1.6'])
ax.set_yticks([0, 0.1, 0.2, 0.3],['0','0.1','0.2','0.3'])
ax.tick_params(axis='y', which='major', pad = 2)
ax.xaxis.set_minor_locator(MultipleLocator(0.2))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))

ax.grid(which='both',linewidth=0.15,alpha=0.4)

xmin, xmax = ax.get_xlim(); xlen = xmax-xmin;
ymin, ymax = ax.get_ylim(); ylen = ymax-ymin;
ax.text(xmin+xlen*subplotposXab,ymin+ylen*subplotposYab,'(b)',fontsize=fontsize,ha='left',va='top')

plt.savefig(folder_save + '/Figure2.pdf',dpi=400)
plt.savefig(folder_save + '/Figure2.png',dpi=400)
plt.show()
