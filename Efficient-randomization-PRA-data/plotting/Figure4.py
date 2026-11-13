import os as _os
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_ROOT))
_os.chdir(_ROOT)
(_ROOT / "Figures").mkdir(exist_ok=True)

folder_data = "./Data_for_Figure"
folder_save = "./Figures"

import os
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator

mpl.rcParams["font.family"] = 'Arial'
mpl.rcParams["mathtext.fontset"] = 'stix'

fontsize=8
fontsize_tick = 7
fontsize_legend = 6
subplotposXa = -0.127
subplotposYa = 1.0
subplotposXb = -0.12
subplotposYb = 1.03

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

Dmax_data = 200
Dmin = 0.15; Dmax = 300
Es = 45

file = os.path.join(folder_data, "Figure4/shared_LE_summary.xlsx")
df = pd.read_excel(file)

d_plot = [0.3,0.425,0.6,0.87,1.05,1.25,1.5,1.73,2,2.4,2.7,2.89,3,3.5,4,4.5,5.2,6,7,8,9,10.8,13.5,16.5,20,23.5,25.5,27.5,33.5,40,48,60,72,90,108,130,160,200]
d_plot = [0.3,0.425,0.6,0.87,1.25,1.73,2.4,3,4,5.2,7,9,10.8,13.5,16.5,20,23.5,27.5,33.5,40,48,60,72,90,108,130,160,200]

df_a = df[df['D'].isin(d_plot)]

df_b = df[(df['D'] <= Dmax_data) & (df['D'] > 0)]

figure_size = (86/25.4,90/25.4)

l = [144]
lb = [18,12]
fig, axes = plt.subplot_mosaic([
    ['blank0']*(sum(l)+sum(lb)),
    ['blank1']*lb[0]+['a']*l[0]+['blank2']*lb[1],
    ['blank3']*(sum(l)+sum(lb)),
    ['blank4']*lb[0]+['b']*(l[0]+lb[1]-2)+['blank5']*2,
    ['blank6']*(sum(l)+sum(lb))],
    constrained_layout=False, figsize=figure_size, gridspec_kw={'height_ratios':[0.02,1.2,0.22,0.93,0.21], 'hspace':0,'wspace':0})
plt.subplots_adjust(left=0, top=1, bottom=0, right=1)

ax = axes['blank0']; ax.axis('off');
ax = axes['blank1']; ax.axis('off');
ax = axes['blank2']; ax.axis('off');
ax = axes['blank3']; ax.axis('off');
ax = axes['blank4']; ax.axis('off');
ax = axes['blank5']; ax.axis('off');
ax = axes['blank6']; ax.axis('off');

ax = axes['a']

pivot_df = df_b.pivot(index='D', columns='spin_energy', values='Average_LE')

X = pivot_df.columns
Y = pivot_df.index/Es
Z = pivot_df.values/Es

im = ax.pcolormesh(X, Y, Z, shading='auto', cmap='viridis', edgecolor='none', linewidth=0, antialiased=True)

ax.set_yscale('log')
ax.set_xlabel(r"$E/\epsilon_s$", labelpad = 1)
ax.set_ylabel(r"$\hbar D_z/\epsilon_s$", labelpad = -0.5)
ax.tick_params(axis='both', which='major', direction = 'out', pad = 0.5)
ax.tick_params(axis='both', which='minor', direction = 'out')
ax.set_xticks([0.4, 0.8,1.2,1.6],['0.4','0.8','1.2','1.6'])
ax.xaxis.set_minor_locator(MultipleLocator(0.2))

xmin, xmax = ax.get_xlim(); xlen = xmax-xmin;
ymin, ymax = ax.get_ylim(); ylen = ymax/ymin

ax.text(xmin+xlen*subplotposXa,ymin*ylen**subplotposYa,'(a)',fontsize=fontsize,ha='left',va='top')

cbar_pos_left = 1.64; cbar_left = (cbar_pos_left -xmin)/(xmax-xmin);
cbar_pos_right = 1.7; cbar_right = (cbar_pos_right -xmin)/(xmax-xmin);
cbar_pos_bottom = ymin; cbar_bottom = (cbar_pos_bottom -ymin)/(ymax-ymin);
cbar_pos_top = ymax*0.93; cbar_top = (cbar_pos_top -ymin)/(ymax-ymin);
ax_cbar = ax.inset_axes((cbar_left,cbar_bottom,cbar_right-cbar_left,cbar_top-cbar_bottom))
cbar = fig.colorbar(im, ax=[ax], cax=ax_cbar, ticks=[0,1],orientation="vertical")

cbar.ax.set_yticklabels([r'0','1'],fontsize=fontsize_legend)
cbar.ax.yaxis.set_tick_params(length = 2, pad = 1)
cbar.ax.set_xlabel(r' $\lambda\tau_s$', rotation=0,fontsize=fontsize_tick, labelpad=3)
cbar.ax.xaxis.set_label_position('top')

ax = axes['b']

ax.scatter(df_a['D']/Es, df_a['Average_LE']/Es, color='black', s=5, linewidths=0, alpha=0.1)

ax.set_xscale("log")
ax.set_xlabel(r"$\hbar D_z/\epsilon_s$", labelpad = 0.1)
ax.set_ylabel(r"$\lambda\tau_s$")
ax.set_xlim(Dmin/Es, Dmax/Es)
ax.set_ylim(-0.3, 1.8)
ax.set_yticks([1.5,1,0.5,0],['1.5','1.0','0.5','0'])
ax.yaxis.set_minor_locator(MultipleLocator(0.5))

xmin, xmax = ax.get_xlim(); xlen = xmax/xmin;
ymin, ymax = ax.get_ylim(); ylen = ymax-ymin

ax.grid(which='both',linewidth=0.15,alpha=0.8)

xmin, xmax = ax.get_xlim(); xlen = xmax/xmin;
ymin, ymax = ax.get_ylim(); ylen = ymax-ymin

ax.text(xmin*xlen**subplotposXb,ymin+ylen*subplotposYb,'(b)',fontsize=fontsize,ha='left',va='top')

plt.savefig(folder_save + '/Figure4.pdf',dpi=400)
plt.savefig(folder_save + '/Figure4.png',dpi=400)
plt.show()
