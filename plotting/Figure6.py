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
import glob
import pandas as pd
import numpy as np
import re

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Rectangle

mpl.rcParams["font.family"] = 'Arial'
mpl.rcParams["mathtext.fontset"] = 'stix'

def analyze_trace_distance(file_path, Tf, dT, dc):
    try:
        df = pd.read_csv(file_path)
        t_col = df.columns[0]
        val_col = df.columns[1]
        df = df.sort_values(by=t_col)
        times = df[t_col].values
        values = df[val_col].values
        mask_d0 = (times >= (Tf - dT)) & (times <= Tf)

        if np.any(mask_d0):
            selected_vals = values[mask_d0]
            d0 = np.mean(selected_vals)
            delta_d0 = np.std(selected_vals)
        else:
            if times[-1] < Tf:
                d0 = values[-1]
            else:
                d0 = np.interp(Tf, times, values)
            delta_d0 = 0.0

        target = dc
        tau = np.inf
        tau_error = np.nan

        if values[0] <= target:
             tau = 0.0
             tau_error = 0.0
        else:
            for i in range(len(values) - 1):

                if times[i+1] > Tf:
                    break
                y1, y2 = values[i], values[i+1]

                if (y1 - target) * (y2 - target) <= 0:
                    t1, t2 = times[i], times[i+1]

                    if y2 != y1:
                        slope = (y2 - y1) / (t2 - t1)
                        tau = t1 + (target - y1) / slope
                    else:
                        tau = t1

                    tau_error = (t2 - t1) / 2
                    break

        filename = os.path.basename(file_path)
        d_match = re.search(r'D=([\d\.]+)', filename)
        init_match = re.search(r'initial(\d+)', filename)

        D_val = float(d_match.group(1)) if d_match else np.nan
        init_val = int(init_match.group(1)) if init_match else np.nan

        return {
            'filename': filename, 'D': D_val, 'initial': init_val,
            'd0': d0, 'delta_d0': delta_d0, 'tau': tau, 'tau_error': tau_error,
            'data_times': times, 'data_values': values
        }

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

fontsize=8
fontsize_tick = 7
fontsize_legend = 6
subplotposXab = -0.147
subplotposYab = 1
subplotposXc = -0.153
subplotposYc = 1.0
markersize = 5

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
colorCF = 'orange'
colorRF = 'lightskyblue'
colorCE = 'red'
colorRE = 'blue'

Es = 45
Dmax_data = 850
Dmin = -50; Dmax = 870

initC = 6; initR = 46
FOLDER_PATH = folder_data + '/Figure6/csvFiles'
N_en = 128**2
dcrit = 1/np.sqrt(N_en)*1
dmin = 1/np.sqrt(N_en)
Tf = 2; dT = 0.1
D0 = 200; D1 = 420.93520018893713; D2 = 700;

search_pattern = os.path.join(FOLDER_PATH, '*.csv')
files = glob.glob(search_pattern)
results = []

for f in files:
    if 'D=' in f and 'initial' in f:
        res = analyze_trace_distance(f,Tf, dT,dcrit)
        if res:
            results.append(res)
df_results = pd.DataFrame(results)

target_subset_D0 = [r for r in results if r['D'] == D0]
target_subset_D1 = [r for r in results if r['D'] == D1]
target_subset_D2 = [r for r in results if r['D'] == D2]

target_subset_D0.sort(key=lambda x: x['initial'])

for item in target_subset_D0:
    if item['initial'] == initC: timesC_D0 = item['data_times']*Es; valuesC_D0 = item['data_values']; tau_valC_D0 = item['tau']
    elif item['initial'] == initR: timesR_D0 = item['data_times']*Es; valuesR_D0 = item['data_values']; tau_valR_D0 = item['tau']
for item in target_subset_D1:
    if item['initial'] == initC: timesC_D1 = item['data_times']*Es; valuesC_D1 = item['data_values']; tau_valC_D1 = item['tau']
    elif item['initial'] == initR: timesR_D1 = item['data_times']*Es; valuesR_D1 = item['data_values']; tau_valR_D1 = item['tau']
for item in target_subset_D2:
    if item['initial'] == initC: timesC_D2 = item['data_times']*Es; valuesC_D2 = item['data_values']; tau_valC_D2 = item['tau']
    elif item['initial'] == initR: timesR_D2 = item['data_times']*Es; valuesR_D2 = item['data_values']; tau_valR_D2 = item['tau']

subsetC = df_results[df_results['initial'] == initC].sort_values('D')
subsetR = df_results[df_results['initial'] == initR].sort_values('D')

D = subsetC['D'].values
d0_C = subsetC['d0'].values
d0_R = subsetR['d0'].values
stdd0_C = subsetC['delta_d0'].values
stdd0_R = subsetR['delta_d0'].values

D_exclude = [1.25,1.5,1,2,3,4,6,7,8,12,17.5,22.5,27.5,35,45,55,60,65,70,80,85,90,95,105,110,115,130,140,160,170]
mask = (D < Dmax_data) & (~np.isin(D, D_exclude))
D_exclude2 = [1.25,1.5,1,2,3,4,6,7,8,12,17.5,22.5,27.5,35,45,55,60,65,70,80,85,90,95,105,110,115,130,140,160,170,205,215,225,230,235,245,405,415,425,435,445,595,605,615,625,635,785,795,805,815,825]
mask2 = (D < Dmax_data) & (~np.isin(D, D_exclude2))

dmax = 5/6;

D_plot = D[mask] / Es

r_C_plot = np.log(dmax / d0_C[mask]) / np.log(dmax / dmin)
r_R_plot = np.log(dmax / d0_R[mask]) / np.log(dmax / dmin)
stdr_C_plot = stdd0_C[mask] * 1/d0_C[mask] / np.log(dmax / dmin)
stdr_R_plot = stdd0_R[mask] * 1/d0_R[mask] / np.log(dmax / dmin)

D_plot2 = D[mask2] / Es

tau_C = subsetC['tau'].values; tau_C_err = subsetC['tau_error'].values
tau_R = subsetR['tau'].values; tau_R_err = subsetR['tau_error'].values

inv_tau_C = 1 / tau_C / Es; inv_tau_C_err = 1 / (tau_C**2) * tau_C_err / Es
inv_tau_R = 1 / tau_R / Es; inv_tau_R_err = 1 / (tau_R**2) * tau_R_err / Es

inv_tau_C_plot = inv_tau_C[mask2]; inv_tau_C_err_plot = inv_tau_C_err[mask2]
inv_tau_R_plot = inv_tau_R[mask2]; inv_tau_R_err_plot = inv_tau_R_err[mask2]

figure_size = (148/25.4,58/25.4)

l = [144,144]
lb = [21,32,3]

fig, axes = plt.subplot_mosaic([
    ['blank0']*(sum(l)+sum(lb)),
    ['blank1']*lb[0]+['a']*l[0]+['blank2']*lb[1]+['c']*l[1]    +['blank3']*lb[2] ,
    ['blank1']*lb[0]+['blank4']*l[0]+['blank2']*lb[1]+['c']*l[1]+['blank3']*lb[2],
    ['blank1']*lb[0]+['b']*l[0]+['blank2']*lb[1]+['c']*l[1]    +['blank3']*lb[2],
    ['blank5']*(sum(l)+sum(lb))],
    constrained_layout=False, figsize=figure_size, gridspec_kw={'height_ratios':[0.07,1,0.06,1,0.35], 'hspace':0,'wspace':0})
plt.subplots_adjust(left=0, top=1, bottom=0, right=1)

ax = axes['blank0']; ax.axis('off');
ax = axes['blank1']; ax.axis('off');
ax = axes['blank2']; ax.axis('off');
ax = axes['blank3']; ax.axis('off');
ax = axes['blank4']; ax.axis('off');
ax = axes['blank5']; ax.axis('off');

ax = axes['a']

ax.errorbar(D_plot, r_C_plot, yerr=stdr_C_plot, linestyle="none", ecolor=colorC, elinewidth=0.4, capsize=1.0, capthick=0.4, zorder=3)
ax.errorbar(D_plot, r_R_plot, yerr=stdr_R_plot, linestyle="none", ecolor=colorR, elinewidth=0.4, capsize=1.0, capthick=0.4, zorder=3)

ax.scatter(D_plot, r_C_plot, marker='D', s = markersize, color=colorCF, edgecolors = colorCE, linewidth = 0.7,  label="Chaotic", zorder=7)
ax.scatter(D_plot, r_R_plot, marker='s', s = markersize, color=colorRF, edgecolors = colorRE, linewidth = 0.7, label="Regular", zorder=6)

ax.hlines(1, Dmin/Es, Dmax/Es, color = 'gray', linewidth = 1, linestyles='dashed')
ax.hlines(0, Dmin/Es, Dmax/Es, color = 'gray', linewidth = 1, linestyles='dotted')
ax.vlines(D1/Es, -0.2, 2.5, color = 'mediumseagreen', linewidth = 1.2, linestyles='--', alpha = 1)
ax.vlines(D2/Es, -0.2, 2.5, color = 'mediumseagreen', linewidth = 1.2, linestyles='-', alpha = 1)

ax.set_xlabel('')
ax.set_ylabel(r'$\mathcal{R}$', labelpad = 3)
ax.set_xticks([0,5,10,15,20])
ax.set_xticklabels([])

ax.set_xlim(-1.5, 20)
ax.set_ylim(-0.1, 1.2)
ax.xaxis.set_minor_locator(MultipleLocator(1))

xmin, xmax = ax.get_xlim(); xlen = xmax-xmin;
ymin, ymax = ax.get_ylim(); ylen = ymax-ymin;

ax.text(xmin+xlen*subplotposXab,ymin+ylen*subplotposYab,'(a)',fontsize=fontsize,ha='left',va='top')

ax.grid(which='both',linewidth=0.15,alpha=0.4)

ax = axes['b']

ax.errorbar(D_plot2, inv_tau_C_plot, yerr=inv_tau_C_err_plot, linestyle="none", ecolor=colorC, elinewidth=0.4, capsize=1.0, capthick=0.4, zorder=3)
ax.errorbar(D_plot2, inv_tau_R_plot, yerr=inv_tau_R_err_plot, linestyle="none", ecolor=colorR, elinewidth=0.4, capsize=1.0, capthick=0.4, zorder=3)

ax.scatter(D_plot2, inv_tau_C_plot, marker='D', s = markersize, color=colorCF, edgecolors = colorCE, linewidth = 0.7,  label="Chaotic", zorder=7)
ax.scatter(D_plot2, inv_tau_R_plot, marker='s', s = markersize, color=colorRF, edgecolors = colorRE, linewidth = 0.7, label="Regular", zorder=6)

ax.vlines(D1/Es,-0.03, 0.14, color = 'mediumseagreen', linewidth = 1.2, linestyles='--', alpha = 1)
ax.vlines(D2/Es,-0.03, 0.14, color = 'mediumseagreen', linewidth = 1.2, linestyles='-', alpha = 1)

ax.set_xlabel(r'$\hbar D_z/\epsilon_s$', labelpad = 2)
ax.set_ylabel(r'$\tau_s/\tau_r$ ',labelpad=3)
ax.set_yticks([0,0.1],['0','0.1'])
ax.set_xticks([0,5,10,15,20])

ax.set_xlim(-1.5, 20)
ax.set_ylim(-0.03, 0.135)
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(MultipleLocator(0.05))

xmin, xmax = ax.get_xlim(); xlen = xmax-xmin;
ymin, ymax = ax.get_ylim(); ylen = ymax-ymin;

ax.text(xmin+xlen*subplotposXab,ymin+ylen*subplotposYab,'(b)',fontsize=fontsize,ha='left',va='top')

ax.grid(which='both',linewidth=0.15,alpha=0.4)

ax = axes['c']

class TextOnly:
    pass

class HandlerTextOnly(HandlerBase):
    def adjust_drawing_area(self, legend, orig_handle,
                            xdescent, ydescent, width, height, fontsize):

        return xdescent, ydescent, 0.0, height

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        dummy = Rectangle(
            (0, 0), 0, 0,
            transform=trans,
            facecolor='none',
            edgecolor='none',
            linewidth=0
        )
        return [dummy]

hC1, = ax.plot(timesC_D1[0:-1:6], valuesC_D1[0:-1:6], '--', color=colorC, linewidth=1, zorder=1)
hR1, = ax.plot(timesR_D1[0:-1:6], valuesR_D1[0:-1:6], '--', color=colorR, linewidth=1, zorder=1)

hC2, = ax.plot(timesC_D2[0:-1:4], valuesC_D2[0:-1:4], '-',  color=colorC, linewidth=0.5, zorder=1)
hR2, = ax.plot(timesR_D2[0:-1:4], valuesR_D2[0:-1:4], '-',  color=colorR, linewidth=0.5, zorder=1)

ax.hlines(1/np.sqrt(N_en), -Tf*Es*0.05, Tf*Es*1.05, color = 'gray', linewidth = 1, linestyles='dashed', zorder=0, label=r'min($\Delta^{(2)}$)=1/$\sqrt{N_{ens}}$')

ax.set_yscale("log")
ax.set_xlabel(r'$t/\tau_s$', labelpad=2)
ax.set_ylabel(r'$\Delta^{(2)}(t)$', labelpad=0)
ax.set_xlim(0, 2*Es)

ax.set_xticks([0,20,40,60,80])

ax.xaxis.set_minor_locator(MultipleLocator(10))
ax.set_ylim([1/np.sqrt(N_en)/2.5, 2.5])
ax.grid(which='both',linewidth=0.15,alpha=0.4)

txtC = TextOnly()
txtR = TextOnly()

handles = [txtR, txtC, hR1, hC1, hR2, hC2]
labels = [
    'Regular', 'Chaotic',
    rf'$\qquad\quad \hbar D_z/\varepsilon_s={D1/Es:.1f}$',
    rf'$\qquad\quad \hbar D_z/\varepsilon_s={D1/Es:.1f}$',
    rf'$\qquad\quad {D2/Es:.1f}$',
    rf'$\qquad\quad {D2/Es:.1f}$',
]

leg1 = ax.legend(
    handles, labels,
    ncol=3,
    loc='lower center',
    bbox_to_anchor=(0.655, 0.815),
    frameon=False,
    borderpad = 0.5,
    columnspacing=1.5,
    handletextpad=-1.7,
    handlelength=2.0,
    borderaxespad=0.2,
    handler_map={TextOnly: HandlerTextOnly()}
)

ax.add_artist(leg1)

xmin, xmax = ax.get_xlim(); xlen = xmax-xmin;
ymin, ymax = ax.get_ylim(); ylen = ymax/ymin;

ax.text(xmin+xlen*subplotposXc,ymin*ylen**subplotposYc,'(c)',fontsize=fontsize,ha='left',va='top')

plt.savefig(folder_save+'/Figure6.pdf',dpi=400)
plt.savefig(folder_save+'/Figure6.png',dpi=400)
plt.show()
