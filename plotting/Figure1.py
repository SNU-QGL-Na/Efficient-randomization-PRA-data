import os as _os
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_ROOT))
_os.chdir(_ROOT)
(_ROOT / "Figures").mkdir(exist_ok=True)

folder_data = "./Data_for_Figure"
folder_save = "./Figures"

import numpy as np
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from spindyn.analysis import State
from spindyn.io import load_data

mpl.rcParams["font.family"] = 'Arial'
mpl.rcParams["mathtext.fontset"] = 'stix'

def read_data(D, r, idx, mode=None, path='./Results/SpinTimeEvolution/Rawdata') :
    if mode is not None : folder = path + f'/{mode} modulation/r={r:.2f}, D={D}/initial{idx}/'
    else : folder = path + f'/r={r:.2f}, D={D}/initial{idx}/'

    loadname = 'State.csv'
    if not os.path.exists(folder): return None

    _, _, psi_load = load_data(folder, loadname, loadmode=2, printmode=0)
    PS_load = State.to(State('psi', psi_load), 'PS')
    return PS_load

def Draw_3D_axis(ax):
    COLOR = 'gray'
    ax.set_xlabel(r"$\theta_s$", labelpad = -12.5)
    ax.set_ylabel(r"$\theta_m$", labelpad = -13)

    ax.text(-2*np.pi, 2.2*np.pi, 1.1, r'$m$', fontsize=fontsize, ha='center', va='bottom')

    ax.set_xlim(-2*np.pi, 2*np.pi)
    ax.set_ylim(-2*np.pi, 2*np.pi);
    ax.set_zlim(-1, 1)
    ax.set_xticks([-2*np.pi,0,2*np.pi], [r'-2$\pi$',r'0',r'2$\pi$'])
    ax.set_yticks([-2*np.pi,0,2*np.pi], [r'-2$\pi$',r'0',r'2$\pi$'])
    ax.set_zticks([-1,0,1], [r'-1',r'0',r'1'])
    ax.tick_params(axis = 'x', which = 'major', pad = -2)
    ax.tick_params(axis = 'y', which = 'major', pad = -4)
    ax.tick_params(axis = 'z', which = 'major', pad = -5)
    ax.xaxis.set_minor_locator(MultipleLocator(np.pi))
    ax.yaxis.set_minor_locator(MultipleLocator(np.pi))
    ax.zaxis.set_minor_locator(MultipleLocator(0.5))
    ax.view_init(elev=elev_angle, azim=azim_angle)

    ax.set_box_aspect((1, 1, 0.9), zoom = 1.18)
    ax.grid = False
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor((1, 1, 1, 0))

        axis._axinfo["grid"]['alpha'] = 0.4
        axis._axinfo["grid"]['linewidth'] = 0.15
        axis.line.set_linewidth(0.5)
        axis.set_rotate_label(False)

    for lab in ax.get_xticklabels():
        lab.set_horizontalalignment('center')
        lab.set_verticalalignment('bottom')

    for lab in ax.get_yticklabels():
        lab.set_horizontalalignment('center')
        lab.set_verticalalignment('center')

    for lab in ax.get_zticklabels():
        lab.set_horizontalalignment('center')
        lab.set_verticalalignment('center')

    diamond = np.array([[0, 2*np.pi], [2*np.pi, 0], [0, -2*np.pi], [-2*np.pi, 0]])
    mmin, mmax = -1.0, 1.00

    top = [[x, y, mmax] for x, y in diamond]
    bot = [[x, y, mmin] for x, y in diamond]

    faces = []; faces.append(top); faces.append(bot)

    for i in range(4):
        j = (i + 1) % 4
        side = [[diamond[i,0], diamond[i,1], mmin], [diamond[j,0], diamond[j,1], mmin], [diamond[j,0], diamond[j,1], mmax], [diamond[i,0], diamond[i,1], mmax]]
        faces.append(side)

    poly = Poly3DCollection(faces, facecolors=COLOR, edgecolors='lightgray', linewidths=0.1, alpha=0.08)
    ax.add_collection3d(poly)
    return ax

fontsize=8
fontsize_tick = 7
fontsize_legend = 6

markersize = 1

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

jump = 67

elev_angle = 20
azim_angle = 35+180

Da = 0; Db = 100; es = 45

idx_chaotic = 6
idx_regular = 46

PS_chaotic = read_data(Da, 0.5, idx_chaotic, None, folder_data+'/Figure1')
PS_regular = read_data(Da, 0.5, idx_regular, None, folder_data+'/Figure1')

rC, tsC, mC, tmC = PS_chaotic.T
rR, tsR, mR, tmR = PS_regular.T

rR_a, tsR_a, mR_a, tmR_a = rR[::jump], tsR[::jump], mR[::jump], tmR[::jump]
rC_b, tsC_b, mC_b, tmC_b = rC[::jump], tsC[::jump], mC[::jump], tmC[::jump]

PS_chaotic = read_data(Db, 0.5, idx_chaotic, None, folder_data+'/Figure1')
PS_regular = read_data(Db, 0.5, idx_regular, None, folder_data+'/Figure1')

rC, tsC, mC, tmC = PS_chaotic.T
rR, tsR, mR, tmR = PS_regular.T

rR_c, tsR_c, mR_c, tmR_c = rR[::jump], tsR[::jump], mR[::jump], tmR[::jump]
rC_d, tsC_d, mC_d, tmC_d = rC[::jump], tsC[::jump], mC[::jump], tmC[::jump]

figure_size = (86/25.4,80/25.4)

l = [100,100]
lb = [12,1,1]

fig, axes = plt.subplot_mosaic([
    ['blank0']*lb[0]+['labelD1']*l[0]+['blank1']*lb[1]+['labelD2']*l[1]+['blank5']*lb[2],
    ['labelRC1']*lb[0]+['a']*l[0]+['blank2']*lb[1]+['b']*l[1]+['blank5']*lb[2],
    ['blank4']*(sum(l)+sum(lb)),
    ['labelRC2']*lb[0]+['c']*l[0]+['blank6']*lb[1]+['d']*l[1]+['blank7']*lb[2],
    ['blank8']*lb[0]+['colorbar1']*(l[0]+lb[1]+l[1])+['blank7']*lb[2]],
    constrained_layout=False, figsize=figure_size, gridspec_kw={'height_ratios':[0.09,1,0.13,1,0.32], 'hspace':0,'wspace':0})
plt.subplots_adjust(left=0, top=1, bottom=0, right=1)

ax = axes['blank0']; ax.axis('off')
ax = axes['blank1']; ax.axis('off')
ax = axes['blank2']; ax.axis('off')

ax = axes['blank4']; ax.axis('off')
ax = axes['blank5']; ax.axis('off')
ax = axes['blank6']; ax.axis('off')
ax = axes['blank7']; ax.axis('off')
ax = axes['blank8']; ax.axis('off')

ax = axes['labelRC1']
xmin, xmax = ax.get_xlim(); xlen = xmax-xmin
ymin, ymax = ax.get_ylim(); ylen = ymax-ymin
ax.text(
    xmin, ymin+ylen/2, rf"$\hbar D_z= 0$",
    rotation = 90, ha='left', va='center', fontsize = fontsize
)
ax.axis('off')

ax = axes['labelRC2']
xmin, xmax = ax.get_xlim(); xlen = xmax-xmin
ymin, ymax = ax.get_ylim(); ylen = ymax-ymin
ax.text(
    xmin, ymin+ylen/2, rf"$\hbar D_z= {Db/es:.1f}\epsilon_s $",
    rotation = 90, ha='left', va='center', fontsize = fontsize
)
ax.axis('off')

ax = axes['labelD1']
xmin, xmax = ax.get_xlim(); xlen = xmax-xmin
ymin, ymax = ax.get_ylim(); ylen = ymax-ymin
ax.text(
    xmin+xlen/2, ymax, f"Regular",
    ha='center', va='top', fontsize = fontsize
)
ax.axis('off')

ax = axes['labelD2']
xmin, xmax = ax.get_xlim(); xlen = xmax-xmin
ymin, ymax = ax.get_ylim(); ylen = ymax-ymin
ax.text(
    xmin+xlen/2, ymax, f"Chaotic",
    ha='center', va='top', fontsize = fontsize
)
ax.axis('off')

spec_a = axes['a'].get_subplotspec()
fig.delaxes(axes['a'])
axes['a'] = fig.add_subplot(spec_a, projection='3d')

ax = axes['a']
ax = Draw_3D_axis(ax)

ax.scatter(tsR_a, tmR_a, mR_a, c=rR_a, cmap='jet', vmin = 0, vmax = 1,  linewidth = 0)

spec_c = axes['c'].get_subplotspec()
fig.delaxes(axes['c'])
axes['c'] = fig.add_subplot(spec_c, projection='3d')

ax = axes['c']
ax = Draw_3D_axis(ax)

sc_c=ax.scatter(tsR_c, tmR_c, mR_c, c=rR_c, cmap='jet',s=markersize, vmin = 0, vmax = 1,  linewidth = 0)

ax = axes['colorbar1']
ax.set_xlim([0,1]); ax.set_ylim([0,1])
ax.axis('off')
ax_cbar = ax.inset_axes((0.2,0.28,0.6,0.168))
cbar = fig.colorbar(sc_c, ax=[ax], cax=ax_cbar, ticks=[0,0.5,1],orientation="horizontal")
cbar.ax.set_xticklabels([r'0','0.5','1'],fontsize=fontsize_legend)
cbar.ax.xaxis.set_tick_params(length = 2, pad = 1, width = 0.5)
cbar.outline.set_linewidth(0.5)
ax.text(0.185,0.25,r'$\rho_0$',fontsize=fontsize, ha = 'right', va = 'bottom')

spec_b = axes['b'].get_subplotspec()
fig.delaxes(axes['b'])
axes['b'] = fig.add_subplot(spec_b, projection='3d')

ax = axes['b']
ax = Draw_3D_axis(ax)

ax.scatter(tsC_b, tmC_b, mC_b, c=rC_b, cmap='jet', vmin = 0, vmax = 1, s=markersize, linewidth = 0)

spec_d = axes['d'].get_subplotspec()
fig.delaxes(axes['d'])
axes['d'] = fig.add_subplot(spec_d, projection='3d')

ax = axes['d']
ax = Draw_3D_axis(ax)

ax.scatter(tsC_d, tmC_d, mC_d, c=rC_d, cmap='jet', vmin = 0, vmax = 1, s=markersize, linewidth = 0)

plt.savefig(folder_save + '/Figure1.pdf',dpi=400)
plt.savefig(folder_save + '/Figure1.png',dpi=400)
plt.savefig(folder_save + '/Figure1.eps',dpi=400)
plt.show()
