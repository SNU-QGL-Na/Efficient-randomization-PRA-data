import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

root = Path(__file__).resolve().parent.parent
os.chdir(root)
(root / "Figures").mkdir(exist_ok=True)
folder_data = "./Data_for_Figure/Figure9"
folder_save = "./Figures"

top_D = 100.0
delta0 = 0.0
Omega = 22.5
bottom_nus = [40.0, 100.0]

fontsize = 8
fontsize_tick = 7
fontsize_legend = 6
mpl.rcParams.update({
    "font.family": "Arial",
    "mathtext.fontset": "stix",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": fontsize,
    "axes.labelsize": fontsize,
    "xtick.labelsize": fontsize_tick,
    "ytick.labelsize": fontsize_tick,
    "legend.fontsize": fontsize_legend,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2.5,
    "ytick.minor.size": 2.5,
})

top = pd.read_excel(folder_data + "/top_LLE_vs_nu_raw.xlsx")
for col in ["g", "delta0", "Omega"]:
    if col not in top.columns:
        top[col] = np.nan
for col in ["idx", "nu", "D", "Average_LE", "g", "delta0", "Omega"]:
    if col in top.columns:
        top[col] = pd.to_numeric(top[col], errors="coerce")
top = top.dropna(subset=["idx", "nu", "D", "Average_LE"])
top["idx"] = top["idx"].astype(int)
top = top.drop_duplicates(subset=["idx", "nu", "D", "delta0", "Omega"], keep="last")
top = top.sort_values(["nu", "idx", "D"])
top = top[np.isclose(top["D"], top_D, atol=1e-12, rtol=0.0)]
for col, value in [("delta0", delta0), ("Omega", Omega)]:
    if np.isfinite(top[col]).any():
        top = top[np.isclose(top[col], value, atol=1e-12, rtol=0.0)]
top = top.sort_values(["nu", "idx"])
x_top = top["nu"].to_numpy(dtype=float)
y_top = (top["Average_LE"] / top["g"]).to_numpy(dtype=float)

bottom = pd.read_excel(folder_data + "/bottom_LLE_vs_D_raw.xlsx")
if "status" in bottom.columns:
    bottom = bottom[bottom["status"].astype(str).str.lower() == "done"]
for col in ["idx", "nu", "D", "Average_LE", "g", "delta0", "Omega", "seed"]:
    if col in bottom.columns:
        bottom[col] = pd.to_numeric(bottom[col], errors="coerce")
for col, value in [("delta0", delta0), ("Omega", Omega)]:
    if col in bottom.columns and np.isfinite(bottom[col]).any():
        bottom = bottom[np.isclose(bottom[col], value, atol=1e-12, rtol=0.0)]
bottom = bottom.dropna(subset=["idx", "nu", "D", "Average_LE"])
bottom["idx"] = bottom["idx"].astype(int)
key_cols = [c for c in ["idx", "seed", "nu", "D", "delta0", "Omega"] if c in bottom.columns]
bottom = bottom.drop_duplicates(subset=key_cols, keep="last")
bottom = bottom.sort_values(["nu", "D", "idx"])

bottom_data = []
for nu in bottom_nus:
    subset = bottom[np.isclose(bottom["nu"], nu, atol=1e-12, rtol=0.0)]
    x = (subset["D"] / subset["nu"]).to_numpy(dtype=float)
    y = (subset["Average_LE"] / subset["g"]).to_numpy(dtype=float)
    order = np.argsort(x)
    bottom_data.append((x[order], y[order]))

all_bottom_y = np.concatenate([y for _, y in bottom_data])
ymin = np.min(all_bottom_y)
ymax = np.max(all_bottom_y)
ypad = 0.20 * (ymax - ymin) if ymax > ymin else 0.05

fig = plt.figure(figsize=(86 / 25.4, 78 / 25.4))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1.8, 1.0], hspace=0.28, wspace=0.18)
ax_top = fig.add_subplot(gs[0, :])
ax_bl = fig.add_subplot(gs[1, 0])
ax_br = fig.add_subplot(gs[1, 1], sharey=ax_bl)

ax_top.set_xlabel(r"$\omega_m/2\pi\ \mathrm{[Hz]}$", labelpad=2)
ax_top.set_ylabel(r"$\lambda \tau_s$", labelpad=2)
ax_top.set_axisbelow(True)
ax_top.grid(True, which="major", color="0.72", alpha=0.35, linewidth=0.6)
ax_top.xaxis.set_major_locator(MaxNLocator(nbins=7))
ax_top.yaxis.set_major_locator(MaxNLocator(nbins=6))
ax_top.xaxis.set_minor_locator(AutoMinorLocator(5))
ax_top.yaxis.set_minor_locator(AutoMinorLocator(5))
xpad = 0.08 * (np.max(x_top) - np.min(x_top))
ypad_top = 0.20 * (np.max(y_top) - np.min(y_top))
ax_top.set_xlim(np.min(x_top) - xpad, np.max(x_top) + xpad)
ax_top.set_ylim(np.min(y_top) - ypad_top, np.max(y_top) + ypad_top)
for xv in [40, 100]:
    ax_top.axvline(xv, color="0.55", linestyle=(0, (3, 3)), linewidth=1.0, alpha=0.85, zorder=2)
ax_top.axvline(60, color="0.35", linestyle=(0, (5, 2.5)), linewidth=1.2, alpha=0.9, zorder=2)
order = np.argsort(x_top)
ax_top.scatter(x_top[order], y_top[order], s=7.0, c="black", alpha=0.04,
               linewidths=0, zorder=3)
ax_top.text(-0.12, 1.0, "(a)", transform=ax_top.transAxes,
            ha="left", va="top", fontsize=fontsize, zorder=5)

for ax, (x, y), nu, label, x_label in zip(
    [ax_bl, ax_br], bottom_data, bottom_nus, ["(b)", "(c)"], [-0.25, -0.13]
):
    ax.scatter(x, y, s=4.0, c="black", alpha=0.04,
               linewidths=0, zorder=3)
    ax.set_xlabel(r"$D_z / \omega_m$", labelpad=2)
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color="0.72", alpha=0.35, linewidth=0.6)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    xpad = 0.12 * (np.max(x) - np.min(x))
    ax.set_xlim(np.min(x) - xpad, np.max(x) + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)
    ax.text(0.94, 0.92, rf"$\omega_m/2\pi = {nu:g}$ Hz",
            transform=ax.transAxes, ha="right", va="top", fontsize=fontsize_tick)
    ax.text(x_label, 1.0, label, transform=ax.transAxes,
            ha="left", va="top", fontsize=fontsize, zorder=5)
ax_bl.set_ylabel(r"$\lambda \tau_s$", labelpad=2)
ax_br.tick_params(labelleft=False)

fig.subplots_adjust(left=0.11, right=0.995, bottom=0.09, top=0.995)
fig.savefig(folder_save + "/Figure9.pdf", bbox_inches="tight", pad_inches=0.01)
fig.savefig(folder_save + "/Figure9.png", bbox_inches="tight", pad_inches=0.01)
plt.show()
plt.close(fig)
