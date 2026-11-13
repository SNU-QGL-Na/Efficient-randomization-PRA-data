import os as _os
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_ROOT))
_os.chdir(_ROOT)
(_ROOT / "Figures").mkdir(exist_ok=True)

folder_data = "./Data_for_Figure/Figure10"
folder_save = "./Figures"

file_common = folder_data + "/shared_LE_summary.xlsx"
file_y = folder_data + "/axis=y/shared_LE_summary.xlsx"
file_z = folder_data + "/axis=z/shared_LE_summary.xlsx"

import os
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

def load_summary_excel(file_path, drive_axis, delta0=None, Omega=None, nu=None, done_only=True, legacy_axis="z"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_excel(file_path, engine="openpyxl")

    required = {"idx", "D", "Average_LE"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if done_only and "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == "done"].copy()

    axis = str(drive_axis).strip().lower()
    legacy_axis = str(legacy_axis).strip().lower()

    if "drive_axis" in df.columns:
        df = df[df["drive_axis"].astype(str).str.lower() == axis].copy()
    else:
        if axis != legacy_axis:
            return df.iloc[0:0].copy()
        df = df.copy()
        df["drive_axis"] = legacy_axis

    if (delta0 is not None) and ("delta0" in df.columns):
        df = df[np.isclose(df["delta0"].to_numpy(dtype=float), float(delta0), atol=1e-12, rtol=0.0)].copy()

    if (Omega is not None) and ("Omega" in df.columns):
        df = df[np.isclose(df["Omega"].to_numpy(dtype=float), float(Omega), atol=1e-12, rtol=0.0)].copy()

    if (nu is not None) and ("nu" in df.columns):
        df = df[np.isclose(df["nu"].to_numpy(dtype=float), float(nu), atol=1e-12, rtol=0.0)].copy()

    df = df.dropna(subset=[c for c in ["idx", "D", "Average_LE"] if c in df.columns]).copy()

    key_cols = [c for c in ["drive_axis", "idx", "seed", "D", "delta0", "Omega", "nu"] if c in df.columns]
    if key_cols:
        df = df.drop_duplicates(subset=key_cols, keep="last").copy()

    sort_cols = [c for c in ["drive_axis", "nu", "D", "idx"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(by=sort_cols).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df

def load_axis_dataframe(file_axis, file_common, axis, delta0=None, Omega=None, nu=None, done_only=True):
    df_list = []

    if os.path.exists(file_axis):
        df_part = load_summary_excel(file_axis, axis, delta0=delta0, Omega=Omega, nu=nu, done_only=done_only).copy()
        if not df_part.empty:
            df_part["source_file"] = file_axis
            df_list.append(df_part)

    if os.path.exists(file_common):
        df_part = load_summary_excel(file_common, axis, delta0=delta0, Omega=Omega, nu=nu, done_only=done_only).copy()
        if not df_part.empty:
            df_part["source_file"] = file_common
            df_list.append(df_part)

    if len(df_list) == 0:
        raise ValueError(f"No rows remain after filtering for axis={axis}. Check input file / nu / delta0 / Omega / status.")

    df = pd.concat(df_list, ignore_index=True, sort=False)
    df["drive_axis_plot"] = axis
    df["drive_axis"] = axis

    key_cols = [c for c in ["drive_axis_plot", "idx", "seed", "D", "delta0", "Omega", "nu"] if c in df.columns]
    if key_cols:
        df = df.drop_duplicates(subset=key_cols, keep="last").copy()

    sort_cols = [c for c in ["drive_axis_plot", "nu", "D", "idx"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(by=sort_cols).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    print("---------------------------------------------------------")
    print(f"axis = {axis}")
    print(f"N_rows = {len(df)}")
    print("---------------------------------------------------------")

    return df

def mask_selected_D_values(d_values, selected_D_values, atol=1e-12):
    d_values = np.asarray(d_values, dtype=float)
    selected_D_values = np.asarray(selected_D_values, dtype=float)
    return np.any(np.isclose(d_values[:, None], selected_D_values[None, :], atol=atol, rtol=0.0), axis=1)

def mean_curve(df):
    stats = (
        pd.DataFrame({
            "x_key": np.round(df["x_plot"].to_numpy(dtype=float), 12),
            "y_plot": df["y_plot"].to_numpy(dtype=float),
        })
        .groupby("x_key", as_index=False)
        .agg(y_mean=("y_plot", "mean"))
        .sort_values("x_key")
    )
    return stats

fontsize = 8
fontsize_tick = 7
fontsize_legend = 7

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

nu = 60.0
delta0 = 0.0
Omega = 22.5
done_only = True

marker_size = 6.0
alpha = 1.0

log_ylim = (-0.20, 1.70)
log_data_xmin = 0.006
log_data_xmax = 4.5

linear_xlim = None
linear_ylim = None

log_mean_lw = 0.8
linear_mean_lw = 0.6

LOG_SELECTED_D_VALUES = [
    0.3, 0.425, 0.6, 0.87, 1.25, 1.73, 2.4, 3.0,
    4.0, 5.2, 7.0, 9.0, 10.8, 13.5, 16.5,
    20.0, 23.5, 27.5, 33.5, 40.0, 48.0, 60.0, 72.0, 90.0,
    108.0, 130.0, 160.0, 200.0,
]

mean_color_z = "#000000"
mean_alpha_z = 0.95
mean_lw_scale_top_z = 1.50
mean_lw_scale_bottom_z = 1.40
mean_linestyle_z = (0, (3, 2))
zorder_mean_z = 5
label_z = r"$\alpha = z$"

color_y = "#D84FA3"
marker_y = "o"
scatter_alpha_top_y = 0.07
scatter_alpha_bottom_y = 0.06
marker_size_scale_y = 1.18
mean_color_y = "#9C145F"
mean_alpha_y = 1.00
mean_lw_scale_top_y = 2.00
mean_lw_scale_bottom_y = 2.00
mean_linestyle_y = "-"
zorder_scatter_y = 6
zorder_mean_y = 8
edgecolor_y = "#7A0F49"
edgewidth_y = 0.22
label_y = r"$\alpha = y$"

df_z = load_axis_dataframe(file_z, file_common, "z", delta0=delta0, Omega=Omega, nu=nu, done_only=done_only)
df_y = load_axis_dataframe(file_y, file_common, "y", delta0=delta0, Omega=Omega, nu=nu, done_only=done_only)

if "g" not in df_z.columns or "g" not in df_y.columns:
    raise ValueError("The summary file must contain a 'g' column to plot D/g on the horizontal axis.")

g_z = df_z["g"].to_numpy(dtype=float)
g_y = df_y["g"].to_numpy(dtype=float)

if np.any(~np.isfinite(g_z)) or np.any(np.isclose(g_z, 0.0, atol=1e-15, rtol=0.0)):
    raise ValueError("All g values for z axis must be finite and nonzero.")
if np.any(~np.isfinite(g_y)) or np.any(np.isclose(g_y, 0.0, atol=1e-15, rtol=0.0)):
    raise ValueError("All g values for y axis must be finite and nonzero.")

df_z["x_plot"] = df_z["D"].to_numpy(dtype=float) / g_z
df_z["y_plot"] = df_z["Average_LE"].to_numpy(dtype=float) / g_z

df_y["x_plot"] = df_y["D"].to_numpy(dtype=float) / g_y
df_y["y_plot"] = df_y["Average_LE"].to_numpy(dtype=float) / g_y

x_all = pd.concat([df_z["x_plot"], df_y["x_plot"]], ignore_index=True).to_numpy(dtype=float)
y_all = pd.concat([df_z["y_plot"], df_y["y_plot"]], ignore_index=True).to_numpy(dtype=float)

x_min = np.nanmin(x_all)
x_max = np.nanmax(x_all)
y_min = np.nanmin(y_all)
y_max = np.nanmax(y_all)

y_pad = 0.10 * (y_max - y_min) if y_max > y_min else 0.05
full_ylim = (y_min - y_pad, y_max + y_pad)

if linear_xlim is None:
    x_range = x_max - x_min
    left_pad = 0.08 * x_range
    right_pad = 0.05 * x_range
    linear_xlim = (x_min - left_pad, x_max + right_pad)

if linear_ylim is None:
    linear_ylim = full_ylim

x_z = df_z["x_plot"].to_numpy(dtype=float)
y_z = df_z["y_plot"].to_numpy(dtype=float)
D_z = df_z["D"].to_numpy(dtype=float)
mask_top_z = (
    (x_z > 0.0) &
    (x_z >= float(log_data_xmin)) &
    (x_z <= float(log_data_xmax)) &
    (y_z >= log_ylim[0]) &
    (y_z <= log_ylim[1]) &
    mask_selected_D_values(D_z, LOG_SELECTED_D_VALUES)
)
mask_bot_z = (
    (x_z >= linear_xlim[0]) &
    (x_z <= linear_xlim[1]) &
    (y_z >= linear_ylim[0]) &
    (y_z <= linear_ylim[1])
)

df_z_top = df_z.loc[mask_top_z].copy()
df_z_bot = df_z.loc[mask_bot_z].copy()
stats_z_top = mean_curve(df_z_top)
stats_z_bot = mean_curve(df_z_bot)

x_y = df_y["x_plot"].to_numpy(dtype=float)
y_y = df_y["y_plot"].to_numpy(dtype=float)
D_y = df_y["D"].to_numpy(dtype=float)
mask_top_y = (
    (x_y > 0.0) &
    (x_y >= float(log_data_xmin)) &
    (x_y <= float(log_data_xmax)) &
    (y_y >= log_ylim[0]) &
    (y_y <= log_ylim[1]) &
    mask_selected_D_values(D_y, LOG_SELECTED_D_VALUES)
)
mask_bot_y = (
    (x_y >= linear_xlim[0]) &
    (x_y <= linear_xlim[1]) &
    (y_y >= linear_ylim[0]) &
    (y_y <= linear_ylim[1])
)

df_y_top = df_y.loc[mask_top_y].copy()
df_y_bot = df_y.loc[mask_bot_y].copy()
stats_y_top = mean_curve(df_y_top)
stats_y_bot = mean_curve(df_y_bot)

figure_size = (86/25.4,86/25.4)

l = [142]
lb = [15,3]

fig, axes = plt.subplot_mosaic([
    ['blank0']*(sum(l)+sum(lb)),
    ['blank1']*lb[0]+['a']*l[0]+['blank4']*lb[1],
    ['blank1']*lb[0]+['blank2']*l[0]+['blank4']*lb[1],
    ['blank1']*lb[0]+['b']*l[0]+['blank4']*lb[1],
    ['blank3']*(sum(l)+sum(lb))],
    constrained_layout=False, figsize=figure_size, gridspec_kw={'height_ratios':[0.01,1.,0.26,1,0.20], 'hspace':0,'wspace':0})
plt.subplots_adjust(left=0, top=1, bottom=0, right=1)

ax = axes['blank0']; ax.axis('off');
ax = axes['blank1']; ax.axis('off');
ax = axes['blank2']; ax.axis('off');
ax = axes['blank3']; ax.axis('off');
ax = axes['blank4']; ax.axis('off');

Dmin = 0.15; Dmax = 300
Es = 45
subplotposXa = -0.105
subplotposYa = 1.0

ax = axes['a']
ax.set_axisbelow(True)

ax.plot(
    stats_z_top["x_key"].to_numpy(dtype=float),
    stats_z_top["y_mean"].to_numpy(dtype=float),
    color=mean_color_z,
    linewidth=log_mean_lw * mean_lw_scale_top_z,
    linestyle=mean_linestyle_z,
    alpha=mean_alpha_z,
    zorder=zorder_mean_z,
    label=label_z,
)

ax.scatter(
    df_y_top["x_plot"].to_numpy(dtype=float),
    df_y_top["y_plot"].to_numpy(dtype=float),
    s=marker_size * marker_size_scale_y,
    marker=marker_y,
    c=color_y,
    alpha=scatter_alpha_top_y * alpha,
    linewidths=edgewidth_y,
    edgecolors=edgecolor_y,
    rasterized=True,
    zorder=zorder_scatter_y,
    label=label_y,
)

ax.plot(
    stats_y_top["x_key"].to_numpy(dtype=float),
    stats_y_top["y_mean"].to_numpy(dtype=float),
    color=mean_color_y,
    linewidth=log_mean_lw * mean_lw_scale_top_y,
    linestyle=mean_linestyle_y,
    alpha=mean_alpha_y,
    zorder=zorder_mean_y,
)

ax.set_xscale("log")
ax.set_xlim(Dmin/Es, Dmax/Es)
ax.set_ylim(-0.3, 1.8)
ax.set_xlabel(r"$\hbar D_\alpha / \varepsilon_s$", labelpad=2)
ax.set_ylabel(r"$\lambda\tau_s$", labelpad= 3)

ax.set_xscale("log")
ax.set_yticks([0, 0.5, 1,1.5],['0','0.5','1.0','1.5'])
ax.tick_params(axis='y', which='major', pad = 2)
ax.yaxis.set_minor_locator(MultipleLocator(0.25))
ax.set_xticks([0.01,0.1,1])

ax.grid(which='both',linewidth=0.15,alpha=0.4)

legend_handles = [
    Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label=r'$\alpha = z$'),
    Line2D([0], [0], marker='o', linestyle='None',
           markerfacecolor=color_y, markeredgecolor=color_y,
           markersize=3, alpha=1.0, label=r'$\alpha = y$')
]

ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.50, 0.98), ncol=2, frameon=False)

xmin, xmax = ax.get_xlim(); xlen = xmax/xmin;
ymin, ymax = ax.get_ylim(); ylen = ymax-ymin;
ax.text(xmin*xlen**subplotposXa,ymin+ylen*subplotposYa,'(a)',fontsize=fontsize,ha='left',va='top')

ax = axes['b']
ax.set_axisbelow(True)

ax.plot(
    stats_z_bot["x_key"].to_numpy(dtype=float),
    stats_z_bot["y_mean"].to_numpy(dtype=float),
    color=mean_color_z,
    linewidth=linear_mean_lw * mean_lw_scale_bottom_z,
    linestyle=mean_linestyle_z,
    alpha=mean_alpha_z,
    zorder=zorder_mean_z,
)

ax.scatter(
    df_y_bot["x_plot"].to_numpy(dtype=float),
    df_y_bot["y_plot"].to_numpy(dtype=float),
    s=marker_size * marker_size_scale_y,
    marker=marker_y,
    c=color_y,
    alpha=scatter_alpha_bottom_y * alpha,
    linewidths=edgewidth_y,
    edgecolors=edgecolor_y,
    rasterized=True,
    zorder=zorder_scatter_y,
)

ax.plot(
    stats_y_bot["x_key"].to_numpy(dtype=float),
    stats_y_bot["y_mean"].to_numpy(dtype=float),
    color=mean_color_y,
    linewidth=linear_mean_lw * mean_lw_scale_bottom_y,
    linestyle=mean_linestyle_y,
    alpha=mean_alpha_y,
    zorder=zorder_mean_y,
)

ax.set_xlim(-1.5, 20)
ax.set_ylim(-0.3, 1.8)
ax.set_xlabel(r"$\hbar D_\alpha / \varepsilon_s$", labelpad=2)
ax.set_ylabel(r"$\lambda\tau_s$", labelpad= 3)
ax.set_xticks([0,5,10,15,20], ['0','5','10','15','20'])
ax.set_yticks([1.5,1,0.5,0],['1.5','1.0','0.5','0'])
ax.tick_params(axis='y', which='major', pad = 2)
ax.yaxis.set_minor_locator(MultipleLocator(0.25))
ax.xaxis.set_minor_locator(MultipleLocator(1))

ax.grid(which='both',linewidth=0.15,alpha=0.4)

xmin, xmax = ax.get_xlim(); xlen = xmax-xmin;
ymin, ymax = ax.get_ylim(); ylen = ymax-ymin;
ax.text(xmin+xlen*subplotposXa,ymin+ylen*subplotposYa,'(b)',fontsize=fontsize,ha='left',va='top')

plt.savefig(folder_save+'/Figure10.pdf',dpi=400)
plt.savefig(folder_save+'/Figure10.png',dpi=400)
plt.show()
