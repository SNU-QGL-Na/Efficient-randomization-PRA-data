import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap

import spindyn.analysis as spinanalysis

# Make a sphere representing magnetization space in input figure
def draw_MagSphere(figure):
    ax = figure.add_subplot(111, projection = '3d')
    ax.grid(False); ax.axis('off')
   
    axis_len = 1.2
    x_start = np.array([-axis_len,0,0]); y_start = np.array([0,-axis_len,0]); z_start = np.array([0,0,-axis_len])
    x_length = np.array([axis_len*2,0,0]); y_length = np.array([0,axis_len*2,0]); z_length = np.array([0,0,axis_len*2])
    ax.quiver(x_start,y_start,z_start,x_length,y_length,z_length, color = [0.2,0.2,0.2], arrow_length_ratio=0.05)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    ls = LightSource(azdeg=0.3, altdeg=0.3)     # Create a light source    
    rgb = ls.shade(z, plt.cm.gray)     # Shade data
    ax.plot_surface(x, y, z, facecolors=rgb, alpha=0.1, rstride=5, cstride=5, linewidth=0)
    
    r = 0.8
    ax.set_xlim(-axis_len*r,axis_len*r);
    ax.set_ylim(-axis_len*r,axis_len*r)
    ax.set_zlim(-axis_len*r,axis_len*r)
    ax.set_box_aspect([1, 1, 1])

    ax.text(axis_len*1.2, 0, 0, r"$\rm{f_x}$", color='k')
    ax.text(0, axis_len*1.2,0, r"$\rm{f_y}$", color='k')
    ax.text(0., 0, axis_len*1.2, r"$\rm{f_z}$", color='k')

    return ax


# Draw time evolution of population of each spin components (1D plot)
def show_data_SG(figure, time, psi):
    psi = spinanalysis.State("psi",psi)
    cpsi = psi.to("cpsi");
    rp = abs(cpsi[:,0])**2; rz = abs(cpsi[:,1])**2; rm = abs(cpsi[:,2])**2; 
    ax = figure.add_subplot(111)
    ax.plot(time,rp,lw=1,zorder = 1, c = 'blue', label = r'$m_z = +1$')
    ax.plot(time,rz,lw=1,zorder = 1, c = 'purple', label = r'$m_z = 0$')
    ax.plot(time,rm,lw=1,zorder = 1, c = 'red', label = r'$m_z = -1$')
    ax.set_xlim([time[0],time[-1]])
    ax.set_ylim([-0.1,1.1])
    ax.legend(frameon=True, fontsize = 8)
    ax.set_xlabel(r'Time [s]', labelpad=-1)
    ax.set_ylabel(r'Spin population $\eta_m$', labelpad=-1)


# Draw trajectory of the spin state in magnetization space (3d plot on sphere)
def show_data_mag_trajectory(figure, time, psi, elev=30, azim = 45, showmode = 2):
    if showmode not in (1,2):
        print(f"Value of showmode should be 1 or 2! (value of {showmode} is entered) - Automatically use mode 2"); printmode = 0
    mag = spinanalysis.State("psi",psi).to("mag")
    ax = draw_MagSphere(figure)
    if showmode == 1:
        ax.plot(mag[:,0],mag[:,1],mag[:,2], color='b')
    else:
        norm = Normalize(vmin=time[0], vmax=time[-1])
        full_cmap = plt.get_cmap("hot")
        cut_cmap = LinearSegmentedColormap.from_list("cut_blues", [full_cmap(i) for i in np.linspace(0.7, 0, 256)])  
        colorset = cut_cmap(norm(time))  
        for i in range(len(time)):
            ax.plot(mag[i:i+2,0],mag[i:i+2,1],mag[i:i+2,2], color=colorset[i])
        sm = ScalarMappable(cmap=cut_cmap, norm=norm)  
        cbar = figure.colorbar(sm, ax=ax)
        cbar.set_label("Time [s]")
        sm.set_clim(time[0],time[-1])
    ax.view_init(elev=elev, azim=azim) 


# Plot spin states in a ensemble in magnetization space (3d plot on sphere)
def show_data_mag_distribution(figure, psi_arr, elev=30, azim = 45):
    mag = spinanalysis.State("psi", psi_arr).to("mag")
    ax = draw_MagSphere(figure)
    ax.scatter(mag[:,0],mag[:,1],mag[:,2], color='b')
    ax.view_init(elev=elev, azim=azim) 
 
    