import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
np.set_printoptions(precision=3, suppress=True)

import spindyn.io as spinio
import spindyn.solve as spinsolve
from sklearn.neighbors import KDTree
from scipy.special import digamma, gamma
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from spindyn.params import SpinParams
from spindyn.analysis import State, get_spin_energy, psi_normalize

# Sample random initial states within energy shell
def sample_init(E0, params, tol=1e-3, max_trials=10000):
    for _ in range(max_trials):
        rho = np.random.uniform(0, 1)
        tP = np.random.uniform(-np.pi, np.pi)
        m = np.random.uniform(-(1 - rho), 1 - rho)
        tM = np.random.uniform(-np.pi, np.pi)

        PS = np.array([rho, (tP+tM)/2, m, (tP-tM)/2])
        E = get_spin_energy(PS, params)

        if np.abs(E - E0)/E0 < tol:
            return PS
    return None

# Harr ensemble (uniform sample)
def harr_ensemble(N):
    state = np.random.randn(N, 6)*10
    state = psi_normalize(state)
    state = State.to(State('psi', state), 'PS')
    return state

# Produce/Read datas for given modulation amplitude
def produce_data(D, idx, PS_init=None, E0 = 1.005, t_final = 100, dt = 0.001):
    # Ready to Solve
    param = SpinParams(delta=0, Omega=22.5, g=45, q=45, D=D, f=60, Phi=0)
    ti_real = 0; tf_real = t_final; tstep_real = dt
    tlist_real = np.arange(ti_real, tf_real+tstep_real*0.1, tstep_real)

    # Initial State
    if PS_init is None :
        while PS_init is None :
            PS_init = sample_init(E0*param.g, param)

    print(PS_init) 
    psi0 = State("PS",PS_init).to("psi")
    psi0 = psi_normalize(psi0)
    
    # Make folder for saving
    savefolder = f'./Results/SpinTimeEvolution/r={param.Omega/param.g:.2f}, D={D}/initial{idx}/'
    os.makedirs(savefolder, exist_ok = True)
    savename = 'State.csv'

    # Time evolution & save result
    sol_psi = spinsolve.evolution(
        psi0, tlist_real, param, savefolder, savename, savemode = 2, printmode = 0, rtol = 10**-8)

def read_data(D, idx) :
    folder = f'./Results/SpinTimeEvolution/r=0.50, D={D}/initial{idx}/'
    loadname = 'State.csv'
    if not os.path.exists(folder): return None

    _, _, psi_load = spinio.load_data(folder, loadname, loadmode=2, printmode=0)
    PS_load = State.to(State('psi', psi_load), 'PS')
    return PS_load

# Get shannon entropy for given PS states
def shannon_entropy(states, bins, log_base=np.e):
    if not State.validate('PS', states):
        raise ValueError("States must be in the form 'PS'")

    # Convert arg components
    states = states.copy()
    tP = (states[:,1] + states[:,3])/2
    tM = (states[:,1] - states[:,3])/2
    states[:, 1], states[:, 3] = tP, tM
    
    # Multidimensional Histogram
    ranges = [(0, 1), (-np.pi, np.pi), (-1, 1), (-np.pi, np.pi)]
    counts, _ = np.histogramdd(states, bins=bins, range=ranges)
    probs = counts.ravel() / states.shape[0]

    # Shannon entropy
    probs = probs[probs > 0] # drop zeros
    S = -np.sum(probs * np.log(probs)) / np.log(log_base)
    return S, len(probs)

# Get approximate shannon entropy
def KD_entropy(states, k=5):
    if not State.validate('PS', states):
        raise ValueError("States must be in the form 'PS'")
    states = states.copy()
    
    N, d = states.shape
    tP = (states[:,1] + states[:,3])/(4*np.pi)
    tM = (states[:,1] - states[:,3])/(4*np.pi)
    states[:,1], states[:,3] = tP, tM

    # Build KD-tree for k-NN search
    tree = KDTree(states)
    dist, _ = tree.query(states, k=k+1)
    eps = dist[:, k]
    c_d = np.pi**(d/2) / gamma(d/2 + 1)

    # KL estimator for differential entropy
    H_data = -digamma(k) + digamma(N) + np.log(c_d) + (d/N)*np.sum(np.log(eps))

    return H_data # H_th = ln(volume) = ln(1) = 0

# Density map for given PS arrays
def density_map(D, N_init, mode='scatter', colorlist=None, showMode=False) : 
    fig = plt.figure()
    ax = fig.add_subplot(111)
       
    if mode == 'scatter':
        initial_points = []
        for i in range(N_init):
            PSs = read_data(D, i+1)
            initial_points.append((PSs[0, 2], PSs[0, 0]))  # (m, rho)
            ax.scatter(PSs[::100,2],PSs[::100,0], s=1, c=colorlist[i], label = rf'idx={i+1}')
        for i, (m, r) in enumerate(initial_points):
            ax.plot(m, r, 'x', color=colorlist[i], markersize=10)

        ax.legend(loc='upper right', fontsize='x-small')
        
    elif mode == 'map':
        initial_points = []; all_r = []; all_m = []
        for i in range(N_init):
            PSs = read_data(D, i+1)
            all_r.append(PSs[:, 0])
            all_m.append(PSs[:, 2])
            initial_points.append((PSs[0, 2], PSs[0, 0]))  # (m, rho)
        for i, (m, r) in enumerate(initial_points):
            ax.plot(m, r, 'x', color=colorlist[i], markersize=10)
        
        r_proj = np.concatenate(all_r)
        m_proj = np.concatenate(all_m)
        
        counts, xedges, yedges, img = ax.hist2d(
            m_proj, r_proj, bins=100, range=[[-1, 1], [0, 1]], cmap='Greys')
        
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label('Count')

    ax.set_xlabel('m'); ax.set_ylabel('rho')
    ax.set_xlim(-1, 1); ax.set_ylim(0, 1)
    ax.set_title(f'Density Heatmap, D={D}')

    if showMode : plt.show()
    else : plt.close()
    fig.savefig('./Results/SpinTimeEvolution/PoincareSection/'+f'Project_r={0.50}, D={D}_{mode}.png')

# Trajectory in PS space
def trajectory(D, idx, dt=0.001):
    PSs = read_data(D, idx)
    L = PSs.shape[0]
    m_proj, r_proj = PSs[::100, 2], PSs[::100, 0]
    
    points = np.array([m_proj, r_proj]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    time_indices = np.linspace(0, L, 100)*dt
    lc = LineCollection(segments, cmap='viridis', array=time_indices[:-1], linewidth=0.5)

    fig, ax = plt.subplots()
    ax.add_collection(lc)
    ax.set_xlim(-1, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('m'); ax.set_ylabel('rho')
    ax.set_title(f"Trajectory (D={D}, idx={idx})")

    cbar = fig.colorbar(lc, ax=ax)
    cbar.set_label('Normalized Time')

    plt.tight_layout(); plt.show()

def trajectory_anim(D, idx):
    PSs = read_data(D, idx)
    m = PSs[::100, 2]; r = PSs[::100, 0]

    fig, ax = plt.subplots()
    ax.set_xlim(-1, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('m'); ax.set_ylabel('rho')
    ax.set_title(f'PS Trajectory anim : D={D}, idx={idx}')

    time_step = 0.001*100
    times = np.arange(len(m))*time_step

    points = np.array([m, r]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection([], cmap='viridis', linewidth=1, 
        norm=Normalize(vmin=times.min(), vmax=times.max()))
    lc.set_array(np.array([]))
    ax.add_collection(lc)

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    
    def init():
        lc.set_segments([]); time_text.set_text('')
        return lc, time_text

    def update(frame):
        i = frame
        lc.set_segments(segments[:i]); lc.set_array(times[:i])
        time_text.set_text(f't={times[i]:.1f} s')

        return lc, time_text

    total_frames = len(m) -1

    ani = animation.FuncAnimation(
        fig, update, frames=total_frames,
        init_func=init, blit=True, interval=10, repeat=False)  # interval in ms)

    plt.show()

# Poincare-Section with given energy shell, θ_m=0
def poincare_section(D_real, N_init, tol=0.1) :
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(N_init):
        # Make folder for saving
        loadfolder = f'./Results/SpinTimeEvolution/r={0.5}, D={D_real}/initial{i+1}/'
        loadname = 'State.csv'
        if not os.path.exists(loadfolder): continue

        # Load saved data (if saved)
        params, t_load, psi_load = spinio.load_data(loadfolder, loadname, loadmode = 2, printmode = 0)

        poinc1_sec = State("psi",psi_load).to("PS")
        init = poinc1_sec[0]
        mask = np.abs(poinc1_sec[:, 3])<tol
        poinc1_sec = poinc1_sec[mask]
        E = get_spin_energy(init, params)

        ax.scatter(poinc1_sec[:,1],poinc1_sec[:,0], s=1, label = rf'$\rho_0={init[0]:.2f},E={E:.1f}$Hz')
    ax.legend(fontsize = 'xx-small')

    # Save figures
    fig.savefig('./Results/SpinTimeEvolution/PoincareSection/'+f'Poincare_r={0.5}, D={D_real}_long.png')    