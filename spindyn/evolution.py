import numpy as np
from spindyn.hamiltonian import *
from spindyn.params import *
from multiprocessing import Pool
from tqdm import tqdm
from scipy.integrate import solve_ivp

# time evolution by e^-iHt, exact diagonalization
def evolution_ED(state0, evals, evecs, t, hbar=1.0):
    state0 = np.asarray(state0, complex).reshape(-1)
    E = np.asarray(evals, float).reshape(-1)
    
    R = np.asarray(evecs, complex)
    if R.shape != (state0.size, E.size): raise ValueError("shapes: evecs must be (dim, nstates)")
    
    t = np.asarray(t, float)
    c0 = R.conj().T @ state0
    phase = np.exp(-1j * np.outer(t.ravel(), E) / hbar)
    psi_t = (R @ (phase * c0).T).T             
    return psi_t[0] if t.ndim == 0 else psi_t

# time evolution with weyl-mean hamiltonian, single trajectory
def evolution_single(args):
    # unpack args
    idx, a_init, t_list, param, rtol = args
    driving = param.driving_array(); modulation = param.modulation_array()

    # initial cpsi state
    ap, a0, am = a_init
    psi0 = np.array([ap.real, ap.imag, a0.real, a0.imag, am.real, am.imag], float)
    
    # solve and save trajectory
    sol = solve_ivp(
        fun = hamiltonian_twa, y0 = psi0,
        t_span = (t_list[0], t_list[-1]), t_eval = t_list,
        args = (driving, modulation),
        method = 'Radau', rtol = rtol)
    a = np.vstack([sol.y[0] + 1j*sol.y[1], sol.y[2] + 1j*sol.y[3], sol.y[4] + 1j*sol.y[5]])
    return idx, a.T

# time evolution by TWA method, ensemble driving ; multiprocessing
def evolution_TWA(PS0, param, t_list, N, n_sample=10000, seed=0, rtol=1e-8):
    rng = np.random.default_rng(seed)
    
    # PS to psi in phasespace
    n0, ts, m, tm = PS0
    nP, nM = (1-n0+m)/2, (1-n0-m)/2
    phip, phim = (ts+tm)/2, (ts-tm)/2
    if min(n0, nP, nM) < 0: raise ValueError("invalid (n0,m)")

    # wigner sampling of coherent state
    a_mean = np.array([np.sqrt(N*nP)*np.exp(1j*phip), np.sqrt(N*n0), np.sqrt(N*nM)*np.exp(1j*phim)])
    noise = (rng.standard_normal((n_sample,3)) + 1j*rng.standard_normal((n_sample,3))) / 2.0
    a0s = a_mean[None,:] + noise

    # ensemble simulation
    t_list = np.asarray(t_list, float)
    trajectory = np.empty((n_sample, t_list.size, 3), complex)
    
    jobs = [(idx, a0s[idx], t_list, param, rtol) for idx in range(n_sample)]
    with Pool(processes=10) as pool:
        for idx, out in tqdm(pool.imap_unordered(evolution_single, jobs, chunksize=50)): trajectory[idx] = out
    
    return trajectory