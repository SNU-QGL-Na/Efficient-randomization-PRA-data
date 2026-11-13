#  Functions for calculating Hamiltonian
import numpy as np
from numba import njit
from scipy.optimize import root_scalar
from spindyn.analysis import State, get_spin_energy
from spindyn.params import SpinParams
np.set_printoptions(precision=3)

@njit(cache=True, fastmath=True) # Decoration for computing acceleration 

def fs(t, psi, driving, modulation, omod):
    # Calculate time derivative of spin state at given t
    # input: t: time when calculate spin state derivative.
    #        psi: normalized wavefunction (norm = 1) at the time (1D array with length 6). [real(psi_p), imag(psi_p), real(psi_z), imag(psi_z), real(psi_m), imag(psi_m)] (p=+1, z=0, m=-1)
    #             the result is dependent on the norm of the wavefunction, because of nonlinear term
    #        parameter, modulation: two 1D arrays for spin Hamiltonian paramters. [\delta, \Omega, \epsilon_s, q], [D, f, \Phi]
    # output: time derivative of the spin state at the given time, 1D array with length 6
    # Hamiltonian : H = Om_t f_x + p_z f_z + q f_z^2 + g/2<f>f, p_z = delta + Dsin(ft+Phi) (modulation)
    # Note that complex wavefunctions are not used due to the differential equation solver does not support complex numbers.

    spr, spi, szr, szi, smr, smi = psi
    rp = spr**2+spi**2; rz = szr**2+szi**2; rm = smr**2+smi**2
    
    # Assign parameter values for Hamiltonian
    delta, Om, g, q = driving
    D, f, Phi  = modulation
    pz = delta + D*np.sin(f*t+Phi)

    # omod: len=2 → [Omega_freq, Omega_phase] (legacy)
    #       len=3 → [Omega1, Omega_freq, Omega_phase]
    #       len>=5 → [Omega1, Omega_freq, Omega_phase, Omega2, Omega2_freq]
    if omod.size == 2:
        Omega1 = 0.0; Omega_freq = omod[0]; Omega_phase = omod[1]
        Omega2 = 0.0; Omega2_freq = 0.0
    else:
        Omega1 = omod[0]; Omega_freq = omod[1]; Omega_phase = omod[2]
        if omod.size >= 5:
            Omega2 = omod[3]; Omega2_freq = omod[4]
        else:
            Omega2 = 0.0; Omega2_freq = 0.0

    # time-dependent Omega
    if Omega_freq == 0.0:
        Om_t = Om
    else:
        Om_t = Om + Omega1 * np.sin(Omega_freq*t + Omega_phase)

    # y-drive
    if (Omega2 == 0.0) or (Omega2_freq == 0.0):
        Py_t = 0.0
    else:
        Py_t = - Omega2 * np.sin(Omega2_freq * t)

    # Allocate an array to store result
    dsdt = np.empty(6, dtype = np.float64)

    # Calculate and assign each components of the derivatives of the spin state. 
    dsdt[0] = g*(-smi*(szr**2-szi**2)+2*smr*szr*szi+spi*(rp+rz-rm))+(pz+q)*spi+Om_t/np.sqrt(2)*szi
    dsdt[1] = -g*(smr*(szr**2-szi**2)+2*smi*szr*szi+spr*(rp+rz-rm))-(pz+q)*spr-Om_t/np.sqrt(2)*szr
    dsdt[2] = g*(-2*szi*(spr*smr-spi*smi)+2*szr*(spr*smi+spi*smr)+szi*(rp+rm))+Om_t/np.sqrt(2)*(spi+smi)
    dsdt[3] = -g*(2*szr*(spr*smr-spi*smi)+2*szi*(spr*smi+spi*smr)+szr*(rp+rm))-Om_t/np.sqrt(2)*(spr+smr)
    dsdt[4] = g*(-spi*(szr**2-szi**2)+2*spr*szr*szi+smi*(rm+rz-rp))+(-pz+q)*smi+Om_t/np.sqrt(2)*szi
    dsdt[5] = -g*(spr*(szr**2-szi**2)+2*spi*szr*szi+smr*(rm+rz-rp))-(-pz+q)*smr-Om_t/np.sqrt(2)*szr

    if Py_t != 0.0:
        dsdt[0] +=  Py_t/np.sqrt(2) * szr
        dsdt[1] +=  Py_t/np.sqrt(2) * szi
        dsdt[2] += -Py_t/np.sqrt(2) * (spr - smr)
        dsdt[3] += -Py_t/np.sqrt(2) * (spi - smi)
        dsdt[4] += -Py_t/np.sqrt(2) * szr
        dsdt[5] += -Py_t/np.sqrt(2) * szi

    return dsdt

def Phasespace(rho0, theta_s, E0, params, Nscan=201): 
    m_min, m_max = -(1-rho0), 1-rho0

    f = lambda m: get_spin_energy(np.array([rho0, theta_s, m, 0]), params) - E0
    ms = np.linspace(m_min, m_max, Nscan)

    for a, b in zip(ms[:-1], ms[1:]):
        fa, fb = f(a), f(b)
        if np.isnan(fa) or np.isnan(fb): continue
        
        if fa*fb < 0: # required for root_scalar function
            try:
                root = root_scalar(f, bracket=[a, b]).root
                return True, np.array([rho0, theta_s, root, 0])
            except: pass
    return False, None

@njit(cache=True, fastmath=True)
def fs_rf(t, psi, driving, rf_params, omod):
    """
    Austin & Harris (2025) style time evolution
      - driving: [delta, Om, g, q]
      - rf_params: [p_drive, omega_plus, omega_minus]
    """
    # unpack wavefunction
    spr, spi, szr, szi, smr, smi = psi
    # occupations
    rp = spr*spr + spi*spi
    rz = szr*szr + szi*szi
    rm = smr*smr + smi*smi

    delta, Om, g, q = driving
    p_drive, ωp, ωm = rf_params  # ωp = omega_plus, ωm = omega_minus

    pz = delta

    drive = p_drive * (np.sin(ωp * t) + np.sin(ωm * t))

    Omega_freq, Omega_phase = omod

    # time-dependent Omega
    if Omega_freq == 0.0:
        Om_t = Om
    else:
        Om_t = Om * np.sin(Omega_freq*t + Omega_phase)

    dsdt = np.empty(6, dtype=np.float64)

    dsdt[0] = ( 
        g * (-smi*(szr*szr - szi*szi) + 2*smr*szr*szi + spi*(rp+rz-rm))
      + (pz+q)*spi
      + (Om_t/np.sqrt(2))*szi
      + drive * spi
    )
    dsdt[1] = (
        -g * (smr*(szr*szr - szi*szi) + 2*smi*szr*szi + spr*(rp+rz-rm))
      - (pz+q)*spr
      - (Om_t/np.sqrt(2))*szr
      - drive * spr
    )
    dsdt[2] = (
        g * (-2*szi*(spr*smr - spi*smi) + 2*szr*(spr*smi + spi*smr) + szi*(rp+rm))
      + (Om_t/np.sqrt(2))*(spi + smi)
      + drive * (spi + smi)
    )
    dsdt[3] = (
        -g * (2*szr*(spr*smr - spi*smi) + 2*szi*(spr*smi + spi*smr) + szr*(rp+rm))
      - (Om_t/np.sqrt(2))*(spr + smr)
      - drive * (spr + smr)
    )
    dsdt[4] = (
        g * (-spi*(szr*szr - szi*szi) + 2*spr*szr*szi + smi*(rm+rz-rp))
      + (-pz+q)*smi
      + (Om_t/np.sqrt(2))*szi
      + drive * smi
    )
    dsdt[5] = (
        -g * (spr*(szr*szr - szi*szi) + 2*spi*szr*szi + smr*(rm+rz-rp))
      -(-pz+q)*smr
      - (Om_t/np.sqrt(2))*szr
      - drive * smr
    )

    return dsdt

# =====================================================================
# Verification / rotated-Hamiltonian support (matrix-form; slower but general)
# =====================================================================

def spin1_matrices():
    """Return spin-1 matrices (fx, fy, fz, fz2) in the (m=+1,0,-1) basis."""
    fx = np.array([[0, 1, 0],
                   [1, 0, 1],
                   [0, 1, 0]], dtype=np.complex128) / np.sqrt(2)

    fy = np.array([[0, -1, 0],
                   [1,  0, -1],
                   [0,  1, 0]], dtype=np.complex128) * (1j / np.sqrt(2))

    fz = np.array([[1, 0, 0],
                   [0, 0, 0],
                   [0, 0, -1]], dtype=np.complex128)

    fz2 = fz @ fz
    return fx, fy, fz, fz2


def d1_y(beta: float) -> np.ndarray:
    """Wigner small-d matrix for spin-1 rotation about y by angle beta."""
    cb = np.cos(beta)
    sb = np.sin(beta)
    s2 = np.sqrt(2.0)
    return np.array([
        [(1.0 + cb) / 2.0,   -sb / s2,        (1.0 - cb) / 2.0],
        [ sb / s2,            cb,            -sb / s2       ],
        [(1.0 - cb) / 2.0,    sb / s2,        (1.0 + cb) / 2.0],
    ], dtype=np.float64)


def rotated_spin_matrices_y(beta: float):
    """Return (fx', fy', fz', fz2') with f_i' = U^\dagger f_i U, U=exp(-i beta F_y)."""
    fx, fy, fz, _ = spin1_matrices()
    U = d1_y(beta).astype(np.complex128)  # real entries, treat as complex
    Udag = U.conj().T

    fx_r = Udag @ fx @ U
    fy_r = Udag @ fy @ U
    fz_r = Udag @ fz @ U
    fz2_r = fz_r @ fz_r
    return fx_r, fy_r, fz_r, fz2_r


def _unpack_omod(omod: np.ndarray):
    """Match the legacy omod packing used in fs():
    omod len=2 -> [Omega_freq, Omega_phase]
    len=3 -> [Omega1, Omega_freq, Omega_phase]
    len>=5 -> [Omega1, Omega_freq, Omega_phase, Omega2, Omega2_freq]
    """
    if omod.size == 2:
        Omega1 = 0.0; Omega_freq = float(omod[0]); Omega_phase = float(omod[1])
        Omega2 = 0.0; Omega2_freq = 0.0
    else:
        Omega1 = float(omod[0]); Omega_freq = float(omod[1]); Omega_phase = float(omod[2])
        if omod.size >= 5:
            Omega2 = float(omod[3]); Omega2_freq = float(omod[4])
        else:
            Omega2 = 0.0; Omega2_freq = 0.0
    return Omega1, Omega_freq, Omega_phase, Omega2, Omega2_freq


def fs_matrix_with_mats(t, psi, driving, modulation, omod, fx, fy, fz, fz2):
    """
    Matrix-form RHS for the spin-1 mean-field ODE.

    psi ordering must match fs():
      spr, spi, szr, szi, smr, smi = psi
    i.e. [Re(psi+), Im(psi+), Re(psi0), Im(psi0), Re(psi-), Im(psi-)].
    """
    spr, spi, szr, szi, smr, smi = psi
    sp = spr + 1j*spi
    sz = szr + 1j*szi
    sm = smr + 1j*smi
    s = np.array([sp, sz, sm], dtype=np.complex128)

    delta, Om, g, q = driving
    D, f, Phi = modulation
    pz = delta + D*np.sin(f*t + Phi)

    Omega1, Omega_freq, Omega_phase, Omega2, Omega2_freq = _unpack_omod(omod)

    # Omega(t)
    if Omega_freq == 0.0:
        Om_t = Om
    else:
        Om_t = Om + Omega1 * np.sin(Omega_freq*t + Omega_phase)

    # y-drive (keep the same sign convention as fs())
    if (Omega2 == 0.0) or (Omega2_freq == 0.0):
        Py_t = 0.0
    else:
        Py_t = - Omega2 * np.sin(Omega2_freq * t)

    # mean-field spin expectation values
    Fx = np.vdot(s, fx @ s)
    Fy = np.vdot(s, fy @ s)
    Fz = np.vdot(s, fz @ s)

    # Hamiltonian operator (scaled Hz units), consistent with fs()
    Hop = (
        pz * fz
        + q * fz2
        + Om_t * fx
        + Py_t * fy
        + g * (Fx * fx + Fy * fy + Fz * fz)
    )

    ds = (-1j) * (Hop @ s)

    return np.array(
        [ds[0].real, ds[0].imag, ds[1].real, ds[1].imag, ds[2].real, ds[2].imag],
        dtype=np.float64
    )

# =======================================
# Quantum chaos
# =======================================

# fock idx -> fock state
def fock(idx:int, N:int): # safety assured for N<=10^4
    Ns = ((N+1)*(N+2))//2
    if (idx>=Ns) or idx<0 : return None 
    S = int(np.ceil(np.sqrt(2*(Ns-idx)+0.25)-1.5))
    Nz = N-S; Np = ((S+1)*(S+2))//2 - (Ns-idx); Nm = (Ns-idx)-(S*(S+1))//2 -1
    return Nz, Np, Nm

# construct 3Hn fock bases
def fock_basis(N):
    Ns = ((N+1)*(N+2))//2
    basis=[]
    for idx in range(Ns):
        Nz, Np, Nm = fock(idx, N)
        basis.append((Nz, Np, Nm))
    return basis

# fock state -> fock idx
def fock_idx(Nz, Np, Nm): # safety assured for N<=10^4
    if Nz<0 or Np<0 or Nm<0 : return None
    N = Nz + Np + Nm
    Ns = ((N+1)*(N+2))//2
    return Ns - ((Np+Nm+1)*(Np+Nm+2))//2 + Np

# quantum hamiltonian in fock basis
def hamiltonian_fock(param, N):
    g = param.g; q = param.q; d0 = param.delta; Omega = param.Omega
    N_state = int((N+2)*(N+1)/2)

    H0 = np.zeros([N_state,N_state])
    for idx in range(N_state):
        Nz, Np, Nm = fock(idx, N); M = Np-Nm
        # diagonal terms
        H0[idx, idx] = g/N*(Nz*(N-Nz)+M*M/2) + q*(N-Nz) + d0*M

        # off-diagonal interaction terms
        int1 = fock_idx(Nz-2, Np+1, Nm+1) 
        if int1 is not None : H0[int1, idx] = g/N*np.sqrt((Np+1)*(Nm+1)*(Nz)*(Nz-1))
        int2 = fock_idx(Nz+2, Np-1, Nm-1) 
        if int2 is not None : H0[int2, idx] = g/N*np.sqrt((Np)*(Nm)*(Nz+1)*(Nz+2))

        # off-diagonal transverse field terms
        tr_p0 = fock_idx(Nz-1, Np+1, Nm)
        if tr_p0 is not None : H0[tr_p0, idx] = Omega/np.sqrt(2)*np.sqrt((Nz)*(Np+1))
        tr_0p = fock_idx(Nz+1, Np-1, Nm)
        if tr_0p is not None : H0[tr_0p, idx] = Omega/np.sqrt(2)*np.sqrt((Nz+1)*(Np))
        tr_m0 = fock_idx(Nz-1, Np, Nm+1)
        if tr_m0 is not None : H0[tr_m0, idx] = Omega/np.sqrt(2)*np.sqrt((Nz)*(Nm+1))
        tr_0m = fock_idx(Nz+1, Np, Nm-1)
        if tr_0m is not None : H0[tr_0m, idx] = Omega/np.sqrt(2)*np.sqrt((Nz+1)*(Nm))

    H0 = (H0 + H0.T)/2 # ensure hermitian
    return H0

# block diagonalize Z2 symmetry
def hamiltonian_Z2_blocks(H, N, check_comm=True, tol=1e-8):
    # find partners (N0, N, M) <-> (N0, N, -M)
    Ns = H.shape[0]
    partner = np.empty(Ns, dtype=int)
    for idx in range(Ns):
        Nz, Np, Nm = fock(idx, N)
        jdx = fock_idx(Nz, Nm, Np)
        partner[idx] = jdx

    # check Z2 symm. [H, P] = 0
    if check_comm:
        P = np.zeros((Ns, Ns), dtype=H.dtype)
        for idx in range(Ns): P[partner[idx], idx] = 1.0
        assert np.max(np.abs(H @ P - P @ H)) < tol, f"{np.max(np.abs(H @ P - P @ H)):.10f}"
    
    # classify idxs into fixed and pairs
    fixed, pairs = [], []
    seen = np.zeros(Ns, dtype=bool)

    for idx in range(Ns):
        if seen[idx]: continue
        jdx = partner[idx]
        seen[idx] = True
        if jdx == idx:
            fixed.append(idx)
        else:
            seen[jdx] = True
            a, b = (idx, jdx) if idx < jdx else (jdx, idx)
            pairs.append((a, b))
    n_plus = len(fixed) + len(pairs)
    n_minus = len(pairs)

    # basis transform matrix U
    U = np.zeros((Ns, n_plus + n_minus), dtype=H.dtype)
    col = 0
    for idx in fixed: # plus bases, fixed
        U[idx, col] = 1.0
        col += 1
    for a, b in pairs: # plus bases, pairs
        U[a, col] = 1/np.sqrt(2)
        U[b, col] = 1/np.sqrt(2)
        col += 1
    for a, b in pairs: # minus bases, pairs
        U[a, col] = 1/np.sqrt(2)
        U[b, col] = -1/np.sqrt(2)
        col += 1

    # unitary transform
    Hp = U[:, :n_plus].T.conj() @ H @ U[:, :n_plus]
    Hm = U[:, n_plus:].T.conj() @ H @ U[:, n_plus:]
    Hp = (Hp + Hp.T.conj())/2
    Hm = (Hm + Hm.T.conj())/2
    return Hp, Hm, U.T.conj() @ H @ U

# weyl-ordered classical hamiltonian for TWA
def hamiltonian_twa(t, psi, driving, modulation):
    apr, api, a0r, a0i, amr, ami = psi # a = sqrt(N)s
    
    # weyl-ordering : N = aa^+ - 1/2 != aa^+ 
    rp = apr*apr+api*api - 0.5      
    r0 = a0r*a0r+a0i*a0i - 0.5
    rm = amr*amr+ami*ami - 0.5

    delta, Om, g, q = driving; D, f, Phi = modulation
    pz = delta + D*np.sin(f*t+Phi)
    dsdt = np.empty(6, dtype=np.float64)

    dsdt[0] = g*(-ami*(a0r*a0r-a0i*a0i)+2*amr*a0r*a0i + api*(rp+r0-rm)) + (pz+q)*api + Om/np.sqrt(2)*a0i
    dsdt[1] = -g*(amr*(a0r*a0r-a0i*a0i)+2*ami*a0r*a0i + apr*(rp+r0-rm)) - (pz+q)*apr - Om/np.sqrt(2)*a0r
    dsdt[2] = g*(-2*a0i*(apr*amr-api*ami)+2*a0r*(apr*ami+api*amr) + a0i*(rp+rm)) + Om/np.sqrt(2)*(api+ami)
    dsdt[3] = -g*(2*a0r*(apr*amr-api*ami)+2*a0i*(apr*ami+api*amr) + a0r*(rp+rm)) - Om/np.sqrt(2)*(apr+amr)
    dsdt[4] = g*(-api*(a0r*a0r-a0i*a0i)+2*apr*a0r*a0i + ami*(rm+r0-rp)) + (-pz+q)*ami + Om/np.sqrt(2)*a0i
    dsdt[5] = -g*(apr*(a0r*a0r-a0i*a0i)+2*api*a0r*a0i + amr*(rm+r0-rp)) - (-pz+q)*amr - Om/np.sqrt(2)*a0r

    return dsdt