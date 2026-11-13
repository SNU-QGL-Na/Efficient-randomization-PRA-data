''' Useful transformations of state variables to analysis/visualize
# ----------------------------------------------------------------------
# 1. State variables: psi, cpsi, PS, mag
# psi: normalized spin state function (norm =1), length 6 and all elements are real. [real(psi_p), imag(psi_p), real(psi_z), imag(psi_z), real(psi_m), imag(psi_m)]
# cpsi: normalized complex spin state function (norm =1), length 3 and elements are complex. [psi_p, psi_z, psi_m]
# PS: phase space variables, length 4. [rho_0, theta_s, m, theta_m] 
#     Variables are defined as.. rho_0 = abs(psi_z)**2 / theta_s = arg(psi_p)+arg(psi_m)-2*arg(psi_z) / m = abs(psi_p)**2-abs(psi_m)**2/ theta_m = arg(psi_p)-arg(psi_m)
#     Each variable is confined, 0 <= rho_0 <= 1, -1 <= m <= 1, -2*pi <= theta_s < 2*pi, -pi <= theta_m < pi
# mag: magnetization (spin expectation value) along each axis, length 3. [Fx, Fy, Fz]
# ----------------------------------------------------------------------
# 2. Functions: psi_normalize, State.to, PS_distance, get_spin_energy
# psi_normalize(psi): Normalize spin wavefunction to make its norm be 1
# State("psi", psi).to("PS"): Transform state function to phase space variables
# State("PS", PS).to("psi"): Transform phase space variables to state function
# State("psi", psi).to("cpsi"): Transform real-component state function to complex state function
# State("cpsi", cpsi).to("psi"): Transform complex state function to real-component state function
# State("psi", psi).to("mag"): Transform state function to magnetization (spin vector)
# PS_distance(PS1, PS2): Calculate distance between two states in phase space
# get_spin_energy(state, params_real, t=0): Calculate spin energy in Hz
# ---------------------------------------------------------------------- '''

import numpy as np
from spindyn.params import SpinParams

def psi_normalize(psi): 
    return psi/np.linalg.norm(psi, axis=-1, keepdims=True)

class State:
    ## Initialization
    def __init__(self, form: str, data: np.ndarray):
        self.form = form
        self.data = np.asarray(data)

        # validate once on construction
        if not self.validate(form, self.data):
            raise ValueError(f"Given data is not the form of '{form}'")
    
    # Transformation API
    def to(self, target: str) -> np.ndarray:
        if target == self.form: return self.data

        key = (self.form, target)
        try:
            convert = self.CONVERTERS[key]
        except KeyError:
            raise ValueError(f"Conversion {self.form} → {target} not implemented")
        return convert(self.data)

    ## Validation API
    @classmethod
    def validate(cls, form: str, arr: np.ndarray) -> bool:
        try:
            return cls.VALIDATORS[form](arr)
        except KeyError:
            raise ValueError(f"No validation function for form '{form}'")

    ## Validation Functions
    @staticmethod
    def is_psi(data):
        return isinstance(data, np.ndarray) and ( 
            (data.ndim == 1 and data.size == 6) or (data.ndim > 1 and data.shape[-1] == 6))
    @staticmethod
    def is_PS(data):
        return isinstance(data, np.ndarray) and (
            (data.ndim == 1 and data.size == 4) or (data.ndim > 1 and data.shape[-1] == 4))
    @staticmethod
    def is_cpsi(data):
        return isinstance(data, np.ndarray) and (
            (data.ndim == 1 and data.size == 3) or (data.ndim > 1 and data.shape[-1] == 3))
    @staticmethod
    def is_mag(data):
        return isinstance(data, np.ndarray) and (
            (data.ndim == 1 and data.size == 3) or (data.ndim > 1 and data.shape[-1] == 3))
    
    ## Converting Functions
    @staticmethod
    def psi_to_PS(psi):
        spr, spi, szr, szi, smr, smi = np.moveaxis(psi, -1, 0) # read psi in last axis
        sp, sz, sm = spr + 1j * spi, szr + 1j * szi, smr + 1j * smi 
        rp, rz, rm = np.abs(sp)**2, np.abs(sz)**2, np.abs(sm)**2
        tp, tz, tm = np.angle(sp), np.angle(sz), np.angle(sm)
        m = rp - rm
        
        tplus = tp-tz; tminus = tm-tz
        tplus = (tplus + np.pi) % (2*np.pi) - np.pi     # -pi ~ pi
        tminus = (tminus + np.pi) % (2*np.pi) - np.pi   # -pi ~ pi
        ts = tplus + tminus
        tM = tplus - tminus
        
        return np.stack([rz, ts, m, tM], axis=-1).astype(np.float64) # stack them to last axis

    @staticmethod
    def PS_to_psi(PS):
        rz, ts, m, tM = np.moveaxis(PS, -1, 0)
        tz = 0.0
        tp, tm = (ts + tM) / 2.0, (ts - tM) / 2.0
        rp, rm = (1 + m - rz) / 2.0, (1 - m - rz) / 2.0

        spr, spi = np.sqrt(rp) * np.cos(tp), np.sqrt(rp) * np.sin(tp)
        szr, szi = np.sqrt(rz) * np.cos(tz), np.sqrt(rz) * np.sin(tz)
        smr, smi = np.sqrt(rm) * np.cos(tm), np.sqrt(rm) * np.sin(tm)
        return np.stack([spr, spi, szr, szi, smr, smi], axis=-1).astype(np.float64)

    @staticmethod
    def psi_to_cpsi(psi):
        spr, spi, szr, szi, smr, smi = np.moveaxis(psi, -1, 0)
        sp, sz, sm = spr + 1j * spi, szr + 1j * szi, smr + 1j * smi
        return np.stack([sp, sz, sm], axis=-1).astype(np.complex128)

    @staticmethod
    def cpsi_to_psi(cpsi):
        sp, sz, sm = np.moveaxis(cpsi, -1, 0)
        return np.stack([
            np.real(sp), np.imag(sp),
            np.real(sz), np.imag(sz),
            np.real(sm), np.imag(sm),
        ], axis=-1).astype(np.float64)

    @staticmethod
    def psi_to_mag(psi):
        cpsi = State.psi_to_cpsi(psi)
        sp, sz, sm = np.moveaxis(cpsi, -1, 0)
        Fz = np.abs(sp) ** 2 - np.abs(sm) ** 2
        Fx = (sp * np.conj(sz) + sz * np.conj(sp) + sm * np.conj(sz) + sz * np.conj(sm)).real / np.sqrt(2)
        Fy = (1j* (sp * np.conj(sz) - sz * np.conj(sp) - sm * np.conj(sz) + sz * np.conj(sm))).real / np.sqrt(2)
        return np.stack([Fx, Fy, Fz], axis=-1).astype(np.float64)

    ## Dictionary of possible existing functions
    VALIDATORS = {
        "psi": is_psi.__func__,
        "PS": is_PS.__func__,
        "cpsi": is_cpsi.__func__, 
        "mag": is_mag.__func__  # cannot instantiate directly
    }

    CONVERTERS = {
        ("psi", "PS"): psi_to_PS.__func__,
        ("PS", "psi"): PS_to_psi.__func__,
        ("psi", "cpsi"): psi_to_cpsi.__func__,
        ("cpsi", "psi"): cpsi_to_psi.__func__,
        ("psi", "mag"): psi_to_mag.__func__,
    }

def PS_distance(PS1, PS2):
    # Check whether input arrays are PS
    if np.shape(PS1) != np.shape(PS2): raise Exception('Two state functions should have same shape')
    elif not State.validate('PS', PS1) : raise Exception('First array is not in the form PS')
    elif not State.validate('PS', PS2) : raise Exception('Second array is not in the form PS')

    drho, _, dm, _ = np.moveaxis(PS1-PS2, -1, 0)
    
    # Wrap angle differences to [-pi, pi]
    def angle_diff(a, b):
        return ((a - b + np.pi) % (2 * np.pi)) - np.pi
    
    dt_plus = angle_diff((PS1[..., 1]+PS1[..., 3])/2, (PS2[..., 1]+PS2[..., 3])/2)
    dt_minus = angle_diff((PS1[..., 1]-PS1[..., 3])/2, (PS2[..., 1]-PS2[..., 3])/2)
    dt_plus /= np.pi ; dt_minus /= np.pi # [-1, 1]

    # Evaluate euclidean distance
    distance = np.sqrt(drho**2+dm**2+dt_plus**2+dt_minus**2)
    return distance

# Get spin energy of the state in Hz
def get_spin_energy(state, params_real: SpinParams, t=0):
    delta, Om, g, q = params_real.driving_array()
    D, f, Phi  = params_real.modulation_array()
    pz = delta + D*np.sin(f*t+Phi)
  
    # Convert given state in to PS form
    if np.ndim(state) == 1:
        if len(state) == 3: state = State("psi", State("cpsi", state).to("psi")).to("PS")
        elif len(state) == 6: state = State("psi", state).to("PS")
        elif len(state) != 4: 
            raise Exception('State function should be one of the forms of psi (length 6), cpsi (length 3) or PS (length 4 along last axis)')
        r = state[0]; ts = state[1]; m = state[2]; tm = state[3];
    else:
        if np.shape(state)[-1] == 3: State("psi", State("cpsi", state).to("psi")).to("PS")
        elif np.shape(state)[-1] == 6: state = State("psi", state).to("PS")
        elif np.shape(state)[-1] != 4: 
            raise Exception('State function should be one of the forms of psi (length 6), cpsi (length 3) or PS (length 4 along last axis)')
        r = state[..., 0]; ts = state[..., 1]; m = state[..., 2]; tm = state[..., 3]
    
    # Inside the sqrt should be positive
    if abs(m) > 1-r or r<0 or r>1 : return np.nan
    
    # Evaluate energy
    BasicEnergy = g*r*((1-r)+np.sqrt((1-r)**2-m**2)*np.cos(ts)) + g/2*m**2+q*(1-r)
    RfEnergy = Om*np.sqrt(r)*(np.sqrt(1-r+m)*np.cos((tm+ts)/2) + np.sqrt(1-r-m)*np.cos((ts-tm)/2))
    ModuleEnergy = pz*m

    return  BasicEnergy + RfEnergy + ModuleEnergy

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

__all__ = ['psi_normalize', 'State', 'PS_distance', 'get_spin_energy', 'shannon_entropy']