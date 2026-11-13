from dataclasses import dataclass
import numpy as np
from itertools import product

@dataclass
class SpinParams:
    delta: float = 0.0; Omega: float = 0.0; Omega1: float = 0.0
    g: float = 0.0; q: float = 0.0
    D: float = 0.0; f: float = 0.0
    Phi: float = 0.0

    # [Austin-Harris 2025] modulation parameters
    p_drive: float = 0.0   # Rabi frequency
    pB: float = 0.0   # linear Zeeman shift (Hz)
    qB: float = 0.0   # quadratic Zeeman q_B (Hz)

    # for Omega modulation: Omega(t) = Omega + Omega1 * sin(Omega_freq*t+Omega_phase)
    Omega_freq:  float = 0.0   # Omega frequency [Hz]
    Omega_phase: float = 0.0   # Omega phase    [rad]

    # y-direction drive: -Omega2 * sin(Omega2_freq * t) * Fy
    Omega2: float = 0.0         # [Hz]
    Omega2_freq: float = 0.0    # [Hz]

    PARAMS_METADATA = {
        "delta" : ("driving", "Hz", r"\delta"),
        "Omega" : ("driving", "Hz", r"\Omega"),
        "Omega1"     : ("rdrive","Hz", r"\Omega1"),
        "g" : ("driving", "Hz", r"g"),
        "q" : ("driving", "Hz", r"q"),
        "D" : ("modulation", "Hz", r"D"),
        "f" : ("modulation", "Hz", r"f"),
        "Phi" : ("modulation", "rad", r"\Phi"),
        "p_drive" : ("rf", "Hz",  r"p_{\rm drive}"),
        "pB"      : ("rf", "Hz",  r"p_B"),
        "qB"      : ("rf", "Hz",  r"q_B"),
        "Omega_freq" : ("rdrive","Hz", r"\Omega_freq"),
        "Omega_phase": ("rdrive","rad", r"\Omega_phase"),
        "Omega2"     : ("rdrive","Hz",  r"\Omega_{2}"),
        "Omega2_freq": ("rdrive","Hz",  r"\Omega_{2,{\rm freq}}"),
    }

    @classmethod
    def keys_by_category(cls, category) :
        return [k for k, (cat, *rest) in cls.PARAMS_METADATA.items() if cat == category]
    
    @classmethod
    def latex_symbol(cls, key) :
        return cls.PARAMS_METADATA[key][2]

    def driving_array(self) : 
        return np.array([getattr(self, k) for k in self.keys_by_category("driving")])

    def modulation_array(self) :
        return np.array([getattr(self, k) for k in self.keys_by_category("modulation")])
    
    def rf_array(self):
        return np.array([getattr(self, k) for k in self.keys_by_category("rf")])
    
    @property
    def omega_plus(self) -> float:
        return 2 * np.pi * (self.pB + self.qB)

    @property
    def omega_minus(self) -> float:
        return 2 * np.pi * (self.pB - self.qB)
    
    def omega_mod_array(self):
        return np.array([self.Omega1, self.Omega_freq, self.Omega_phase])

    def omega2_array(self):
        return np.array([self.Omega2, self.Omega2_freq])

    def rescale(self, time_real) :
        drive = self.driving_array()
        if np.all(drive == 0):
            raise ValueError("At least one driving parameter must be non-zero")

        scale = self.g if self.g != 0 else drive[drive != 0][0]

        scaled = SpinParams(
            delta=self.delta / scale, Omega=self.Omega / scale,
            g=self.g / scale, q=self.q / scale,
            D=self.D / scale, f=self.f / scale,
            Phi=self.Phi,  # do not rescale angular phase
            p_drive=self.p_drive / scale,
            pB=self.pB / scale,
            qB=self.qB / scale,
            Omega1=self.Omega1 / scale,
            Omega_freq=self.Omega_freq / scale,
            Omega_phase=self.Omega_phase,
            Omega2=self.Omega2 / scale,
            Omega2_freq=self.Omega2_freq / scale
        )
        time_scaled = time_real * 2 * np.pi * scale
        return scaled, time_scaled

    def __str__(self) :
        string = []
        for category in ["driving", "modulation", "rf", "rdrive"] :
            keys = self.keys_by_category(category)
            line = f"{category.capitalize()} parameters: "
            parts = []
            
            for k in keys : 
                val = getattr(self, k)
                unit = self.PARAMS_METADATA[k][1]
                name = self.PARAMS_METADATA[k][2]
                
                if unit == "rad" : parts.append(f"{name} = {val/np.pi:.1f}π{unit}")
                else : parts.append(f"{name} = {val:.1f}{unit}")

            line += ", ".join(parts)
            string.append(line)

        return "\n".join(string)


def generate_param_grid(varying: dict, default: SpinParams = None) :
    """Generate list of SpinParams from varying grid + default values"""
    default_dict = default.__dict__ if default else {k: 0.0 for k in SpinParams.__annotations__}
    keys, values = zip(*varying.items())
    grid = []
    for combo in product(*values):
        kwargs = {k: v for k, v in zip(keys, combo)}
        full = {**default_dict, **kwargs}
        grid.append(SpinParams(**full))
    return grid
