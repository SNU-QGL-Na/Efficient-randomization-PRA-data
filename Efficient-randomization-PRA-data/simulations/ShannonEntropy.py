import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spindyn.analysis import State, get_spin_energy
from spindyn.io import load_data
from spindyn.phasespace import harr_ensemble, shannon_entropy

STATE_FILE = ROOT / "Results" / "TimeEvolution" / "State.csv"
N_SAMPLES = 100000
BINS = 8
SEED = 42
OUTPUT_FILE = ROOT / "Results" / "Entropy" / "ShannonEntropy.csv"

if __name__ == "__main__":
    params, times, psi = load_data(str(STATE_FILE.parent) + os.sep,
                                   STATE_FILE.name, loadmode=2, printmode=0)
    ps = State("psi", psi[:N_SAMPLES]).to("PS")
    np.random.seed(SEED)
    s_haar = shannon_entropy(harr_ensemble(len(ps)), bins=BINS)[0]
    s = shannon_entropy(ps, bins=BINS)[0]
    result = {
        "energy_over_g": get_spin_energy(ps[0], params) / params.g,
        "S": s,
        "S_Haar": s_haar,
        "S_minus_S_Haar": s - s_haar,
        "coverage": np.exp(s - s_haar),
    }
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(OUTPUT_FILE, index=False)
    print(f"Saved: {OUTPUT_FILE}")
