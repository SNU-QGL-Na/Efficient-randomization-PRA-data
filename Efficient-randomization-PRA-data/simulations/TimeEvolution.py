import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spindyn.analysis import State, psi_normalize
from spindyn.initial_conditions import ic
from spindyn.params import SpinParams
from spindyn.solve import evolution

INITIAL_STATE = ic[6]
PARAMS = SpinParams(delta=0, Omega=22.5, g=45, q=45, D=0, f=60, Phi=0)
T_FINAL = 100
DT = 0.001
OUTPUT_DIR = ROOT / "Results" / "TimeEvolution"

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    times = np.arange(0, T_FINAL + DT * 0.1, DT)
    psi0 = psi_normalize(State("PS", np.asarray(INITIAL_STATE)).to("psi"))
    evolution(psi0, times, PARAMS, str(OUTPUT_DIR) + os.sep,
              "State.csv", savemode=2, printmode=0, rtol=1e-8)
    print(f"Saved: {OUTPUT_DIR / 'State.csv'}")
