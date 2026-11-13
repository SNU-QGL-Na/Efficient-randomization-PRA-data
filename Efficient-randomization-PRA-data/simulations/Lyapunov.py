import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spindyn.analysis import State, PS_distance, psi_normalize, get_spin_energy
from spindyn.initial_conditions import ic
from spindyn.params import SpinParams
from spindyn.solve import evolution

INITIAL_INDEX = 6
INITIAL_STATE = ic[INITIAL_INDEX]
PARAMS = SpinParams(delta=0, Omega=22.5, g=45, q=45, D=0, f=60, Phi=0)
N_ITERATION = 2000
T_ITERATION = 0.05
DT = 0.001
D0 = 1e-6
SEED = 42 + INITIAL_INDEX
OUTPUT_FILE = ROOT / "Results" / "Lyapunov" / "Lyapunov.csv"

if __name__ == "__main__":
    ps0 = np.asarray(INITIAL_STATE, dtype=float)
    rng = np.random.default_rng(SEED)
    perturbation = rng.random(4)
    ps1 = ps0 + perturbation / PS_distance(np.zeros(4), perturbation) * D0
    psi0 = State("PS", ps0).to("psi")
    psi1 = State("PS", ps1).to("psi")

    local_le = []
    for i in range(N_ITERATION):
        times = np.arange(i * T_ITERATION, (i + 1) * T_ITERATION + 0.1 * DT, DT)
        sol0 = evolution(psi0, times, PARAMS, savefolder=None, savename=None,
                         savemode=0, printmode=0, rtol=1e-8)
        sol1 = evolution(psi1, times, PARAMS, savefolder=None, savename=None,
                         savemode=0, printmode=0, rtol=1e-8)

        ps0_initial = State("psi", sol0[0]).to("PS")
        ps1_initial = State("psi", sol1[0]).to("PS")
        ps0_final = State("psi", psi_normalize(sol0[-1])).to("PS")
        ps1_final = State("psi", psi_normalize(sol1[-1])).to("PS")
        distance_initial = PS_distance(ps0_initial, ps1_initial)
        distance_final = PS_distance(ps0_final, ps1_final)
        local_le.append(np.log(distance_final / distance_initial) / (times[-1] - times[0]))

        ps1_reset = ps0_final + (ps1_final - ps0_final) / distance_final * D0
        psi0 = State("PS", ps0_final).to("psi")
        psi1 = State("PS", ps1_reset).to("psi")

    last_half = np.asarray(local_le[N_ITERATION // 2:])
    result = {
        "initial_index": INITIAL_INDEX,
        "energy_over_g": get_spin_energy(ps0, PARAMS) / PARAMS.g,
        "average_LE": np.mean(last_half),
        "standard_error": np.std(last_half, ddof=1) / np.sqrt(len(last_half)) if len(last_half) > 1 else np.nan,
    }
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(OUTPUT_FILE, index=False)
    print(f"Saved: {OUTPUT_FILE}")
