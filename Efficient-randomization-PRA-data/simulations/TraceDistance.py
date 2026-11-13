import os
import shutil
import sys
from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import io
from scipy.linalg import eigvalsh

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spindyn.analysis import State, psi_normalize
from spindyn.io import load_data
from spindyn.params import SpinParams
from spindyn.solve import evolution

INITIAL_STATE = [0.7, 0.0, 0.279017, 0.0]
PARAMS = SpinParams(delta=0, Omega=22.5, g=45, q=45, D=0, f=60, Phi=0)
N_ENSEMBLE = 128 * 128
PERTURBATION = 0.005
SEED = 42
T_FINAL = 2
DT = 0.002
OUTPUT_DIR = ROOT / "Results" / "TraceDistance"
HAAR_FILE = ROOT / "simulations" / "Haar_sample.mat"


def moments(states, k):
    moment = np.zeros((3**k, 3**k), dtype=complex)
    for phi in states:
        rho = np.outer(phi, np.conjugate(phi))
        term = 1
        for _ in range(k):
            term = np.kron(term, rho)
        moment += term / len(states)
    return moment


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    work_dir = OUTPUT_DIR / "EnsembleEvolution"
    rng = np.random.default_rng(SEED)
    psi0 = psi_normalize(State("PS", np.asarray(INITIAL_STATE)).to("psi"))
    psi_ensemble = psi_normalize(psi0 + rng.standard_normal((N_ENSEMBLE, 6)) * PERTURBATION)
    times = np.arange(0, T_FINAL + 0.1 * DT, DT)

    for i, psi in enumerate(psi_ensemble, start=1):
        folder = work_dir / f"ensemble{i}"
        folder.mkdir(parents=True, exist_ok=True)
        evolution(psi, times, PARAMS, str(folder) + os.sep, "State.csv",
                  savemode=2, printmode=0, rtol=1e-8)

    trajectories = []
    for i in range(1, N_ENSEMBLE + 1):
        folder = work_dir / f"ensemble{i}"
        _, t_load, psi = load_data(str(folder) + os.sep, "State.csv", loadmode=2, printmode=0)
        trajectories.append(State("psi", psi).to("cpsi"))
    trajectories = np.stack(trajectories)

    haar = io.loadmat(HAAR_FILE)["Haar"][0][1].toarray() / comb(4, 2)
    distances = []
    for t in range(len(t_load)):
        distances.append(0.5 * np.sum(np.abs(eigvalsh(haar - moments(trajectories[:, t, :], 2)))))

    pd.DataFrame({"time": t_load, "trace_distance": distances}).to_csv(
        OUTPUT_DIR / "TraceDistance.csv", index=False)
    fig, ax = plt.subplots()
    ax.semilogy(t_load, distances)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"2nd trace distance $\Delta^{(2)}$")
    ax.set_xlim(0, max(t_load))
    ax.set_ylim(1 / (2 * np.sqrt(N_ENSEMBLE)), 1)
    ax.hlines(1 / np.sqrt(N_ENSEMBLE), 0, max(t_load), linestyles="dashed")
    fig.savefig(OUTPUT_DIR / "TraceDistance.png")
    plt.close(fig)
    shutil.rmtree(work_dir)
    print(f"Saved: {OUTPUT_DIR / 'TraceDistance.csv'}")
