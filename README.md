# Code and data for “Efficient ensemble randomization by tuning chaos in a nonlinear spin-1 system”

This repository accompanies:

**Efficient ensemble randomization by tuning chaos in a nonlinear spin-1 system**  
Jongmin Kim, Minsung Jeong, Jongyoon Han, and Y. Shin, *Physical Review A* (accepted 9 November 2026).

Article: [10.1103/z7bd-8hzz](https://doi.org/10.1103/z7bd-8hzz)  
Preprint: [arXiv:2605.27448](https://arxiv.org/abs/2605.27448)  
Code: [GitHub repository](https://github.com/SNU-QGL-Na/Efficient-randomization-PRA-data)

This GitHub repository contains plotting scripts and numerical routines for spin-1 dynamics. The `Data_for_Figure/` directory is not included. Download the inputs for Figures 1–10, including four example trajectories, from [Zenodo](https://doi.org/10.5281/zenodo.24161774).

## Repository layout

```text
Data_for_Figure/Figure1/ ... Figure10/  Figure inputs downloaded from Zenodo
plotting/Figure1.py ... Figure10.py     Figure-generation scripts
simulations/                          Numerical calculation scripts
spindyn/                              Spin dynamics and analysis routines
requirements.txt                      Python dependencies
```

## Requirements and quick start

Download the archive from [Zenodo](https://doi.org/10.5281/zenodo.24161774) and copy its `Data_for_Figure/` folder into the repository root, alongside `plotting/` and `simulations/`. This data folder is not included in the GitHub repository.

From the repository root, install the dependencies and run a plotting script:

```bash
python -m pip install -r requirements.txt
python plotting/Figure2.py
```

Run `plotting/FigureN.py` for Figure N. Each script reads `Data_for_Figure/FigureN/` and saves `Figures/FigureN.pdf` and `.png`. Figures are also displayed when a graphical Matplotlib backend is available. The downloaded data can be plotted without rerunning the simulations.

## Running simulations

Run the commands below from the repository root. Each script performs one calculation using the constants near its top. Set `OUTPUT_DIR` or `OUTPUT_FILE` to choose where the results are saved.

`INITIAL_STATE` uses phase-space coordinates `[rho_0, theta_s, m, theta_m]`. The phases are in radians. Hamiltonian frequencies in `SpinParams` are in Hz, and times are in seconds. The evolution scripts use the following default Hamiltonian and solver tolerances `rtol=1e-8`, `atol=1e-6`:

```python
SpinParams(delta=0, Omega=22.5, g=45, q=45, D=0, f=60, Phi=0)
```

`DT` sets the interval between requested trajectory samples; the solver uses adaptive internal integration steps.

### TimeEvolution.py

Evolves one initial spin state and saves its wavefunction trajectory. No input file is required.

```bash
python simulations/TimeEvolution.py
```

| Setting | Default | Meaning |
| --- | --- | --- |
| `INITIAL_STATE` | `ic[6]` = `[0.7, 0, 0.279017, 0]` | Initial phase-space coordinates |
| `PARAMS` | Hamiltonian above | Hamiltonian and drive parameters |
| `T_FINAL` | `100` | Final time in seconds |
| `DT` | `0.001` | Output sampling interval in seconds |
| `OUTPUT_DIR` | `Results/TimeEvolution/` | Directory for both output files |

**Output: `State.csv`**

A tab-delimited text file with **no header**. Each row is one time sample. The default run contains **100,001 rows and 7 columns**, covering 0–100 s. Values are written with 16 decimal places.

| Column | Value | Unit |
| ---: | --- | --- |
| 1 | `time` | s |
| 2 | `Re(psi_+1)` | Dimensionless |
| 3 | `Im(psi_+1)` | Dimensionless |
| 4 | `Re(psi_0)` | Dimensionless |
| 5 | `Im(psi_0)` | Dimensionless |
| 6 | `Re(psi_-1)` | Dimensionless |
| 7 | `Im(psi_-1)` | Dimensionless |

For example, read the trajectory and reconstruct the three complex components with:

```python
import numpy as np

data = np.loadtxt("Results/TimeEvolution/State.csv", delimiter="\t")
time = data[:, 0]
psi = data[:, 1::2] + 1j * data[:, 2::2]  # shape: (number of samples, 3)
populations = np.abs(psi)**2             # columns: m_F = +1, 0, -1
```

**Output: `Run_variables.txt`**

A text file with **no header**, containing one parameter name and numeric value per line, separated by two tabs. It records these 18 fields in order:

```text
delta, Omega, g, q, D, f, Phi,
p_drive, pB, qB, Omega1, Omega_freq, Omega_phase, Omega2, Omega2_freq,
Start_time, End_time, Time_step
```

Frequency parameters are in Hz; `Phi` and `Omega_phase` are in radians; the final three fields are in seconds. The initial state and solver tolerances are specified in the script rather than this file. Keep `Run_variables.txt` alongside `State.csv` when using the trajectory as input to `ShannonEntropy.py`.

### Lyapunov.py

Estimates the largest Lyapunov exponent using two nearby trajectories and repeated separation rescaling. No input file is required.

```bash
python simulations/Lyapunov.py
```

The main settings are `INITIAL_INDEX=6`, `INITIAL_STATE=ic[INITIAL_INDEX]`, `PARAMS`, `N_ITERATION=2000`, `T_ITERATION=0.05` s, `DT=0.001` s, `D0=1e-6`, and `SEED=42+INITIAL_INDEX`. `D0` is the initial/reset phase-space separation, and `T_ITERATION` is the time between resets.

**Output: `Results/Lyapunov/Lyapunov.csv`**

A comma-delimited file with a header and **one summary row**:

```csv
initial_index,energy_over_g,average_LE,standard_error
```

| Column | Meaning | Unit |
| --- | --- | --- |
| `initial_index` | Value of `INITIAL_INDEX` used to label the calculation | Integer |
| `energy_over_g` | Initial-state energy divided by g | Dimensionless |
| `average_LE` | Mean local LE over the latter half of the iterations | s⁻¹ |
| `standard_error` | Sample standard deviation of the same local LEs (`ddof=1`), divided by the square root of their count | s⁻¹ |

With the default settings, the total evolution time is 100 s. Both statistics use iterations **1001–2000**, corresponding to the final 50 s. This script saves the summary; it does not save the individual local-LE series or the two trajectories.

### ShannonEntropy.py

Calculates phase-space Shannon entropy and coverage from an existing wavefunction trajectory.

```bash
python simulations/ShannonEntropy.py
```

**Input:** `STATE_FILE`, defaulting to `Results/TimeEvolution/State.csv`, and the `Run_variables.txt` in the same directory. Run `TimeEvolution.py` first, or point `STATE_FILE` to another trajectory in the same format.

The script uses the first `N_SAMPLES=100000` rows, or all available rows if the trajectory is shorter. `BINS=8` sets eight bins along each histogram coordinate `[rho_0, theta_+, m, theta_-]`, where `theta_+ = (theta_s + theta_m)/2` and `theta_- = (theta_s - theta_m)/2`. Their ranges are `[0, 1]`, `[-pi, pi]`, `[-1, 1]`, and `[-pi, pi]`, respectively. `SEED=42` controls the Haar sample, whose size equals the number of trajectory samples used.

**Output: `Results/Entropy/ShannonEntropy.csv`**

A comma-delimited file with a header and **one summary row**:

```csv
energy_over_g,S,S_Haar,S_minus_S_Haar,coverage
```

| Column | Meaning | Unit |
| --- | --- | --- |
| `energy_over_g` | Energy of the first selected state divided by g | Dimensionless |
| `S` | Histogram Shannon entropy of the selected trajectory samples | nats, using natural logarithms |
| `S_Haar` | Histogram entropy of the Haar reference sample | nats |
| `S_minus_S_Haar` | `S - S_Haar` | nats |
| `coverage` | `exp(S - S_Haar)` | Dimensionless |

### TraceDistance.py

Generates an initial ensemble around `INITIAL_STATE`, evolves its members, and calculates the trace distance between the ensemble's second moment and the Haar second moment.

```bash
python simulations/TraceDistance.py
```

**Input:** the included `simulations/Haar_sample.mat`, selected by `HAAR_FILE`. Trajectories are generated by this script.

The default settings are `INITIAL_STATE=[0.7, 0, 0.279017, 0]`, `PARAMS` as above, `N_ENSEMBLE=128*128`, `PERTURBATION=0.005`, `SEED=42`, `T_FINAL=2` s, and `DT=0.002` s. `PERTURBATION` sets the Gaussian perturbation amplitude in the six real wavefunction components before each ensemble member is normalized.

**Output: `Results/TraceDistance/TraceDistance.csv`**

A comma-delimited file with a header and one row per time sample. The default run produces **1,001 data rows** from 0 to 2 s:

```csv
time,trace_distance
```

| Column | Meaning | Unit |
| --- | --- | --- |
| `time` | Evolution time | s |
| `trace_distance` | One half of the trace norm of the difference between the ensemble and Haar second moments | Dimensionless |

The script also saves `Results/TraceDistance/TraceDistance.png`, showing the same time series on a logarithmic vertical axis. Per-member trajectories are written temporarily under `EnsembleEvolution/` and removed after the final outputs are saved.


## Figure data

The input files on Zenodo are organized under `Data_for_Figure/FigureN/`, where N is the figure number. Figures 8–10 appear in Appendices A–C.

| Folder | Contents |
| --- | --- |
| `Figure1/` | Four example trajectories and their calculation parameters |
| `Figure2/` | LLE versus initial energy and phase-space coverage |
| `Figure3/` | LLE and phase-space coverage versus modulation amplitude |
| `Figure4/` | z-drive LLE parameter scan |
| `Figure5/`, `Figure6/` | Second-moment trace-distance time series |
| `Figure7/` | Trajectory and LLE examples, with an LLE summary table |
| `Figure8/` | Lyapunov calculation diagnostics |
| `Figure9/` | LLE scans versus modulation frequency and amplitude |
| `Figure10/` | y- and z-drive LLE scans |

### File formats

- **Trajectories:** The Figure 1 trajectories use the `State.csv` format described under `TimeEvolution.py`. Folder labels `initial6` and `initial46` correspond to `ic[6]` (chaotic) and `ic[46]` (regular), the same initial states used in Figures 3, 5, and 6. For each initial state, trajectories are supplied at D = 0 and 100 Hz, with all other Hamiltonian parameters and solver tolerances as specified under `TimeEvolution.py`, over 0–100 s at 0.001 s intervals.
- **Figure 2:** `data_Fig_2b.xlsx` supplies the LLE values for panel (a), with columns `idx`, `energy`, `Average LE`, and `Std LE`. `energy` is E/g; the LE and SEM are in Hz and are divided by g = 45 Hz for plotting. `data_Fig_2a.txt` supplies panel (b), with columns `initial index`, `E/g`, and `S-S_Haar`.
- **Figure 3:** `data_regular_Fig3.txt` and `data_chaotic_Fig3.txt` contain `D`, `S-S_Haar`, and the sample count. For Figures 2–3, coverage is `exp(S-S_Haar)`, calculated with a rectangular four-dimensional histogram using eight bins per coordinate. The LLE table's `cum_LE_std_1` and `cum_LE_std_2` columns contain SEM values.
- **Trace distance:** Figure 5–6 CSVs have headers `time,state1`. `time` is in seconds and `state1` is the second-moment trace distance. The filename identifies the drive amplitude. The labels `initial6` and `initial46` correspond to `ic[6]` and `ic[46]`: ensembles centered on the chaotic and regular initial states used in Figure 3, respectively.
- **LLE tables:** `Average_LE` contains the LLE in Hz. The Figure 7 `Std_LE` column contains the plotted SEM. Calculation settings are recorded in the tables where available. The running averages in Figures 7(d) and 8(b) use the latter half of the iterations available at each point.

### Scope of the data release

The Zenodo archive contains the inputs needed to reproduce the figures, including processed observables, four example wavefunction trajectories for Figure 1, and selected trajectory observables and Lyapunov diagnostics for Figures 7–8. Full wavefunction trajectories for every parameter-scan point and the individual ensemble-member trajectories underlying Figures 5–6 are not included; the supplied trace-distance time series are sufficient to reproduce those figures. In the companion code, `simulations/TraceDistance.py` treats individual ensemble-member trajectories as temporary intermediate files and removes them after saving the trace-distance outputs.

## Citation

Please cite the article and the associated dataset.

```bibtex
@article{kim_spin1_randomization_2026,
  author  = {Kim, Jongmin and Jeong, Minsung and Han, Jongyoon and Shin, Y.},
  title   = {Efficient ensemble randomization by tuning chaos in a nonlinear spin-1 system},
  journal = {Physical Review A},
  year    = {2026},
  doi     = {10.1103/z7bd-8hzz},
  url     = {https://doi.org/10.1103/z7bd-8hzz},
  note    = {Accepted}
}

@dataset{kim_spin1_data_2026,
  author    = {Kim, Jongmin and Jeong, Minsung and Han, Jongyoon and Shin, Y.},
  title     = {Data and analysis scripts for "Efficient ensemble randomization by tuning chaos in a nonlinear spin-1 system"},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24161774},
  url       = {https://doi.org/10.5281/zenodo.24161774}
}
```

## License

Code is distributed under the [MIT License](https://opensource.org/license/mit). Data are distributed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

