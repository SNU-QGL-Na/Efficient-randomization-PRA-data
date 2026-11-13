import spindyn.global_string as sdgs
import spindyn.hamiltonian as sdh
from spindyn.params import SpinParams
from scipy.integrate import solve_ivp

import numpy as np
import datetime as dt
import time
import os


# Verification: rotated Hamiltonian switch
_ROTATED_HAM_ENABLED = False
_ROT_BETA = 0.0

def set_rotated_hamiltonian(enabled: bool, beta: float = 0.0):
    """
    If enabled=True, evolution() (non-RF branch) integrates with a matrix-form RHS
    using rotated spin operators f_i' = U^† f_i U (y-axis, angle beta).

    This does NOT modify SpinParams, so existing utilities remain compatible.
    Intended for verification/debug (slower than fs()).
    """
    global _ROTATED_HAM_ENABLED, _ROT_BETA
    _ROTATED_HAM_ENABLED = bool(enabled)
    _ROT_BETA = float(beta)

def clear_rotated_hamiltonian():
    """Disable rotated Hamiltonian integration."""
    set_rotated_hamiltonian(False, 0.0)

def evolution(
    psi0,
    tlist_real,
    params_real: SpinParams,
    savefolder=0,
    savename=sdgs.default_result_file,
    savemode=1,
    printmode=1,
    rtol=1e-8,
    atol=1e-6,
):
    # Evolving initial state under specific spin Hamiltonian and saving the states at times included in time list.
    # input: Psi0: initial normalized state (1D array with length 6). [real(psi_p), imag(psi_p), real(psi_z), imag(psi_z), real(psi_m), imag(psi_m)]
    #        tlist_real: 1D array of times to evaluate spin state in [s]
    #        params_real : SpinParams object with Hamiltonian parameters in [Hz]. [\delta, \Omega, \epsilon_s, q, D, f, \Phi]
    #        savefolder: folder name string to save result.
    #        savename: file name string to save result. (include extension : e.g. .csv, .txt ..)
    #        savemode: how save the results in some files. one of 0,1,2. 
    #                   0: not save as file. 
    #                   1: save in seperate files for different times. 
    #                   2: save in a file for whole times. 
    #        printmode: how print the solving information and process on the terminal. one of 0,1.
    #                   0: not print
    #                   1: print on the terminal
    # output: solution of the time evolution - 2D array with size L by 6 (L is length of tlist_real). 
    #         savemode = 0 -> no folder/file is created
    #         savemode = 1 -> files in savefolder 'savename1.csv','savename2.csv', ... and 'Run_variables.txt' are created
    #         savemode = 2 -> files in savefolder 'savename.csv' and 'Run_variables.txt' are created        
    #         in 'Run_variables.txt', the Hamiltonian parameters and times are saved.
    #         each row in the saved file 'savename.csv' represents [time[s], real(psi_p), imag(psi_p), real(psi_z), imag(psi_z), real(psi_m), imag(psi_m)]
    #         resulting files are saved all at onse after entire time evolution is solved.
    #         the result is not dependent on time step.
    
    # Check savemode and printmode
    if printmode not in (0,1): 
        printmode = 0
        print(f"Value of printmode should be 0 or 1! (value of {printmode} is entered) - Automatically use mode 0")
    if savemode not in (0,1,2): 
        savemode = 0
        print(f"Value of savemode should be one of 0,1,2! (value of {savemode} is entered) - Automatically use mode 0") 
    
    # Print simulation informations
    if printmode == 1:
        print(f"\n--------------------------------------------------------\n")
        print(f"\nEvolution from {tlist_real[0]}s to {tlist_real[-1]}s, timestep {tlist_real[1]-tlist_real[0]}s")
        print(str(params_real)) # print used parameter values       
        print(f"Evolution starts ..."); start = time.time()
    
    # Simulation: legacy modulation vs. [Austin-Harris 2025] RF drive
    params, tlist = params_real.rescale(tlist_real)
    omod = np.concatenate([params.omega_mod_array(), params.omega2_array()]).astype(np.float64)
    if getattr(params_real, 'use_rf', False):
        rf_params = (params_real.p_drive, params_real.omega_plus, params_real.omega_minus)
        sol = solve_ivp(
            fun = sdh.fs_rf,
            t_span = [tlist[0], tlist[-1]],
            t_eval = tlist,
            y0     = psi0,
            args   = (params.driving_array(), rf_params, omod),
            method = 'RK45',
            rtol   = rtol,
            atol   = atol
        )
    else:
        # Verification: rotated Hamiltonian H' = U^† H U (matrix-form; slower).
        if _ROTATED_HAM_ENABLED:
            fx, fy, fz, fz2 = sdh.rotated_spin_matrices_y(_ROT_BETA)
            sol = solve_ivp(
                fun = sdh.fs_matrix_with_mats,
                t_span = [tlist[0], tlist[-1]],
                t_eval = tlist,
                y0     = psi0,
                args   = (params.driving_array(), params.modulation_array(), omod, fx, fy, fz, fz2),
                method = 'Radau',
                rtol   = rtol,
                atol   = atol
            )
        else:
            sol = solve_ivp(
                fun = sdh.fs,
                t_span = [tlist[0], tlist[-1]],
                t_eval = tlist,
                y0     = psi0,
                args   = (params.driving_array(), params.modulation_array(), omod),
                method = 'Radau',
                rtol   = rtol,
                atol   = atol
            )
    time_psi = np.vstack([tlist_real.reshape(1, -1), sol.y])

    # Print time took for simulation
    if printmode == 1: 
        print(f"Evolution is done (from {tlist_real[0]}s to {tlist_real[-1]}s): Took {time.time()-start:.1f}sec.")
    
    # Save simulation results
    if savemode == 0: pass
    else:
        if savefolder == 0: 
           print(f"Invalid folder name is entered. Folder name is automatically determined.")
           now = dt.datetime.now()
           savefolder = './SpinDynamics/'+now.strftime("%Y%m%d_%H%M%S")+'/'; os.makedirs(savefolder, exist_ok = True)

        if savemode == 2:
            # Save all categories: driving, modulation, rf, rdrive
            cats = ["driving", "modulation", "rf", "rdrive"]
            data_names = [name for c in cats for name in SpinParams.keys_by_category(c)] + sdgs.time_names
            data_values = np.array(
                [getattr(params_real, name) for c in cats for name in SpinParams.keys_by_category(c)]
                + [tlist_real[0], tlist_real[-1], tlist_real[1]-tlist_real[0]],
                dtype=float
            )
            
            with open(savefolder+sdgs.default_var_file, "w") as f:
                for name, value in zip(data_names, data_values):
                    f.write(f"{name}\t\t{value}\n")

            if printmode == 1: print(f"\nRun variables are saved: {savefolder+sdgs.default_var_file}")        

        np.savetxt(savefolder+savename, time_psi.T, fmt = '%.16f', delimiter = '\t')     
        if printmode == 1: print(f"Evolution result is saved: {savefolder+savename}")

    return np.transpose(sol.y)