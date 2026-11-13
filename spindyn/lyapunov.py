import numpy as np
import spindyn.solve as spinsolve
from spindyn.analysis import *

def lyapunov_exponent(PS0, parameter_real, tstep_real, d0, tau, N_iter):
    # set another state (PS1) displaced from PS0 by d0
    dist_raw = np.random.rand(4)
    dist_norm = dist_raw / PS_distance(dist_raw, np.array([0, 0, 0,0])) * d0
    PS1 = PS0 + dist_norm

    # Convert to wavefunction
    psi0 = State("PS", PS0).to("psi")
    psi1 = State("PS", PS1).to("psi")

    Lyapunov_exp_arr = []
    direction = np.zeros((int(N_iter/1)+2,4))
    direction[0] = dist_raw

    for i in range(N_iter):
        tlist_real = np.arange(tau * i, tau * (i + 1) + tstep_real * 0.1, tstep_real) 

        # Time evolution
        psi0_sol = spinsolve.evolution(psi0, tlist_real, parameter_real, 
                                       savefolder=None, savename=None, savemode=0, printmode=0, rtol=1e-8)
        psi1_sol = spinsolve.evolution(psi1, tlist_real, parameter_real, 
                                       savefolder=None, savename=None, savemode=0, printmode=0, rtol=1e-8)
        # Normalize
        psi0_f = psi_normalize(psi0_sol[-1, :])
        psi1_f = psi_normalize(psi1_sol[-1, :])
        
        # Distance in phase space
        PS0_f = State("psi", psi0_f).to("PS")
        PS1_f = State("psi", psi1_f).to("PS")
        dist = PS_distance(PS0_f, PS1_f)

        # Lyapunov exponent: log(distance growth) / time
        Lyapunov_exp = np.log(dist / d0) / (tlist_real[-1] - tlist_real[0])
        Lyapunov_exp_arr.append(Lyapunov_exp)

        # Reset distance
        PS1_f_moved = PS0_f + (PS1_f-PS0_f)/PS_distance(PS0_f, PS1_f)*d0
        if i%1 == 0 : direction[int(i/1)+1] = (PS1_f-PS0_f)/PS_distance(PS1_f, PS0_f)

        # Update initial states for next iteration
        psi0 = State("PS", PS0_f).to("psi")
        psi1 = State("PS", PS1_f_moved).to("psi")

    # Compute cumulative average
    Lyapunov_exp_arr = np.array(Lyapunov_exp_arr)
    Cumulative_Lyapunov_exp = np.cumsum(Lyapunov_exp_arr) / np.arange(1, N_iter + 1)

    return Cumulative_Lyapunov_exp[-1], Cumulative_Lyapunov_exp, direction