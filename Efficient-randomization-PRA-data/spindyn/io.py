import spindyn.global_string as sdgs
from spindyn.params import SpinParams
import numpy as np

def load_data(loadfolder, loadname = sdgs.default_result_file, loadmode = 2, printmode = 1):
    if printmode not in (0,1): 
        printmode = 0
        print(f"Value of printmode should be 0 or 1! (value of {printmode} is entered) - Automatically use mode 0")
    if loadmode not in (1,2):
        loadmode = 2 
        print(f"Value of printmode should be 1 or 2! (value of {loadmode} is entered) - Automatically use mode 2"); loadmode = 2

    parameter = []; modulation = []

    if printmode == 1: 
        print(f"\n--------------------------------------------------------\n")
    
    if loadmode == 2:   
        driving_keys = SpinParams.keys_by_category("driving")
        modulation_keys = SpinParams.keys_by_category("modulation")

        params_dict = {name: None for name in driving_keys}
        modul_dict = {name: None for name in modulation_keys}
        
        with open(loadfolder+sdgs.default_var_file, "r") as file:
            for line in file:
                parts = line.split()
                if len(parts) == 2: 
                    name, value = parts[0], float(parts[1])
                    if name in params_dict: params_dict[name] = value
                    if name in modul_dict: modul_dict[name] = value
        
        parameter = np.array([params_dict[name] for name in driving_keys], dtype=np.float64)
        modulation = np.array([modul_dict[name] for name in modulation_keys], dtype=np.float64)
        
        parameter_real = SpinParams(
           delta=parameter[0], Omega=parameter[1], g=parameter[2], q=parameter[3],  # Driving parameters
            D=modulation[0], f=modulation[1], Phi=modulation[2])                    # Modulation

        if printmode == 1:
            print("Run variables are loaded:")
            print(" Driving parameters:"+" ".join([f"{name} = {val:.1f}{SpinParams.PARAMS_METADATA[name][1]}" for name, val, in zip(driving_keys, parameter)]))
            print(" Modulation parameters:"+" ".join([f"{name} = {val:.1f}{SpinParams.PARAMS_METADATA[name][1]}" for name, val, in zip(modulation_keys, modulation)]))
    
    time_psi = np.loadtxt(loadfolder+loadname)
    t = time_psi[:,0] # read first column (t) 
    s = time_psi[:,1:7] # read psi(t) (dim 6)
    if printmode == 1: print(f"Evolution result is loaded: \n from {t[0]}s to {t[-1]}s, timestep {t[1]-t[0]}s, initial state {s[0]}\n")
    
    return parameter_real, t, s