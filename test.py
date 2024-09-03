import numpy as np

import recombination 
import collisional_ionization 
import uv_background as uvb
from tqdm import tqdm

# Define the recombination and collisional ionization functiomns
rec_func = recombination.recombination_helium
col_ion_func = collisional_ionization.coll_ion_generic
uvb_pi_func = uvb.uvbg_rates_interp_list_helium

redshift = 2.5
pi_uvb_rates = np.array([
    float(uvb_pi_func[i](redshift)) for i in range(len(uvb_pi_func))
])
pi_uvb_rates *= 0.0

element = "helium"
n_ions = 3
atomic_number = n_ions - 1

min_xion = 1e-10

# Define the number density
n_element = 1e-5

def get_ne(dX,n_element):
    ne = 0.0
    for i in range(n_ions):
        ne += i * n_element * dX[i]
    return ne

def reduce_xion(dX):
    dX[dX < min_xion] = min_xion
    return dX / dX.sum()

all_res = []

# Define the temperature [K]
TTT = 10.**np.arange(2,8.00001,0.01)

ddt = 3.15e7 * 1e9  # 10^9 years
for TK in TTT:
    print(f"working with {TK}")

    # Initialize the ionization fractions
    dx_old = np.ones(n_ions) * min_xion
    dx_old[-1] = 1.0 # initialize everything to ionized

    # infinite loop
    success_counter = 0
    convergence_counter = 0
    model_converged = False
    total_iterations = 0
    while True:
        total_iterations += 1
        # First get the electron fraction
        ne = get_ne(dx_old,n_element)

        # set dx_new to zero
        dx_new = np.copy(dx_old)

        # Loop over the ions and update
        ok = 0
        
        for im in range(n_ions):
            """
            #! Creation
            """
            cr = 0.0

            #! Recombinations of the more excited ionization state
            if (im < atomic_number): 
                cr += rec_func(TK, im+1) * ne * dx_new[im+1]  

            #! Collisional ionization of the less excited state
            if (im > 0): 
                cr += col_ion_func(TK, im-1, element) * ne * dx_new[im-1] 

            #! Photoionization from the less excited state
            if (im > 0): 
                cr += pi_uvb_rates[im-1] * dx_new[im-1] 

            """
            #! Destruction = collisional ionization + recombination
            """
            de = 0.0

            #! Collisional ionization 
            if (im < atomic_number): 
                de += col_ion_func(TK, im, element) * ne 

            #! Recombination
            if (im > 0):
                de += rec_func(TK, im) * ne   

            #! Photoionization 
            if (im < atomic_number): 
                de += pi_uvb_rates[im] 

            """
            #! Update
            """
            dx_new[im] = (cr*ddt + dx_new[im])/(1.+de*ddt)        #!  The update
            # Make sure ions sum to 1
            dx_new = reduce_xion(dx_new)
            # Get the new electron fraction
            ne = get_ne(dx_old,n_element)

            """
            #! Convergence
            """
            # X% rule
            if (np.abs(dx_new[im] - dx_old[im])/dx_old[im]) > 0.1:
                # If the timestep is too long, half the timestep
                ddt = ddt / 2.
                break
            else:
                ok += 1

        # If we successfully completed the timestep
        # perform the update
        if (ok == n_ions):

            # Check if the model has converged
            if max(np.abs((dx_new-dx_old))/dx_old) < 1e-3:
                convergence_counter += 1
            else:
                # Reset
                convergence_counter = 0

            # Check if we are converged
            if convergence_counter > 100:
                model_converged = True

            # Copy the new ion fractions
            dx_old = np.copy(dx_new)

            # update the success counters
            success_counter += 1

            # If we have had ten consecutive successes
            # double the time step and reset the counter
            if (success_counter >= 10):
                ddt *= 2.0
                success_counter = 0
        else:
            success_counter = 0
            convergence_counter = 0

        # Finish the calculation if model converged
        if model_converged:
            break

    all_res.append(dx_old)
    print(f"finished {TK} in {total_iterations} iterations")
all_res = np.array(all_res)


# aa = np.array([8.40e-14, 1.40e-13, 2.62e-13, 4.44e-13, 1.16e-12, 2.10e-12, 3.24e-12, 4.61e-12, 6.00e-12, 7.50e-12, 8.96e-12, 1.08e-11, 1.58e-11, 2.12e-11])
# bb = np.array([1.66e-11, 2.89e-11, 3.88e-11, 1.12e-12, 7.95e-12, 1.88e-11, 3.76e-11, 4.19e-11, 6.40e-11, 8.79e-11, 5.06e-11, 1.10e-32, 6.55e-34, 0.00e+00])

# for i in range(1,15):
#     print(i,aa[i-1]+bb[i-1],rec_func(3e5,i))