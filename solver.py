import numpy as np

from recombination import recombination_oxygen
from collisional_ionization import collisional_ionization_oxygen
from tqdm import tqdm

element_dict = {
    "oxygen": {
        "atomic_number": 8,
        "rec_func": recombination_oxygen,
        "col_ion_func": collisional_ionization_oxygen,
    }
}

def get_ne(dX,n_ions,n_element):
    ne = 0.0
    for i in range(n_ions):
        ne += i * n_element * dX[i]
    return ne

def reduce_xion(dX,min_xion):
    dX[dX < min_xion] = min_xion
    return dX / dX.sum()

def solver(element, n_element=1e-5, min_xion=1e-10, ddt=3.15e16,
          x_percent_rule=0.1, tol=1e-4, conv_its=100):
    atomic_number = element_dict[element]["atomic_number"]
    n_ions = atomic_number + 1
    rec_func = element_dict[element]["rec_func"]
    col_func = element_dict[element]["col_ion_func"]

    all_res = []

    # Define the temperature [K]
    TTT = 10.**np.arange(5,6.00001,0.05)

    for TK in TTT:
        print(f"working with {TK}")

        # Initialize the ionization fractions
        dx_old = np.ones(9) * min_xion
        dx_old[3] = 1.0 # initialize everything to ionized

        # infinite loop
        success_counter = 0
        convergence_counter = 0
        model_converged = False
        total_iterations = 0
        while True:
            total_iterations += 1
            # First get the electron fraction
            ne = get_ne(dx_old,n_ions,n_element)

            # set dx_new to zero
            dx_new = np.copy(dx_old)

            # Loop over the ions and update
            ok = 0
            
            for im in range(n_ions):
                #! Creation
                cr = 0.0

                #! Recombinations of the more excited ionization state
                if (im < atomic_number): 
                    cr = cr + rec_func(TK, im+1) * ne * dx_new[im+1] 

                #! Collisional ionization of the less excited state
                if (im > 0): 
                    cr = cr + col_func(TK, im-1) * ne * dx_new[im-1] 

                #! Destruction = collisional ionization + recombination
                de = 0.0

                #! Collisional ionization 
                if (im < atomic_number): 
                    de = de + col_func(TK, im) * ne 

                #! Recombination
                if (im > 0):
                    de = de + rec_func(TK, im) * ne  

                dx_new[im] = (cr*ddt + dx_new[im])/(1.+de*ddt)        #!  The update

                # X% rule
                if (np.abs(dx_new[im] - dx_old[im])/dx_old[im]) > x_percent_rule:
                    # If the timestep is too long, half the timestep
                    ddt = ddt / 2.
                    break
                else:
                    ok += 1

            # Make sure ions sum to 1
            dx_new = reduce_xion(dx_new,min_xion)
            # Get the new electron fraction
            ne = get_ne(dx_old,n_ions,n_element)

            # If we successfully completed the timestep
            # perform the update
            if (ok == n_ions):

                # Check if the model has converged
                if max(np.abs((dx_new-dx_old))/dx_old) < tol:
                    convergence_counter += 1
                else:
                    # Reset
                    convergence_counter = 0

                # Check if we are converged
                if convergence_counter > conv_its:
                    model_converged = True

                # Copy the new ion fractions
                dx_old = np.copy(dx_new)

                # update the success counters
                success_counter += 1

                # If we have had N consecutive successes
                # double the time step and reset the counter
                if (success_counter >= 10):
                    ddt *= 2.0
                    success_counter = 0
            else:
                # otherwise reset the success counter
                success_counter = 0
                convergence_counter = 0

            # Finish the calculation if model converged
            if model_converged:
                break

        all_res.append(dx_old)
        print(f"finished {TK} in {total_iterations} iterations")
    all_res = np.array(all_res)
    return(all_res)
