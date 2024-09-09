import numpy as np

import recombination 
import collisional_ionization 
import uv_background as uvb

# Define the recombination and collisional ionization functions dict

def initialize_element_dict():
    """
    Initialize element dictionary with recombination, collisional 
    ionization, and UV background functions as well as various
    other atomic data
    """

    data_dict = {
        "hydrogen": {
            "rec_func": recombination.recombination_hydrogen,
            "col_ion_func": collisional_ionization.coll_ion_generic,
            "uvb_pi_func": uvb.uvbg_rates_interp_list_hydrogen,
            "n_ions": 2,
            "atomic_number": 1,
            "solar": 1.0,
        },
        "helium": {
            "rec_func": recombination.recombination_helium,
            "col_ion_func": collisional_ionization.coll_ion_generic,
            "uvb_pi_func": uvb.uvbg_rates_interp_list_helium,
            "n_ions": 3,
            "atomic_number": 2,
            "solar": 8.51E-02,
        },
        "carbon": {
            "rec_func": recombination.recombination_carbon,
            "col_ion_func": collisional_ionization.coll_ion_generic,
            "uvb_pi_func": uvb.uvbg_rates_interp_list_carbon,
            "n_ions": 7,
            "atomic_number": 6,
            "solar": 2.69E-04,
        },
        "nitrogen": {
            "rec_func": recombination.recombination_nitrogen,
            "col_ion_func": collisional_ionization.coll_ion_generic,
            "uvb_pi_func": uvb.uvbg_rates_interp_list_nitrogen,
            "n_ions": 8,
            "atomic_number": 7,
            "solar": 6.76E-05,
        },
        "oxygen": {
            "rec_func": recombination.recombination_oxygen,
            "col_ion_func": collisional_ionization.coll_ion_generic,
            "uvb_pi_func": uvb.uvbg_rates_interp_list_oxygen,
            "n_ions": 9,
            "atomic_number": 8,
            "solar": 4.90E-04,
        },
        "neon": {
            "rec_func": recombination.recombination_neon,
            "col_ion_func": collisional_ionization.coll_ion_generic,
            "uvb_pi_func": uvb.uvbg_rates_interp_list_neon,
            "n_ions": 11,
            "atomic_number": 10,
            "solar": 8.51E-05,
        },
        "magnesium": {
            "rec_func": recombination.recombination_magnesium,
            "col_ion_func": collisional_ionization.coll_ion_generic,
            "uvb_pi_func": uvb.uvbg_rates_interp_list_magnesium,
            "n_ions": 13,
            "atomic_number": 12,
            "solar": 3.98E-05,
        },
        "silicon": {
            "rec_func": recombination.recombination_silicon,
            "col_ion_func": collisional_ionization.coll_ion_generic,
            "uvb_pi_func": uvb.uvbg_rates_interp_list_silicon,
            "n_ions": 15,
            "atomic_number": 14,
            "solar": 3.24E-05,
        },
        "sulfur": {
            "rec_func": recombination.recombination_sulfur,
            "col_ion_func": collisional_ionization.coll_ion_generic,
            "uvb_pi_func": uvb.uvbg_rates_interp_list_sulfur,
            "n_ions": 17,
            "atomic_number": 16,
            "solar": 1.32E-05
        },
        "iron": {
            "rec_func": recombination.recombination_iron,
            "col_ion_func": collisional_ionization.coll_ion_generic,
            "uvb_pi_func": uvb.uvbg_rates_interp_list_iron,
            "n_ions": 27,
            "atomic_number": 26,
            "solar": 3.16E-05,
        },
    }

    return data_dict

def get_ne(ion_fracs,el_dict,min_ne=1e-10):
    """
    Calculates the electron number density from 
    a given ion distribution [cm^-3]
    """
    # Initialize to zero
    ne = 0.0

    # Loop over elements
    for el in el_dict.keys():
        n_el = el_dict[el]["n"]
        electron_number = np.arange(len(ion_fracs[el]["new"]))
        ne += n_el * (ion_fracs[el]["new"] * electron_number).sum()

    return max(ne,min_ne)

def reduce_xion(dX,min_xion=1e-10):
    """
    Makes sure that ion abundances don't do below some minimum 
    threshold and then ensures that everything sums to 1
    """
    dX[dX < min_xion] = min_xion

    return dX / dX.sum()

def chem_eqm(data_dict,redshift=6,min_xion=1e-10,metallicity=1e-2,
             nH=1e-3,TK=1e4,ddt=3.15e16,uvb_scale_fac=1.0):
    """
    Calculates chemical equilibrium at constant temperature
    
    Parameters:
    data_dict: dictionary with recomb., coll. ion., and uvb. functions
    redshift: redshift needed for the UVB
    min_xion: minimum ion fraction (needed for stability)
    metallicity: fraction of solar metallicity (using cloudy abundance patterns)
    nH: hydrogen number density [cm^-3]
    TK: temperature [K]
    ddt: initial time step [s]
    uvb_scale_fac: linear scale factor for the UVB

    returns a dictionary with the ion abundances and various convergence
    statistics
    """

    # initialize the UVB rates
    for el in data_dict.keys():
        uvb_pi_func = data_dict[el]["uvb_pi_func"]
        pi_uvb_rates = np.array([
            float(uvb_pi_func[i](redshift)) for i in range(len(uvb_pi_func))
        ])
        data_dict[el]["uvb_pi_rates"] = pi_uvb_rates * uvb_scale_fac

    # initialize the metallicity
    for el in data_dict.keys():
        data_dict[el]["n"] = data_dict[el]["solar"] * nH

    # Initialize the dictionary with ion fractions
    ion_fracs = {
        el: {
            "old": min_xion * np.ones(data_dict[el]["n_ions"]),
            "new": min_xion * np.ones(data_dict[el]["n_ions"]),
        } for el in data_dict.keys()
    }
    # Set everything to completely ionized
    for el in ion_fracs.keys():
        ion_fracs[el]["old"][-1] = 1.0
        ion_fracs[el]["new"][-1] = 1.0
        ion_fracs[el]["old"] = reduce_xion(ion_fracs[el]["old"],min_xion)
        ion_fracs[el]["new"] = reduce_xion(ion_fracs[el]["new"],min_xion)

    """
    Run the model
    """
    print(f"working with {TK}")

    # Initialize 
    success_counter = 0
    convergence_counter = 0
    model_converged = False
    total_iterations = 0
    total_time = 0.0

    # infinite loop
    while True:
        total_iterations += 1

        # set dx_new to dx_old
        for el in ion_fracs.keys():
            ion_fracs[el]["new"] = np.copy(ion_fracs[el]["old"])

        # First get the electron fraction
        ne = get_ne(ion_fracs,data_dict)
    
        # Check that the X% rule isn't violated
        x_percent_rule = True

        # Convergence
        convergence_bool = True

        # Loop over all elements
        for el in ion_fracs.keys():

            # set the rec function
            rec_func      = data_dict[el]["rec_func"]
            col_ion_func  = data_dict[el]["col_ion_func"]
            pi_uvb_rates  = data_dict[el]["uvb_pi_rates"]
            n_ions        = data_dict[el]["n_ions"]
            atomic_number = data_dict[el]["atomic_number"]

            # set dx_new and dx_old
            dx_new = np.copy(ion_fracs[el]["new"])
            dx_old = np.copy(ion_fracs[el]["old"])

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
                    cr += col_ion_func(TK, im-1, el) * ne * dx_new[im-1] 

                #! Photoionization from the less excited state
                if (im > 0): 
                    cr += pi_uvb_rates[im-1] * dx_new[im-1] 

                """
                #! Destruction = collisional ionization + recombination
                """
                de = 0.0

                #! Collisional ionization 
                if (im < atomic_number): 
                    de += col_ion_func(TK, im, el) * ne 

                #! Recombination
                if (im > 0):
                    de += rec_func(TK, im) * ne   

                #! Photoionization 
                if (im < atomic_number): 
                    de += pi_uvb_rates[im] 

                """
                #! Update
                """
                #!  The update
                dx_new[im] = (cr*ddt + dx_new[im])/(1. + de*ddt)  

                # Make sure ions sum to 1
                dx_new = reduce_xion(dx_new,min_xion)

                # Get the new electron fraction
                ne = get_ne(ion_fracs,data_dict)

                """
                #! Convergence
                """
                # X% rule
                if (np.abs(dx_new[im] - dx_old[im])/(dx_old[im] + min_xion)) > 0.1:
                    # If the timestep is too long, half the timestep
                    ddt /= 2.
                    x_percent_rule = False
                    print(el,im,ddt)
                    break
        
            # If the X% rule is violated restart with shorter timestep
            if not x_percent_rule:
                success_counter = 0
                break

            # If step completed for element, set "new" to dx_new
            ion_fracs[el]["new"] = np.copy(dx_new)

            # Check if the model has converged
            if max(np.abs(dx_new - dx_old)/(dx_old + min_xion)) > 1e-2:
                convergence_bool = False

        # Check if the model has converged
        if convergence_bool:
            # If yes, increment counter
            convergence_counter += 1
        else:
            # Reset
            convergence_counter = 0

        # Check if we are converged
        if convergence_counter > 100:
            model_converged = True

        # Loop over elements and copy new and old
        for el in ion_fracs.keys():
            ion_fracs[el]["old"] = np.copy(ion_fracs[el]["new"])

        # update the success counters
        success_counter += 1

        # If we have had ten consecutive successes
        # double the time step and reset the counter
        if (success_counter >= 10):
            ddt *= 2.0
            success_counter = 0

        # Update the total time
        total_time += ddt

        # Finish the calculation if model converged
        if model_converged:
            break

    # Reformat the dictionary
    ion_fracs_new = {
        el: ion_fracs[el]["old"] for el in ion_fracs.keys()
    }
    ion_fracs_new["stats"] = {
        "total_time": total_time,
        "total_iterations": total_iterations,
    }

    return ion_fracs_new
