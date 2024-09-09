import get_eqm_abundances_const_T as gea

# Initialize the element dictionary
data_dict = gea.initialize_element_dict()

# Run the model to equilibrium
ion_fracs = gea.chem_eqm(
    data_dict,
    nH=1e0,
    TK=2e4,
    uvb_scale_fac=0.0
    )