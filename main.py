"""
Porting RTZ into python

Here we fix the temperature and the photon number density
so that we can evolve into equilibrium
"""
import periodictable

element_dict = [
    periodictable.elements.H,
    periodictable.elements.He,
    periodictable.elements.C,
    periodictable.elements.N,
    periodictable.elements.O,
    periodictable.elements.Ne,
    periodictable.elements.Mg,
    periodictable.elements.Si,
    periodictable.elements.S,
    periodictable.elements.Ca,
    periodictable.elements.Fe,
]

def get_ne(element_number_densities,ion_fractions):
    """
    Returns the electron number density (cm^-3)

    element_number_densities: element number densities (cm^-3)
    ion_fractions: ion fractions for each element (list of lists)
    """

    # Initialize elements
    ne = 0.0

    # Loop over elements
    for i in range(len(element_number_densities)):
        for j in range(len(ion_fractions[i])):
            # ne = n_element * electrons/ion * ion_fraction
            ne += element_number_densities[i] * j * ion_fractions[j]

    return ne
    

def rt_solve_cooling(TK, 
                     xH, xHe,
                     nH, nHe,          
                     xC, xN, xO, xNe, xMg, xSi, xS, xCa, xFe,  
                     nC, nN, nO, nNe, nMg, nSi, nS, nCa, nFe   
                    ):
    """
    """

    element_number_densities = [
        nH, nHe, nC, nN, nO, nNe, nMg, nSi, nS, nCa, nFe
    ]
    
    ion_fractions = [
        list(xH),
        list(xHe),
        list(xC),
        list(xN),
        list(xO),
        list(xNe),
        list(xMg),
        list(xSi),
        list(xS),
        list(xCa),
        list(xFe)
    ]

