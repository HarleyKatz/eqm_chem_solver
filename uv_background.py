import numpy as np

from scipy.interpolate import interp1d

# Load in the data
hm12_dir = "./data/HM12"
redshifts          = np.loadtxt(f"{hm12_dir}/redshifts.dat")
hydrogen_pi_rates  = np.loadtxt(f"{hm12_dir}/hydrogen_pi.dat")
helium_pi_rates    = np.loadtxt(f"{hm12_dir}/helium_pi.dat")
carbon_pi_rates    = np.loadtxt(f"{hm12_dir}/carbon_pi.dat")
nitrogen_pi_rates  = np.loadtxt(f"{hm12_dir}/nitrogen_pi.dat")
oxygen_pi_rates    = np.loadtxt(f"{hm12_dir}/oxygen_pi.dat")
neon_pi_rates      = np.loadtxt(f"{hm12_dir}/neon_pi.dat")
magnesium_pi_rates = np.loadtxt(f"{hm12_dir}/magnesium_pi.dat")
silicon_pi_rates   = np.loadtxt(f"{hm12_dir}/silicon_pi.dat")
sulfur_pi_rates    = np.loadtxt(f"{hm12_dir}/sulfur_pi.dat")
iron_pi_rates      = np.loadtxt(f"{hm12_dir}/iron_pi.dat")

uvbg_rates_interp_list_hydrogen = [
    interp1d(redshifts,hydrogen_pi_rates)
]
uvbg_rates_interp_list_helium = [
    interp1d(redshifts,helium_pi_rates[:,i]) for i in range(helium_pi_rates.shape[1])
]
uvbg_rates_interp_list_carbon = [
    interp1d(redshifts,carbon_pi_rates[:,i]) for i in range(carbon_pi_rates.shape[1])
]
uvbg_rates_interp_list_nitrogen = [
    interp1d(redshifts,nitrogen_pi_rates[:,i]) for i in range(nitrogen_pi_rates.shape[1])
]
uvbg_rates_interp_list_oxygen = [
    interp1d(redshifts,oxygen_pi_rates[:,i]) for i in range(oxygen_pi_rates.shape[1])
]
uvbg_rates_interp_list_neon = [
    interp1d(redshifts,neon_pi_rates[:,i]) for i in range(neon_pi_rates.shape[1])
]
uvbg_rates_interp_list_magnesium = [
    interp1d(redshifts,magnesium_pi_rates[:,i]) for i in range(magnesium_pi_rates.shape[1])
]
uvbg_rates_interp_list_silicon = [
    interp1d(redshifts,silicon_pi_rates[:,i]) for i in range(silicon_pi_rates.shape[1])
]
uvbg_rates_interp_list_sulfur = [
    interp1d(redshifts,sulfur_pi_rates[:,i]) for i in range(sulfur_pi_rates.shape[1])
]
uvbg_rates_interp_list_iron = [
    interp1d(redshifts,iron_pi_rates[:,i]) for i in range(iron_pi_rates.shape[1])
]
