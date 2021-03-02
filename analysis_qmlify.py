import numpy as np
import os

from qmlify.analysis import aggregate_per_pair_works, fully_aggregate_work_dict


ligand_indices = [(4,0)]

work_pair_dictionary = aggregate_per_pair_works(ligand_indices, annealing_steps = {'complex': 5000, 'solvent':5000}) #this may take a few minutes
aggregated_work_dictionary, concatenated_work_dictionary = fully_aggregate_work_dict(work_pair_dictionary) #process the pair dictionary into something that is easier to analyze
np.savez('work_dictionaries.npz', aggregated_work_dictionary, concatenated_work_dictionary) #save into a compressed file

work_dictionaries = np.load('work_dictionaries.npz', allow_pickle=True)
agg_dict, concat_dict = work_dictionaries['arr_0'].item(), work_dictionaries['arr_1'].item()
# experimental = np.load('tyk2_experimental.npz')['arr_0']

from qmlify.analysis import compute_BAR
agg_BAR = compute_BAR(agg_dict)
concat_BAR = compute_BAR(concat_dict)

from qmlify.plotting import *
generate_work_distribution_plots(concat_dict, concat_BAR, 'hpk1')


corrected_concat_BAR = {}
for key, val in concat_BAR.items():
    new_val = {_key: _val[0] for _key, _val in val.items()}
    corrected_concat_BAR[key] = new_val

from openmmtools.constants import kB
from simtk import unit

temperature = 300.0 * unit.kelvin
kT = temperature*kB

def _per_ligand_correction(ml_corrections, kT):
    """Take the output corrections for both complex and solvent phases and turn into a single, per-ligand correction
    Parameters
    ----------
    ml_corrections : dict(int:dict)
        ML/MM corrections from qmlify
    kT : unit.Quantity(float) compatible with unit.kilocalories_per_mole
        thermal energy
    Returns
    -------
    dict
        dictionary of format ligand_ID:(correction, uncertainty)
    """
    binding_corrections = {}
    for ligand in ml_corrections.keys():
        binding_corr = ((ml_corrections[ligand]['complex'][0] - ml_corrections[ligand]['solvent'][0])*kT).value_in_unit(unit.kilocalorie_per_mole)
        binding_corr_err = ((ml_corrections[ligand]['complex'][1]**2 + ml_corrections[ligand]['solvent'][1]**2)**0.5*kT).value_in_unit(unit.kilocalorie_per_mole)
        binding_corrections[ligand] = (binding_corr, binding_corr_err)
    return binding_corrections

def _relative_corrections(corrections):
    """Shift the MM->ML/MM corrections to have a mean of zero.
    Parameters
    ----------
    corrections : dict
        dictionary of corrections
    Returns
    -------
    dict
        dictionary of corrections, with mean of zero
    """
    shift = np.min(list(corrections.values()))
#     print(f'shift is {shift}')
    for lig in corrections.keys():
        corrections[lig] = corrections[lig] - shift
    return corrections


binding_corrections = _per_ligand_correction(corrected_concat_BAR, kT) #query and compute the binding free energy corrections
print(binding_corrections)
shifted_corrections = _relative_corrections(binding_corrections) #shift the binding free energy corrections off from the minimal correction (not strictly necessary)
print(shifted_corrections)