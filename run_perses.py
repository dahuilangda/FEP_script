import yaml 
import sys
import itertools
import os
from rdkit import Chem

import simtk.unit as unit
from perses.samplers.multistate import HybridRepexSampler
from openmmtools.multistate import MultiStateReporter
from perses.annihilation.lambda_protocol import LambdaProtocol

def run_relative_perturbation(ligA, ligB,tidy=True):
    print(f'Starting relative calcluation of ligand {ligA} to {ligB}')
    trajectory_directory = f'lig{ligA}to{ligB}'
    new_yaml = f'ani_{ligA}to{ligB}.yaml'
    
    # rewrite yaml file
    with open(f'ani.yaml', "r") as yaml_file:
        options = yaml.load(yaml_file, Loader=yaml.FullLoader)
    options['old_ligand_index'] = ligA
    options['new_ligand_index'] = ligB
    options['ligand_file'] = sys.argv[1]
    options['trajectory_directory'] = f'{trajectory_directory}'
    options['save_setup_pickle_as'] = f'lig{ligA}to{ligB}.pkl'
    with open(new_yaml, 'w') as outfile:
        yaml.dump(options, outfile)
    
    # run the simulation
    os.system(f'perses-relative {new_yaml}')

    print(f'Relative calcluation of ligand {ligA} to {ligB} complete')

    if tidy:
        os.remove(new_yaml)

    return

sdf = Chem.SDMolSupplier(sys.argv[1])
mols = [s for s in sdf]
n_mols = len(mols)

suffix = 'run'; selection = 'not water'; checkpoint_interval = 10; n_states = 100; n_cycles = 5000
#lambda_protocol = LambdaProtocol(functions='default')
#ligand_pairs = []
for i,j in zip([0]*n_mols, range(1,n_mols)):
    #ligand_pairs.append((i,j))
    #ligand_pairs = (i,j)
    #ligand1, ligand2 = ligand_pairs[int(sys.argv[1])-1] # jobarray starts at 1 
    ligand1, ligand2 = i, j
    if not os.path.exists(f'lig{ligand2}to{ligand1}'):

        #running forwards
        run_relative_perturbation(ligand2, ligand1)
        
        #mcmc
        reporter = MultiStateReporter(storage=f'lig{ligand2}to{ligand1}/out-complex.nc')
        simulation = HybridRepexSampler.from_storage(reporter)
        simulation.extend(n_cycles)

        reporter = MultiStateReporter(storage=f'lig{ligand2}to{ligand1}/out-solvent.nc')
        simulation = HybridRepexSampler.from_storage(reporter)
        simulation.extend(n_cycles)

        reporter = MultiStateReporter(storage=f'lig{ligand2}to{ligand1}/out-vacuum.nc')
        simulation = HybridRepexSampler.from_storage(reporter)
        simulation.extend(n_cycles)
