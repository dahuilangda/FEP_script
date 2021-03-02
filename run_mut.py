from __future__ import absolute_import

import simtk.openmm as openmm
import simtk.openmm.app as app
from openmmtools.constants import kB
import simtk.unit as unit
import pickle
import os


ENERGY_THRESHOLD = 1e-2
temperature = 300 * unit.kelvin
kT = kB * temperature
beta = 1.0/kT
ring_amino_acids = ['TYR', 'PHE', 'TRP', 'PRO', 'HIS']

# Set up logger
import logging
_logger = logging.getLogger()
_logger.setLevel(logging.INFO)


protein_path = 'Rec_fixed.pdb'
ligands_path = 'Lig.sdf'
chain_id = '1'
residue_id = '718'
mut_residue = 'THR'
output_dir = chain_id +'_'+ residue_id +'_'+ mut_residue
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

from perses.app.relative_point_mutation_setup import PointMutationExecutor
pm_delivery = PointMutationExecutor(receptor_filename=protein_path,
                            mutation_chain_id=chain_id,
                            mutation_residue_id=residue_id,
                                proposed_residue=mut_residue,
                                phase='complex',
                                conduct_endstate_validation=False,
                                ligand_filename=ligands_path,
                                ligand_index=0,
                                forcefield_files=['amber14/protein.ff14SB.xml', 'amber14/tip3p.xml'],
                                barostat=openmm.MonteCarloBarostat(1.0 * unit.atmosphere, temperature, 50),
                                forcefield_kwargs={'removeCMMotion': False, 'ewaldErrorTolerance': 1e-4, 'constraints' : app.HBonds, 'hydrogenMass' : 4 * unit.amus},
                                small_molecule_forcefields='openff-1.2.1')

complex_htf = pm_delivery.get_complex_htf()
pickle.dump(complex_htf, open(os.path.join(output_dir, f"htf-complex-hybrid_factory.pickle"), "wb" ))
apo_htf = pm_delivery.get_apo_htf()
pickle.dump(apo_htf, open(os.path.join(output_dir, f"htf-solvent-hybrid_factory.pickle"), "wb" ))

# Now we can build the hybrid repex samplers
from perses.annihilation.lambda_protocol import LambdaProtocol
from openmmtools.multistate import MultiStateReporter
from perses.samplers.multistate import HybridRepexSampler
from openmmtools import mcmc
suffix = 'run'; selection = 'not water'; checkpoint_interval = 10; n_states = 11; n_cycles = 5000
for phase, htf in zip(['complex', 'solvent'],[complex_htf, apo_htf]):
    lambda_protocol = LambdaProtocol(functions='default')
    reporter_file = os.path.join(output_dir, f'reporter-{phase}.nc')
    reporter = MultiStateReporter(reporter_file, analysis_particle_indices = htf.hybrid_topology.select(selection), checkpoint_interval = checkpoint_interval)
    hss = HybridRepexSampler(mcmc_moves=mcmc.LangevinSplittingDynamicsMove(timestep= 4.0 * unit.femtoseconds,
                                                                            collision_rate=5.0 / unit.picosecond,
                                                                            n_steps=250,
                                                                            reassign_velocities=False,
                                                                            n_restart_attempts=20,
                                                                            splitting="V R R R O R R R V",
                                                                            constraint_tolerance=1e-06),
                                                                            hybrid_factory=htf, online_analysis_interval=10)
    hss.setup(n_states=n_states, temperature=300*unit.kelvin, storage_file=reporter, lambda_protocol=lambda_protocol, endstates=False)
    hss.extend(n_cycles)