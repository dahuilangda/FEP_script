import sys
from perses.analysis.load_simulations import Simulation

from rdkit import Chem

sim = Simulation(sys.argv[1])
sdf = Chem.SDMolSupplier('ligands_shifted.sdf')
ligand2 = sys.argv[1][3:].split('to')[0]
for step, s in enumerate(sdf):
    if step == int(ligand2):
        print(Chem.MolToSmiles(s))
print(sim.report())