from qmlify.executables import perses_extraction_admin
from qmlify.executables import extract_perses_repex_to_local
from qmlify.executor import run
import os
import functools
from p_tqdm import p_map
import numpy as np

ligand_indices = [(1,0)]
i = ligand_indices[0][0]
j = ligand_indices[0][1]
input_dir = f'lig{ligand_indices[0][0]}to{ligand_indices[0][1]}'
output_dir = f'ml_lig{ligand_indices[0][0]}to{ligand_indices[0][1]}'
#output_dir = './'
#print(input_dir, output_dir)

if not os.path.exists(output_dir):
    extract_perses_repex_to_local(input_dir, output_dir, ['solvent', 'complex'])

if not os.path.exists('template_empty.sh'):
    with open('template_empty.sh', 'w') as f:
        f.write('')

def p_func(idx, output_idx, output_dir, state, ligand, direction):
    prefix = f'lig{ligand_indices[0][0]}to{ligand_indices[0][1]}'
    #direction_list = ['forward', 'ani_endstate', 'ani_endstate']
    if ligand == 'old':
        ligand_2 = 'A'
        lb = 0
    else:
        ligand_2 = 'B'
        lb = 1
    try:
        if not os.path.exists(f'{output_dir}/{prefix}.{state}.{ligand}.{direction}.idx_{idx}.5000_steps.positions.npz'):
            if direction == 'forward':
                #print(f'{output_dir}/ligand{ligand_2}lambda{lb}_{state}.positions.npz')
                #print(f'{output_dir}/{prefix}.{state}.{ligand}.{direction}.idx_{output_idx}.5000_steps.positions.npz')
                run({'system': f'{output_dir}/{state}.{ligand}_system.xml', 
                    'subset_system': f'{output_dir}/vacuum.{ligand}_system.xml', 
                    'topology': f'{output_dir}/{state}.{ligand}_topology.pkl', 
                    'subset_topology': f'{output_dir}/vacuum.{ligand}_topology.pkl', 
                    'positions_cache_filename': f'{output_dir}/ligand{ligand_2}lambda{lb}_{state}.positions.npz', 
                    'position_extraction_index': idx, 
                    'integrator_kwargs': None, 
                    'direction': 'forward', 
                    'num_steps': 5000, 
                    'out_positions_npz': f'{output_dir}/{prefix}.{state}.{ligand}.{direction}.idx_{output_idx}.5000_steps.positions.npz', 
                    'out_works_npz': f'{output_dir}/{prefix}.{state}.{ligand}.{direction}.idx_{output_idx}.5000_steps.works.npz'})
            elif direction == 'ani_endstate':
                print(f'{output_dir}/{prefix}.{state}.{ligand}.forward.idx_{idx}.5000_steps.positions.npz')
                print(f'{output_dir}/{prefix}.{state}.{ligand}.ani_endstate.idx_{output_idx}.5000_steps.positions.npz')
                run({'system': f'{output_dir}/{state}.{ligand}_system.xml',
                    'subset_system': f'{output_dir}/vacuum.{ligand}_system.xml',
                    'topology': f'{output_dir}/{state}.{ligand}_topology.pkl',
                    'subset_topology': f'{output_dir}/vacuum.{ligand}_topology.pkl',
                    'positions_cache_filename': f'{output_dir}/{prefix}.{state}.{ligand}.forward.idx_{idx}.5000_steps.positions.npz',
                    'position_extraction_index': 0,
                    'integrator_kwargs': None,
                    'direction': 'ani_endstate',
                    'num_steps': 5000,
                    'out_positions_npz': f'{output_dir}/{prefix}.{state}.{ligand}.ani_endstate.idx_{output_idx}.5000_steps.positions.npz',
                    'out_works_npz': f'{output_dir}/{prefix}.{state}.{ligand}.ani_endstate.idx_{output_idx}.5000_steps.works.npz'})
            elif direction == 'backward':
                print(f'{output_dir}/{prefix}.{state}.{ligand}.ani_endstate.idx_{idx}.5000_steps.positions.npz')
                print(f'{output_dir}/{prefix}.{state}.{ligand}.backward.idx_{output_idx}.5000_steps.positions.npz')
                run({'system': f'{output_dir}/{state}.{ligand}_system.xml',
                    'subset_system': f'{output_dir}/vacuum.{ligand}_system.xml',
                    'topology': f'{output_dir}/{state}.{ligand}_topology.pkl',
                    'subset_topology': f'{output_dir}/vacuum.{ligand}_topology.pkl',
                    'positions_cache_filename': f'{output_dir}/{prefix}.{state}.{ligand}.ani_endstate.idx_{idx}.5000_steps.positions.npz',
                    'position_extraction_index': 0,
                    'integrator_kwargs': None,
                    'direction': 'backward',
                    'num_steps': 5000,
                    'out_positions_npz': f'{output_dir}/{prefix}.{state}.{ligand}.backward.idx_{output_idx}.5000_steps.positions.npz',
                    'out_works_npz': f'{output_dir}/{prefix}.{state}.{ligand}.backward.idx_{output_idx}.5000_steps.works.npz'})
            else:
                pass
    except Exception as e:
        print(f'Error {e}')

def extract_and_subsample_forward_works(i,j,phase,state,annealing_steps, parent_dir, num_resamples, resample=True):
    """
    after forward annealing, query the output positions and work files;
    in the event that some fail, they will not be written;
    pull the index labels of the work/position files, match them accordingly, and assert that the size of the work array is appropriately defined;
    normalize the works and return a subsampled array of size num_resamples
    arguments
        i : int
            lig{i}to{j}
        j : int
            lig{i}to{j}
        phase : str
            the phase
        state : str
            'old' or 'new'
        annealing_steps : int
            number of annealing steps to extract
        parent_dir : str
            full path of the parent dir of lig{i}to{j}
        num_resamples : int
            number of resamples to pull
        resample : bool, default True
            whether to actually resample or just return the full list
    returns
        resamples : np.array(num_resamples)
            resampled indices
    """
    import glob
    import os
    import numpy as np
    from qmlify.utils import exp_distribution

    #define a posiiton and work template
    positions_template = os.path.join(parent_dir, f"lig{i}to{j}.{phase}.{state}.forward.*.{annealing_steps}_steps.positions.npz")
    works_template = os.path.join(parent_dir, f"lig{i}to{j}.{phase}.{state}.forward.*.{annealing_steps}_steps.works.npz")

    #query the positions/work template
    positions_filenames = glob.glob(positions_template)
    #position_index_extractions = {int(i.split('.')[4][4:]): i for i in positions_filenames if int(i.split('.')[4][4:]) != 10} #make a dict of indices
    position_index_extractions = {int(i.split('.')[4][4:]): i for i in positions_filenames}
    works_filenames = glob.glob(works_template)
    #corresponding_work_filenames = {int(i.split('.')[4][4:]): i for i in works_filenames if int(i.split('.')[4][4:]) != 10}
    corresponding_work_filenames = {int(i.split('.')[4][4:]): i for i in works_filenames}

    #iterate through posiiton indices; if there is a work file and it has the approproate number of annealing steps, append it
    full_dict = {}; works = {}
    for index in position_index_extractions.keys():
        if index in list(corresponding_work_filenames.keys()):
            work_array = np.load(corresponding_work_filenames[index])['works']
            if len(work_array) == annealing_steps + 1:
                full_dict[index] = (position_index_extractions[index], corresponding_work_filenames[index])
                works[index] = work_array[-1]

    #normalize
    work_indices, work_values = list(works.keys()), np.array(list(works.values()))

    if resample:
        normalized_work_values = exp_distribution(work_values)

        assert all(len(item)>0 for item in [work_indices, work_values, normalized_work_values])

        resamples = np.random.choice(work_indices, num_resamples, p = normalized_work_values)
    else:
        resamples = work_indices

    return resamples

def backward_extractor(i,j,phase, state, annealing_steps, parent_dir):
    """
    pull the indices of all existing position files
    """
    import os
    import glob
    positions_template = os.path.join(parent_dir, f"lig{i}to{j}.{phase}.{state}.ani_endstate.*.{annealing_steps}_steps.positions.npz")
    positions_filenames = glob.glob(positions_template)
    position_index_extractions = {int(filename.split('.')[4][4:]): filename for filename in positions_filenames}
    return list(position_index_extractions.keys())

state_list = ['solvent', 'complex']
direction_list = ['forward', 'ani_endstate', 'backward']
prefix = f'lig{ligand_indices[0][0]}to{ligand_indices[0][1]}'
for direction in direction_list:
    for ligand in ['old', 'new']:
        for state in state_list:

            if direction == 'forward':
                extraction_indices = list(range(11))
            if direction == 'ani_endstate':         
                extraction_indices = extract_and_subsample_forward_works(i,j,state,ligand,5000,output_dir, 11, resample=True)
                #extraction_indices = extraction_indices[:-1]
                resample_file = f'{output_dir}/{prefix}.{state}.{ligand}.5000_steps.forward_resamples.npz'
                assert not os.path.exists(resample_file), f"{resample_file} already exists; aborting"
                np.savez(resample_file, extraction_indices)
            if direction == 'backward':
                extraction_indices = backward_extractor(i,j,state,ligand,5000,output_dir)
                resample_file = f'{output_dir}/{prefix}.{state}.{ligand}.5000_steps.backward_samples.npz'
                assert not os.path.exists(resample_file), f"{resample_file} already exists; aborting"
                np.savez(resample_file, extraction_indices)

            print(extraction_indices)
            print(f'direction is {direction}')
            func_partial = functools.partial(p_func, output_dir=output_dir, state=state, ligand=ligand, direction=direction)
            p_map(func_partial, extraction_indices, range(len(extraction_indices)))
