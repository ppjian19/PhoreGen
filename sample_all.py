import argparse
import sys, os, time
import numpy as np
import json
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from utils.misc import *
from models.diffusion import PhoreDiff
from datasets.get_phore_data import PhoreData, PhoreData_New
from utils.training_utils import get_transform
from utils.sample_utils import unbatch_data, decode_data, reconstruct_from_generated_with_edges, MolReconsError


def print_status(pool, n_finished, n_failed):
    print(f"[All] Finished {n_finished} | Failed {n_failed}", end='\t')
    print(f'[single] Finished {len(pool.finished)} | Failed {len(pool.failed)}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/train_dock-cpx-phore.yml')
    parser.add_argument('--num_samples', type=int, default=1000, help='total number of samples to generate per pharmacophore')
    parser.add_argument('--batch_size', type=int, default=30, help='number of samples to generate per step')
    parser.add_argument('--outdir', type=str, default='./results/test')
    parser.add_argument('--check_point',type=str,default='./ckpt/crossdocked_pdbbind_trained.pt',help='load the parameter')
    parser.add_argument('--phore_file_list', type=str, default='./data/phores_for_sampling/file_index.json', help='pharmacophore file specified for generation')
    parser.add_argument('--add_edge', type=str, default='predicted')
    parser.add_argument('--save_traj_prob', type=float, default=0.0, help='save the trajectory probability')
    parser.add_argument('--pos_guidance_opt', type=json.loads, default=None, help='position guidance option')
    parser.add_argument('--sample_nodes_mode', type=str, default='uniform')
    parser.add_argument('--normal_scale', type=float, default=4.0)
    parser.add_argument('--seed', type=int, default=2032)
    args = parser.parse_args()
    config = load_config(args.config)
    print(config)
    # seed_all(config.train.seed)
    seed_all(args.seed)

    print(args)
    if config.dataset.data_name in ['zinc_300', 'pdbbind']:
        config.model.phore_feat_dim += 2
        print(f"[W] The model parameter of 'phore_feat_dim' plus 2 when 'data_name' is '{config.dataset.data_name}'.")


    ## define the transform function (process the data again) and the test dataset
    transform = get_transform(config.train)
    if args.phore_file_list is not None:
        if config.dataset.data_name in ['zinc_300', 'pdbbind']:
            test_set = PhoreData_New(read_json(args.phore_file_list), transform=transform, data_name=config.dataset.data_name)
        else:
            test_set = PhoreData(read_json(args.phore_file_list), transform=transform, data_name=config.dataset.data_name)
    else:
        sys.exit('[E] No pharmacophore file specified for generation.')

    ## define the model and load parameters
    model = PhoreDiff(config.model, data_name=config.dataset.data_name).to(config.train.device)
    model_state = torch.load(args.check_point, map_location='cpu')['model']
    model.load_state_dict(model_state)
    model.eval()
    print(f"[I] Loading Diffusion model checkpoint from '{args.check_point}'")

    timestr = f"[{time.strftime('%Y/%m/%d-%H:%M:%S')}]"
    st_time = time.time()
    print(f">> {timestr} Generating molecules based on {read_json(args.phore_file_list)}\n")

    n_finished = 0
    n_failed = 0
    for pidx, data in enumerate(test_set):
        pool = EasyDict({
            'failed': [],
            'finished': [],
            'phore_name': []
        })
        t1 = time.time()
        timestr = f"[{time.strftime('%Y/%m/%d-%H:%M:%S')}]"
        print(f"\n>> {timestr} Start for {pidx+1} / {len(test_set)} -> {data.name}...")
        data = data.to(config.train.device)
        while len(pool.finished) < args.num_samples:
            if len(pool.failed) > 3 * args.num_samples:
                print(f'[E] Too many failed samples. Skip this pharmacophore: `{data.name}`.')
                break
            n_graphs = min(args.batch_size, args.num_samples - len(pool.finished))

        # num_batch = int(np.ceil(args.num_samples / args.batch_size))
        # for i in range(num_batch):
        #     n_graphs = args.batch_size if i < num_batch - 1 else args.num_samples - args.batch_size * (num_batch - 1)
            
            ## sample from the model
            try:
                results = model.sample(
                    data, n_graphs, config.train.device, pos_guidance_opt=args.pos_guidance_opt, 
                    sample_mode=args.sample_nodes_mode, normal_scale=args.normal_scale
                    )
            except Exception as e:
                if 'out of memory' in str(e):
                    print(f'| WARNING: ran out of memory, skipping sample batch: ({data.name}) / {n_graphs} graphs | Message: {e}')
                    pool.failed.extend([ [0] for _ in range(n_graphs) ])
                    continue
                else:
                    print(f'[E] Failed to sample the batch: ({data.name}) / {n_graphs} graphs')
                    raise e
                
            results = {key:[v.cpu() for v in value if v is not None] for key, value in results.items()}

            ## unbatch pos, node type, edge type
            outputs_list = unbatch_data(results, n_graphs, include_bond=config.model.bond_diffusion)

            ## decode data to molecules
            gen_list = []
            for i, output_mol in enumerate(outputs_list):
                mol_info = decode_data(
                    pred_info=output_mol['pred'],
                    edge_index=output_mol['edge_index'], 
                    include_bond=config.model.bond_diffusion
                )
                # reconstruct the molecule
                try:
                    rdmol = reconstruct_from_generated_with_edges(mol_info, add_edge=args.add_edge)
                except MolReconsError:
                    pool.failed.append(mol_info)
                    n_failed += 1
                    print('[E] Reconstruction error encountered.')
                    continue
                
                mol_info['rdmol'] = rdmol
                smiles = Chem.MolToSmiles(rdmol)
                mol_info['smiles'] = smiles
                if '.' in smiles:
                    print(f"[E] Incomplete molecule generated: {smiles}")
                    pool.failed.append(mol_info)
                    n_failed += 1
                else:
                    print(f"[I] Success: {smiles}")
                    n_finished += 1
                    p_save_traj = np.random.rand() 
                    if p_save_traj < args.save_traj_prob:
                        traj_info = [decode_data(
                            pred_node=output_mol['traj'][0][t], 
                            pred_pos=output_mol['traj'][1][t],
                            pred_edge=output_mol['traj'][2][t],
                            edge_index=output_mol['edge_index']
                        ) for t in range(len(output_mol['traj'][0]))]
                        mol_traj = []
                        for t in range(len(traj_info)):
                            try:
                                mol_traj.append(reconstruct_from_generated_with_edges(traj_info[t], add_edge=args.add_edge))
                            except MolReconsError:
                                mol_traj.append(Chem.MolFromSmiles('O'))
                        mol_info['traj'] = mol_traj

                    gen_list.append(mol_info)

            ## save sdf mols
            sdf_dir = os.path.join(args.outdir, 'sdf_results')
            os.makedirs(sdf_dir, exist_ok=True)
            with open(os.path.join(args.outdir, f'{data.name}_SMILES_all.txt'), 'a') as smiles_f:
                for i, data_finished in enumerate(gen_list):
                    smiles_f.write(data_finished['smiles'] + '\n')
                    rd_mol = data_finished['rdmol']
                    rd_mol.SetProp("_Name", f"Molecule_{pidx}_{i+len(pool.finished)}_for_phore_{data.name}")
                    Chem.MolToMolFile(rd_mol, os.path.join(sdf_dir, f'{pidx}_{data.name}_{i+len(pool.finished)}.sdf'))
                    
                    if 'traj' in data_finished:
                        with Chem.SDWriter(os.path.join(sdf_dir, f"traj_{pidx}_{i+len(pool.finished)}.sdf")) as w:
                            for m in data_finished['traj']:
                                try:
                                    w.write(m)
                                except:
                                    w.write(Chem.MolFromSmiles('O'))

            pool.finished.extend(gen_list)
        print_status(pool, n_finished, n_failed)
        pool.phore_name = data.name
        torch.save(pool, os.path.join(args.outdir, f'{data.name}_samples_all.pt'))
        t2 = time.time()
        time_chain_info = (data.name, len(pool.finished), t2 - t1)
        with open(os.path.join(args.outdir, 'time_chain.txt'), 'a') as f:
            f.write(str(time_chain_info) + '\n')

    ed_time = time.time()
    timestr = f"[{time.strftime('%Y/%m/%d-%H:%M:%S')}]"
    print(f'\n>> {timestr} Generation done! {n_finished + n_failed} molecules generated at {convert_to_min_sec(ed_time - st_time)}.')
