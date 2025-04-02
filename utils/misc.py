import os, json, pickle, yaml, torch, random
from easydict import EasyDict
import numpy as np
from rdkit import Chem


def read_json(json_file):
    with open(json_file,'r') as f:
        data = json.load(f)
    return data


def read_pkl(pkl_file):
    with open(pkl_file,'rb') as f:
        data = pickle.load(f)
    return data


def write_pkl(data_list, pkl_file):
    with open(pkl_file, 'wb') as f:
        pickle.dump(data_list, f)


def load_config(path):
    with open(path, 'r') as f:
        return EasyDict(yaml.safe_load(f))


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_yaml_file(path, content):
    assert isinstance(path, str), f'path must be a string, got {path} which is a {type(path)}'
    content = yaml.dump(data=content)
    if '/' in path and os.path.dirname(path) and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write(content)


def check_mol(mol):
    if isinstance(mol, str) and os.path.exists(mol):
        if os.path.splitext(mol)[1] == ".sdf":
            mol = next(iter(Chem.SDMolSupplier(mol)))
        elif os.path.splitext(mol)[1] == ".mol":
            mol = Chem.MolFromMolFile(mol)
        else:
            raise NotImplementedError(f"Unsupported file: `{mol}`")
    elif isinstance(mol, Chem.Mol):
        mol = mol
    else:
        raise NotImplementedError(f"Unsupported objects: `{mol}`")
    return mol


def prepare_args(args):
    if not args.charge_weight:
        args.in_node_nf -= 1
    
    if args.include_hybrid:
        if args.hybrid_one_hot:
            args.in_node_nf += 4
        else:
            args.in_node_nf += 1
    
    if args.add_core_atoms:
        args.in_node_nf += 2
    
    if args.include_valencies:
        args.in_node_nf += 1

    if args.data_name == 'zinc_300':
        args.in_node_nf_2 += 2

    if args.include_ring:
        args.in_node_nf += 2
    
    if args.include_aromatic:
        args.in_node_nf += 2

    if args.include_neib_dist:
        args.in_node_nf += 2

    return args


# convert seconds to minutes and seconds
def convert_to_min_sec(seconds):
    m, s = divmod(seconds, 60)
    return f'{m} mins {s:.2f} secs'

