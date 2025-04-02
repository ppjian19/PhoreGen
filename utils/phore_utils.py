import pickle
import copy, os, random, warnings, time
from collections import namedtuple

import numpy as np
import pandas as pd
import scipy
import scipy.spatial as spa
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, GetPeriodicTable, RemoveHs
from rdkit.Chem import rdMolDescriptors as rdescriptors
from rdkit.Geometry import Point3D


DEBUG = False
periodic_table = GetPeriodicTable()
NUM_PHORETYPE = 11
PI = float(np.pi)
eps=1e-12
PHORETYPES = ['MB', 'HD', 'AR', 'PO', 'HA', 'HY', 'NE', 'CV', 'CR', 'XB', 'EX']
PHORETYPES1 = ['MB', 'HD', 'AR', 'PO', 'HA', 'HY', 'NE', 'CV1', 'CV2', 'CV3', 'CV4', 'CR', 'XB', 'EX']
PHORE_SMARTS = {
    'MB': {
        '*-P(-O)(-O)=O': [2, 3, 4],
        '*-S(-O)=O': [2, 3],
        '*-S(=O)(-O)=O': [2, 3, 4],
        '*-S(-*)=O': [3],
        '*-C(-O)=O': [2, 3],
        '[O^3]': [0],
        '*-C(-C(-F)(-F)-F)=O': [6],
        '[OH1]-P(-*)(-*)=O': [0, 4],
        '*-C(-N-*)=O': [4],
        '*-[CH1]=O': [2],
        '*-N(-*)-N=O': [4],
        '*-C(-S-*)=O': [4],
        'O=C(-C-O-*)-C-[OH1]': [0],
        '*-C(-S-*)=O': [4],
        '*-C(-C(-[OH1])=C)=O': [5],
        '[S^3D2]': [0],
        '*=N-C=S': [3],
        'S=C(-N-C(-*)=O)-N-C(-*)=O': [0],
        '[#7^2,#7^3;!$([n;H0;X3]);!+;!+2;!+3]': [0],
        '[C,#1]-[Se&H1]': [1],
        'C1:C:C:C:S:1': [4],
        'O2:C:C:C:C:2': [0],
        'a[O,NH2,NH1,SX2,SeH1]': [1],
        '[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@[O,N,P,S])]),$([nH0,o,s;+0])]' : [0]
    },

    'NE': {
        '[CX3,SX3,PD3](=[O,S])[O;H0&-1,OH1]': [1, 2],
        '[PX4](=[O,S])([O;H0&-1,OH1])[O;H0&-1,OH1]': [1, 2, 3],
        '[PX4](=[O,S])([O;H0&-1,OH1])[O][*;!H]': [1, 2],
        '[SX4](=[O,S])(=[O,S])([O;H0&-1,OH1])': [1, 2, 3]
    },

    'PO': {
        '[+;!$([N+]-[O-])]': [0],
        'N-C(-N)=N': [1]
    },

    'HD': {
        '[#7,#8,#16;+0,+1,+2;!H0]': [0]
    },

    'HA': {'[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@[O,N,P,S])]),$([nH0,o,s;+0])]' : [0]
        # '[S;$(S=C);-0,-1,-2,-3]': [0],
    },
    # 'HA': {
    #     '[#7;!$([#7]~C=[N,O,S]);!$([#7]~S=O);!$([n;H0;X3]);!$([N;H1;X3]);-0,-1,-2,-3]': [0],
    #     '[O,F;-0,-1,-2,-3]': [0],
    #     '[S^3;X2;H0;!$(S=O);-0,-1,-2,-3]': [0],
    #     # '[S;$(S=C);-0,-1,-2,-3]': [0],
    # },

    'CV':{
        '[N]#[C]-[C,#1]': [1],
        '[C,#1]-[C]1-[C](-[C,#1])-[O]-1': [1, 2],
        '[C]=[C]-[C](-[N&H1]-[C,#1])=[O]': [0],
        '[S&H1]-[C,#1]': [0],
        '[C,#1]-[C]1-[C](-[C,#1])-[N]-1': [1, 2],
        '[C]=[C]-[S](=[O])(-[C,#1])=[O]': [0],
        '[F,Cl,Br,I]-[C]-[C,#1]': [1],
        '[C,#1]-[C](-[F,Cl,Br,I])-[C](-[C,N,O]-[C,#1])=[O]': [1],
        '[O]=[C](-[N]-[C,#1])-[C]#[C]': [5],
        '[C,#1]-[S](-[C,#1])=[O]': [1],
        '[C,#1]-[Se&H1]': [1],
        '[O]=[C](-[O]-[C,#1])-[C]#[C]': [5],
        '[S]=[C]=[N]-[C,#1]': [1],
        '[C,#1]-[S]-[S]-[C,#1]': [1, 2],
        '[C,#1]-[N,O]-[C](-[N,O]-[C,#1])=[O]': [2],
        '[C,#1]-[C](-[C](-[N]-[C,#1])=[O])=[O]': [1],
        '[C,#1]-[B](-[O&H1])-[O&H1]': [1],
        '[C,#1]-[C&H1]=[O]': [1],
        '[C,#1]-[S](-[F])(=[O])=[O]': [1],
        '[C,#1]-[S](-[C]=[C])(=[O])=[O]': [3],
        '[F,Cl,Br,I]-[C]-[C](-[C,#1])=[O]': [1]
    },
    'AR': {'[a]': [0]},
    'CR': {'[a]': [0], 
           '[+;!$([N+]-[O-])]': [0],
           'N-C(-N)=N': [1],
           },
    'XB': {'[#6]-[Cl,Br,I;X1]': [1]},
    'HY': {
            # refered to hydrophobic atom in 
            # /home/worker/software/anaconda3/envs/diffphore/lib/python3.9/site-packages/rdkit/Data/BaseFeatures.fdef
            '[c,s,S&H0&v2,Br,I,$([#6;+0;!$([#6;$([#6]~[#7,#8,#9])])])]': [0]
        },
    # CV-SH
    'CV1': {
        "C(-[F,Cl,Br,I])(-[!F;!Cl;!Br;!I])(-[!F;!Cl;!Br;!I])": [0],
        "C#N": [0],
        "C1-O-C-1": [0, 2],
        "[CX2]#C-C-N": [0],
        "C(=O)-O-C-C=O": [3],
        "C=C-C(=O)": [0],
        "S(=O)(=O)-C=C": [4],
        "C=C-[N+](-[O-])=O": [0],
        "N-C-[CH1,CH2]-C(=O)-C": [1],
        "[#6]1:[#7]:[#6]2:[#7H]:[#6]:[#7]:[#6]:2:[#6](-[#8]-[#6]-[#6]2:[#6]:[#6]:[#6]:[#6]:[#6]:2):[#7]:1": [10],
        "[#6]-C(=O)-[#1]": [1],
        "[#8]=[#6]1-[#6]2:[#6]:[#6]:[#6]:[#6]:[#6]:2-[#6]=[#6]-[#6]-1=[#8]": [10],
        "N-C(=O)-C(=O)": [3],
        "O-C(=O)-N": [1],
        "N-C(=O)-S": [1],
        "[#8]=[#6](-[#7])-[#7]1:[#6]:[#7]:[#6]:[#6]:1": [1],
        "[#8]=[#6]1:[#8]:[#6](-[#8]):[#6]:[#6]2:[#6]:[#6]:[#6]:[#6]:[#6]:1:2": [1],
        "C1(=O)-O-C=N-N-1": [0],
        "N-C(=O)-n1:n:n:n:c:1": [1],
        "C1(=O)-[N,O]-C-C-1": [0],
        "[#6](=O)-,:[#8]": [0],
        "[NX3]-C(=S)-[NX3]": [1],
        "[#6]-[#16+](-[#6])-[#6]-[#6](-[#7])=[#8]": [1],
        "F-[SX4](=O)(=O)": [1],
        "[#6]-[SX2]-[H,CX4]": [1],
        "c1:c:c(-C(-F)(-F)(-F)):c:n:c:1-S(=O)(=O)-C": [9],
        "[#6]-[SX2]-[SX2]-[#6]": [1, 2],
        "C1=C-C-[NX3]-C-C-1": [0],
        "[#6]1:[#7H]:[#6](:[#6]2:[#6](:[#7]:1):[#7]:[#6]:[#7]:2)=[#8]": [0],
        "[NX2]=C=S": [0],
        "C=N-[OX2]-[#6]": [1],
        '[C,#1]-[C]1-[C](-[C,#1])-[N]-1': [1, 2],
        '[C,#1]-[C](-[F,Cl,Br,I])-[C](-[C,N,O]-[C,#1])=[O]': [1],
        '[O]=[C](-[N]-[C,#1])-[C]#[C]': [5],
        '[C,#1]-[S](-[C,#1])=[O]': [1],
        '[C,#1]-[Se&H1]': [1],
        '[O]=[C](-[O]-[C,#1])-[C]#[C]': [5],
        "[BX3](-O)(-O)": [0]
    },
    # CV-OH
    'CV2': {
        "C(-[F,Cl,Br,I])(-[!F;!Cl;!Br;!I])(-[!F;!Cl;!Br;!I])": [0],
        "C#N": [0],
        "C1-O-C-1": [0, 2],
        "C=C-C(=O)": [0],
        "[#6]-C(=O)-[#1]": [1],
        "O-C(=O)-N": [1],
        "[#8]=[#6]1-[#6]2:[#6]:[#6]:[#6]:[#6]:[#6]:2-[#6]=[#6]-[#6]-1=[#8]": [10],
        "N-C(=O)-C(=O)": [3],
        "O-C(=O)-N": [1],
        "N-C(=O)-S": [1],
        "[#8]=[#6](-[#7])-[#7]1:[#6]:[#7]:[#6]:[#6]:1": [1],
        "[#8]=[#6]1:[#8]:[#6](-[#8]):[#6]:[#6]2:[#6]:[#6]:[#6]:[#6]:[#6]:1:2": [1],
        "C1(=O)-O-C=N-N-1": [0],
        "N-C(=O)-n1:n:n:n:c:1": [1],
        "C1(=O)-[N,O]-C-C-1": [0],
        "[#6](=O)-,:[#8]": [0],
        "[#6]-[#16+](-[#6])-[#6]-[#6](-[#7])=[#8]": [1],
        "F-[SX4](=O)(=O)": [1],
        "[NX2]=C=S": [0],
        "[CX3]1-N=N-1": [0],
        '[C,#1]-[N,O]-[C](-[N,O]-[C,#1])=[O]': [2],
        '[C,#1]-[C](-[C](-[N]-[C,#1])=[O])=[O]': [1],
        '[C,#1]-[B](-[O&H1])-[O&H1]': [1],
        "[BX3](-O)(-O)": [0],
        '[C,#1]-[C&H1]=[O]': [1]

    },
    # CV-NH2
    'CV3': {
        "C(-[F,Cl,Br,I])(-[!F;!Cl;!Br;!I])(-[!F;!Cl;!Br;!I])": [0],
        "C#N": [0],
        "C1-O-C-1": [0, 2],
        "C=C-C(=O)": [0],
        "S(=O)(=O)-C=C": [4],
        "[#6]-C(=O)-[#1]": [1],
        "[#8]=[#6]1-[#6]2:[#6]:[#6]:[#6]:[#6]:[#6]:2-[#6]=[#6]-[#6]-1=[#8]": [10],
        "N-C(=O)-C(=O)": [3],
        "O-C(=O)-N": [1],
        "N-C(=O)-S": [1],
        "[#8]=[#6](-[#7])-[#7]1:[#6]:[#7]:[#6]:[#6]:1": [1],
        "[#8]=[#6]1:[#8]:[#6](-[#8]):[#6]:[#6]2:[#6]:[#6]:[#6]:[#6]:[#6]:1:2": [1],
        "C1(=O)-O-C=N-N-1": [0],
        "N-C(=O)-n1:n:n:n:c:1": [1],
        "C1(=O)-[N,O]-C-C-1": [0],
        "[#6](=O)-,:[#8]": [0],
        "[#6]-[#16+](-[#6])-[#6]-[#6](-[#7])=[#8]": [1],
        "F-[SX4](=O)(=O)": [1],
        "[NX2]=C=S": [0],
        "[OX2]-[PX4](=O)(-[F,Cl,Br,I])": [0],
        "[BX3](-O)(-O)": [0],
        "[NX2]=[N+]=[N-]": [0]
    },
    # CV-COOH
    'CV4': {
        "C(-[F,Cl,Br,I])(-[!F;!Cl;!Br;!I])(-[!F;!Cl;!Br;!I])": [0],
        "C1-O-C-1": [0, 2],
        '[F,Cl,Br,I]-[C]-[C](-[C,#1])=[O]': [1]
    }
}

atom_radiuses = [periodic_table.GetRvdw(n) for n in range(1, 119)] + [0.0]

Phore = namedtuple('Phore', ['id', 'features', 'exclusion_volumes', 'clusters'])
PhoreFeature = namedtuple('PhoreFeature', ['type', 'alpha', 'weight', 'factor', 'coordinate', 
                                           'has_norm', 'norm', 'label', 'anchor_weight'])
Coordinate = namedtuple('Coordinate', ['x', 'y', 'z'])


def extend_exclusion_volumes(phore, mol, low=3, up=5, theta_cavity=10, theta=15, num_ex=5, rounds=100, 
                             ex_dis=0.8, only_surface_ex=True, debug=False, trim=False):
    _mol = Chem.RemoveHs(mol)
    atom_coords = _mol.GetConformer().GetPositions()
    center = np.mean(atom_coords, axis=0)
    norms = (atom_coords - center) / (np.linalg.norm(atom_coords - center, axis=1, keepdims=True) + eps)
    origin_ex = np.array([[ex.coordinate.x, ex.coordinate.y, ex.coordinate.z] for ex in phore.exclusion_volumes])
    random_exs, exclusion_volumes = np.empty((0, 3)), np.empty((0, 3))
    _atom_ids = [i for i, at_coord in enumerate(atom_coords) if cavity_detection(at_coord, norms[i], origin_ex, theta_cavity)]
    if debug:
        print(f'[I] Expanding random exclusion volumes: `{len(_atom_ids)}` atoms to process.')
    for idx in _atom_ids:
        random_exs = generate_ex_by_shell(atom_coords[idx], norms[idx], low=low, up=up, theta=theta, 
                                          num_ex=num_ex, rounds=rounds, ex_dis=ex_dis, 
                                          exclusion_volumes=np.concatenate([exclusion_volumes, origin_ex], axis=0))
        random_exs = exclude_clashed_ex(random_exs, phore, atom_coords, exclusion_volumes, ex_dis=ex_dis)
        random_exs = ex_in_range(random_exs, atom_coords, up, return_axis=0)
        exclusion_volumes = np.concatenate([exclusion_volumes, random_exs], axis=0)

    if trim:
        exclusion_volumes = trim_weird_ex(origin_ex, exclusion_volumes, center)
    
    exclude_idx = []
    if only_surface_ex:
        exclude_idx = filter_surface_ex(atom_coords, exclusion_volumes)
        if debug:
            print(f'[I] Filtering exclusion volumes not on the surface: `{len(exclude_idx)}` EX to exclude.')

    exclusion_volumes = [PhoreFeature(type='EX', alpha=0.837, weight=0.5, factor=1, 
                                      coordinate=Coordinate(x=float(ex[0]), y=float(ex[1]), z=float(ex[2])),
                                      has_norm=0, norm=Coordinate(0, 0, 0), label='0', anchor_weight=1)
                         for idx, ex in enumerate(exclusion_volumes) if idx not in exclude_idx]
    if debug:
        print(f'[I] Expanding random exclusion volumes: `{len(exclusion_volumes)}` EX to add.')
    _phore = copy.deepcopy(phore)._replace(exclusion_volumes=exclusion_volumes+phore.exclusion_volumes)
    return _phore


def trim_weird_ex(origin_ex, exclusion_volumes, center):
    max_radius = np.linalg.norm(center - origin_ex, axis=1).max()
    radius = np.linalg.norm(center - exclusion_volumes, axis=1)
    return exclusion_volumes[radius <= max_radius]


def filter_surface_ex(ligand_coords, ex_coords, cutoff=30.0, cutoff_num=15, exclude_far=True, debug=False):
    dis_mat = spa.distance_matrix(ligand_coords, ex_coords)
    sorted_index = dis_mat.argsort()
    total_list = []
    mask_d = np.sort(dis_mat, axis=1) <= 7.0
    for i in range(len(sorted_index)):
        nearby_ex = sorted_index[i][mask_d[i]]
        if len(nearby_ex) >= 2:
            # print(f"len(nearby_ex) = {nearby_ex}")
            total_list.extend(stack_analysis(nearby_ex, i, ligand_coords, ex_coords, cutoff_angle=cutoff, dis_mat=dis_mat))
    too_far = np.arange(len(ex_coords))[np.sort(dis_mat, axis=0)[0, :] > 6.0].tolist()
    # total_list.extend(too_far)
    remove_list = []
    counts = {}
    if total_list:
        counts = pd.Series(total_list).value_counts().to_dict()
        remove_list = [int(k) for k, v in counts.items() if int(v) >= cutoff_num]
    if exclude_far:
        remove_list.extend(too_far)
    remove_list = [k for k in remove_list if k not in sorted_index[:, 0]]
    remove_list = list(set(remove_list))
    if debug:
        print(f"Angle: {cutoff}, Number: {cutoff_num}, Too Far: {too_far}, Remove: {remove_list} -->", counts)
    return remove_list


def cavity_detection(at_coord, norm, exclusion_volumes, angle_cutoff=5):
    ex_norm = exclusion_volumes - at_coord.reshape(-1, 3)
    ex_norm = ex_norm / (np.linalg.norm(ex_norm, axis=1, keepdims=True) + eps)
    angles = np.rad2deg(np.arccos((ex_norm * norm).sum(axis=1)))
    return np.sum(angles <= angle_cutoff) == 0


def float_eq(a, b, epsilon=1e-6):
    return abs(a - b) <= epsilon


def write_mol_with_coords(mol, new_coords, path, skip_h=True):
    conf = mol.GetConformer()
    idx = 0
    for i in range(mol.GetNumAtoms()):
        # print(f"{mol.GetAtomWithIdx(i).GetSymbol()}")
        if skip_h and mol.GetAtomWithIdx(i).GetAtomicNum() == 1:
            continue
        x,y,z = new_coords.astype(np.double)[idx]
        idx += 1
        conf.SetAtomPosition(i,Point3D(x,y,z))
    if path.endswith(".sdf"):
        w = Chem.SDWriter(path)
        w.write(mol)
        w.close()
    elif path.endswith(".mol"):
        Chem.MolToMolFile(mol, path)


def write_mol_with_multi_coords(mol, multi_new_coords, path, name, marker="", properties=None):
    w = Chem.SDWriter(path)
    if properties is not None:
        w.SetProps(list(properties.keys()))

    _mol = copy.deepcopy(mol)
    
    for idx, new_coords in enumerate(multi_new_coords):
        _mol.SetProp("_Name", f"{name}_{marker}_{idx}")
        if properties is not None:
            for k, v in properties.items():
                _mol.SetProp(f"{k}", f"{v[idx]}")

        conf = _mol.GetConformer()
        for i in range(_mol.GetNumAtoms()):
        # print(f"{mol.GetAtomWithIdx(i).GetSymbol()}")
            x,y,z = new_coords.astype(np.double)[i]
            conf.SetAtomPosition(i,Point3D(x,y,z))
        w.write(_mol)
    w.close()


def read_molecule(molecule_file, sanitize=False, calc_charges=False, remove_hs=False):
    if molecule_file.endswith('.mol2'):
        mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.mol'):
        mol = Chem.MolFromMolFile(molecule_file, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.sdf'):
        supplier = Chem.SDMolSupplier(molecule_file, sanitize=False, removeHs=False)
        mol = supplier[0]
    elif molecule_file.endswith('.pdbqt'):
        with open(molecule_file) as file:
            pdbqt_data = file.readlines()
        pdb_block = ''
        for line in pdbqt_data:
            pdb_block += '{}\n'.format(line[:66])
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(molecule_file, sanitize=False, removeHs=False)
    else:
        raise ValueError('Expect the format of the molecule_file to be '
                         'one of .mol2, .sdf, .pdbqt and .pdb, got {}'.format(molecule_file))

    try:
        if sanitize or calc_charges:
            Chem.SanitizeMol(mol)

        if calc_charges:
            # Compute Gasteiger charges on the molecule.
            try:
                AllChem.ComputeGasteigerCharges(mol)
            except:
                warnings.warn('Unable to compute charges for the molecule.')

        if remove_hs:
            mol = Chem.RemoveHs(mol, sanitize=sanitize)
    except Exception as e:
        print(e)
        print("RDKit was unable to read the molecule.")
        return None

    return mol


def read_sdf_or_mol2(sdf_fileName, mol2_fileName):

    mol = Chem.MolFromMolFile(sdf_fileName, sanitize=False)
    problem = False
    try:
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
    except Exception as e:
        problem = True
    if problem:
        mol = Chem.MolFromMol2File(mol2_fileName, sanitize=False)
        try:
            Chem.SanitizeMol(mol)
            mol = Chem.RemoveHs(mol)
            problem = False
        except Exception as e:
            problem = True

    return mol, problem


def check_nearby_phore(phore, atom_coord, lig_phorefp=None, cutoff=2, strict=True):
    matches = []
    nearby = False
    for feature in phore.features:
        coord = feature.coordinate
        coord = np.array([coord.x, coord.y, coord.z])
        if np.sqrt(np.sum((coord - atom_coord) ** 2)) < cutoff:
            if strict and isinstance(lig_phorefp, list):
                phorefp = np.array([int(x == feature.type) for x in ['MB', 'HD', 'AR', 'PO', 'HA', 'HY', 'NE', 'CV', 'CR', 'XB', 'EX']])
                match = phorefp * np.array(lig_phorefp)
                if sum(match) > 0:
                    matches.append(match)
                    nearby = True
            else:
                nearby = True
    if matches:
        lig_phorefp = [0 if i == 0 else 1 for i in sum(matches).tolist()]
        # print(f"matches: {matches}")
    return nearby


def extract_random_phore_from_origin(phore, up_num=8, low_num=4, 
                                     sample_num=10, max_rounds=100, **kwargs):
    phores = []
    collection = []
    coords = phore.clusters.keys()
    _round = 0
    while sample_num != 0:
        _round += 1
        num = min(random.choice(list(range(low_num, up_num))), len(coords))
        clusters = random.sample(coords, num)
        ex, feat = [], []
        for cluster in clusters:
            selected = random.choice(phore.clusters[cluster])
            if selected.type == "EX":
                ex.append(selected)
            else:
                feat.append(selected)
        collect = set(ex+feat)
        if collect not in collection:
            phores.append(Phore(f"{phore.id}_{sample_num}", 
                                copy.deepcopy(feat), copy.deepcopy(ex), {}))
            collection.append(collect)
            sample_num -= 1
        if _round >= max_rounds:
            break
    return phores


def generate_ex_by_shell(at_pos, norm, exclusion_volumes=None, low=3, up=5, 
                         ex_dis=0.8, theta=np.pi/12, num_ex=5, rounds=100, debug=DEBUG):
    random_exs = np.empty((0, 3))

    _not_max_rounds = True
    _not_max_num_ex = True
    n = 0
    st = time.time()
    
    while _not_max_rounds and _not_max_num_ex:
        _norm = generate_perpendicular_vector(norm)
        angle = np.random.uniform(0, theta)
        rotation = spa.transform.Rotation.from_rotvec(_norm * angle)
        # rotmat = axis_angle_to_rotate_matrix(_norm, angle)
        curr_ex = rotation.apply(norm) * np.random.uniform(low, up) + at_pos
        if len(random_exs) == 0:
            curr_ex = curr_ex.reshape(-1, 3)
        else:
            curr_ex = exclude_clashed_ex([curr_ex], exclusion_volumes=random_exs, ex_dis=ex_dis)
            
        if exclusion_volumes is not None:
            curr_ex = exclude_clashed_ex(curr_ex, exclusion_volumes=exclusion_volumes, ex_dis=ex_dis)
        
        random_exs = np.concatenate([random_exs, curr_ex.reshape(-1, 3)], axis=0)
        if debug:
            print(len(random_exs), 'EX generated')

        n += 1
        if rounds == n:
            _not_max_rounds = False
        if len(random_exs) == num_ex:
            _not_max_num_ex = False

    if debug and not _not_max_rounds and _not_max_num_ex:
        print("[W] Max round reached. Not enough exclusion spheres added.")
    if debug and not _not_max_num_ex:
        print(f"[I] {num_ex} exclusion spheres added within {n} rounds {time.time()-st:.3f} seconds.")

    return random_exs


def generate_perpendicular_vector(v, norm=True, epsilon=1e-12):
    a, b = np.random.uniform(0.1, 1,size=(2))
    if v[2] != 0:
        c = - (a * v[0] + b * v[1]) / v[2] 
    else:
        assert not (v[0] == 0 and v[1] == 0)
        a = -v[1]
        b = v[0]
        c = 0
    vec = np.array([a, b, c])
    if norm:
        vec = vec / (np.linalg.norm(vec, axis=-1) + epsilon)
    return vec


def exclude_clashed_ex(random_exs, phore=None, lig_coords=None, 
                       exclusion_volumes=None, low=3.0, ex_dis=0.8, debug=DEBUG):
    random_exs = np.array(random_exs).reshape(-1, 3)
    num_ex = len(random_exs)
    # print(f'phore: {phore}')
    if phore is not None:
        phore_coords = np.array([[feat.coordinate.x, feat.coordinate.y, feat.coordinate.z] \
                                 for feat in phore.features])
        random_exs = ex_not_clashed(random_exs, phore_coords, low, return_axis=0)
    if lig_coords is not None:
        random_exs = ex_not_clashed(random_exs, lig_coords, distance=low, return_axis=0)
    
    if exclusion_volumes is not None:
        exclusion_volumes = np.array(exclusion_volumes).reshape(-1, 3)
        random_exs = ex_not_clashed(random_exs, exclusion_volumes, distance=ex_dis, return_axis=0)
        if debug:
            print(num_ex - len(random_exs), "points abandoned.")

    return random_exs


def ex_not_clashed(points1, points2, distance, return_axis=0):
    return [points1, points2][return_axis][np.all(spa.distance_matrix(points1, points2) > distance, axis=[1, 0][return_axis])]


def ex_in_range(points1, points2, distance, return_axis=0):
    return [points1, points2][return_axis][np.any(spa.distance_matrix(points1, points2) <= distance, axis=[1, 0][return_axis])]


def stack_analysis(ex_index, lig_index, l_coords, e_coords, dis_mat, cutoff_angle=10.0):
    remove_list = []
    for idx1 in range(len(ex_index)):
        for idx2 in range(idx1+1, len(ex_index)):
            i = ex_index[idx1]
            j = ex_index[idx2]
            # if i in remove_list or j in remove_list:
            #     continue
            vec_1 = e_coords[i] - l_coords[lig_index]
            vec_2 = e_coords[j] - l_coords[lig_index]
            len_vec_1 = dis_mat[lig_index, i]
            len_vec_2 = dis_mat[lig_index, j]
            # assert sum(vec_1 ** 2)**0.5  - len_vec_1 < 1e-12
            angle = np.rad2deg(np.arccos(vec_1.dot(vec_2) / len_vec_1 / len_vec_2))
            delta_len = len_vec_2 - len_vec_1
            if  angle <= cutoff_angle and delta_len >= 1.0:
                remove_list.append(j)
            # print(f"EX{ex_index[idx2]} --> EX{ex_index[idx1]} :: {angle}")
    return remove_list 


def count_ph(phore):
    count = {k: 0 for k in PHORETYPES1}
    try:
        for ph in phore.features:
            if ph.type != 'CV':
                count[ph.type] += 1
            else:
                for cvtype in ph.label:
                    count[ph.type+cvtype] += 1
    except Exception as e:
        print(f"[E] Failed to count phore: {phore.id} -> {phore}. {e}")
    count['PH'] = phore.features.__len__()
    count['CLST'] = phore.clusters.__len__()
    return count


def extract_phore_info(src_path, dst_path, para=True, nworker=30):
    multi = False
    if para and nworker > 1:
        from pandarallel import pandarallel
        pandarallel.initialize(nb_workers=nworker)
        multi = True
    os.makedirs(dst_path, exist_ok=True)
    dumped_files = []
    for idx, filename in enumerate(os.listdir(src_path)):
        dst_file = os.path.join(dst_path, filename)
        if not os.path.exists(dst_file):
            phore_pkl = os.path.join(src_path, filename)
            df_phore_info = pickle.load(open(phore_pkl, 'rb'))
            if not multi:
                df_phore_info['phore_count'] = df_phore_info['phore'].map(lambda x: count_ph(x))
                df_phore_info['heavy_atom'] = df_phore_info['mol'].map(lambda x: x.GetNumHeavyAtoms())
                df_phore_info['mw'] = df_phore_info['mol'].map(lambda x: Descriptors.MolWt(x))
            else:
                df_phore_info['phore_count'] = df_phore_info.parallel_apply(lambda x: count_ph(x['phore']), axis=1)
                df_phore_info['heavy_atom'] = df_phore_info.parallel_apply(lambda x: x['mol'].GetNumHeavyAtoms(), axis=1)
                df_phore_info['mw'] = df_phore_info.parallel_apply(lambda x: Descriptors.MolWt(x['mol']), axis=1)

            for phtype in PHORETYPES1 + ['CLST', 'PH']:
                df_phore_info[phtype] = df_phore_info['phore_count'].map(lambda x: x[phtype])
            drop_list = ['phore_count', 'mol', 'smiles', 'num_flag', 'metal_flag', 'rdkit_flag', 'num_phore', 
                         'min_energy', 'multiple', 'quality_flag', 'random_phores', 'phore_file', 'EX']
            df_phore_info.drop(drop_list, axis=1, inplace=True)
            pickle.dump(df_phore_info, open(dst_file, 'wb'))
            print(f"[I] Phore info file dumped: {dst_file}. ({idx+1}/{len(os.listdir(src_path))})")
        else:
            print(f"[I] Phore info file already exists: {dst_file}. ({idx+1}/{len(os.listdir(src_path))})")
        dumped_files.append(dst_file)
    dfs = []
    for dumped_file in dumped_files:
        df_phore_info = pickle.load(open(dumped_file, 'rb'))
        dfs.append(df_phore_info)
    df_all = pd.concat(dfs, axis=0)
    return df_all


def extract_basic_info(src_path, out_path, selected=[], B=False):
    bo3 = Chem.MolFromSmarts('[BX4](-O)(-O)(-O)')
    bo2 = Chem.MolFromSmarts('[BX3](-O)(-O)')
    if selected:
        df_selected = pd.DataFrame(selected, columns=['zinc_id'])
        df_selected['zinc_id'] = df_selected['zinc_id'].astype(str)
    
    if not os.path.exists(out_path):
        dfs = []
        col = ['zinc_id', 'mol', 'phore']
        for idx, filename in enumerate(os.listdir(src_path)):
            phore_pkl = os.path.join(src_path, filename)
            df_phore_info = pickle.load(open(phore_pkl, 'rb'))
            df_phore_info['zinc_id'] = df_phore_info['zinc_id'].astype(str)
            df_phore_info = df_phore_info[col]
            if B:
                df_phore_info['Bsp2'] = df_phore_info['mol'].map(lambda x: len(x.GetSubstructMatches(bo2))>0)
                df_phore_info['Bsp3'] = df_phore_info['mol'].map(lambda x: len(x.GetSubstructMatches(bo3))>0)
                df_phore_info = df_phore_info[df_phore_info['Bsp2'] | df_phore_info['Bsp3']]

            if selected:
                df_phore_info = pd.merge(df_phore_info, df_selected, on='zinc_id', how='inner')

            dfs.append(df_phore_info)

            print(f"[I] Basic info extracted{' (B)' if B else ''}: {phore_pkl} ({idx+1}/{len(os.listdir(src_path))})")
        df_all = pd.concat(dfs, axis=0)
        pickle.dump(df_all, open(out_path, 'wb'))
    else:
        df_all = pickle.load(open(out_path, 'rb'))

    return df_all


def coordinate_in_range(a, b, cutoff):
    return distance_of_phorepoints(a, b) <= cutoff


def distance_of_phorepoints(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2 ) ** 0.5


def write_phore_to_file(phore, path, name=None, overwrite=False):
    name = name if name is not None else phore.id
    filename = os.path.join(path, f"{name}.phore") if os.path.isdir(path) else path
    if not os.path.exists(filename) or overwrite:
        with open(filename, 'w') as f:
            f.write(f"{name}\n")
            for feat in phore.features:
                out_string = [feat.type, feat.alpha, feat.weight, feat.factor, 
                            feat.coordinate.x, feat.coordinate.y, feat.coordinate.z, 
                            int(feat.has_norm), feat.norm.x, feat.norm.y, feat.norm.z, 
                            feat.label, feat.anchor_weight]
                f.write("\t".join([f"{x:.3f}" if isinstance(x, float) else str(x) for x in out_string ]) + '\n')

            for ex in phore.exclusion_volumes:
                out_string = [ex.type, ex.alpha, ex.weight, ex.factor, 
                            ex.coordinate.x, ex.coordinate.y, ex.coordinate.z, 
                            int(ex.has_norm), ex.norm.x, ex.norm.y, ex.norm.z, 
                            ex.label, ex.anchor_weight]
                f.write("\t".join([f"{x:.3f}" if isinstance(x, float) else str(x) for x in out_string ]) + '\n')
            f.write("$$$$\n")
    return filename


def parse_phore(phore_file=None, name=None, data_path=None, 
                skip_wrong_lines=True, verbose=False, epsilon=1e-6, skip_ex=False, cvs=False):
    if name is not None and data_path is not None:
        phore_file = os.path.join(data_path, f"{name}/{name}_complex.phore")

    phores = []
    if phore_file is not None and os.path.exists(phore_file):
        with open(phore_file, 'r') as f:
            started, finished, correct = False, False, True
            id = ""
            phore_feats = []
            exclusion_volumes = []
            clusters = {}
            while True:
                record = f.readline().strip()
                if record:
                    if not started:
                        id = record
                        started = True
                    else:
                        phore_feat = parse_phore_line(record, skip_wrong_lines, cvs) if correct else False
                        if phore_feat is None:
                            finished = True
                        elif phore_feat == False:
                            correct = False
                        else:
                            if phore_feat.type != 'EX':
                                phore_feats.append(phore_feat)
                            else:
                                if not skip_ex:
                                    exclusion_volumes.append(phore_feat)
                            add_phore_to_cluster(phore_feat, clusters, epsilon)
                    if finished:
                        if len(phore_feats) and correct:
                            phore = Phore(id, copy.deepcopy(phore_feats), 
                                          copy.deepcopy(exclusion_volumes), copy.deepcopy(clusters))
                            phores.append(phore)
                        phore_feats = []
                        exclusion_volumes = []
                        clusters = {}
                        started, finished = False, False
                else:
                    break
    else:
        raise FileNotFoundError(f"The specified pharmacophore file (*.phore) is not found: `{phore_file}`")
    if verbose:
        if len(phores) == 0:
            print(f"[W] No pharmacophores read from the phore file `{phore_file}`")

    return phores


def parse_phore_line(record, skip_wrong_lines=False, cvs=True):
    if record == "$$$$":
        return None
    else:
        try:
            phore_type, alpha, weight, factor, x, y, z, \
                has_norm, norm_x, norm_y, norm_z, label, anchor_weight = record.split("\t")
            phore_type = phore_type if cvs else phore_type[:2]
            coordinate = Coordinate(float(x), float(y), float(z))
            norm = Coordinate(float(norm_x), float(norm_y), float(norm_z))
            has_norm = bool(int(has_norm))
            alpha, weight, factor, anchor_weight = float(alpha), float(weight), float(factor), float(anchor_weight)
            return PhoreFeature(phore_type, alpha, weight, factor, coordinate, has_norm, norm, label, anchor_weight)
        except:
            print(f"[E]: Failed to parse the line:\n {record}")
            if not skip_wrong_lines:
                raise SyntaxError("Invalid phore feature syntax from the specified phore file.")
            else:
                return False


def add_phore_to_cluster(phore_feat, clusters, epsilon=1e-6):
    if phore_feat.coordinate in clusters:
        clusters[phore_feat.coordinate].append(phore_feat)
    else:
        if len(clusters) == 0:
            clusters[phore_feat.coordinate] = [phore_feat]
        else:
            flag = False
            for stored_coord in clusters:
                curr_coord = np.array([phore_feat.coordinate.x, phore_feat.coordinate.y, 
                                       phore_feat.coordinate.z])
                stored_coord = np.array([stored_coord.x, stored_coord.y, stored_coord.z])
                if np.sqrt(np.sum((stored_coord - curr_coord) ** 2)) <= epsilon:
                    clusters[stored_coord].append(phore_feat)
                    flag = True
                    break
            if not flag:
                clusters[phore_feat.coordinate] = [phore_feat]
    return clusters

