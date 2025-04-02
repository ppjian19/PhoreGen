import copy
import pickle
import lmdb
import pandas as pd
import numpy as np
import os
import torch
from torch.nn import functional as F
from rdkit import Chem
from torch_geometric.data import Dataset, HeteroData, Batch
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from datasets.generate_phorefp import generate_ligand_phore_feat
from datasets.get_phore_data import PhoreData, PhoreData_New
from utils.misc import read_pkl, write_pkl, check_mol, read_json
from models.common import get_neib_dist_feat
# from generate_phorefp import generate_ligand_phore_feat

## Global Parameters
PDBBIND_OR_ZINC_ATOM_TYPES = [0, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]  # masked B C N O F Si P S Cl Br I
PHORETYPES = ['MB', 'HD', 'AR', 'PO', 'HA', 'HY', 'NE', 'CV', 'CR', 'XB', 'EX']
PHORETYPES1 = ['MB', 'HD', 'AR', 'PO', 'HA', 'HY', 'NE', 'CV1', 'CV2', 'CV3', 'CV4', 'XB', 'EX']


class pz_dataset(Dataset):
    """
        Using PDBBind or ZINC dataset.
    """
    def __init__(self, dataset_file, dataset='pdbbind', max_node=78, center='phore', 
                 include_charges=True, transform=None, remove_H=True, include_hybrid=False, 
                 hybrid_one_hot=True, add_core_atoms=False, include_valencies=False, 
                 include_ring=False, include_aromatic=False, include_neib_dist=False, 
                 hierarchical=False, **kwargs):
        super(pz_dataset, self).__init__(transform=transform)
        self.dataset = dataset
        self.datalist = pd.read_pickle(dataset_file)
        if dataset == 'pdbbind':
            self.datalist = [g for g in self.datalist if g['ligand'].x.size(0) < max_node]
        self.center = center
        self.include_charges = include_charges
        self.remove_H = remove_H
        self.include_hybrid = include_hybrid
        self.hybrid_one_hot = hybrid_one_hot
        self.add_core_atoms = add_core_atoms
        self.include_valencies = include_valencies
        self.include_ring = include_ring
        self.include_aromatic = include_aromatic
        self.include_neib_dist = include_neib_dist
        self.hierarchical = hierarchical


    def len(self):
        return len(self.datalist)


    def get(self, idx):
        graph = HeteroData()
        _graph = copy.deepcopy(self.datalist[idx]) if self.dataset == 'pdbbind' \
            else pd.read_pickle(self.datalist[idx])
        graph.name = _graph.name
        graph.mol = Chem.RemoveAllHs(_graph.mol)

        hydrogen_mask = _graph['ligand'].x[:, 0] != 0 if self.remove_H else torch.ones_like(_graph['ligand'].x[:, 0]).bool()

        ## Get pos
        _graph['ligand'].pos = _graph['ligand'].pos[hydrogen_mask]
        ligand_orig_pos = torch.tensor(_graph['ligand'].orig_pos).float()[hydrogen_mask]
        phore_orig_pos = _graph['phore'].pos + _graph.original_center
        if self.center == 'phore':
            graph['ligand'].pos = _graph['ligand'].pos
            graph['phore'].pos = _graph['phore'].pos
            graph.center = _graph.original_center
        elif self.center == 'ligand':
            graph['ligand'].pos = ligand_orig_pos - ligand_orig_pos.mean(dim=0)
            graph['phore'].pos = phore_orig_pos - ligand_orig_pos.mean(dim=0)
            graph.center = ligand_orig_pos.mean(dim=0)
        else:
            graph['ligand'].pos = ligand_orig_pos
            graph['phore'].pos = phore_orig_pos
            graph.center = 0

        ## Get x
        _graph['ligand'].x = _graph['ligand'].x[hydrogen_mask]
        charges = (_graph['ligand'].x[:, 0]+1).unsqueeze(-1)
        atom_types = (charges == torch.LongTensor(PDBBIND_OR_ZINC_ATOM_TYPES).view(1, -1)).float()
        if self.include_charges:
            graph['ligand'].x = torch.cat([atom_types, charges], dim=-1)
        else:
            graph['ligand'].x = atom_types
        graph['phore'].x = torch.cat([F.one_hot(_graph['phore'].x[:, 0].long(), len(PHORETYPES)).float(), # phore_type
                                      _graph['phore'].x[:, 3:4],                                          # alpha
                                      F.one_hot(_graph['phore'].x[:, 2].long(), 2).float(),               # has_norm
                                      F.one_hot(_graph['phore'].x[:, 1].long(), 2).float(), ], dim=-1)    # exclusion_volume
        ## Get atom_count
        graph['ligand'].atom_count = charges.numel()

        ## Get phore_norm
        graph['ligand'].phorefp = _graph['ligand'].phorefp
        graph['phore'].norm = _graph['phore'].norm

        ## Get edge_attr and edge_index
        # graph['ligand', 'lig_bond', 'ligand'].edge_attr = _graph['ligand', 'lig_bond', 'ligand'].edge_attr.argmax(dim=-1) + 1
        # graph['ligand', 'lig_bond', 'ligand'].edge_index = _graph['ligand', 'lig_bond', 'ligand'].edge_index
        edge_attr = to_dense_adj(_graph['ligand', 'lig_bond', 'ligand'].edge_index, 
                                torch.zeros(hydrogen_mask.size(0)).long(),
                                _graph['ligand', 'lig_bond', 'ligand'].edge_attr.argmax(dim=-1) + 1).squeeze()
        edge_attr = edge_attr[hydrogen_mask.view(-1, 1) * hydrogen_mask.view(1, -1)].view(hydrogen_mask.sum().item(), hydrogen_mask.sum().item())
        graph['ligand', 'lig_bond', 'ligand'].edge_index, graph['ligand', 'lig_bond', 'ligand'].edge_attr= dense_to_sparse(edge_attr)

        ## Get hybridization type
        if self.include_hybrid:
            hybrid_map = {
                Chem.rdchem.HybridizationType.SP: 1, 
                Chem.rdchem.HybridizationType.SP2: 2, 
                Chem.rdchem.HybridizationType.SP3: 3
                }
            hybridization = [hybrid_map.get(atom.GetHybridization(), 0) for atom in graph.mol.GetAtoms()]
            hybridization = torch.tensor(hybridization, dtype=torch.long)[hydrogen_mask]
            if self.hybrid_one_hot:
                hybrid_one_hot = F.one_hot(hybridization, 4)
                graph['ligand'].x = torch.cat([graph['ligand'].x, hybrid_one_hot], dim=-1)
            else:
                graph['ligand'].x = torch.cat([graph['ligand'].x, hybridization.unsqueeze(-1)], dim=-1)

        ## Add core atom features
        if self.add_core_atoms:
            is_core = graph['ligand'].phorefp.sum(dim=-1) > 0
            core_one_hot = F.one_hot(is_core.long(), 2).float()[hydrogen_mask]
            graph['ligand'].x = torch.cat([graph['ligand'].x, core_one_hot], dim=-1)

        ## Add valencies
        if self.include_valencies:
            valencies = torch.tensor([atom.GetTotalValence() for atom in graph.mol.GetAtoms()], dtype=torch.long)[hydrogen_mask]
            graph['ligand'].x = torch.cat([graph['ligand'].x, valencies.unsqueeze(-1)], dim=-1)

        ## Add distance
        if self.include_neib_dist:
            neib_dist_feat = get_neib_dist_feat(graph['ligand'].pos)
            graph['ligand'].x = torch.cat([graph['ligand'].x, neib_dist_feat], dim=-1)

        ## Add is_ring
        if self.include_ring:
            is_ring = torch.tensor([atom.IsInRing() for atom in graph.mol.GetAtoms()], dtype=torch.long)
            ring_one_hot = F.one_hot(is_ring, 2).float()[hydrogen_mask]
            graph['ligand'].x = torch.cat([graph['ligand'].x, ring_one_hot], dim=-1)

        ## Add aromatic
        if self.include_aromatic:
            aromatic = torch.tensor([atom.GetIsAromatic() for atom in graph.mol.GetAtoms()], dtype=torch.long)
            aromatic_one_hot = F.one_hot(aromatic, 2).float()[hydrogen_mask]
            graph['ligand'].x = torch.cat([graph['ligand'].x, aromatic_one_hot], dim=-1)

        return graph


class mol_dataset(Dataset):
    """
        Input: mol object and phore file.
        Args:
            file_list: [(mol1, phore1), (mol2, phore2), ...]
            center: 'phore' or 'ligand'
    """
    def __init__(self, file_list, center='phore', transform=None, remove_H=True, save_path=None, 
                 include_charges=False, include_hybrid=False, hybrid_one_hot=False, add_core_atoms=False,
                 include_valencies=False, include_ring=False, include_aromatic=False, 
                 include_neib_dist=False, data_name='zinc_300', hierarchical=False, **kwargs):
        super(mol_dataset, self).__init__(transform=transform)
        if isinstance(file_list, list):
            self.file_list = file_list
        else:
            self.file_list = read_pkl(file_list)
        self.center = center
        self.remove_H = remove_H
        self.save_path = save_path
        self.include_charges = include_charges
        self.include_hybrid = include_hybrid
        self.hybrid_one_hot = hybrid_one_hot
        self.add_core_atoms = add_core_atoms
        self.include_valencies = include_valencies
        self.include_ring = include_ring
        self.include_aromatic = include_aromatic
        self.include_neib_dist = include_neib_dist
        self.data_name = data_name
        self.hierarchical = hierarchical


    def parse_mol(self, mol, data):
        mol = check_mol(mol)
        
        if self.remove_H:
            mol = Chem.RemoveAllHs(mol)
            atomic_numbers = torch.LongTensor([0, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53])  # masked B C N O F Si P S Cl Br I
        else:
            atomic_numbers = torch.LongTensor([0, 1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53])  # masked H B C N O F Si P S Cl Br I

        data.mol = mol

        ## get element
        element = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()])
        element_t = (element.view(-1, 1) == atomic_numbers.view(1, -1)).float()
        
        ## get position and center of mass
        conformer = mol.GetConformer()
        pos = torch.tensor([conformer.GetAtomPosition(i) for i in range(element.size(0))])

        ## get bond index and bond type
        bond_index, bond_type = [], []
        bond_type_map = {
            Chem.BondType.SINGLE: 1,
            Chem.BondType.DOUBLE: 2,
            Chem.BondType.TRIPLE: 3,
            Chem.BondType.AROMATIC: 4,
        }
        for bond in mol.GetBonds():
            atom1_idx = bond.GetBeginAtomIdx()
            atom2_idx = bond.GetEndAtomIdx()
            bond_index.append([atom1_idx, atom2_idx])
            bond_index.append([atom2_idx, atom1_idx])
            bond_type.extend([bond_type_map.get(bond.GetBondType(), 0)]*2)
        bond_index = torch.tensor(bond_index, dtype=torch.long).T
        bond_type = torch.tensor(bond_type, dtype=torch.long)

        ## get phorefp
        lig_phorefp = torch.tensor(generate_ligand_phore_feat(mol, self.remove_H), dtype=torch.float)

        ## get hybridization type
        hybrid_map = {
            Chem.rdchem.HybridizationType.SP: 1, 
            Chem.rdchem.HybridizationType.SP2: 2, 
            Chem.rdchem.HybridizationType.SP3: 3
            }
        hybridization = [hybrid_map.get(atom.GetHybridization(), 0) for atom in mol.GetAtoms()]
        hybrid_t = F.one_hot(torch.tensor(hybridization, dtype=torch.long), 4)

        ## get core atom feature
        is_core = lig_phorefp.sum(dim=-1) > 0
        core_t = F.one_hot(is_core.long(), 2).float()

        ## get valencies
        valencies = torch.tensor([atom.GetTotalValence() for atom in mol.GetAtoms()], dtype=torch.long)

        ## Add distance
        neib_dist_feat = get_neib_dist_feat(pos)

        ## Add is_ring
        is_ring = torch.tensor([atom.IsInRing() for atom in mol.GetAtoms()], dtype=torch.long)
        ring_one_hot = F.one_hot(is_ring, 2).float()

        ## Add aromatic
        aromatic = torch.tensor([atom.GetIsAromatic() for atom in mol.GetAtoms()], dtype=torch.long)
        aromatic_one_hot = F.one_hot(aromatic, 2).float()

        ## remove all hydrogens
        is_H_ligand = (element == 1)
        if self.remove_H and is_H_ligand.int().sum().item() > 0:
            not_H_ligand = ~ is_H_ligand
            element, element_t, pos = element[not_H_ligand], element_t[not_H_ligand], pos[not_H_ligand]
            hybrid_t, lig_phorefp, core_t = hybrid_t[not_H_ligand], lig_phorefp[not_H_ligand], core_t[not_H_ligand]
            valencies, ring_one_hot, aromatic_one_hot = valencies[not_H_ligand], ring_one_hot[not_H_ligand], aromatic_one_hot[not_H_ligand]
            neib_dist_feat = neib_dist_feat[not_H_ligand]
            ## bond
            index_atom_H = torch.nonzero(is_H_ligand)[:, 0]
            index_changer = -torch.ones(len(not_H_ligand), dtype=torch.int64)
            index_changer[not_H_ligand] = torch.arange(torch.sum(not_H_ligand))
            ind_bond_with_H = torch.tensor([(bond_i in index_atom_H) | (bond_j in index_atom_H) for bond_i, bond_j in zip(*bond_index)])
            ind_bond_without_H = ~ind_bond_with_H
            old_ligand_bond_index = bond_index[:, ind_bond_without_H]
            bond_index = index_changer[old_ligand_bond_index]
            bond_type = bond_type[ind_bond_without_H]

        data['ligand'].pos = pos
        data['ligand'].x = element_t
        data['ligand'].atom_count = element.size(0)
        data['ligand'].charge = element.float().unsqueeze(-1)
        data['ligand'].center_of_mass = pos.mean(dim=0)
        data['ligand'].phorefp = lig_phorefp
        data['ligand'].hybrid = hybrid_t
        data['ligand'].is_core = core_t
        data['ligand'].valencies = valencies.float().unsqueeze(-1)
        data['ligand'].is_ring = ring_one_hot
        data['ligand'].aromatic = aromatic_one_hot
        data['ligand'].neib_dist_feat = neib_dist_feat
        data['ligand', 'lig_bond', 'ligand'].edge_index = bond_index
        data['ligand', 'lig_bond', 'ligand'].edge_attr = bond_type

        return data


    def parse_phore_file(self, phore_file, data):
        if phore_file is not None and os.path.exists(phore_file):
            all_phore_type = PHORETYPES1 if self.data_name in ['zinc_300', 'pdbbind'] else PHORETYPES
            possible_phore_type = {phore_type: index for index, phore_type in enumerate(all_phore_type)}
            phore_type_list, alpha_list, pos_list, has_norm_list, norm_list = [], [], [], [], []
            with open(phore_file, 'r') as f:
                title = f.readline().strip()
                record = f.readline()
                while record:
                    record = record.strip()
                    if record != "$$$$":
                        try:
                            phore_type, alpha, weight, factor, x, y, z, \
                                has_norm, norm_x, norm_y, norm_z, label, anchor_weight = record.split("\t")
                            if phore_type == 'CR':
                                # print(f"Unsported phore type: {phore_type}")
                                record = f.readline()
                                continue
                            if phore_type == 'CV':
                                phore_type += label[0]
                            phore_type_list.append(possible_phore_type[phore_type])
                            alpha_list.append(float(alpha))
                            pos_list.append([float(x), float(y), float(z)])
                            has_norm_list.append(int(has_norm))
                            norm_list.append([float(norm_x), float(norm_y), float(norm_z)])
                        except:
                            print(f"[E]: Failed to parse the line:\n {record}")
                    else:
                        break
                    record = f.readline()
                    
            phore_type_t = F.one_hot(torch.tensor(phore_type_list, dtype=torch.long), num_classes=len(possible_phore_type)).float()
            exclusion_volume = F.one_hot(phore_type_t[:, -1:].long().squeeze(-1), 2).float()
            alpha_t = torch.tensor(alpha_list, dtype=torch.float).unsqueeze(-1)
            has_norm_t = F.one_hot(torch.tensor(has_norm_list, dtype=torch.long), 2).float()

            ## get unit norm
            norm_t = torch.tensor(norm_list, dtype=torch.float)
            norms = norm_t.norm(dim=-1, keepdim=True)
            no_zero = (norms != 0).squeeze(-1)
            unit_norm = torch.zeros_like(norm_t)
            unit_norm[no_zero] = norm_t[no_zero] / norms[no_zero]

            data.name = title
            data['phore'].pos = torch.tensor(pos_list, dtype=torch.float)
            data['phore'].x = torch.cat((phore_type_t, alpha_t, has_norm_t, exclusion_volume), dim=-1)
            data['phore'].norm = unit_norm
            data['phore'].center_of_mass = torch.tensor(pos_list, dtype=torch.float).mean(dim=0)

            return data
        else:
            raise FileNotFoundError(f"The specified pharmacophore file (*.phore) is not found: `{phore_file}`")


    def move_to_center(self, data):
        if self.center == "phore":
            data["ligand"].pos -= data["phore"].center_of_mass
            data["phore"].pos -= data["phore"].center_of_mass
            data.center = "phore"
        elif self.center == "ligand":
            data["ligand"].pos -= data["ligand"].center_of_mass
            data["phore"].pos -= data["ligand"].center_of_mass
            data.center = "ligand"
        else:
            raise ValueError(f"The center should be `phore` or `ligand`, but got `{self.center}`")
        return data


    def get_graph(self, data):
        graph = HeteroData()
        graph.name = data.name
        graph.mol = Chem.RemoveAllHs(data.mol) if self.remove_H else data.mol
        if data.center == "phore":
            graph.center = data["phore"].center_of_mass
        elif data.center == "ligand":
            graph.center = data["ligand"].center_of_mass
        else:
            graph.center = 0

        ## ligand
        graph['ligand'].pos = data['ligand'].pos
        graph['ligand'].x = data['ligand'].x.argmax(-1) - 1  # removed masked_atom, [5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]
        assert -1 not in graph['ligand'].x, f' {graph["ligand"].x}'
        graph['ligand'].atom_count = data['ligand'].atom_count
        graph['ligand'].phorefp = data['ligand'].phorefp

        ## phore
        graph['phore'].pos = data['phore'].pos
        graph['phore'].x = data['phore'].x
        graph['phore'].norm = data['phore'].norm


        ## ligand - ligand
        graph['ligand', 'lig_bond', 'ligand'].edge_index = data['ligand', 'lig_bond', 'ligand'].edge_index
        graph['ligand', 'lig_bond', 'ligand'].edge_attr = data['ligand', 'lig_bond', 'ligand'].edge_attr

        return graph


    def len(self):
        return len(self.file_list)
    

    def get(self, idx):
        _data = HeteroData()
        mol, phore_file = self.file_list[idx]
        data_pkl = os.path.join(self.save_path, os.path.splitext(os.path.basename(phore_file))[0]+".pkl")
        if self.data_name == 'pdbbind' and os.path.basename(self.save_path) == 'pdbbind_pkl':
            data_pkl = os.path.join(self.save_path, os.path.basename(phore_file).split('_')[0]+"_ligand.pkl")

        parse_flag = False
        if os.path.exists(data_pkl):
            try:
                data = read_pkl(data_pkl)
                graph = self.get_graph(data)
            except:
                # print(f"[W] {data_pkl} is corrupted, reparse...")
                parse_flag = True
        else:
            parse_flag = True

        if parse_flag:
            data = self.parse_mol(mol, _data)
            data = self.parse_phore_file(phore_file, data)
            data = self.move_to_center(data)
            write_pkl(data, os.path.join(self.save_path, f"{data.name}.pkl"))
            graph = self.get_graph(data)
        return graph


