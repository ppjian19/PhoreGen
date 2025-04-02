import copy
from typing import Any
import numpy as np
from numpy import random
import torch
from torch_geometric.utils import k_hop_subgraph
import scipy.spatial as spa

REGIONPHORE = torch.tensor([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
REGIONPHORE1 = torch.tensor([0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0])


class MaskByPhore():
    """
        Masking atoms based on the distance from the pharmacophore.
        Args:
            phore_threshold: distance between pharmacophore and pharmacophore  
                to form phormacophore groups.
            ligand_threshold: distance between pharmacophore and ligand atoms 
    """
    def __init__(self, phore_threshold=1.5, ligand_threshold=1.5, mask_one_phore=False, 
                 min_ratio=0.0, max_ratio=1.0, min_num_masked=1, min_num_unmasked=0, 
                 random=False):
        super().__init__()
        self.phore_threshold = phore_threshold
        self.ligand_threshold = ligand_threshold
        self.mask_one_phore = mask_one_phore
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.min_num_masked = min_num_masked
        self.min_num_unmasked = min_num_unmasked
        self.random = random

    def __call__(self, data):
        phore_type = data["phore"].x[:, :11]
        phore_pos = data["phore"].pos
        ligand_pos = data["ligand"].pos

        # 1. Excluding EX Phore_type/position
        last_column_index = phore_type[:, -1]
        rows_to_keep = torch.nonzero(last_column_index == 0).squeeze()
        phore_pos_noEX = phore_pos[rows_to_keep]
        
        # 2. Get distance < threshold phore_group
        phore_pos_group = self.get_phore_group(phore_pos_noEX, self.phore_threshold)

        # 3. Get masked_phore_idx/context_phore_idx
        if self.mask_one_phore:
            num_phore_masked = 1
        else:
            if not self.random:
                np.random.seed(2023)
                random.seed(2023)
            ratio = np.clip(random.uniform(self.min_ratio, self.max_ratio), 0.0, 1.0)
            num_phore_group = len(phore_pos_group)
            num_phore_masked = int(num_phore_group * ratio)

        if num_phore_masked < self.min_num_masked:
            num_phore_masked = self.min_num_masked
        if (num_phore_group - num_phore_masked) < self.min_num_unmasked:
            num_phore_masked = num_phore_group - self.min_num_unmasked

        idx = np.arange(num_phore_group).tolist()
        np.random.shuffle(idx)
        masked_phore_idx = idx[:num_phore_masked]

        # 4. Get masked_phore_pos/context_phore_pos
        masked_phore_pos = []
        for i in masked_phore_idx:
            masked_phore_pos.append(phore_pos_group[i])

        # 5. Get distance < threshold masked_ligand_pos
        masked_ligand_pos = self.get_ligand_pos(data, ligand_pos, masked_phore_pos, self.ligand_threshold)
        if masked_ligand_pos is None:
            # return HeteroData()
            return data
        
        # 6. Get masked_ligand_idx/context_ligand_idx/context_ligand_pos
        masked_ligand_idx = []
        for i, coord in enumerate(ligand_pos):
            exist = any(torch.equal(coord, ligand_coord) for ligand_coord in masked_ligand_pos)
            if exist:
                masked_ligand_idx.append(i)

        # 7. Get mask/x/masked_x
        atom_list = list(range(data["ligand"].atom_count))
        mask = torch.tensor([True if i in masked_ligand_idx else False for i in atom_list])

        masked_x = copy.deepcopy(data['ligand'].x)
        masked_x[mask, 0] = 1
        masked_x[mask, 1:] = 0
        
        data["ligand"].mask = mask
        data["ligand"].masked_x = masked_x
        return data
    
    def get_phore_group(self, phore_pos_noEX, threshold):
        if phore_pos_noEX.ndim == 1:
            print(f"[E]: Only one phore!")
            phore_pos_noEX = torch.unsqueeze(phore_pos_noEX, dim=0)
        distances = torch.cdist(phore_pos_noEX, phore_pos_noEX)
        close_pairs = (distances < threshold).nonzero()
        phore_group_idx = []
        for i in range(phore_pos_noEX.shape[0]):
            indices = close_pairs[close_pairs[:, 0] == i][:, 1].tolist()
            phore_group_idx.append(indices)
            
        phore_group_idx = [lst for i, lst in enumerate(phore_group_idx) if lst not in phore_group_idx[:i]]
        
        phore_group_dict = {}
        for i, indices in enumerate(phore_group_idx):
            phore_group_dict[i] = phore_pos_noEX[indices].tolist()
        return phore_group_dict
    
    def euclidean_distance(self, x, y):
        return torch.norm(x - y, dim=1)

    def get_ligand_pos(self, data, ligand_all_pos, phore_pos, threshold):
        if len(phore_pos) != 0:
            max_length = max(len(sublist) for sublist in phore_pos)
            phore_pos = torch.tensor([
                sublist + [[float('inf')]*len(sublist[0])] * (max_length - len(sublist))
                for sublist in phore_pos
            ])
            selected_ligand_pos = []
            for i in range(ligand_all_pos.size(0)):
                current_ligand = ligand_all_pos[i]
                distances = self.euclidean_distance(current_ligand, phore_pos.view(-1, 3))

                if torch.any(distances < threshold):
                    selected_ligand_pos.append(current_ligand)
            if len(selected_ligand_pos) == 0:
                print(f"[E]: have no ligand_atom under {threshold}A") if 0 not in data['ligand'].x.shape else None
                return None

            selected_ligand_pos = torch.stack(selected_ligand_pos)
            return selected_ligand_pos
        else:
            print(f"[E]: have no masked phore!")
            return None


class MaskByPhore_hop():
    def __init__(self, k_hop=3, dis_threshold=1.8, min_ratio=0.0, max_ratio=1.0, 
                 min_num_masked=1, min_num_unmasked=0, mol_no_mask=0.05, data_name='zinc'):
        super().__init__()
        self.k_hop = k_hop
        self.dis_threshold = dis_threshold
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.min_num_masked = min_num_masked
        self.min_num_unmasked = min_num_unmasked
        self.mol_no_mask = mol_no_mask
        self.data_name = data_name

    def __call__(self, data):
        phore_type = data['phore'].x[:, :13] if self.data_name == 'zinc_300' else data['phore'].x[:, :11]
        ligand_pos = data["ligand"].pos

        ## get region phore
        rows_to_keep = torch.nonzero(phore_type[:, -1] == 0).squeeze()
        phore_pos_noEX = data["phore"].pos[rows_to_keep]
        phore_type_noEX = phore_type[rows_to_keep][:, :12] if self.data_name == 'zinc_300' else phore_type[rows_to_keep][:, :10]
        region_phore_label = REGIONPHORE1 if self.data_name == 'zinc_300' else REGIONPHORE
        region_phore = (phore_type_noEX * region_phore_label).sum(-1).bool()
    
        ## get masked phore idx
        pro = torch.rand(1)
        if pro <= self.mol_no_mask:
            data['ligand'].mask = torch.zeros(ligand_pos.size(0)).bool()
            data['ligand'].masked_x = copy.deepcopy(data['ligand'].x)
        else:
            masked_ratio = np.clip(random.uniform(self.min_ratio, self.max_ratio), 0.0, 1.0)
            num_phore_masked = int(len(phore_type_noEX) * masked_ratio)
            if num_phore_masked < self.min_num_masked:
                num_phore_masked = self.min_num_masked
            if (len(phore_type_noEX) - num_phore_masked) < self.min_num_unmasked:
                num_phore_masked = len(phore_type_noEX) - self.min_num_unmasked
            idx = torch.randperm(len(phore_type_noEX))
            masked_phore_idx = idx[:num_phore_masked]
            assert masked_phore_idx.size(0) >= self.min_num_masked, \
                f"The number of masked pharmacophores must >= {self.min_num_masked}"
            
            ## get masked region phore and no region phore
            masked_region_phore = region_phore[masked_phore_idx]
            masked_phore_pos = phore_pos_noEX[masked_phore_idx]
            reg_masked_phore_pos = masked_phore_pos[masked_region_phore]
            no_reg_masked_phore_pos = masked_phore_pos[~masked_region_phore]

            ## get masked region ligand idx
            reg_msked_ligand_pos = torch.cdist(ligand_pos, reg_masked_phore_pos) < self.dis_threshold
            reg_masked_ligand_lab = reg_msked_ligand_pos.int().sum(1) > 0
            ligand_idx = torch.arange(ligand_pos.size(0))
            reg_masked_ligand_idx = ligand_idx[reg_masked_ligand_lab]

            ## get masked no region ligand idx
            phore2ligand_pos = torch.cdist(ligand_pos, no_reg_masked_phore_pos) < 0.1
            phore2ligand_lab = phore2ligand_pos.int().sum(1) > 0
            phore2ligand_idx = ligand_idx[phore2ligand_lab]
            no_reg_masked_ligand_idx, _, _, _ = k_hop_subgraph(phore2ligand_idx, self.k_hop, 
                                                    data['ligand', 'ligand'].edge_index, relabel_nodes=True)

            ## get masked no region and in ring ligand idx
            masked_ring = torch.empty(0, dtype=torch.long)
            if hasattr(data, 'mol'): 
                ri = data.mol.GetRingInfo()
                masked_ring = []
                for ring in ri.AtomRings():
                    for atom in phore2ligand_idx:
                        if atom.item() in ring:
                            masked_ring.append(list(ring))
                masked_ring = torch.tensor([atom for ring in masked_ring for atom in ring]).unique()

            all_masked_ligand_idx = torch.cat([no_reg_masked_ligand_idx, reg_masked_ligand_idx, masked_ring], dim=0).unique()
            # all_masked_ligand_idx = torch.cat([no_reg_masked_ligand_idx, reg_masked_ligand_idx], dim=0).unique()
            if all_masked_ligand_idx.size(0) == 0:
                print(f"[W] Have no masked ligand_atom!")
                return data

            mask = torch.tensor([True if i in all_masked_ligand_idx else False for i in ligand_idx])
            masked_x = copy.deepcopy(data['ligand'].x)
            masked_x[mask, 0] = 1
            masked_x[mask, 1:] = 0
            data['ligand'].mask = mask
            data["ligand"].masked_x = masked_x
        return data
    

class MaskByPhore_mixed():
    def __init__(self, k_hop=2, dis_threshold=1.8, min_ratio=0.0, max_ratio=1.0, 
                 min_num_masked=1, min_num_unmasked=0, mol_no_mask=0.05, 
                 p_random_mask=0.0, data_name='zinc'):
        super().__init__()
        self.k_hop = k_hop
        self.dis_threshold = dis_threshold
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.min_num_masked = min_num_masked
        self.min_num_unmasked = min_num_unmasked
        self.mol_no_mask = mol_no_mask
        self.p_random_mask = p_random_mask
        self.data_name = data_name


    def get_all_masked_ligand_idx(self, data, k_hop=2, dis_threshold=1.8, min_ratio=0.0, max_ratio=1.0):
        ligand_pos = data["ligand"].pos

        ## get masked region phore and no region phore
        _, reg_masked_phore_pos, no_reg_masked_phore_pos = get_masked_phore_pos(data, 
                        min_ratio, max_ratio, data_name=self.data_name)

        ## get masked region ligand idx
        reg_msked_ligand_pos = torch.cdist(ligand_pos, reg_masked_phore_pos) < dis_threshold
        reg_masked_ligand_lab = reg_msked_ligand_pos.int().sum(1) > 0
        ligand_idx = torch.arange(ligand_pos.size(0))
        reg_masked_ligand_idx = ligand_idx[reg_masked_ligand_lab]

        ## get masked no region ligand idx
        phore2ligand_pos = torch.cdist(ligand_pos, no_reg_masked_phore_pos) < 0.1
        phore2ligand_lab = phore2ligand_pos.int().sum(1) > 0
        phore2ligand_idx = ligand_idx[phore2ligand_lab]
        no_reg_masked_ligand_idx, _, _, _ = k_hop_subgraph(phore2ligand_idx, k_hop, 
                                            data['ligand', 'ligand'].edge_index, relabel_nodes=True)

        ## get masked no region and in ring ligand idx
        masked_ring = torch.empty(0, dtype=torch.long)
        if hasattr(data, 'mol'): 
            _, _, masked_ring = get_masked_ring(data.mol, phore2ligand_idx)

        all_masked_ligand_idx = torch.cat([no_reg_masked_ligand_idx, reg_masked_ligand_idx, masked_ring], dim=0).unique()

        return ligand_idx, all_masked_ligand_idx


    def get_random_masked_ligand_idx(self, atom_count):
        ligand_idx = torch.arange(atom_count)
        r_lig_idx = torch.randperm(atom_count)
        masked_ratio = np.clip(random.uniform(0.0, 1.0), 0.0, 1.0)
        num_lig_masked = int(masked_ratio * atom_count)

        if num_lig_masked < self.min_num_masked:
            num_lig_masked = self.min_num_masked
        if (atom_count - num_lig_masked) < self.min_num_unmasked:
            num_lig_masked = atom_count - self.min_num_unmasked

        masked_lig_idx = r_lig_idx[:num_lig_masked]
        return ligand_idx, masked_lig_idx


    def __call__(self, data):
        pro = torch.rand(1)
        if pro <= self.mol_no_mask:
            data['ligand'].mask = torch.zeros(data["ligand"].pos.size(0)).bool()
            data['ligand'].masked_x = copy.deepcopy(data['ligand'].x)
        else:
            ## get masked ligand idx
            if pro <= self.p_random_mask: 
                ligand_idx, all_masked_ligand_idx = self.get_random_masked_ligand_idx(data['ligand'].pos.size(0))
            else:
                ligand_idx, all_masked_ligand_idx = self.get_all_masked_ligand_idx(data, self.k_hop, self.dis_threshold, 
                                                                                self.min_ratio, self.max_ratio)
            if all_masked_ligand_idx.size(0) == 0:
                print(f"[W] Have no masked ligand atoms!")
                return data
            
            mask = torch.tensor([True if i in all_masked_ligand_idx else False for i in ligand_idx])
            masked_x = copy.deepcopy(data['ligand'].x)
            masked_x[mask, 0] = 1
            masked_x[mask, 1:] = 0
            data['ligand'].mask = mask
            data["ligand"].masked_x = masked_x

        return data
    

class AddLigandPhoreEdges():
    """ 
    Add the affiliation of pharmacophore with ligand. 
    """
    def __init__(self, k_hop=2, dis_threshold=1.8, data_name='zinc'):
        super().__init__()
        self.k_hop = k_hop
        self.dis_threshold = dis_threshold
        self.data_name = data_name


    def get_no_reg_aff(self, k_hop, edge_index, phore2ligand_pos, phore2ligand_idx, masked_ring, atom_in_ring):
        subset_list = []
        for node_idx in phore2ligand_idx:
            subset, _, _, _ = k_hop_subgraph(node_idx.tolist(), num_hops=k_hop, edge_index=edge_index)
            subset_list.append(subset)

        hop_phore2ligand_pos = copy.deepcopy(phore2ligand_pos)
        for i, j in enumerate(torch.where(phore2ligand_pos)[1]):
            hop_phore2ligand_pos[:, j][subset_list[i]] = True

        ring_phore2ligand_pos = copy.deepcopy(hop_phore2ligand_pos)
        idx = 0
        for i, j in enumerate(torch.where(phore2ligand_pos)[1]):
            if phore2ligand_idx[i] in atom_in_ring:
                ring_phore2ligand_pos[:, j][masked_ring[idx]] = True
                idx += 1

        return ring_phore2ligand_pos


    def get_lig_phore_aff(self, data, k_hop=2, dis_threshold=1.8):
        ligand_pos = data["ligand"].pos

        ## get masked region phore and no region phore
        masked_region_phore, reg_masked_phore_pos, no_reg_masked_phore_pos = get_masked_phore_pos(data, 
                                    min_ratio=1.0, data_name=self.data_name)

        ## get masked region ligand idx
        reg_msked_ligand_pos = torch.cdist(ligand_pos, reg_masked_phore_pos) < dis_threshold
        ligand_idx = torch.arange(ligand_pos.size(0))

        ## get masked no region ligand idx
        phore2ligand_pos = torch.cdist(ligand_pos, no_reg_masked_phore_pos) < 0.1
        phore2ligand_lab = phore2ligand_pos.int().sum(1) > 0
        phore2ligand_idx = ligand_idx[phore2ligand_lab]

        ## get masked no region and in ring ligand idx
        masked_ring = torch.empty(0, dtype=torch.long)
        atom_in_ring = torch.empty(0, dtype=torch.long)
        if hasattr(data, 'mol'): 
            masked_ring, atom_in_ring, _ = get_masked_ring(data.mol, phore2ligand_idx)

        no_reg_msked_ligand_pos = self.get_no_reg_aff(k_hop, data['ligand', 'ligand'].edge_index, 
                                    phore2ligand_pos, phore2ligand_idx, masked_ring, atom_in_ring)

        lig_phore_aff = torch.zeros(ligand_pos.size(0), masked_region_phore.size(0))
        lig_phore_aff[:, masked_region_phore] = reg_msked_ligand_pos.float()
        lig_phore_aff[:, ~masked_region_phore] = no_reg_msked_ligand_pos.float()

        return lig_phore_aff
    

    def __call__(self, data):
        lig_phore_aff = self.get_lig_phore_aff(data, self.k_hop, self.dis_threshold)
        rows = torch.arange(lig_phore_aff.shape[0])
        cols = torch.arange(lig_phore_aff.shape[1])
        row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")

        edge_index = torch.stack((row_grid.flatten(), col_grid.flatten()), dim=0)
        data['ligand', 'to', 'phore'].edge_index = edge_index
        data['ligand', 'to', 'phore'].edge_type = lig_phore_aff.flatten().long()

        return data


def get_masked_phore_pos(data, min_ratio=0.0, max_ratio=1.0, min_num_masked=1, min_num_unmasked=0, data_name='zinc'):
    phore_type = data['phore'].x[:, :13] if data_name == 'zinc_300' else data['phore'].x[:, :11]

    ## get region phore
    rows_to_keep = torch.nonzero(phore_type[:, -1] == 0).squeeze()
    phore_pos_noEX = data["phore"].pos[rows_to_keep]
    phore_type_noEX = phore_type[rows_to_keep][:, :12] if data_name == 'zinc_300' else phore_type[rows_to_keep][:, :10]
    region_phore_label = REGIONPHORE1 if data_name == 'zinc_300' else REGIONPHORE
    region_phore = (phore_type_noEX * region_phore_label).sum(-1).bool()

    ## get masked phore idx
    if min_ratio == 1.0:
        masked_phore_idx = torch.arange(len(phore_type_noEX))
    else:
        masked_ratio = np.clip(random.uniform(min_ratio, max_ratio), 0.0, 1.0)
        num_phore_masked = int(len(phore_type_noEX) * masked_ratio)
        if num_phore_masked < min_num_masked:
            num_phore_masked = min_num_masked
        if (len(phore_type_noEX) - num_phore_masked) < min_num_unmasked:
            num_phore_masked = len(phore_type_noEX) - min_num_unmasked
        idx = torch.randperm(len(phore_type_noEX))
        masked_phore_idx = idx[:num_phore_masked]
        assert masked_phore_idx.size(0) >= min_num_masked, \
            f"The number of masked pharmacophores must >= {min_num_masked}"

    ## get masked region phore and no region phore
    masked_region_phore = region_phore[masked_phore_idx]
    masked_phore_pos = phore_pos_noEX[masked_phore_idx]
    reg_masked_phore_pos = masked_phore_pos[masked_region_phore]
    no_reg_masked_phore_pos = masked_phore_pos[~masked_region_phore]

    return masked_region_phore, reg_masked_phore_pos, no_reg_masked_phore_pos


def get_masked_ring(mol, phore2ligand_idx):
    ri = mol.GetRingInfo()
    masked_ring = []
    atom_in_ring = []
    for ring in ri.AtomRings():
        for atom in phore2ligand_idx:
            if atom.item() in ring:
                atom_in_ring.append(atom.item())
                masked_ring.append(list(ring))
    masked_ring_t = torch.tensor([atom for ring in masked_ring for atom in ring]).unique()
    
    return masked_ring, atom_in_ring, masked_ring_t


class AddPhoreNoise():
    def __init__(self, noise_std, angle):
        super().__init__()
        self.noise_std = noise_std
        self.angle = angle


    def generate_perpendicular_vector(self, v: np.array, norm=True, epsilon=1e-12):
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


    def generate_perturbed_norm(self, norm: np.array, theta: np.array):
        axis = self.generate_perpendicular_vector(norm)  # Rotation around x-axis
        rotation = spa.transform.Rotation.from_rotvec(axis * theta)
        new_norm = rotation.apply(norm)
        return torch.tensor(new_norm)


    def __call__(self, data):
        ## add noise to phore pos
        data['phore'].pos = data['phore'].pos + torch.randn_like(data['phore'].pos) * self.noise_std

        ## add noise to phore norm
        all_norm = copy.deepcopy(data['phore'].norm)
        for i in range(all_norm.size(0)):
            theta = np.random.uniform(0, np.pi / 180 * self.angle)
            if not torch.all(all_norm[i] == 0):
                if torch.rand(1) <= 0.5:
                    data['phore'].norm[i] = self.generate_perturbed_norm(all_norm[i], theta)
        return data


class FeaturizeLigandBond():
    def __init__(self) -> None:
        super().__init__()
        

    def __call__(self, data):
        n_atoms = data['ligand'].pos.size(0)
        full_dst = torch.repeat_interleave(torch.arange(n_atoms), n_atoms)
        full_src = torch.arange(n_atoms).repeat(n_atoms)
        mask = full_dst != full_src
        full_dst, full_src = full_dst[mask], full_src[mask]
        data['ligand', 'ligand'].f_edge_index = torch.stack([full_src, full_dst], dim=0)

        bond_matrix = torch.zeros(n_atoms, n_atoms).long()
        src, dst = data['ligand', 'ligand'].edge_index
        bond_matrix[src, dst] = data['ligand', 'ligand'].edge_attr
        data['ligand', 'ligand'].f_edge_attr = bond_matrix[data['ligand', 'ligand'].f_edge_index[0], 
                                                           data['ligand', 'ligand'].f_edge_index[1]]
        return data
    
    