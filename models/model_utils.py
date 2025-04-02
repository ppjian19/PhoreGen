import math
import copy
import pickle
import os
import numpy as np
from typing import Dict
from rdkit.Chem import AllChem as Chem
from rdkit import Geometry
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.utils import remove_self_loops
from torch_scatter import scatter
from torch.distributions.categorical import Categorical


ATOM_TYPES = [0, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]
ELEMENT_SYMBOLS = {5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}

class EMA():
    def __init__(self, beta, parameters):
        super().__init__()
        self.beta = beta
        self.shadow_params = [p.clone() for p in parameters]

    def update_model_average(self, current_model):
        for current_params, ma_params in zip(current_model.parameters(), self.shadow_params):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def state_dict(self):
        return {'beta': self.beta, 'shadow_params': self.shadow_params}
    
    def load_state_dict(self, state_dict, device):
        self.beta = state_dict['beta']
        self.shadow_params = [tensor.to(device) for tensor in state_dict['shadow_params']]


class SampleDataset(Dataset):
    def __init__(self, n_samples: int, num_nodes: torch.Tensor, data: HeteroData):
        super(SampleDataset, self).__init__()
        self.n_nodes = torch.flatten(num_nodes)
        self.data = data
        # self.n_samples = n_samples
        self.phore_index = torch.repeat_interleave(torch.arange(len(data['phore'].ptr) - 1).to(num_nodes.device), 
                                                   n_samples)

    def len(self):
        return len(self.n_nodes)

    def get(self, idx):
        _data = HeteroData()
        _data['ligand'].n_nodes = self.n_nodes[idx].view(-1)
        _data.name = copy.deepcopy(self.data[self.phore_index][idx].name)
        _data['phore'].pos = copy.deepcopy(self.data[self.phore_index][idx]['phore'].pos)
        _data['phore'].x = copy.deepcopy(self.data[self.phore_index][idx]['phore'].x)
        _data['phore'].norm = copy.deepcopy(self.data[self.phore_index][idx]['phore'].norm)
        _data['phore'].center_of_mass = copy.deepcopy(self.data[self.phore_index][idx]['phore'].center_of_mass.unsqueeze(0))
        _data['phore'].id = self.phore_index[idx].view(-1)
        return _data


class AtomMdn(nn.Module):
    def __init__(self, fhidden_dim, atom_mlp_factor, n_gaussians, device):
        super().__init__()
        self.fhidden_dim = fhidden_dim
        self.atom_mlp_factor = atom_mlp_factor
        self.n_gaussians = n_gaussians
        self.mlp = nn.Sequential(nn.Linear(self.fhidden_dim, self.fhidden_dim*self.atom_mlp_factor), nn.ReLU(), 
                                              nn.Linear(self.fhidden_dim*self.atom_mlp_factor, self.fhidden_dim))
        self.mu_net = nn.Linear(self.fhidden_dim, self.n_gaussians)
        self.logsigma_net = nn.Linear(self.fhidden_dim, self.n_gaussians)
        self.pi_net = nn.Linear(self.fhidden_dim, self.n_gaussians)
        self.to(device)

    def forward(self, h_p, batch):
        z_h_p = scatter(self.mlp(h_p), batch, dim=-2, reduce='mean')
        mu = self.mu_net(z_h_p)
        logsigma = self.logsigma_net(z_h_p)
        sigma = torch.exp(logsigma)
        pi = self.pi_net(z_h_p)
        pi = F.softmax(pi, dim=1)
        return mu, sigma, pi


def get_neib_center(x, batch_l):
    all_edge_index_l = fully_connect_two_graphs(batch_l, batch_l)
    a_src, a_dst = all_edge_index_l
    a_bond_mask = (torch.norm(x[a_dst] - x[a_src], dim=-1, keepdim=True)) > 3.0
    neib_bond_index = all_edge_index_l[:, ~a_bond_mask.squeeze(-1)]
    n_src, n_dst = neib_bond_index
    neib_center = scatter(x[n_dst], n_src, dim=0, reduce='mean') 
    return neib_center


def sample_gaussian_with_mask(size, device, node_mask, sigma=1.0):
    # x = torch.normal(0, sigma, size, device=device)
    x = torch.randn(size, device=device) * sigma
    x_masked = x * node_mask
    return x_masked


def fully_connect_two_graphs(batch_1, batch_2, mask_1=None, mask_2=None, return_batch=False):
    mask_1 = torch.ones_like(batch_1).bool() if mask_1 is None else mask_1
    mask_2 = torch.ones_like(batch_2).bool() if mask_2 is None else mask_2
    index_1 = torch.arange(len(batch_1)).to(batch_1.device)
    index_2 = torch.arange(len(batch_2)).to(batch_2.device)
    masked_index_1 = index_1[mask_1]
    masked_index_2 = index_2[mask_2]
    masked_batch_1 = batch_1[mask_1]
    masked_batch_2 = batch_2[mask_2]
    new_index = []
    batch = []
    for i in torch.unique(masked_batch_1):
        _mask_1 = masked_batch_1 == i
        _mask_2 = masked_batch_2 == i
        _masked_index_1 = masked_index_1[_mask_1]
        _masked_index_2 = masked_index_2[_mask_2]
        len_1 = _masked_index_1.shape[0]
        len_2 = _masked_index_2.shape[0]
        new_index.append(torch.concat([_masked_index_1.unsqueeze(-1).tile([1, len_2]).reshape(1, -1), 
                                       _masked_index_2.unsqueeze(-1).tile([1, len_1]).T.reshape(1, -1)], 
                                      axis=0))
        if return_batch:
            batch += [i] * (len_1 * len_2)
    new_index = torch.concat(new_index, axis=1).long()

    if return_batch:
        return new_index, torch.tensor(batch).long().to(batch_1.device)
    return new_index


def get_act_func(act_func):
    if act_func == 'relu':
        return nn.ReLU()
    elif act_func == 'sigmoid':
        return nn.Sigmoid()
    elif act_func == 'tanh':
        return nn.Tanh()
    elif act_func == 'softmax':
        return nn.Softmax(dim=1)
    elif act_func == 'identity':
        return nn.Identity()
    elif act_func == 'gelu':
        return nn.GELU()
    elif act_func == 'leakyrelu':
        return nn.LeakyReLU()
    elif act_func == 'elu':
        return nn.ELU()
    elif act_func == 'selu':
        return nn.SELU()
    elif act_func == 'glu':
        return nn.GLU()
    elif act_func == 'silu':
        return nn.SiLU()
    elif act_func == 'softplus':
        return nn.Softplus()
    elif callable(act_func):
        return act_func
    else:
        print(f'[W] Invalid activation function (`{act_func}`) sepcified, using nn.RelU instead.')
        return nn.ReLU()


def gaussian_KL(q_mu, q_sigma, p_mu, p_sigma):
    # print(f"q_mu.shape = {q_mu.shape}")
    # print(f"q_sigma.shape = {q_sigma.shape}")
    # print(f"p_mu.shape = {p_mu.shape}")
    # print(f"p_sigma.shape = {p_sigma.shape}")
    return torch.log(p_sigma / (q_sigma + 1e-8) + 1e-8) \
                + 0.5 * (q_sigma**2 + (q_mu - p_mu)**2) / (p_sigma**2) \
                - 0.5


def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(-1)


def cdf_standard_gaussian(x):
    return 0.5 * (1. + torch.erf(x / math.sqrt(2)))


def gaussian_KL_for_dimension(q_mu, q_sigma, p_mu, p_sigma, d):
    """Computes the KL distance between two normal distributions.

        Args:
            q_mu: Mean of distribution q.
            q_sigma: Standard deviation of distribution q.
            p_mu: Mean of distribution p.
            p_sigma: Standard deviation of distribution p.
        Returns:
            The KL distance, summed over all dimensions except the batch dim.
        """
    mu_norm2 = sum_except_batch((q_mu - p_mu)**2)
    assert len(q_sigma.size()) == 1
    assert len(p_sigma.size()) == 1
    return (d * torch.log(p_sigma / (q_sigma + 1e-8) + 1e-8) 
            + 0.5 * (d * q_sigma**2 + mu_norm2) / (p_sigma**2) - 0.5 * d)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   

def remove_mean_with_mask(x, node_mask):
    masked_max_abs_value = (x * (1 - node_mask.float())).abs().sum().item()
    assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x


def expm1(x: torch.Tensor) -> torch.Tensor:
    return torch.expm1(x)


def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)


def sample_center_gravity_zero_gaussian_with_mask(size, device, node_mask, sigma=1.0, remove_mean=True):
    assert len(size) == 3
    # x = torch.normal(0, sigma, size, device=device)
    x = torch.randn(size, device=device) * sigma

    x_masked = x * node_mask

    if remove_mean:
        # This projection only works because Gaussian is rotation invariant around
        # zero and samples are independent!
        x_projected = remove_mean_with_mask(x_masked, node_mask)
    else:
        x_projected = x_masked
    return x_projected


def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = (alphas2[1:] / alphas2[:-1])

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def polynomial_schedule(timesteps: int, s=1e-4, power=3.):
    """
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power))**2

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2


def write_xyz(path, atomic_numbers, coords, sam_idx, idx, p_idx):
    with open(f'{path}/phore{p_idx}_{sam_idx}_{idx}.xyz', 'w') as xyz_file:
        xyz_file.write("%d\n\n" % len(atomic_numbers))
        for i in range(len(atomic_numbers)):
            xyz_file.write("%s %.9f %.9f %.9f\n" % (
                ELEMENT_SYMBOLS[atomic_numbers[i]], 
                coords[i][0], coords[i][1], coords[i][2]))
            

def write_sdf(path, mol, name, sam_idx, idx, p_idx):
    mol.SetProp("_Name", f"Molecule {name}")
    mol_block = Chem.MolToMolBlock(mol)
    with open(f'{path}/phore{p_idx}_{sam_idx}_{idx}.sdf', 'w') as sdf_file:
        sdf_file.write(mol_block)


def reconstruct_from_batch(atomic_batch, edge_batch, atomic_numbers_batch, 
                           xyz_batch, phore_center, bond_type_batch, bond_index_batch, 
                           sdf_path, name_batch, sam_idx, p_idx):
    """ 
    Reconstructing atomic_numbers, atomic_position and bond_type into sdf file.
    """
    unique_atomic_batches = torch.unique(atomic_batch)
    atomic_list = [atomic_numbers_batch[atomic_batch == b] for b in unique_atomic_batches]
    xyz_list = [xyz_batch[atomic_batch == b] for b in unique_atomic_batches]
    
    unique_edge_batches = torch.unique(edge_batch)
    bond_type_list = [bond_type_batch[edge_batch == b] for b in unique_edge_batches] if bond_type_batch is not None else None
    bond_index_list = [bond_index_batch[:, edge_batch == b] for b in unique_edge_batches]
    
    assert len(unique_atomic_batches) == len(unique_edge_batches), \
    f'[E] Atomic_batch numbers must be equal to edge_batch numbers, '
    f'but got {len(unique_atomic_batches)} and {len(unique_edge_batches)}'

    for idx in range(len(unique_atomic_batches)):
        atomic_numbers = [ATOM_TYPES[i] for i in atomic_list[idx]]
        xyz_list[idx] += phore_center[idx]
        try:
            assert 0 not in atomic_numbers, \
            f'[E] Atomic number cannot be zero, {sam_idx}_{idx} skipped!'
        except:
            continue
        
        if bond_type_list is None:
            print(f"[I] phore{p_idx}_{sam_idx}_{idx}: No bond type information, only xyz files are generated!")
            write_xyz(sdf_path, atomic_numbers, xyz_list[idx].tolist(), sam_idx, idx, p_idx)
        else:
            bond_index = bond_index_list[idx]
            if idx > 0 and len(atomic_list) > 1:
                bond_index -= len(atomic_list[idx-1])

            # 1. add atomic numbers and position
            rd_mol = Chem.RWMol()
            rd_conf = Chem.Conformer(len(atomic_numbers))
            for atomic_num, coord in zip(atomic_numbers, xyz_list[idx].tolist()):
                rd_atom = Chem.Atom(atomic_num)
                atom_id = rd_mol.AddAtom(rd_atom)
                rd_coords = Geometry.Point3D(coord[0], coord[1], coord[2])
                rd_conf.SetAtomPosition(atom_id, rd_coords)
            rd_mol.AddConformer(rd_conf)

            # 2. add bond type
            for id, bond_value in enumerate(bond_type_list[idx].tolist()):
                i = bond_index.tolist()[0][id]
                j = bond_index.tolist()[1][id]
                bond_ij = rd_mol.GetBondBetweenAtoms(i, j)
                if bond_value == 1 and not bond_ij:
                    rd_mol.AddBond(i, j, Chem.BondType.SINGLE)
                elif bond_value == 2 and not bond_ij:
                    rd_mol.AddBond(i, j, Chem.BondType.DOUBLE)
                elif bond_value == 3 and not bond_ij:
                    rd_mol.AddBond(i, j, Chem.BondType.TRIPLE)
                elif bond_value == 4 and not bond_ij:
                    rd_mol.AddBond(i, j, Chem.BondType.AROMATIC)
            
            # 3. if rd_mol is available, save as sdf file, otherwise save as xyz file
            try:
                # determining whether rd_mol is available
                assert rd_mol.GetNumBonds() > 0, f"[I] Molecule {sam_idx}_{idx} has no bonds, converting to `xyz` file."
                Chem.SanitizeMol(rd_mol)
                write_sdf(sdf_path, rd_mol, name_batch[idx], sam_idx, idx, p_idx)
            except:
                print(f"[I] Molecule {sam_idx}_{idx} in {name_batch[idx]} is not available, converting to `xyz` file.")
                write_xyz(sdf_path, atomic_numbers, xyz_list[idx].tolist(), sam_idx, idx, p_idx)


def save_pkl_file(path, content, sam_idx, p_idx):
    """ 
    Save tensor data as pkl file.
    """
    tensor_path = os.path.join(path, 'tensor_results')
    os.makedirs(tensor_path, exist_ok=True)

    with open(f"{tensor_path}/phore{p_idx}_{sam_idx}.pkl", 'wb') as file:
        pickle.dump(content, file)
        

def save_sdf_file(path, content: Dict[str, torch.Tensor], sam_idx, p_idx):
    """ 
    Save atomic_numbers, atomic_position and bond_type as sdf file.
    """
    sdf_path = os.path.join(path, 'sdf_results')
    os.makedirs(sdf_path, exist_ok=True)

    atomic_numbers_batch = content['h'][:, :12].argmax(dim=-1)
    xyz_batch = content['x']
    phore_center = content['sam_data']['phore'].center_of_mass
    bond_type_batch = (F.softmax(content['bond_type'], dim=-1)).argmax(dim=-1) if content['bond_type'] is not None else None
    atomic_batch = content['sam_data'].batch_l
    bond_index, bond_batch = fully_connect_two_graphs(atomic_batch, atomic_batch, return_batch=True)
    bond_batch = bond_batch[bond_index[0] != bond_index[1]]
    bond_index_batch, _ = remove_self_loops(bond_index)

    reconstruct_from_batch(atomic_batch, bond_batch, atomic_numbers_batch, 
                           xyz_batch, phore_center, bond_type_batch, bond_index_batch, 
                           sdf_path, content['sam_data'].name, sam_idx, p_idx)


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def qd_loss(y_true, y_l, y_u, a=0.05, s=160, nd=15, epsilon=1e-12):
    # captured samples
    n = y_true.shape[0]
    if s <= 0:
        k_u = torch.relu(torch.sign(y_u - y_true))
        k_l = torch.relu(torch.sign(y_true - y_l))
    else:
        # soft uses sigmoid fn
        k_u = torch.sigmoid((y_u - y_true) * s)
        k_l = torch.sigmoid((y_true - y_l) * s)

    k = k_u * k_l

    # combine for loss fn
    MPIW_c = torch.sum((y_u - y_l) * k) / (torch.sum(k) + epsilon)
    PICP = torch.mean(k)
    loss = MPIW_c + (torch.relu((1-a) - PICP) ** 2) * (n ** 0.5) * nd
    return loss


def mdn_loss(label, mu, sigma, pi):
    if torch.isnan(mu).int().sum() > 0 or torch.isnan(sigma).int().sum() > 0:
        print(f"[W] NaN detected in mu or sigma.")
        return torch.full((1,), float('nan')).to(mu.device)
    m = torch.distributions.Normal(loc=mu, scale=sigma)
    loss = torch.exp(m.log_prob(label.unsqueeze(-1)))
    loss = torch.sum(loss.squeeze(-1) * pi, dim=1)
    loss = -torch.log(loss + 1e-16)
    return torch.mean(loss)


def compute_true_atom(h_true, h_recon, batch):
    ## compute the number of molecules in each batch with exactly the right atom type
    not_match = (h_recon.softmax(dim=-1).argmax(dim=-1) != h_true.argmax(dim=-1)).float()
    not_match_batch = (scatter(not_match, batch) == 0).int().sum().item()
    return not_match_batch


def sample_from_mdn(mu, sigma, pi):
    if torch.isnan(pi).int().sum() > 0:
        print(f"[W] NaN detected in pi.")
        return torch.full((pi.size(0), 1), float('nan')).to(pi.device)
    pis = Categorical(pi).sample().view(pi.size(0), 1, 1)
    gaussian_noise = torch.randn(
        (sigma.size(2), sigma.size(0)), requires_grad=False).to(sigma.device)
    variance_samples = sigma.gather(1, pis).detach().squeeze()
    mean_samples = mu.gather(1, pis).detach().squeeze()
    count_pred = (gaussian_noise * variance_samples + mean_samples).transpose(0, 1)
    return count_pred


def compute_true_count(count_true, count_pred, count_type, count_normalized=False, 
                      min_atom=4, max_atom=78):
    c_true = 0
    if count_type == 'classification':
        c_true = ((count_true - (F.softmax(count_pred, dim=-1).argmax(dim=-1) + 3)).abs() <= 3).int().sum().item()

    elif count_type in ['regression', 'boundary', 'mdn']:
        if count_type == 'mdn':
            ## sampling atomic number from mixture gaussians network
            ## count_pred -> mu, sigma, pi
            count_pred = sample_from_mdn(*count_pred)

        if count_normalized:
            ## mapping 0-1 back to the true number of atoms
            count_true = (count_true * (max_atom - min_atom) + min_atom).round()
            if count_type in ['regression', 'mdn']:
                count_pred = (count_pred * (max_atom - min_atom) + min_atom).round()
            elif count_type == 'boundary':
                ## min and max number of atoms
                count_pred = (count_pred[0] * (max_atom - min_atom) + min_atom).round(), \
                             (count_pred[1] * (max_atom - min_atom) + min_atom).round()
                
        if count_type in ['regression', 'mdn']:
            ## get the true atomic number of molecules that are within 3 standard deviations
            c_true = ((count_true - count_pred.squeeze(-1)).abs() <= 3).int().sum().item()

        else:
            ## get the true atomic number of molecules that are within min and max numbers
            count_l, count_u = count_pred
            c_true = ((count_true - count_l.squeeze(-1) >= 0) * (count_u.squeeze(-1) - count_true >= 0)).int().sum().item()

    return c_true


def create_mask(x):
    node_mask = torch.as_tensor(x[:, :, 0], dtype=torch.bool, device=x.device).long()
    zero_diag = torch.ones((node_mask.shape[-1], node_mask.shape[-1]), device=node_mask.device) - \
                torch.eye(node_mask.shape[-1], device=node_mask.device)
    pair_mask = node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2) * zero_diag
    return node_mask.unsqueeze(-1), pair_mask.unsqueeze(-1)


def get_bond(name='zinc'):
    if name == 'zinc':
        bond = torch.tensor([
            [[0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
             [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.75, 0.00, 0.00],
             [0.00, 0.00, 1.54, 1.47, 1.43, 1.35, 1.85, 1.84, 1.82, 1.77, 1.94, 2.14],
             [0.00, 0.00, 1.47, 1.45, 1.40, 1.36, 0.00, 1.77, 1.68, 1.75, 2.14, 2.22],
             [0.00, 0.00, 1.43, 1.40, 1.48, 1.42, 1.63, 1.63, 1.51, 1.64, 1.72, 1.94],
             [0.00, 0.00, 1.35, 1.36, 1.42, 1.42, 1.60, 1.56, 1.58, 1.66, 1.78, 1.87],
             [0.00, 0.00, 1.85, 0.00, 1.63, 1.60, 2.33, 0.00, 2.00, 2.02, 2.15, 2.43],
             [0.00, 0.00, 1.84, 1.77, 1.63, 1.56, 0.00, 2.21, 2.10, 2.03, 2.22, 0.00],
             [0.00, 0.00, 1.82, 1.68, 1.51, 1.58, 2.00, 2.10, 2.04, 2.07, 2.25, 2.34],
             [0.00, 1.75, 1.77, 1.75, 1.64, 1.66, 2.02, 2.03, 2.07, 1.99, 2.14, 0.00],
             [0.00, 0.00, 1.94, 2.14, 1.72, 1.78, 2.15, 2.22, 2.25, 2.14, 2.28, 0.00],
             [0.00, 0.00, 2.14, 2.22, 1.94, 1.87, 2.43, 0.00, 2.34, 0.00, 0.00, 2.66]],
            [[0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
             [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
             [0.00, 0.00, 1.34, 1.29, 1.20, 0.00, 0.00, 0.00, 1.60, 0.00, 0.00, 0.00],
             [0.00, 0.00, 1.29, 1.25, 1.21, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
             [0.00, 0.00, 1.20, 1.21, 1.21, 0.00, 0.00, 1.50, 0.00, 0.00, 0.00, 0.00],
             [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
             [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
             [0.00, 0.00, 0.00, 0.00, 1.50, 0.00, 0.00, 0.00, 1.86, 0.00, 0.00, 0.00],
             [0.00, 0.00, 1.60, 0.00, 0.00, 0.00, 0.00, 1.86, 0.00, 0.00, 0.00, 0.00],
             [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
             [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
             [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]],
            [[0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
             [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
             [0.00, 0.00, 1.20, 1.16, 1.13, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
             [0.00, 0.00, 1.16, 1.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
             [0.00, 0.00, 1.13, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
             [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
             [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
             [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
             [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
             [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
             [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
             [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]]
        ])
        bond_mask = torch.tensor([
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
             [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
             [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1],
             [0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
             [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
             [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
             [0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1]],
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0],
             [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        ], dtype=torch.long)
    else:
        raise NotImplementedError('Unsupported dataset')
    margin = torch.tensor([0.1, 0.05, 0.03])
    return bond.permute(1, 2, 0) + margin.unsqueeze(0).unsqueeze(0), bond_mask.permute(1, 2, 0)


def get_direction_feat(x, edge_index, batch_l, phore_pos=None, phore_norm=None):
    src, dst = edge_index
    neib_center = get_neib_center(x, batch_l)
    if phore_norm is None:
        vec_1 = neib_center[src] - x[src]
        vec_2 = neib_center[dst] - x[dst]
        vec_3 = x[src] - x[dst]
    else:
        vec_1 = neib_center[src] - x[src]
        vec_2 = phore_norm[dst]
        vec_3 = x[src] - phore_pos[dst]
    dot_1 = (vec_1 * vec_2).sum(-1, keepdim=True)
    dot_2 = (vec_1 * vec_3).sum(-1, keepdim=True)
    dot_3 = (vec_2 * vec_3).sum(-1, keepdim=True)
    return torch.cat([dot_1, dot_2, dot_3], dim=-1)


def get_phore_mask(h_p, edge_index_lp, data_name='zinc'):
    phore_type = h_p[:, :13] if data_name == 'zinc_300' else h_p[:, :11]
    ex_mask = phore_type.argmax(dim=-1) == (phore_type.shape[1] - 1)
    phore_node = torch.arange(phore_type.shape[0], device=phore_type.device)[~ex_mask]
    phore_mask = torch.isin(edge_index_lp[1, :], phore_node)
    return phore_mask


def compute_true_aff(aff_pred, aff_true, aff_batch):
    aff_pred_value = aff_pred.softmax(dim=-1).argmax(dim=-1)
    mat = (aff_pred_value == aff_true).float()
    accuracy = scatter(mat, aff_batch, reduce='mean').mean().item()
    return accuracy


def focal_loss(inputs: torch.Tensor,
               targets: torch.Tensor,
               alpha: float = -1,
               gamma: float = 2,
               reduction: str = "mean") -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
            balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                'none': No reduction will be applied to the output.
                'mean': The output will be averaged.
                'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    ce_loss = F.cross_entropy(inputs, targets, reduction="none")
    inputs = inputs.softmax(dim=-1).argmax(dim=-1)
    p_t = inputs * targets + (1 - inputs) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def get_neib_dist_feat(x: torch.Tensor):
    # get neighbor bonde index
    node_index = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    all_edge_index_l, _ = remove_self_loops(fully_connect_two_graphs(node_index, node_index))
    a_src, a_dst = all_edge_index_l
    a_bond_mask = (torch.norm(x[a_dst] - x[a_src], dim=-1, keepdim=True)) > 3.0
    neib_bond_index = all_edge_index_l[:, ~a_bond_mask.squeeze(-1)]

    # get neighbor count
    n_neib = neib_bond_index[0].unique(return_counts=True)[1]

    # get neighbor distance
    neighbors = [neib_bond_index[1][neib_bond_index[0] == atom].unique() for atom in range(x.shape[0])]
    combin = [torch.combinations(sublist, r=2).T for sublist in neighbors]
    neib_dist = torch.tensor([(torch.norm(x[com[0]] - x[com[1]], dim=-1).sum() / com.shape[1] \
                        if com.shape[1] > 0 else torch.tensor(0)).item() for com in combin], device=x.device)
    
    if n_neib.shape[0] != x.shape[0] or n_neib.shape[0] != neib_dist.shape[0]:
        neib_dist_feat = torch.zeros((x.shape[0], 2), device=x.device)
    else:
        neib_dist_feat = torch.cat([n_neib.unsqueeze(-1), neib_dist.unsqueeze(-1)], dim=-1)
    return neib_dist_feat

