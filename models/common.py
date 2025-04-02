import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn_graph
from torch_scatter import scatter
from torch_geometric.utils import remove_self_loops

# ----- denoiser related -----

class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50, fix_offset=True):
        super(GaussianSmearing, self).__init__()
        self.start = start
        self.stop = stop
        if fix_offset:
            # customized offset
            offset = torch.tensor([0, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 10])
            self.num_gaussians = 20
        else:
            offset = torch.linspace(start, stop, num_gaussians)
            self.num_gaussians = num_gaussians
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def __repr__(self):
        return f'GaussianSmearing(start={self.start}, stop={self.stop}, num_gaussians={self.num_gaussians})'

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class TimeGaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=10.0, num_gaussians=50, type_='exp'):
        super().__init__()
        self.start = start
        self.stop = stop
        if type_ == 'exp':
            offset = torch.exp(torch.linspace(start=np.log(start+1), end=np.log(stop+1), steps=num_gaussians)) - 1
        elif type_ == 'linear':
            offset = torch.linspace(start=start, end=stop, steps=num_gaussians)
        else:
            raise NotImplementedError('type_ must be either exp or linear')
        diff = torch.diff(offset)
        diff = torch.cat([diff[:1], diff])
        coeff = -0.5 / (diff**2)
        self.register_buffer('coeff', coeff)
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.clamp_min(self.start)
        dist = dist.clamp_max(self.stop)
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift
    

class AngularEncoding(nn.Module):

    def __init__(self, num_funcs=3):
        super().__init__()
        self.num_funcs = num_funcs
        self.register_buffer('freq_bands', torch.FloatTensor(
            [i+1 for i in range(num_funcs)] + [1./(i+1) for i in range(num_funcs)]
        ))

    def get_out_dim(self, in_dim):
        return in_dim * (1 + 2*2*self.num_funcs)

    def forward(self, x):
        """
        Args:
            x:  (E, ).
        """
        x = x.unsqueeze(-1)  # (E, 1)
        code = torch.cat([x, torch.sin(x * self.freq_bands), torch.cos(x * self.freq_bands)], dim=-1)   # (E, 4f+1)

        return code


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class MLP(nn.Module):
    """MLP with the same hidden dim across all layers."""

    def __init__(self, in_dim, out_dim, hidden_dim, num_layer=2, norm=True, act_fn='relu', act_last=False):
        super().__init__()
        layers = []
        for layer_idx in range(num_layer):
            if layer_idx == 0:
                layers.append(nn.Linear(in_dim, hidden_dim))
            elif layer_idx == num_layer - 1:
                layers.append(nn.Linear(hidden_dim, out_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            if layer_idx < num_layer - 1 or act_last:
                if norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(get_act_func(act_fn))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


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
    elif act_func == 'swish':
        return Swish()
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


def outer_product(*vectors):
    for index, vector in enumerate(vectors):
        if index == 0:
            out = vector.unsqueeze(-1)
        else:
            out = out * vector.unsqueeze(1)
            out = out.view(out.shape[0], -1).unsqueeze(-1)
    return out.squeeze()


def find_index_after_sorting(size_all, size_p, size_l, sort_idx, device):
    # find phore/ligand index in ctx
    ligand_index_in_ctx = torch.zeros(size_all, device=device)
    ligand_index_in_ctx[size_p:size_p + size_l] = torch.arange(1, size_l + 1, device=device)
    ligand_index_in_ctx = torch.sort(ligand_index_in_ctx[sort_idx], stable=True).indices[-size_l:]
    ligand_index_in_ctx = ligand_index_in_ctx.to(device)

    phore_index_in_ctx = torch.zeros(size_all, device=device)
    phore_index_in_ctx[:size_p] = torch.arange(1, size_p + 1, device=device)
    phore_index_in_ctx = torch.sort(phore_index_in_ctx[sort_idx], stable=True).indices[-size_p:]
    phore_index_in_ctx = phore_index_in_ctx.to(device)
    return phore_index_in_ctx, ligand_index_in_ctx


def compose_context(h_phore, h_ligand,
                    pos_phore, pos_ligand,
                    batch_phore, batch_ligand, ligand_atom_mask=None):
    # previous version has problems when ligand atom types are fixed
    # (due to sorting randomly in case of same element)
    batch_ctx = torch.cat([batch_phore, batch_ligand], dim=0)
    # sort_idx = batch_ctx.argsort()
    # pj: convert to (phore + ligand + phore + ligand ...) in a batch
    sort_idx = torch.sort(batch_ctx, stable=True).indices

    mask_ligand = torch.cat([
        torch.zeros([batch_phore.size(0)], device=batch_phore.device).bool(),
        torch.ones([batch_ligand.size(0)], device=batch_ligand.device).bool(),
    ], dim=0)[sort_idx]

    if ligand_atom_mask is None:
        mask_ligand_atom = mask_ligand
    else:
        mask_ligand_atom = torch.cat([
            torch.zeros([batch_phore.size(0)], device=batch_phore.device).bool(),
            ligand_atom_mask
        ], dim=0)[sort_idx]

    batch_ctx = batch_ctx[sort_idx]
    h_ctx = torch.cat([h_phore, h_ligand], dim=0)[sort_idx]  # (N_protein+N_ligand, H)
    pos_ctx = torch.cat([pos_phore, pos_ligand], dim=0)[sort_idx]  # (N_protein+N_ligand, 3)
    phore_index_in_ctx, ligand_index_in_ctx = find_index_after_sorting(
        len(h_ctx), len(h_phore), len(h_ligand), sort_idx, batch_phore.device)
    return h_ctx, pos_ctx, batch_ctx, mask_ligand, mask_ligand_atom, phore_index_in_ctx, ligand_index_in_ctx


def hybrid_edge_connection(ligand_pos, protein_pos, k, ligand_index, protein_index):
    # fully-connected for ligand atoms
    dst = torch.repeat_interleave(ligand_index, len(ligand_index))
    src = ligand_index.repeat(len(ligand_index))
    mask = dst != src
    dst, src = dst[mask], src[mask]
    ll_edge_index = torch.stack([src, dst])

    # knn for ligand-protein edges
    ligand_protein_pos_dist = torch.unsqueeze(ligand_pos, 1) - torch.unsqueeze(protein_pos, 0)
    ligand_protein_pos_dist = torch.norm(ligand_protein_pos_dist, p=2, dim=-1)
    knn_p_idx = torch.topk(ligand_protein_pos_dist, k=k, largest=False, dim=1).indices
    knn_p_idx = protein_index[knn_p_idx]
    knn_l_idx = torch.unsqueeze(ligand_index, 1)
    knn_l_idx = knn_l_idx.repeat(1, k)
    pl_edge_index = torch.stack([knn_p_idx, knn_l_idx], dim=0)
    pl_edge_index = pl_edge_index.view(2, -1)
    return ll_edge_index, pl_edge_index


def batch_hybrid_edge_connection(x, k, mask_ligand, batch, add_p_index=False):
    batch_size = batch.max().item() + 1
    batch_ll_edge_index, batch_pl_edge_index, batch_p_edge_index = [], [], []
    with torch.no_grad():
        for i in range(batch_size):
            ligand_index = ((batch == i) & (mask_ligand == 1)).nonzero()[:, 0]
            protein_index = ((batch == i) & (mask_ligand == 0)).nonzero()[:, 0]
            ligand_pos, protein_pos = x[ligand_index], x[protein_index]
            ll_edge_index, pl_edge_index = hybrid_edge_connection(
                ligand_pos, protein_pos, k, ligand_index, protein_index)
            batch_ll_edge_index.append(ll_edge_index)
            batch_pl_edge_index.append(pl_edge_index)
            if add_p_index:
                all_pos = torch.cat([protein_pos, ligand_pos], 0)
                p_edge_index = knn_graph(all_pos, k=k, flow='source_to_target')
                p_edge_index = p_edge_index[:, p_edge_index[1] < len(protein_pos)]
                p_src, p_dst = p_edge_index
                all_index = torch.cat([protein_index, ligand_index], 0)
                p_edge_index = torch.stack([all_index[p_src], all_index[p_dst]], 0)
                batch_p_edge_index.append(p_edge_index)

    if add_p_index:
        edge_index = [torch.cat([ll, pl, p], -1) for ll, pl, p in zip(
            batch_ll_edge_index, batch_pl_edge_index, batch_p_edge_index)]
    else:
        edge_index = [torch.cat([ll, pl], -1) for ll, pl in zip(batch_ll_edge_index, batch_pl_edge_index)]
    edge_index = torch.cat(edge_index, -1)
    return edge_index


def qd_loss(y_true, y_l, y_u, a=0.05, s=160, nd=15, factor=1, epsilon=1e-12, mode='soft'):
    # captured samples
    n = y_true.shape[0]

    k_u_h = torch.relu(torch.sign(y_u - y_true))
    k_l_h = torch.relu(torch.sign(y_true - y_l))

    # soft uses sigmoid fn
    k_u_s = torch.sigmoid((y_u - y_true) * s)
    k_l_s = torch.sigmoid((y_true - y_l) * s)

    k_s = k_u_s * k_l_s
    k_h = k_u_h * k_l_h

    # combine for loss fn
    MPIW_c = torch.sum((y_u - y_l) * k_h) / (torch.sum(k_h) + epsilon) * factor
    PICP = torch.mean(k_s) if mode == 'soft' else torch.mean(k_h)

    # QD Loss
    loss = MPIW_c + (torch.relu((1-a) - PICP) ** 2) * (n ** 0.5) * nd
    return loss


def get_node_accuracy(h_true, h_recon, batch):
    ## compute the number of molecules in each batch with exactly the right atom type
    not_match = (h_recon.softmax(dim=-1).argmax(dim=-1) != h_true).float()
    not_match_batch = (scatter(not_match, batch) == 0).int().sum().item()
    num_graphs = batch.unique().numel()
    return not_match_batch / num_graphs


def get_edge_accuracy(edge_true, edge_recon, batch):
    ## compute the number of molecules in each batch with exactly the right bond type
    not_match = (edge_recon.softmax(dim=-1).argmax(dim=-1) != edge_true).float()
    not_match_batch = (scatter(not_match, batch) == 0).int().sum().item()
    num_graphs = batch.unique().numel()
    return not_match_batch / num_graphs


def get_neib_norm(l_x, l_batch):
    neib_edge_index = knn_graph(l_x, k=3, batch=l_batch)
    n_src, n_dst = neib_edge_index
    l_neib_norm = scatter(l_x[n_src], n_dst, dim=0, reduce='mean') - l_x
    return l_neib_norm


def get_direction_feat(x, phore_norm, edge_index, mask_ligand, batch):
    # [center postion of neighbor atoms - atom position]
    l_neib_norm = get_neib_norm(x[mask_ligand], batch[mask_ligand])
    
    # combine phore norm and neib center norm
    comb_norm = torch.zeros_like(x).to(x.device)
    comb_norm[~mask_ligand] = phore_norm
    comb_norm[mask_ligand] = l_neib_norm

    src, dst = edge_index
    vec_1 = comb_norm[src]
    vec_2 = comb_norm[dst]
    vec_3 = x[src] - x[dst]

    dot_1 = (vec_1 * vec_2).sum(-1, keepdim=True)
    dot_2 = (vec_1 * vec_3).sum(-1, keepdim=True)
    dot_3 = (vec_2 * vec_3).sum(-1, keepdim=True)
    dire_feat = torch.cat([dot_1, dot_2, dot_3], dim=-1)

    return dire_feat


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



# ----- torch utils -----

def to_torch_const(x):
    x = torch.from_numpy(x).float()
    x = nn.Parameter(x, requires_grad=False)
    return x

def log_1_min_a(a):
    return np.log(1 - np.exp(a) + 1e-40)



# ----- categorical diffusion related -----

def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x


def extract(coef, t, batch, ndim=2):
    out = coef[t][batch]
    # warning: test wrong code!
    # out = coef[batch]
    # return out.view(-1, *((1,) * (len(out_shape) - 1)))
    if ndim == 1:
        return out
    elif ndim == 2:
        return out.unsqueeze(-1)
    elif ndim == 3:
        return out.unsqueeze(-1).unsqueeze(-1)
    else:
        raise NotImplementedError('ndim > 3')


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def log_sample_categorical(logits):
    uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    sample_index = (gumbel_noise + logits).argmax(dim=-1)
    # sample_onehot = F.one_hot(sample, self.num_classes)
    # log_sample = index_to_log_onehot(sample, self.num_classes)
    return sample_index


def categorical_kl(log_prob1, log_prob2):
    kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=-1)
    return kl


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=-1)



# ----- beta  schedule -----

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)


def advance_schedule(timesteps, scale_start, scale_end, width, return_alphas_bar=False):
    k = width
    A0 = scale_end
    A1 = scale_start

    a = (A0-A1)/(sigmoid(-k) - sigmoid(k))
    b = 0.5 * (A0 + A1 - a)

    x = np.linspace(-1, 1, timesteps)
    y = a * sigmoid(- k * x) + b
    # print(y)
    
    alphas_cumprod = y 
    alphas = np.zeros_like(alphas_cumprod)
    alphas[0] = alphas_cumprod[0]
    alphas[1:] = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas
    betas = np.clip(betas, 0, 1)
    if not return_alphas_bar:
        return betas
    else:
        return betas, alphas_cumprod


def segment_schedule(timesteps, time_segment, segment_diff):
    assert np.sum(time_segment) == timesteps
    alphas_cumprod = []
    for i in range(len(time_segment)):
        time_this = time_segment[i] + 1
        params = segment_diff[i]
        _, alphas_this = advance_schedule(time_this, **params, return_alphas_bar=True)
        alphas_cumprod.extend(alphas_this[1:])
    alphas_cumprod = np.array(alphas_cumprod)
    
    alphas = np.zeros_like(alphas_cumprod)
    alphas[0] = alphas_cumprod[0]
    alphas[1:] = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas
    betas = np.clip(betas, 0, 1)
    return betas


def sigmoid(x):
    return 1 / (np.exp(-x) + 1)


def get_beta_schedule(beta_schedule, num_timesteps, **kwargs):
    
    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    kwargs['beta_start'] ** 0.5,
                    kwargs['beta_end'] ** 0.5,
                    num_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            kwargs['beta_start'], kwargs['beta_end'], num_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = kwargs['beta_end'] * np.ones(num_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_timesteps, 1, num_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        s = dict.get(kwargs, 's', 6)
        betas = np.linspace(-s, s, num_timesteps)
        betas = sigmoid(betas) * (kwargs['beta_end'] - kwargs['beta_start']) + kwargs['beta_start']
    elif beta_schedule == "cosine":
        s = dict.get(kwargs, 's', 0.008)
        betas = cosine_beta_schedule(num_timesteps, s=s)
    elif beta_schedule == "advance":
        scale_start = dict.get(kwargs, 'scale_start', 0.999)
        scale_end = dict.get(kwargs, 'scale_end', 0.001)
        width = dict.get(kwargs, 'width', 2)
        betas = advance_schedule(num_timesteps, scale_start, scale_end, width)
    elif beta_schedule == "segment":
        betas = segment_schedule(num_timesteps, kwargs['time_segment'], kwargs['segment_diff'])
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_timesteps,)
    return betas

