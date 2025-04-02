import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter
from torch_geometric.data import Batch
from .transition import ContigousTransition, GeneralCategoricalTransition
from .common import TimeGaussianSmearing, GaussianSmearing, ShiftedSoftplus, \
            compose_context, get_beta_schedule, qd_loss, log_sample_categorical, \
            get_node_accuracy, get_edge_accuracy, fully_connect_two_graphs
from . import get_denoiser_net, get_phore_encoder

# import sys
# sys.path.append("../")
from utils.sample_utils import make_edge_data, compute_batch_atom_prox_loss, \
            compute_batch_center_prox_loss, sample_from_interval, \
            get_fully_connected_edge


class PhoreDiff(nn.Module):
    def __init__(self, config, data_name, **kwargs):
        super().__init__()
        self.config = config
        self.data_name = data_name
        self.num_node_types = config.num_atom_classes  # [5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53] + masked_atom -> 12
        self.num_edge_types = config.num_bond_classes  # explicit bond type: 0, 1, 2, 3, 4 + masked_bond -> 6 ?
        self.bond_len_loss = config.bond_len_loss
        self.bond_diffusion = config.bond_diffusion
        self.bond_net_type = config.bond_net_type
        self.count_pred_type = config.count_pred_type
        self.max_atom = 78
        self.min_atom = 4
        self.loss_weight = getattr(config, 'loss_weight', [1, 100, 100])  # [pos, node, edge]
        self.count_factor = getattr(config, 'count_factor', 1)
        self.hp_emb_with_pos = getattr(config, 'hp_emb_with_pos', False)

        ## define beta and alpha
        self.define_betas_alphas(config.diff)
        
        self.node_embedder = nn.Linear(self.num_node_types, config.hidden_dim-config.diff.time_dim, bias=False)  # element type
        self.edge_embedder = nn.Linear(self.num_edge_types, config.hidden_dim-config.diff.time_dim, bias=False)  # bond type
        self.time_emb = nn.Sequential(
            TimeGaussianSmearing(stop=self.num_timesteps, num_gaussians=config.diff.time_dim, type_='linear'),
        )

        ## phore_embedding
        self.phore_embedding = nn.Linear(config.phore_feat_dim, config.hidden_dim)
        if self.hp_emb_with_pos:
            self.phore_encoder = get_phore_encoder(config.denoiser)

        ## denioser
        assert config.denoiser.hidden_dim == config.hidden_dim
        self.denoiser = get_denoiser_net(config.denoiser)

        ## atom type prediction
        self.v_inference = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(config.hidden_dim, self.num_node_types),
        )

        ## bond type prediction
        if self.bond_diffusion:
            self.distance_expansion = GaussianSmearing(0., 5., 
                                        num_gaussians=config.denoiser.num_r_gaussian, fix_offset=False)
            if self.bond_net_type == 'pre_att':
                bond_input_dim = config.denoiser.num_r_gaussian + config.hidden_dim
            elif self.bond_net_type == 'lin':
                bond_input_dim = config.hidden_dim
            else:
                raise ValueError(self.bond_net_type)
            self.bond_inference = nn.Sequential(
                nn.Linear(bond_input_dim, config.hidden_dim),
                ShiftedSoftplus(),
                nn.Linear(config.hidden_dim, self.num_edge_types)
            )
        else:
            self.distance_embedding = nn.Linear(1, config.hidden_dim-config.diff.time_dim)

        ## atom number prediction
        if self.count_pred_type == 'boundary':
            self.atom_mlp = nn.Sequential(nn.Linear(config.hidden_dim, config.hidden_dim*2), nn.ReLU(), 
                                        nn.Linear(config.hidden_dim*2, 1), nn.Sigmoid())
            self.atom_mlp_1 = nn.Sequential(nn.Linear(config.hidden_dim, config.hidden_dim*2), nn.ReLU(), 
                                        nn.Linear(config.hidden_dim*2, 1), nn.Sigmoid())
        else:
            raise ValueError(f'Invalid prediction type ({self.count_pred_type}) for atom counter.')


    def define_betas_alphas(self, config):
        self.num_timesteps = config.num_timesteps
        self.categorical_space = config.categorical_space

        ## try to get the scaling
        if self.categorical_space == 'continuous':
            self.scaling = getattr(config, 'scaling', [1., 1., 1.])
        else:
            self.scaling = [1., 1., 1.]  # actually not used for discrete space (defined for compatibility)

        ## diffusion for pos
        pos_betas = get_beta_schedule(
            num_timesteps=self.num_timesteps, 
            **config.diff_pos
            )
        assert self.scaling[0] == 1, 'scaling for pos should be 1'
        self.pos_transition = ContigousTransition(pos_betas)

        ## diffusion for node type
        node_betas = get_beta_schedule(
            num_timesteps=self.num_timesteps,
            **config.diff_atom
        )
        if self.categorical_space == 'discrete':
            init_prob = config.diff_atom.init_prob
            self.node_transition = GeneralCategoricalTransition(node_betas, self.num_node_types,
                                                            init_prob=init_prob)
        elif self.categorical_space == 'continuous':
            scaling_node = self.scaling[1]
            self.node_transition = ContigousTransition(node_betas, self.num_node_types, scaling_node)
        else:
            raise ValueError(f"Unsupported categorical space: `{self.categorical_space}`")
        
        ## diffusion for edge type
        edge_betas = get_beta_schedule(
            num_timesteps=self.num_timesteps,
            **config.diff_bond
        )
        if self.categorical_space == 'discrete':
            init_prob = config.diff_bond.init_prob
            self.edge_transition = GeneralCategoricalTransition(edge_betas, self.num_edge_types,
                                                            init_prob=init_prob)
        elif self.categorical_space == 'continuous':
            scaling_edge = self.scaling[2]
            self.edge_transition = ContigousTransition(edge_betas, self.num_edge_types, scaling_edge)
        else:
            raise ValueError(f"Unsupported categorical space: `{self.categorical_space}`")


    def sample_time(self, num_graphs, device, **kwargs):
        # sample time
        time_step = torch.randint(
            0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=device)
        time_step = torch.cat(
            [time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]
        pt = torch.ones_like(time_step).float() / self.num_timesteps
        return time_step, pt
    

    def predict_atom_count(self, h_p, batch_p, _h_p):
        atom_count = self.atom_mlp(h_p)
        atom_count = scatter(atom_count, batch_p, dim=-2, reduce='mean')
        if self.count_pred_type == 'boundary':
            if self.data_name in ['zinc_300', 'pdbbind']:
                mask_l = _h_p[:, 12] != 1
            else:
                mask_l = _h_p[:, 10] != 1
            atom_count_l = self.atom_mlp_1(h_p[mask_l, :])
            atom_count_l = scatter(atom_count_l, batch_p[mask_l], dim=-2, reduce='mean')
            atom_count_u = atom_count_l + F.relu(atom_count - atom_count_l)
            atom_count = atom_count_l, atom_count_u
        else:
            raise NotImplementedError
        
        return atom_count


    def compute_count_loss(self, true_count, pred_count):
        true_count = (true_count - self.min_atom) / (self.max_atom - self.min_atom)  # normalized
        if self.count_pred_type == 'boundary':
            loss_count = qd_loss(true_count.unsqueeze(-1).float(), *pred_count, s=160, nd=15, factor=self.count_factor)
        else:
            raise NotImplementedError
        return loss_count


    def forward(self, h_node_pert, pos_pert, batch_node,
                h_edge_pert, edge_index, batch_edge, time_step, 
                h_phore, pos_phore, phore_norm, batch_phore
                ):
        # 1 ligand node and edge embedding + time embedding + phore embedding
        time_embed_node = self.time_emb(time_step.index_select(0, batch_node))
        h_node_pert = torch.cat([self.node_embedder(h_node_pert), time_embed_node], dim=-1)

        time_embed_edge = self.time_emb(time_step.index_select(0, batch_edge))

        ## phore embedding
        h_phore_emb = self.phore_embedding(h_phore)
        if self.hp_emb_with_pos:
            f_edge_index_p = fully_connect_two_graphs(batch_phore, batch_phore)
            src, dst = f_edge_index_p
            dist_feat = torch.norm(pos_phore[dst] - pos_phore[src], p=2, dim=-1, keepdim=True)
            h_phore_emb = self.phore_encoder(h_phore_emb, dist_feat, f_edge_index_p)
        
        h_all, pos_all, batch_all, mask_ligand, mask_ligand_atom, p_index_in_ctx, l_index_in_ctx = compose_context(
            h_phore=h_phore_emb, 
            h_ligand=h_node_pert, 
            pos_phore=pos_phore, 
            pos_ligand=pos_pert, 
            batch_phore=batch_phore, 
            batch_ligand=batch_node
        )
        bond_index_in_all = l_index_in_ctx[edge_index]  # pj:bond index in ligand -> bond index in complex

        ## edge embedding
        if self.bond_diffusion:
            h_edge_pert = torch.cat([self.edge_embedder(h_edge_pert), time_embed_edge], dim=-1)
        else:
            src, dst = bond_index_in_all
            dist_feat = self.distance_embedding(torch.norm(pos_all[src] - pos_all[dst], dim=-1, keepdim=True))
            h_edge_pert = torch.cat([dist_feat, time_embed_edge], dim=-1)

        # 2 diffuse to get the updated node embedding and bond embedding
        outputs = self.denoiser(
            h=h_all, x=pos_all, group_idx=None,
            bond_index=bond_index_in_all, h_bond=h_edge_pert,
            mask_ligand=mask_ligand,
            mask_ligand_atom=mask_ligand_atom,  # dummy node is marked as 0
            batch=batch_all,
            phore_norm=phore_norm, 
            return_all=False
        )
        final_pos, final_h = outputs['x'], outputs['h']
        final_ligand_pos, final_ligand_h = final_pos[mask_ligand_atom], final_h[mask_ligand_atom]
        final_ligand_v = self.v_inference(final_ligand_h)

        pred_bond = None
        if self.bond_diffusion:
            # bond inference input
            if self.bond_net_type == 'pre_att':
                src, dst = bond_index_in_all
                dist = torch.norm(final_pos[dst] - final_pos[src], p=2, dim=-1, keepdim=True)
                r_feat = self.distance_expansion(dist)
                if self.bond_net_type == 'pre_att':
                    hi, hj = final_h[dst], final_h[src]
                    bond_inf_input = torch.cat([r_feat, (hi + hj) / 2], -1)
                else:
                    raise NotImplementedError
            elif self.bond_net_type == 'lin':
                bond_inf_input = outputs['h_bond']
            else:
                raise ValueError(self.bond_net_type)
            pred_bond = self.bond_inference(bond_inf_input)

        ## atom count prediction
        pred_count = self.predict_atom_count(h_phore_emb, batch_phore, h_phore)

        return final_ligand_v, final_ligand_pos, pred_bond, pred_count


    def compute_loss(self, data):
        ## 1. sample noise levels
        time_step, _ = self.sample_time(data.num_graphs, data['ligand'].x.device)

        ## 2. perturb pos, node, edge(fully-connected | directional)
        ligand_pos_pert = self.pos_transition.add_noise(data['ligand'].pos, time_step, data['ligand'].batch)
        ligand_node_pert = self.node_transition.add_noise(data['ligand'].x, time_step, data['ligand'].batch)
        edge_pert = self.edge_transition.add_noise(data['ligand', 'ligand'].f_edge_attr, time_step, 
                                                    data['ligand', 'ligand'].f_edge_attr_batch)
        
        if self.categorical_space == 'discrete':
            h_node_pert, log_node_t, log_node_0 = ligand_node_pert
            h_edge_pert, log_edge_t, log_edge_0 = edge_pert
        else:
            h_node_pert, h_node_0 = ligand_node_pert
            h_edge_pert, h_edge_0 = edge_pert

        ## 3. forward diffusion process
        pred_node, pred_pos, pred_edge, pred_count = self.forward(
            h_node_pert=h_node_pert,
            pos_pert=ligand_pos_pert, 
            batch_node=data['ligand'].batch, 
            h_edge_pert=h_edge_pert, 
            edge_index=data['ligand', 'ligand'].f_edge_index, 
            batch_edge=data['ligand', 'ligand'].f_edge_attr_batch, 
            time_step=time_step, 
            h_phore=data['phore'].x, 
            pos_phore=data['phore'].pos, 
            phore_norm=data['phore'].norm, 
            batch_phore=data['phore'].batch
        )

        ## 4. compute loss
        ## pos loss
        loss_pos = F.mse_loss(pred_pos, data['ligand'].pos) * self.loss_weight[0]
        
        ## bond len loss
        if self.bond_len_loss == True:
            src, dst = data['ligand', 'ligand'].edge_index
            true_length = torch.norm(data['ligand'].pos[src] - data['ligand'].pos[dst], dim=-1)
            pred_length = torch.norm(pred_pos[src] - pred_pos[dst], dim=-1)
            loss_len = F.mse_loss(pred_length, true_length)

        if self.categorical_space == 'discrete':
            ## node type loss
            log_node_recon = F.log_softmax(pred_node, dim=-1)
            log_node_post_true = self.node_transition.q_v_posterior(log_node_0, 
                                    log_node_t, time_step, data['ligand'].batch, v0_prob=True
                                    )
            log_node_post_pred = self.node_transition.q_v_posterior(log_node_recon, log_node_t, 
                                    time_step, data['ligand'].batch, v0_prob=True
                                    )
            kl_node = self.node_transition.compute_v_Lt(log_node_post_true, log_node_post_pred, 
                                    log_node_0, t=time_step, batch=data['ligand'].batch
                                    )
            loss_node = torch.mean(kl_node) * self.loss_weight[1]

            if self.bond_diffusion:
                ## edge type loss
                log_edge_recon = F.log_softmax(pred_edge, dim=-1)
                log_edge_post_true = self.edge_transition.q_v_posterior(log_edge_0, 
                                        log_edge_t, time_step, data['ligand', 'ligand'].f_edge_attr_batch, 
                                        v0_prob=True
                                        )
                log_edge_post_pred = self.edge_transition.q_v_posterior(log_edge_recon, 
                                        log_edge_t, time_step, data['ligand', 'ligand'].f_edge_attr_batch, 
                                        v0_prob=True
                                        )
                kl_edge = self.edge_transition.compute_v_Lt(log_edge_post_true, log_edge_post_pred, 
                                log_edge_0, t=time_step, batch=data['ligand', 'ligand'].f_edge_attr_batch
                                )
                loss_edge = torch.mean(kl_edge) * self.loss_weight[2]
        else:
            loss_node = F.mse_loss(pred_node, h_node_0)  * 30
            if self.bond_diffusion:
                loss_edge = F.mse_loss(pred_edge, h_edge_0) * 30

        ## atom count loss
        true_count = data['ligand'].ptr[1:] - data['ligand'].ptr[:-1]
        loss_count = self.compute_count_loss(true_count, pred_count)


        ## total loss
        loss_total = loss_pos + loss_node + (loss_edge if self.bond_diffusion else 0) + \
            loss_count + (loss_len if self.bond_len_loss else 0)
        
        loss_dict = {
            'loss': loss_total.item(),
            'loss_pos': loss_pos.item(),
            'loss_node': loss_node.item(),
            'loss_count': loss_count.item()
        }
        if self.bond_len_loss == True:
            loss_dict['loss_len'] = loss_len.item()
        if self.bond_diffusion:
            loss_dict['loss_edge'] = loss_edge.item()

        ## atomic type accuracy
        loss_dict['node_acc'] = get_node_accuracy(data['ligand'].x, pred_node, data['ligand'].batch)
        ## edge type accuracy
        if self.bond_diffusion:
            loss_dict['edge_acc'] = get_edge_accuracy(data['ligand', 'ligand'].f_edge_attr, 
                                    pred_edge, data['ligand', 'ligand'].f_edge_attr_batch)
        return loss_total, loss_dict


    @torch.no_grad()
    def sample_nodes(self, data, batch_size, device, sample_mode='uniform', normal_scale=4.0):
        # batch_p = torch.zeros(data['phore'].num_nodes, dtype=torch.long).to(device)
        h_p = self.phore_embedding(data['phore'].x)
        if self.hp_emb_with_pos:
            f_edge_index_p = get_fully_connected_edge(data['phore'].num_nodes).to(device)
            src, dst = f_edge_index_p
            dist_feat = torch.norm(data['phore'].pos[dst] - data['phore'].pos[src], p=2, dim=-1, keepdim=True)
            h_p = self.phore_encoder(h_p, dist_feat, f_edge_index_p)
        
        atom_count = self.atom_mlp(h_p).mean(dim=0, keepdim=True)
        # atom_count = self.atom_mlp(h_p)
        # atom_count = scatter(atom_count, batch_p, dim=-2, reduce='mean')
        if self.count_pred_type == 'boundary':
            if self.data_name in ['zinc_300', 'pdbbind']:
                mask_l = data['phore'].x[:, 12] != 1
            else:
                mask_l = data['phore'].x[:, 10] != 1
            atom_count_l = self.atom_mlp_1(h_p[mask_l, :]).mean(dim=0, keepdim=True)
            # atom_count_l = self.atom_mlp_1(h_p[mask_l, :])
            # atom_count_l = scatter(atom_count_l, batch_p[mask_l], dim=-2, reduce='mean')
            atom_count_u = atom_count_l + F.relu(atom_count - atom_count_l)
            assert atom_count_l.shape == atom_count_u.shape, \
                'the shape of atom count lower and upper boundary should be the same.'
            atom_count_l = (atom_count_l * (self.max_atom - self.min_atom) + self.min_atom).round().int().squeeze(-1)
            atom_count_u = (atom_count_u * (self.max_atom - self.min_atom) + self.min_atom).round().int().squeeze(-1)
            ligand_num_atoms = sample_from_interval(
                atom_count_l.item(), atom_count_u.item(), 
                batch_size, mode=sample_mode, scale=normal_scale
                ).to(device)
        else:
            raise NotImplementedError
        return ligand_num_atoms


    @torch.no_grad()
    def sample(self, data, n_graphs, device, pos_guidance_opt=None, sample_mode='uniform', normal_scale=4.0, **kwargs):
        
        ## sample number nodes
        ligand_num_atoms = self.sample_nodes(data, n_graphs, device, sample_mode, normal_scale)
        # ligand_num_atoms = torch.randint(low=3, high=11, size=(n_graphs, )).to(device)
        ligand_batch = torch.repeat_interleave(torch.arange(n_graphs).to(device), 
                                ligand_num_atoms).to(device)
        center = data.center
        batch_phore = Batch.from_data_list([data.clone()] * n_graphs).to(device)

        ## get the init values(pos, node, edge[fully-connected | directional])
        ligand_edge_index, ligand_edge_batch = make_edge_data(ligand_num_atoms)
        n_node_all = len(ligand_batch)
        n_edge_all = len(ligand_edge_batch)

        ligand_pos_init = self.pos_transition.sample_init([n_node_all, 3]) - center  # center pos
        ligand_node_init = self.node_transition.sample_init(n_node_all)
        ligand_edge_init = self.edge_transition.sample_init(n_edge_all)

        if self.categorical_space == 'discrete':
            _, h_node_init, log_node_type = ligand_node_init
            _, h_edge_init, log_edge_type = ligand_edge_init
        else:
            h_node_init = ligand_node_init
            h_edge_init = ligand_edge_init

        ## log init
        node_traj = torch.zeros([self.num_timesteps+1, n_node_all, h_node_init.shape[-1]],
                                dtype=h_node_init.dtype).to(device)
        pos_traj = torch.zeros([self.num_timesteps+1, n_node_all, 3], 
                               dtype=ligand_pos_init.dtype).to(device)
        edge_traj = torch.zeros([self.num_timesteps+1, n_edge_all, h_edge_init.shape[-1]],
                                    dtype=h_edge_init.dtype).to(device)
        node_traj[0] = h_node_init
        pos_traj[0] = ligand_pos_init
        edge_traj[0] = h_edge_init

        ## sample loop
        h_node_pert = h_node_init
        pos_pert = ligand_pos_init
        h_edge_pert = h_edge_init
        for i, step in enumerate(range(self.num_timesteps)[::-1]):
            time_step = torch.full(size=(n_graphs,), fill_value=step, dtype=torch.long).to(device)
            
            ## inference
            pred_node, pred_pos, pred_edge, _ = self.forward(
                h_node_pert=h_node_pert, 
                pos_pert=pos_pert, 
                batch_node=ligand_batch, 
                h_edge_pert=h_edge_pert, 
                edge_index=ligand_edge_index, 
                batch_edge=ligand_edge_batch, 
                time_step=time_step, 
                h_phore=batch_phore['phore'].x, 
                pos_phore=batch_phore['phore'].pos, 
                phore_norm=batch_phore['phore'].norm, 
                batch_phore=batch_phore['phore'].batch
            )

            ## get the t-1 state : node and edge
            if self.categorical_space == 'discrete':
                # node types
                log_node_recon = F.log_softmax(pred_node, dim=-1)
                log_node_type = self.node_transition.q_v_posterior(log_node_recon, log_node_type, 
                                                        time_step, ligand_batch, v0_prob=True)
                node_type_prev = log_sample_categorical(log_node_type)
                h_node_prev = self.node_transition.onehot_encode(node_type_prev)

                h_edge_prev = 0
                if self.bond_diffusion:
                    # edge types
                    log_edge_recon = F.log_softmax(pred_edge, dim=-1)
                    log_edge_type = self.edge_transition.q_v_posterior(log_edge_recon, log_edge_type, 
                                                            time_step, ligand_edge_batch, v0_prob=True)
                    edge_type_prev = log_sample_categorical(log_edge_type)
                    h_edge_prev = self.edge_transition.onehot_encode(edge_type_prev)
            else:
                h_node_prev = self.node_transition.get_prev_from_recon(
                    x_t=h_node_pert, x_recon=pred_node, t=time_step, batch=ligand_batch)
                h_edge_prev = 0
                if self.bond_diffusion:
                    h_edge_prev = self.edge_transition.get_prev_from_recon(
                        x_t=h_edge_pert, x_recon=pred_edge, t=time_step, batch=ligand_edge_batch)
            
            ## validity guidance
            all_energy_grad = 0.
            if pos_guidance_opt is not None:
                xt = pos_pert
                xt.requires_grad = True

                for drift in pos_guidance_opt:
                    energy_grad = 0.
                    if drift['type'] == 'atom_prox' and self.bond_diffusion:
                        with torch.enable_grad():
                            energy = compute_batch_atom_prox_loss(
                                xt, ligand_batch, h_edge_prev, ligand_edge_index, ligand_edge_batch, 
                                min_d=drift['min_d'], max_d=drift['max_d']
                                )
                            if energy != torch.tensor(0.):
                                energy_grad = torch.autograd.grad(energy, xt)[0]
                    elif drift['type'] == 'center_prox':
                        with torch.enable_grad():
                            if self.data_name in ['zinc_300', 'pdbbind']:
                                p_mask = data['phore'].x[:, 12] != 1
                            else:
                                p_mask = data['phore'].x[:, 10] != 1
                            onlyPhoreCenter = data['phore'].pos[p_mask].mean(dim=0)
                            energy = compute_batch_center_prox_loss(
                                xt, ligand_batch, onlyPhoreCenter
                                )
                            energy_grad = torch.autograd.grad(energy, xt)[0]
                    all_energy_grad += energy_grad

            ## get the t-1 state : pos
            pos_prev = self.pos_transition.get_prev_from_recon(
                x_t=pos_pert, x_recon=pred_pos, t=time_step, batch=ligand_batch, 
                energy_grad=all_energy_grad)

            # log update
            node_traj[i+1] = h_node_prev
            pos_traj[i+1] = pos_prev + center
            edge_traj[i+1] = h_edge_prev

            # update t-1
            h_node_pert = h_node_prev
            pos_pert = pos_prev
            h_edge_pert = h_edge_prev

        pred_pos = pred_pos + center
        return {
            'pred': [pred_node, pred_pos, pred_edge],
            'traj': [node_traj, pos_traj, edge_traj], 
            'lig_info': [ligand_num_atoms, ligand_batch, 
                         ligand_edge_index, ligand_edge_batch]
        }
