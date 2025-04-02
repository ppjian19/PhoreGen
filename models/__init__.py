from .uni_denoiser import UniTransformerO2TwoUpdateGeneralBond
from .uni_denoiser import NodeUpdateLayer


def get_denoiser_net(config):
    if config.name == "uni_node_edge":
        denoiser = UniTransformerO2TwoUpdateGeneralBond(
            num_blocks=config.num_blocks,
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            n_heads=config.n_heads,
            k=config.knn,
            edge_feat_dim=config.edge_feat_dim,
            num_r_gaussian=config.num_r_gaussian,
            act_fn=config.act_fn,
            norm=config.norm,
            cutoff_mode=config.cutoff_mode,
            r_max=config.r_max,
            x2h_out_fc=config.x2h_out_fc,
            h_node_in_bond_net=config.h_node_in_bond_net, 
            direction_match=getattr(config, 'direction_match', False)
        )
    else:
        raise NotImplementedError(f"Denoiser: `{config.name}` is not implemented")
    
    return denoiser


def get_phore_encoder(config):
    phore_encoder = NodeUpdateLayer(
        config.hidden_dim, config.hidden_dim, config.hidden_dim,
        n_heads=config.n_heads, edge_feat_dim=1, 
        act_fn=config.act_fn, norm=config.norm, out_fc=config.x2h_out_fc
    )
    return phore_encoder