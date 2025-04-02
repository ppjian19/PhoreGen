import os
import numpy as np
import yaml, pickle
from easydict import EasyDict
import torch
from torch_geometric.transforms import Compose
from datasets.transform import AddPhoreNoise, FeaturizeLigandBond
from datasets.phoregen import pz_dataset, mol_dataset



def get_parameter_number(model):                                                                                                                                                         
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"['Total parameters': {total_num / 1e6:.4f} M, 'Trainable parameters': {trainable_num / 1e6:.4f} M]"


def freeze_parameters(model, cfg):
    if getattr(cfg, 'freeze_pos', False):
        print(f"[I] Freezing position update layers...")
        for i in range(model.denoiser.num_layers):
            for p in model.denoiser.base_block[i].pos_layer_with_edge.parameters():
                p.requires_grad = False
            for p in model.denoiser.base_block[i].pos_layer_with_bond.parameters():
                p.requires_grad = False
    return model


def load_config(path):
    with open(path, 'r') as f:
        return EasyDict(yaml.safe_load(f))


def easydict_to_dict(d):
    if isinstance(d, EasyDict):
        d = {k: easydict_to_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        d = [easydict_to_dict(v) for v in d]
    return d


def save_yaml_file(path, content):
    assert isinstance(path, str), f'path must be a string, got {path} which is a {type(path)}'
    content = yaml.dump(data=content)
    if '/' in path and os.path.dirname(path) and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write(content)


def get_optimizer(model, cfg, freezed=False):
    if freezed:
        para = filter(lambda p: p.requires_grad, model.parameters())  # filter out the parameters that don't require gradients
    else:
        para = model.parameters()

    if cfg.type == 'adam':
        return torch.optim.Adam(
            para,
            lr=cfg.lr
        )
    elif cfg.type == 'adamw':
        return torch.optim.AdamW(
            para,
            lr=cfg.lr,
            amsgrad=True, 
            weight_decay=cfg.weight_decay
        )
    else:
        raise NotImplementedError('Optimizer not supported: %s' % cfg.type)


def get_scheduler(optimizer, cfg):
    if cfg.type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=cfg.lr_decay_factor,
            patience=cfg.scheduler_patience,
            min_lr=cfg.min_lr
        )
    else:
        raise NotImplementedError('Scheduler not supported: %s' % cfg.type)


def get_transform(conf):
    trans_list = [FeaturizeLigandBond()]
    if conf.add_phore_noise:
        trans_list.append(AddPhoreNoise(noise_std=conf.phore_noise_std, 
                                        angle=conf.phore_norm_angle))
    return Compose(trans_list)


def cut_dataset(train_filelist, valid_filelist, test_filelist):
    print('[I] Cuting data...')
    root_path = os.path.dirname(train_filelist)
    subset_train_file = os.path.splitext(train_filelist)[0].split('/')[-1] + '_subset' + '.pkl'
    subset_valid_file = os.path.splitext(valid_filelist)[0].split('/')[-1] + '_subset' + '.pkl'
    subset_test_file = os.path.splitext(test_filelist)[0].split('/')[-1] + '_subset' + '.pkl'
    train_filelist = os.path.join(root_path, subset_train_file)
    valid_filelist = os.path.join(root_path, subset_valid_file)
    test_filelist = os.path.join(root_path, subset_test_file)
    return train_filelist, valid_filelist, test_filelist


def get_dataset(args, transform):
    # print(f'[I] Using PhoreGenDataset_{args.pg_data}.')
    if args.pg_data == 'pz':
        num_train = 120000
        num_valid = 10002
        num_test = 10002
        zinc_dataset = pz_dataset(dataset_file=args.zinc_filelist, dataset=args.data_name, max_node=args.max_atom, 
                                    center=args.center, include_charges=args.charge_weight, transform=transform, 
                                    remove_H=args.remove_H, include_hybrid=args.include_hybrid, hybrid_one_hot=args.hybrid_one_hot, 
                                    add_core_atoms=args.add_core_atoms, include_valencies=args.include_valencies, 
                                    include_ring=args.include_ring, include_aromatic=args.include_aromatic, 
                                    include_neib_dist=args.include_neib_dist
                                    )
        if args.cut_data:
            print('[I] Cuting data...')
            train_set = zinc_dataset[:num_train]
        else:
            train_set = zinc_dataset[:-num_valid-num_test]
        valid_set = zinc_dataset[-num_valid-num_test:-num_test]
        test_set = zinc_dataset[-num_test:]
    
    elif args.pg_data == "mol_phore":
        if args.data_name == 'zinc_300':
            # assert os.path.basename(args.save_path).split('_')[0] == 'ZINC'
            train_filelist = args.zinc_train_filelist
            valid_filelist = args.zinc_valid_filelist
            test_filelist = args.zinc_test_filelist
            if args.cut_data: 
                train_filelist, valid_filelist, test_filelist = cut_dataset(train_filelist, valid_filelist, test_filelist)
        elif args.data_name == 'pdbbind':
            # assert os.path.basename(args.save_path) == 'pdbbind_pkl'
            pdbbind_filelist = pickle.load(open(args.pdbbind_filelist, 'rb'))
            train_filelist = pdbbind_filelist['pdbbind_train']
            valid_filelist = pdbbind_filelist['pdbbind_valid']
            test_filelist = pdbbind_filelist['pdbbind_test']

        train_set = mol_dataset(file_list=train_filelist, center=args.center, transform=transform, remove_H=args.remove_H, 
                        save_path=args.save_path, include_charges=args.charge_weight, include_hybrid=args.include_hybrid, 
                        hybrid_one_hot=args.hybrid_one_hot, add_core_atoms=args.add_core_atoms, include_valencies=args.include_valencies, 
                        include_ring=args.include_ring, include_aromatic=args.include_aromatic, 
                        include_neib_dist=args.include_neib_dist, data_name=args.data_name
                        )
        valid_set = mol_dataset(file_list=valid_filelist, center=args.center, transform=transform, remove_H=args.remove_H, 
                        save_path=args.save_path, include_charges=args.charge_weight, include_hybrid=args.include_hybrid, 
                        hybrid_one_hot=args.hybrid_one_hot, add_core_atoms=args.add_core_atoms, include_valencies=args.include_valencies, 
                        include_ring=args.include_ring, include_aromatic=args.include_aromatic, 
                        include_neib_dist=args.include_neib_dist, data_name=args.data_name
                        )
        test_set = mol_dataset(file_list=test_filelist, center=args.center, transform=transform, remove_H=args.remove_H, 
                        save_path=args.save_path, include_charges=args.charge_weight, include_hybrid=args.include_hybrid, 
                        hybrid_one_hot=args.hybrid_one_hot, add_core_atoms=args.add_core_atoms, include_valencies=args.include_valencies, 
                        include_ring=args.include_ring, include_aromatic=args.include_aromatic, 
                        include_neib_dist=args.include_neib_dist, data_name=args.data_name
                        )
    return train_set, valid_set, test_set


class Queue():
    def __init__(self, max_len=50):
        self.items = []
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)


def gradient_clipping(flow, gradnorm_queue, verbose=False):
    # Allow gradient norm to be 150% + 2 * stdev of the recent history.
    max_grad_norm = 1.5 * gradnorm_queue.mean() + 2 * gradnorm_queue.std()

    # Clips gradient and returns the norm
    grad_norm = torch.nn.utils.clip_grad_norm_(
        flow.parameters(), max_norm=max_grad_norm, norm_type=2.0)

    if float(grad_norm) > max_grad_norm:
        gradnorm_queue.add(float(max_grad_norm))
    else:
        gradnorm_queue.add(float(grad_norm))

    if verbose and float(grad_norm) > max_grad_norm:
        print(f'Clipped gradient with value {grad_norm:.1f} '
              f'while allowed {max_grad_norm:.1f}')
    return grad_norm


def print_parm_log(args):
    if args.clip_grad_mode == 'queue':
        print(f"[I] Clip the gradient by 'Queue'.")
    else:
        print(f"[I] Clip the gradient by 'torch.nn.utils.clip_grad_norm_()', max_grad_norm is '{args.max_grad_norm}'.")
    print(f"[I] Add 'noise' to the coordinates of ligand, std: {args.lig_noise_std}.") if args.add_lig_noise else None
    print(f"[I] Add 'noise' to the pharmacophore, std: {args.phore_noise_std}, angle: {args.phore_norm_angle}.") if args.add_phore_noise else None
    print(f"[I] Start training model.\n")
