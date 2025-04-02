import copy, json, pickle
import os
import shutil
import time
from tensorboardX import SummaryWriter
import torch
import yaml
from utils.training_utils import save_yaml_file, easydict_to_dict
from utils.misc import seed_all
from easydict import EasyDict
import torch.distributed as dist

class Logger():
    def __init__(self, args) -> None:
        args = self.prepare_args(args)
        seed_all(args.train.seed)
        self.run_dir = os.path.join(args.logger.result, args.logger.run_name)
        self.run_name = args.logger.run_name
        self.result = args.logger.result
        self.restart = args.logger.restart
        self.tensorboard = args.logger.tensorboard
        self.dataset_args = []
        self.init_status(args)
        self.check_existence(args, self.restart, self.run_dir, 
                             restart_dir=args.logger.restart_dir, model_ckp=args.logger.model_ckp)
        self.init_writer()
        self.args = args
        

    def check_existence(self, args, restart, run_dir, restart_dir=None, model_ckp='last'):
        """
        Check the running status
        Args: 
            restart: ['restart', 'inplace', 'backup', 'finetuning', 'none', 'overwrite']
        """
        prev_dir = None
        exist_flag = os.path.exists(run_dir)
        if exist_flag:
            if restart == 'none':
                raise RuntimeError(f"[E] The run directory `{run_dir}` exists already."+\
                                   " Please use an another run name instead.")
            elif restart == 'overwrite':
                os.system(f"rm -rf {run_dir}")
                # shutil.rmtree(run_dir)
                print("[W] Restart mode set as `overwrite`, removing the existing directory.")
            else:
                if restart == 'backup':
                    prev_dir = run_dir

            if restart_dir is not None:
                assert os.path.abspath(run_dir) != os.path.abspath(restart_dir), \
                    f"[E] Same run directory with the restart directory is not allowed. Please change the run name."+\
                        f"\n> Run directory: {run_dir}\n> Restart directory: {restart_dir}"
        # Loading previous status if available
        prev_dir = restart_dir if restart_dir is not None else prev_dir
        if prev_dir is not None:
            self.load_previous_status(args, prev_dir, run_dir, model_ckp)
        # Dumping model args and overall parameters.
        os.makedirs(run_dir, exist_ok=True)
        save_yaml_file(self.parameter_file, easydict_to_dict(args))
        json.dump(self.model_conf.__dict__, open(self.model_conf_file, 'w'), indent=4)


    def prepare_args(self, args):
        if args.config:
            print(f"[I] Reading parameters from config file `{args.config}` which is privileged.")
            args = EasyDict(yaml.safe_load(open(args.config, 'r')))
            # args.__dict__ = conf_args
        else:
            raise RuntimeError("[E] No config file is specified.")

        if args.train.device == 'cuda' and not torch.cuda.is_available():
            args.train.device = 'cpu'
            print('[W] The model is specified to train on `CUDA` but no devices available. Switch to `CPU`.')

        if args.dataset.charge_weight:
            args.lig_feat_dim += 1
            print("[W] The model parameter of 'lig_feat_dim' plus 1 when 'charge_weight' is not 0.")

        if args.dataset.include_hybrid:
            if args.dataset.hybrid_one_hot:
                args.model.lig_feat_dim += 4
                print("[W] The model parameter of 'lig_feat_dim' plus 4 when 'include_hybrid' is true and encode hybrid into one-hot.")
            else:
                args.model.lig_feat_dim += 1
                print("[W] The model parameter of 'lig_feat_dim' plus 1 when 'include_hybrid' is true.")

        if args.dataset.add_core_atoms:
            args.model.lig_feat_dim += 2
            print("[W] The model parameter of 'lig_feat_dim' plus 2 when 'add_core_atoms' is true.")

        if args.dataset.include_valencies:
            args.lig_feat_dim += 1
            print("[W] The model parameter of 'lig_feat_dim' plus 1 when 'include_valencenes' is true.")

        if args.dataset.data_name in ['zinc_300', 'pdbbind']:
            args.model.phore_feat_dim += 2
            print(f"[W] The model parameter of 'phore_feat_dim' plus 2 when 'data_name' is '{args.dataset.data_name}'.")

        if args.dataset.include_ring:
            args.model.lig_feat_dim += 2
            print(f"[W] The model parameter of 'lig_feat_dim' plus 2 when 'include_ring' is true.")

        if args.dataset.include_aromatic:
            args.model.lig_feat_dim += 2
            print(f"[W] The model parameter of 'lig_feat_dim' plus 2 when 'include_aromatic' is true.")
    
        if args.dataset.include_neib_dist:
            args.model.lig_feat_dim += 2
            print(f"[W] The model parameter of 'lig_feat_dim' plus 2 when 'include_neib_dist' is true.")
        return args
         
            
    def find_valid_path(self, path):
        if os.path.exists(path):
            path += "_BACK"
            return self.find_valid_path(path)
        else:
            return path


    def init_status(self, args):
        # Status Variables
        self.history = {'train': [], 
                        'test': {'best': {}, 'last': {}}, 
                        'best': {'epoch': 0, 'valid_loss': 1e10}}
        self.epoch = 0
        self.lr = args.train.optimizer.lr
        self.best = False
        self.epoch_iter = range(args.train.epochs)  # 2000
        self.model_conf = args.model
        self.model_conf_file = os.path.join(self.run_dir, 'model.conf')
        self.parameter_file = os.path.join(self.run_dir, 'parameters.yml')

        # Cache Files
        self.hist_file = os.path.join(self.run_dir, 'history.log')
        self.model_pt = {'best': os.path.join(self.run_dir, 'best_model.pt'),
                         'last': os.path.join(self.run_dir, 'last_model.pt')}
        
        self.prev = {}


    def init_writer(self):
        if self.tensorboard:
            train_path = os.path.join(self.run_dir, 'train')
            valid_path = os.path.join(self.run_dir, 'valid')
            self.train_writer = SummaryWriter(train_path)
            self.valid_writer = SummaryWriter(valid_path)
        else:
            self.train_writer = None
            self.valid_writer = None


    def load_previous_status(self, args, prev_dir, run_dir, model_ckp):
        """
        Load previous status
        Args:
            prev_dir, path to the restart directory
            run_dir, path to current running directory
            model_ckp, 'last' or 'best' to specify the model to reload.
        """
        prev_model_pt = os.path.join(prev_dir, f'{model_ckp}_model.pt')
        self.prev['model_state_dict'] = torch.load(prev_model_pt, map_location=args.train.device) \
                if os.path.exists(prev_model_pt) else None
        if self.restart == 'finetuning':
            if self.prev['model_state_dict'] is None:
                raise ValueError(f'Invalid model checkpoint in `{prev_model_pt}`')
            else:
                print(f'[I] Loading previous model checkpoint from `{prev_model_pt}`')
        else:
            prev_hist_file = os.path.join(prev_dir, 'history.log')
            self.prev['history'] = json.load(open(prev_hist_file, 'r')) \
                if os.path.exists(prev_hist_file) else None
            prev_conf_file = os.path.join(prev_dir, 'parameters.yml')
            self.prev['args'] = EasyDict(**yaml.load(open(prev_conf_file, 'r'), Loader=yaml.FullLoader)) \
                if os.path.exists(prev_conf_file) else None
            prev_model_conf_file = os.path.join(prev_dir, 'model.conf')
            self.prev['model_conf'] = EasyDict(**json.load(open(prev_model_conf_file, 'r'))) \
                if os.path.exists(prev_model_conf_file) else None
            prev_model_pt = os.path.join(prev_dir, f'{model_ckp}_model.pt')
            self.prev['model_state_dict'] = torch.load(prev_model_pt, map_location=args.train.device) \
                if os.path.exists(prev_model_pt) else None

            if None in self.prev.values():
                print("[W] Invalid restart directory, run as usual.")
                if self.restart == 'backup':
                    os.system(f"rm -rf {run_dir}")
                    # shutil.rmtree(run_dir)
                self.prev.clear()
            else:
                self.model_conf = copy.deepcopy(self.prev['model_conf'])
                if self.restart in ['backup', 'inplace']:
                    prev_epoch = min(self.prev['model_state_dict']['epoch'], self.prev['history']['train'][-1]['epoch'])
                    if prev_epoch >= self.prev['args'].train.epochs:
                        print("[W] The run directory to restart from has already been finished.")
                        exit(0)
                    print(f"[I] Restarting from the {prev_epoch}th epoch of previous")
                    self.epoch = prev_epoch
                    self.epoch_iter = range(prev_epoch, self.prev['args'].train.epochs)
                    self.history = copy.deepcopy(self.prev['history'])
                    self.history['train'] = self.history['train'][:prev_epoch]
                    args.update(copy.deepcopy(self.prev['args']))
                    self.model_conf = copy.deepcopy(args.model)


    def add_record(self, record, mode=''):
        if mode not in self.record:
            self.record[mode] = {} 
        for k, v in record.items():
            if k not in self.record[mode]:
                self.record[mode][k] = [v]
            else:
                self.record[mode][k].append(v)


    def save_status(self, model, optimizer, scheduler, ema):
        _record = {}
        for mode, rec in self.record.items():
            for k, v in rec.items():
                _record[f"{mode}_{k}"] = v
        _record['lr'] = self.lr
        _record['epoch'] = self.epoch
        self.history['train'].append(copy.deepcopy(_record))

        self.dump_history()
        torch.save({'model': model.state_dict() if not hasattr(model, 'module') else model.module.state_dict(),
                    'optimizer': optimizer.state_dict() if optimizer is not None else None,
                    'scheduler': scheduler.state_dict() if scheduler is not None else None,
                    'ema': ema.state_dict() if ema is not None else None,
                    'epoch': self.epoch, 'config': self.args}, 
                   self.model_pt['last'])
        if self.best:
            torch.save({'model': model.state_dict() if not hasattr(model, 'module') else model.module.state_dict(),
                        'optimizer': optimizer.state_dict() if optimizer is not None else None,
                        'scheduler': scheduler.state_dict() if scheduler is not None else None,
                        'ema': ema.state_dict() if ema is not None else None,
                        'epoch': self.epoch, 'config': self.args}, 
                       self.model_pt['best'])

        if self.args.dataset.data_name != 'pdbbind' and self.epoch in [160, 250]:
            copy_model = os.path.join(self.run_dir, f"best_model_epoch{self.history['best']['epoch']}.pt")
            os.system(f"cp {self.model_pt['best']} {copy_model}")


    def add_new_epoch(self, epoch):
        self.epoch = epoch
        self.clear_record()
        self.best = False


    def clear_record(self):
        self.record = {}


    def summarize_a_epoch(self, mode):
        self.record[mode] = {k: sum(v)/len(v) if k not in ['m_true', 'num_mols', 'c_true'] else sum(v) \
                             for k, v in self.record[mode].items()}  # Get the mean values of loss
        if 'm_true' in self.record[mode] and 'c_true' in self.record[mode] and 'num_mols' in self.record[mode]:
            self.record[mode]['atomic_type_accuracy'] = self.record[mode]['m_true'] / self.record[mode]['num_mols']
            self.record[mode]['molecular_count_accuracy'] = self.record[mode]['c_true'] / self.record[mode]['num_mols']
        self.record[mode]['time_cost'] = self.ed_time - self.st_time
        if mode == 'valid' and self.record[mode]['loss'] <= self.history['best']['valid_loss']:
            self.history['best']['epoch'] = self.epoch
            self.history['best']['valid_loss'] = self.record[mode]['loss']
            self.best = True

        writer = None
        if mode == 'train':
            writer = self.train_writer
        elif mode == 'valid':
            writer = self.valid_writer

        if writer is not None:
            for k, v in self.record[mode].items():
                writer.add_scalar(k, v, self.epoch)
            if mode == 'train':
                writer.add_scalar('lr', self.lr, self.epoch)
        
        self.log(mode)


    def start(self):
        self.st_time = time.time()


    def end(self):
        self.ed_time = time.time()


    def log(self, mode):
        timestr = f"[{time.strftime('%Y/%m/%d-%H:%M:%S')}]"
        record_str = ", ".join([f"{k}: {v:.6f}" for k, v in self.record[mode].items()])
        print(f">> {timestr} Epoch {self.epoch} {mode.upper()}: {record_str}")


    def dump_history(self):
        json.dump(self.history, open(self.hist_file, 'w'), indent=4)


    def save_test_result(self):
        self.history['test']['best'] = copy.deepcopy(self.record['best'])
        self.history['test']['last'] = copy.deepcopy(self.record['last'])
        self.dump_history()


class LoggerDdp():
    def __init__(self, args) -> None:
        args = self.prepare_args(args)
        seed_all(args.train.seed)
        self.run_dir = os.path.join(args.logger.result, args.logger.run_name)
        self.run_name = args.logger.run_name
        self.result = args.logger.result
        self.restart = args.logger.restart
        self.tensorboard = args.logger.tensorboard
        self.dataset_args = []
        self.init_status(args)
        if dist.get_rank() == 0:
            self.check_existence(args, self.restart, self.run_dir, 
                                restart_dir=args.logger.restart_dir, model_ckp=args.logger.model_ckp)
        self.init_writer()
        self.args = args
        

    def check_existence(self, args, restart, run_dir, restart_dir=None, model_ckp='last'):
        """
        Check the running status
        Args: 
            restart: ['restart', 'inplace', 'backup', 'finetuning', 'none', 'overwrite']
        """
        prev_dir = None
        exist_flag = os.path.exists(run_dir)
        if exist_flag:
            if restart == 'none':
                raise RuntimeError(f"[E] The run directory `{run_dir}` exists already."+\
                                   " Please use an another run name instead.")
            elif restart == 'overwrite':
                os.system(f"rm -rf {run_dir}")
                # shutil.rmtree(run_dir)
                print("[W] Restart mode set as `overwrite`, removing the existing directory.")
            else:
                if restart == 'backup':
                    prev_dir = run_dir

            if restart_dir is not None:
                assert os.path.abspath(run_dir) != os.path.abspath(restart_dir), \
                    f"[E] Same run directory with the restart directory is not allowed. Please change the run name."+\
                        f"\n> Run directory: {run_dir}\n> Restart directory: {restart_dir}"
        # Loading previous status if available
        prev_dir = restart_dir if restart_dir is not None else prev_dir
        if prev_dir is not None:
            self.load_previous_status(args, prev_dir, run_dir, model_ckp)
        # Dumping model args and overall parameters.
        os.makedirs(run_dir, exist_ok=True)
        save_yaml_file(self.parameter_file, easydict_to_dict(args))
        json.dump(self.model_conf.__dict__, open(self.model_conf_file, 'w'), indent=4)


    def prepare_args(self, args):
        if args.config:
            # print(f"[I] Reading parameters from config file `{args.config}` which is privileged.")
            args = EasyDict(yaml.safe_load(open(args.config, 'r')))
            # args.__dict__ = conf_args
        else:
            raise RuntimeError("[E] No config file is specified.")

        if args.train.device == 'cuda' and not torch.cuda.is_available():
            args.train.device = 'cpu'
            print('[W] The model is specified to train on `CUDA` but no devices available. Switch to `CPU`.')

        # if args.dataset.charge_weight:
        #     args.lig_feat_dim += 1
        #     print("[W] The model parameter of 'lig_feat_dim' plus 1 when 'charge_weight' is not 0.")

        # if args.dataset.include_hybrid:
        #     if args.dataset.hybrid_one_hot:
        #         args.model.lig_feat_dim += 4
        #         print("[W] The model parameter of 'lig_feat_dim' plus 4 when 'include_hybrid' is true and encode hybrid into one-hot.")
        #     else:
        #         args.model.lig_feat_dim += 1
        #         print("[W] The model parameter of 'lig_feat_dim' plus 1 when 'include_hybrid' is true.")

        # if args.dataset.add_core_atoms:
        #     args.model.lig_feat_dim += 2
        #     print("[W] The model parameter of 'lig_feat_dim' plus 2 when 'add_core_atoms' is true.")

        # if args.dataset.include_valencies:
        #     args.lig_feat_dim += 1
        #     print("[W] The model parameter of 'lig_feat_dim' plus 1 when 'include_valencenes' is true.")

        if args.dataset.data_name in ['zinc_300', 'pdbbind']:
            args.model.phore_feat_dim += 2
            if dist.get_rank() == 0:            
                print(f"[W] The model parameter of 'phore_feat_dim' plus 2 when 'data_name' is '{args.dataset.data_name}'.")

        # if args.dataset.include_ring:
        #     args.model.lig_feat_dim += 2
        #     print(f"[W] The model parameter of 'lig_feat_dim' plus 2 when 'include_ring' is true.")

        # if args.dataset.include_aromatic:
        #     args.model.lig_feat_dim += 2
        #     print(f"[W] The model parameter of 'lig_feat_dim' plus 2 when 'include_aromatic' is true.")
    
        # if args.dataset.include_neib_dist:
        #     args.model.lig_feat_dim += 2
        #     print(f"[W] The model parameter of 'lig_feat_dim' plus 2 when 'include_neib_dist' is true.")
        return args
         
            
    def find_valid_path(self, path):
        if os.path.exists(path):
            path += "_BACK"
            return self.find_valid_path(path)
        else:
            return path


    def init_status(self, args):
        # Status Variables
        self.history = {'train': [], 
                        'test': {'best': {}, 'last': {}}, 
                        'best': {'epoch': 0, 'valid_loss': 1e10}}
        self.epoch = 0
        self.lr = args.train.optimizer.lr
        self.best = False
        self.epoch_iter = range(args.train.epochs)  # 2000
        self.model_conf = args.model
        self.model_conf_file = os.path.join(self.run_dir, 'model.conf')
        self.parameter_file = os.path.join(self.run_dir, 'parameters.yml')

        # Cache Files
        self.hist_file = os.path.join(self.run_dir, 'history.log')
        self.model_pt = {'best': os.path.join(self.run_dir, 'best_model.pt'),
                         'last': os.path.join(self.run_dir, 'last_model.pt')}
        
        self.prev = {}


    def init_writer(self):
        if self.tensorboard and dist.get_rank() == 0:
            train_path = os.path.join(self.run_dir, 'train')
            valid_path = os.path.join(self.run_dir, 'valid')
            self.train_writer = SummaryWriter(train_path)
            self.valid_writer = SummaryWriter(valid_path)
        else:
            self.train_writer = None
            self.valid_writer = None


    def load_previous_status(self, args, prev_dir, run_dir, model_ckp):
        """
        Load previous status
        Args:
            prev_dir, path to the restart directory
            run_dir, path to current running directory
            model_ckp, 'last' or 'best' to specify the model to reload.
        """
        prev_model_pt = os.path.join(prev_dir, f'{model_ckp}_model.pt')
        self.prev['model_state_dict'] = torch.load(prev_model_pt, map_location=args.train.device) \
                if os.path.exists(prev_model_pt) else None
        if self.restart == 'finetuning':
            if self.prev['model_state_dict'] is None:
                raise ValueError(f'Invalid model checkpoint in `{prev_model_pt}`')
            else:
                print(f'[I] Loading previous model checkpoint from `{prev_model_pt}`')
        else:
            prev_hist_file = os.path.join(prev_dir, 'history.log')
            self.prev['history'] = json.load(open(prev_hist_file, 'r')) \
                if os.path.exists(prev_hist_file) else None
            prev_conf_file = os.path.join(prev_dir, 'parameters.yml')
            self.prev['args'] = EasyDict(**yaml.load(open(prev_conf_file, 'r'), Loader=yaml.FullLoader)) \
                if os.path.exists(prev_conf_file) else None
            prev_model_conf_file = os.path.join(prev_dir, 'model.conf')
            self.prev['model_conf'] = EasyDict(**json.load(open(prev_model_conf_file, 'r'))) \
                if os.path.exists(prev_model_conf_file) else None
            prev_model_pt = os.path.join(prev_dir, f'{model_ckp}_model.pt')
            self.prev['model_state_dict'] = torch.load(prev_model_pt, map_location=args.train.device) \
                if os.path.exists(prev_model_pt) else None

            if None in self.prev.values():
                print("[W] Invalid restart directory, run as usual.")
                if self.restart == 'backup':
                    # os.system(f"rm -rf {run_dir}")
                    shutil.rmtree(run_dir)
                self.prev.clear()
            else:
                self.model_conf = copy.deepcopy(self.prev['model_conf'])
                if self.restart in ['backup', 'inplace']:
                    prev_epoch = min(self.prev['model_state_dict']['epoch'], self.prev['history']['train'][-1]['epoch'])
                    if prev_epoch >= self.prev['args'].train.epochs:
                        print("[W] The run directory to restart from has already been finished.")
                        exit(0)
                    print(f"[I] Restarting from the {prev_epoch}th epoch of previous")
                    self.epoch = prev_epoch
                    self.epoch_iter = range(prev_epoch, self.prev['args'].train.epochs)
                    self.history = copy.deepcopy(self.prev['history'])
                    self.history['train'] = self.history['train'][:prev_epoch]
                    args.update(copy.deepcopy(self.prev['args']))
                    self.model_conf = copy.deepcopy(args.model)


    def add_record(self, record, mode=''):
        if mode not in self.record:
            self.record[mode] = {} 
        for k, v in record.items():
            if k not in self.record[mode]:
                self.record[mode][k] = [v]
            else:
                self.record[mode][k].append(v)


    def save_status(self, model, optimizer, scheduler, ema):
        _record = {}
        for mode, rec in self.record.items():
            for k, v in rec.items():
                _record[f"{mode}_{k}"] = v
        _record['lr'] = self.lr
        _record['epoch'] = self.epoch
        self.history['train'].append(copy.deepcopy(_record))

        self.dump_history()
        torch.save({'model': model.state_dict() if not hasattr(model, 'module') else model.module.state_dict(),
                    'optimizer': optimizer.state_dict() if optimizer is not None else None,
                    'scheduler': scheduler.state_dict() if scheduler is not None else None,
                    'ema': ema.state_dict() if ema is not None else None,
                    'epoch': self.epoch, 'config': self.args}, 
                   self.model_pt['last'])
        if self.best:
            torch.save({'model': model.state_dict() if not hasattr(model, 'module') else model.module.state_dict(),
                        'optimizer': optimizer.state_dict() if optimizer is not None else None,
                        'scheduler': scheduler.state_dict() if scheduler is not None else None,
                        'ema': ema.state_dict() if ema is not None else None,
                        'epoch': self.epoch, 'config': self.args}, 
                       self.model_pt['best'])

        if self.args.dataset.data_name != 'pdbbind' and self.epoch in [160, 250]:
            copy_model = os.path.join(self.run_dir, f"best_model_epoch{self.history['best']['epoch']}.pt")
            os.system(f"cp {self.model_pt['best']} {copy_model}")


    def add_new_epoch(self, epoch):
        self.epoch = epoch
        self.clear_record()
        self.best = False


    def clear_record(self):
        self.record = {}


    def summarize_a_epoch(self, mode):
        self.record[mode] = {k: sum(v)/len(v) if k not in ['m_true', 'num_mols', 'c_true'] else sum(v) \
                             for k, v in self.record[mode].items()}  # Get the mean values of loss
        if 'm_true' in self.record[mode] and 'c_true' in self.record[mode] and 'num_mols' in self.record[mode]:
            self.record[mode]['atomic_type_accuracy'] = self.record[mode]['m_true'] / self.record[mode]['num_mols']
            self.record[mode]['molecular_count_accuracy'] = self.record[mode]['c_true'] / self.record[mode]['num_mols']
        self.record[mode]['time_cost'] = self.ed_time - self.st_time
        if mode == 'valid' and self.record[mode]['loss'] <= self.history['best']['valid_loss']:
            self.history['best']['epoch'] = self.epoch
            self.history['best']['valid_loss'] = self.record[mode]['loss']
            self.best = True

        writer = None
        if mode == 'train':
            writer = self.train_writer
        elif mode == 'valid':
            writer = self.valid_writer

        if writer is not None:
            for k, v in self.record[mode].items():
                writer.add_scalar(k, v, self.epoch)
            if mode == 'train':
                writer.add_scalar('lr', self.lr, self.epoch)
        
        if dist.get_rank() == 0:
            self.log(mode)


    def start(self):
        self.st_time = time.time()


    def end(self):
        self.ed_time = time.time()


    def log(self, mode):
        timestr = f"[{time.strftime('%Y/%m/%d-%H:%M:%S')}]"
        record_str = ", ".join([f"{k}: {v:.6f}" for k, v in self.record[mode].items()])
        print(f">> {timestr} Epoch {self.epoch} {mode.upper()}: {record_str}")


    def dump_history(self):
        json.dump(self.history, open(self.hist_file, 'w'), indent=4)


    def save_test_result(self):
        self.history['test']['best'] = copy.deepcopy(self.record['best'])
        self.history['test']['last'] = copy.deepcopy(self.record['last'])
        self.dump_history()
