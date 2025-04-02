import os
import torch
from torch_geometric.loader import DataLoader, DataListLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from models.diffusion import PhoreDiff
from models.model_utils import EMA
from run.logger import Logger, LoggerDdp
from utils.training_utils import Queue, gradient_clipping, print_parm_log, get_parameter_number, \
                                get_optimizer, get_scheduler, get_transform, get_dataset, freeze_parameters


class Run():
    def __init__(self, *args, **kwargs) -> None:
        self.parallel = False


    def train(self, args):
        self.logger = Logger(args)
        args = self.logger.args
        print(args)
        model, optimizer, scheduler, ema = self.init_model(args)
        # model = model.to(args.train.device)
        loaders = self.init_dataloader(args)
        gradnorm_queue = Queue()
        gradnorm_queue.add(3000)
        print_parm_log(args.train)

        for epoch in self.logger.epoch_iter:  # 2000
            # train
            self.logger.add_new_epoch(epoch)

            self.run_on_epoch(args.train, model, loaders['train'], self.logger, optimizer, ema, 
                              gradnorm_queue=gradnorm_queue,  mode='train')
            self.run_on_epoch(args.train, model, loaders['valid'], self.logger, optimizer, ema, mode='valid')
            
            self.logger.save_status(model, optimizer, scheduler, ema)
            if scheduler is not None:
                scheduler.step(self.logger.history['train'][-1]['valid_loss'])

        return self.logger.history


    def init_model(self, args):
        print(f"[I] Model initializing...")
        model_conf = self.logger.model_conf
        if model_conf.name == 'diffusion':
            model = PhoreDiff(model_conf, args.dataset.data_name).to(args.train.device)
        else:
            raise NotImplementedError
        
        if args.dataset.data_name == 'pdbbind' and args.dataset.checkpoint is not None and os.path.exists(args.dataset.checkpoint):
            model.load_state_dict(torch.load(args.dataset.checkpoint, map_location=args.train.device)['model'])
            print(f"[I] Loading pretrained zinc checkpoint from: {args.dataset.checkpoint}")
        
        print(f'** Model initialized with {get_parameter_number(model)} parameters **')
        optimizer = get_optimizer(model, args.train.optimizer)
        scheduler = get_scheduler(optimizer, args.train.scheduler)
        ema = EMA(parameters=model.parameters(), beta=args.train.ema_decay) if args.train.ema else None

        if len(self.logger.prev) != 0 and self.logger.restart in ['backup', 'inplace']:
            prev_state = self.logger.prev['model_state_dict']
            model.load_state_dict(prev_state['model'])
            if 'optimizer' in prev_state:
                optimizer.load_state_dict(prev_state['optimizer'])
            if 'scheduler' in prev_state and scheduler is not None:
                scheduler.load_state_dict(prev_state['scheduler'])
            if 'ema' in prev_state is not None:
                ema.load_state_dict(prev_state['ema'], device=args.train.device)

        if args.train.parallel and torch.cuda.device_count() > 1 and args.train.device=='cuda':
            model = torch.nn.DataParallel(model)
            self.parallel = True
            print("[I] Using multiple GPUs.")

        return model, optimizer, scheduler, ema


    def init_dataloader(self, args):
        print(f"[I] Loading dataset...")
        loaders = {}
        transform = get_transform(args.train)
        train_set, valid_set, test_set = get_dataset(args.dataset, transform=transform)
        print(f"** '{args.dataset.data_name}' dataset loaded with [Train: {len(train_set)}, Valid: {len(valid_set)}, Test: {len(test_set)}] **")

        if self.parallel:
            loaders['train'] = DataListLoader(train_set, batch_size=args.train.batch_size, shuffle=True, 
                                              num_workers=args.train.num_workers, follow_batch=['f_edge_attr'])
            loaders['valid'] = DataListLoader(valid_set, batch_size=args.train.batch_size, shuffle=True, 
                                              num_workers=args.train.num_workers, follow_batch=['f_edge_attr'])
            loaders['test'] = DataListLoader(test_set, batch_size=args.train.batch_size, shuffle=True, 
                                             num_workers=args.train.num_workers, follow_batch=['f_edge_attr']) if test_set is not None else None
        else:
            loaders['train'] = DataLoader(train_set, batch_size=args.train.batch_size, shuffle=True, 
                                          num_workers=args.train.num_workers, follow_batch=['f_edge_attr'])
            loaders['valid'] = DataLoader(valid_set, batch_size=args.train.batch_size, shuffle=True, 
                                          num_workers=args.train.num_workers, follow_batch=['f_edge_attr'])
            loaders['test'] = DataLoader(test_set, batch_size=args.train.batch_size, shuffle=True, 
                                         num_workers=args.train.num_workers, follow_batch=['f_edge_attr']) if test_set is not None else None

        return loaders


    def run_on_epoch(self, args, model: PhoreDiff, dataloader, logger: Logger, optimizer=None, ema=None, 
                     gradnorm_queue=None, mode='train'):
        if mode == 'train':
            model = model.train()
            logger.lr = optimizer.param_groups[0]['lr']
        else:
            model = model.eval()
        with torch.set_grad_enabled(mode=='train'):
            n_batch = len(dataloader)
            logger.start()
            for idx, data in enumerate(dataloader):
                if mode == 'train': 
                    optimizer.zero_grad()
                data = data.to(args.device)
                if args.add_lig_noise:
                    # Add noise eps ~ N(0, lig_noise_std) around points.
                    data['ligand'].pos += torch.randn_like(data['ligand'].pos) * args.lig_noise_std
                try:
                    # pred = model(data)
                    loss, loss_record = model.compute_loss(data)
                    logger.add_record(loss_record, mode=mode)
                    if mode == 'train':
                        loss.backward()
                        if gradnorm_queue is not None:
                            if args.clip_grad:
                                if args.clip_grad_mode == 'queue':
                                    grad_norm = gradient_clipping(model, gradnorm_queue)
                                else:
                                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                            else:
                                grad_norm = 0.
                        optimizer.step()
                        if args.ema_decay < 0 and ema is not None:
                            ema.update_model_average(model)
                        if idx != 0 and idx % args.n_report_steps == 0:
                            print(f"\rEpoch: {logger.epoch}, Batch: {idx}/{n_batch}, "
                                f"Loss {loss.item():.2f}"
                                f", GradNorm: {grad_norm:.1f}" if gradnorm_queue is not None else "" )
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print(f'| WARNING: ran out of memory, skipping batch: {idx}({data.name}) | Message: {e}')
                        for p in model.parameters():
                            if p.grad is not None:
                                del p.grad  # free some memorry
                        torch.cuda.empty_cache()
                        continue
                    else:
                        print(f'[E] Failed to calculate the batch: {idx}({data.name})')
                        raise e
            logger.end()
            logger.summarize_a_epoch(mode)



class RunDdp():
    def __init__(self, *args, **kwargs) -> None:
        self.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(self.local_rank)  # set device
        dist.init_process_group(backend='nccl', init_method='env://')  # nccl backend (commonly used)
        # device = torch.device("cuda", self.local_rank)  # Transfer to the device

        # limit # of CPU threads to be used per worker.
        torch.set_num_threads(9)


    def train(self, args):
        self.logger = LoggerDdp(args)
        args = self.logger.args
        if dist.get_rank() == 0:
            print(args)
        model, optimizer, scheduler, ema = self.init_model(args)
        # model = model.to(args.train.device)
        loaders = self.init_dataloader(args)
        gradnorm_queue = Queue()
        gradnorm_queue.add(3000)
        if dist.get_rank() == 0:
            print_parm_log(args.train)

        for epoch in self.logger.epoch_iter:  # 2000
            # train
            loaders['train'].sampler.set_epoch(epoch)
            self.logger.add_new_epoch(epoch)

            self.run_on_epoch(args.train, model, loaders['train'], self.logger, optimizer, ema, 
                              gradnorm_queue=gradnorm_queue,  mode='train')
            if dist.get_rank() == 0:
                self.run_on_epoch(args.train, model, loaders['valid'], self.logger, optimizer, ema, mode='valid')
                
                self.logger.save_status(model, optimizer, scheduler, ema)
                if scheduler is not None:
                    scheduler.step(self.logger.history['train'][-1]['valid_loss'])
        
        dist.destroy_process_group()
        return self.logger.history


    def init_model(self, args):
        if dist.get_rank() == 0:
            print(f"[I] Model initializing...")
        model_conf = self.logger.model_conf
        if model_conf.name == 'diffusion':
            model = PhoreDiff(model_conf, args.dataset.data_name).to(self.local_rank)
        else:
            raise NotImplementedError
        
        if args.dataset.data_name == 'pdbbind' and args.dataset.checkpoint is not None and os.path.exists(args.dataset.checkpoint):
            model.load_state_dict(torch.load(args.dataset.checkpoint, map_location=args.train.device)['model'])
            if dist.get_rank() == 0:
                print(f"[I] Loading pretrained zinc checkpoint from: {args.dataset.checkpoint}")
        
        model = freeze_parameters(model, args.train)

        if dist.get_rank() == 0:
            print(f'** Model initialized with {get_parameter_number(model)} parameters **')
        optimizer = get_optimizer(model, args.train.optimizer, getattr(args.train, 'freeze_pos', False))
        scheduler = get_scheduler(optimizer, args.train.scheduler)
        ema = EMA(parameters=model.parameters(), beta=args.train.ema_decay) if args.train.ema else None

        if len(self.logger.prev) != 0 and self.logger.restart in ['backup', 'inplace']:
            prev_state = self.logger.prev['model_state_dict']
            model.load_state_dict(prev_state['model'])
            if 'optimizer' in prev_state:
                optimizer.load_state_dict(prev_state['optimizer'])
            if 'scheduler' in prev_state and scheduler is not None:
                scheduler.load_state_dict(prev_state['scheduler'])
            if 'ema' in prev_state is not None:
                ema.load_state_dict(prev_state['ema'], device=args.train.device)

        model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)

        return model, optimizer, scheduler, ema


    def init_dataloader(self, args):
        if dist.get_rank() == 0:
            print(f"[I] Loading dataset...")
        loaders = {}
        transform = get_transform(args.train)
        train_set, valid_set, test_set = get_dataset(args.dataset, transform=transform)
        if dist.get_rank() == 0:
            print(f"** '{args.dataset.data_name}' dataset loaded with [Train: {len(train_set)}, Valid: {len(valid_set)}, Test: {len(test_set)}] **")

        train_sampler = DistributedSampler(train_set, shuffle=True)
        valid_sampler = DistributedSampler(valid_set, shuffle=True)

        loaders['train'] = DataLoader(train_set, batch_size=args.train.batch_size, num_workers=args.train.num_workers, 
                                      follow_batch=['f_edge_attr'], sampler=train_sampler)
        loaders['valid'] = DataLoader(valid_set, batch_size=args.train.batch_size, num_workers=args.train.num_workers, 
                                      follow_batch=['f_edge_attr'], sampler=valid_sampler)
        loaders['test'] = DataLoader(test_set, batch_size=args.train.batch_size, shuffle=True, 
                                    num_workers=args.train.num_workers, follow_batch=['f_edge_attr']) if test_set is not None else None
        return loaders



    def run_on_epoch(self, args, model, dataloader, logger: Logger, optimizer=None, ema=None, 
                     gradnorm_queue=None, mode='train'):
        if mode == 'train':
            model = model.train()
            logger.lr = optimizer.param_groups[0]['lr']
        else:
            model = model.eval()
        with torch.set_grad_enabled(mode=='train'):
            n_batch = len(dataloader)
            logger.start()
            for idx, data in enumerate(dataloader):
                if mode == 'train': 
                    optimizer.zero_grad()
                data = data.to(self.local_rank)
                if args.add_lig_noise:
                    # Add noise eps ~ N(0, lig_noise_std) around points.
                    data['ligand'].pos += torch.randn_like(data['ligand'].pos) * args.lig_noise_std
                try:
                    # pred = model(data)
                    loss, loss_record = model.module.compute_loss(data)
                    logger.add_record(loss_record, mode=mode)
                    if mode == 'train':
                        loss.backward()
                        if gradnorm_queue is not None:
                            if args.clip_grad:
                                if args.clip_grad_mode == 'queue':
                                    grad_norm = gradient_clipping(model, gradnorm_queue)
                                else:
                                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                            else:
                                grad_norm = 0.
                        optimizer.step()
                        if args.ema_decay < 0 and ema is not None:
                            ema.update_model_average(model)
                        if dist.get_rank() == 0 and idx != 0 and idx % args.n_report_steps == 0:
                            print(f"\rEpoch: {logger.epoch}, Batch: {idx}/{n_batch}, "
                                f"Loss {loss.item():.2f}"
                                f", GradNorm: {grad_norm:.1f}" if gradnorm_queue is not None else "" )
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print(f'| WARNING: ran out of memory, skipping batch: {idx}({data.name}) | Message: {e}')
                        for p in model.parameters():
                            if p.grad is not None:
                                del p.grad  # free some memorry
                        torch.cuda.empty_cache()
                        continue
                    else:
                        print(f'[E] Failed to calculate the batch: {idx}({data.name})')
                        raise e
            logger.end()
            logger.summarize_a_epoch(mode)

