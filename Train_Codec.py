import torch
import os
import sys

import time
import datetime
from modules.torch_utils import fix_seed, seed_worker
from tqdm import tqdm
import yaml

from torch.utils.tensorboard import SummaryWriter
from importlib import import_module
import argparse
from TempDataset.Temp_Dataset import TempDataset
from Eval import eval_image
from contextlib import nullcontext

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from collections import defaultdict
from typing import Optional


def main(model_name: str, exp_name: str, train_config_name: str, data_path: str, save_path: str) -> None:
    """
    Main function for training an image compression model.

    Args:
        model_name (str): The name of the compression model, corresponding to the model config file in './config/model'.
        exp_name (str): The postfix for saving the experiment.
        train_config_name (str): The name of the training configuration, corresponding to the files in './config/train'.
        data_path (str): The path to the data.
        save_path (str): The directory where training results will be saved.

    Returns:
        None
    """
    USE_CUDA = torch.cuda.is_available()

    # Check the number of GPUs for training
    num_gpus = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    use_ddp = True if num_gpus > 1 else False

    rank = 0 if not use_ddp else None

    if use_ddp:
        dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=9000))
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
        world_size = dist.get_world_size()
        print(f'World size: {world_size}') if rank == 0 else None

    device = torch.cuda.current_device() if USE_CUDA else 'cpu'
    print(f'Device: {device} is used\n')

    model_exp_name = f'{model_name}_{exp_name}' if exp_name != "" else model_name

    ''' Set logging dir '''
    _base_log_dir = os.path.join(save_path, 'Eval_log', '{}', model_exp_name)
    tensorboard_dir = os.path.join(save_path, 'Train_record', model_exp_name, "tensorboard")

    ''' Get train configure '''
    train_conf_file = f'./config/train/{train_config_name}.yaml'
    with open(train_conf_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        args = argparse.Namespace(**config)
        args.optim = config['optimizer']
        if rank == 0:
            print(vars(args))

    ''' Fix random seed'''
    fix_seed(args.seed)

    ''' Tensorboard '''
    writer = SummaryWriter(tensorboard_dir)
    print(f"\nSave dir: {os.path.join(save_path, 'Train_record', model_exp_name)}\n") if rank == 0 else None

    ''' Get model '''
    model_conf_file = f'./config/model/{model_name}.yaml'
    with open(model_conf_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        module_name = config['model']['module_name']
        class_name = config['model']['args']['class']

    model = getattr(import_module(f'modules.{module_name}.models'), class_name)(model_conf_file, device)
    if rank == 0:
        print(f"Model '{class_name}' with configure file '{model_name}' is loaded")
        print(f"Loaded model details: {config}\n")

    training_consumed_sec = 0

    ''' Get dataloader '''
    train_dataset = TempDataset(data_path, 'temp_train', is_train=True, input_resolution=args.input_resolution)

    ''' Create DistributedSampler '''
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if use_ddp else None

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,
                                                   num_workers=args.num_workers, pin_memory=False, drop_last=True,
                                                   worker_init_fn=seed_worker, shuffle=(sampler is None))

    # Get Test Dataloader (Temp)
    test_dataset = TempDataset(data_path, 'temp_test', is_train=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1,
                                                   pin_memory=False, drop_last=False)

    ''' Optimizer '''
    optimizer = getattr(import_module(f"modules.{module_name}.optimizers"), 'CustomOptimizer')(model, train_conf_file)

    ''' Make distributed data parallel module '''
    model = DistributedDataParallel(model, device_ids=[device], output_device=device) if use_ddp else model
    module = model.module if isinstance(model, DistributedDataParallel) else model

    # postfix = ""
    # model_dir = os.path.join(save_path, 'Train_record', model_exp_name, f"Param{postfix}.pth")
    # module.load(model_dir)

    ''' Train Loop '''
    for epoch in range(args.epoch):
        module.train(True)
        postfix = f'_{epoch}' if epoch is not None else ""

        _base_log_dir = os.path.join(save_path, 'Eval_log', '{}', model_exp_name)
        tensorboard_dir = os.path.join(save_path, 'Train_record', model_exp_name, "tensorboard")

        total_loss_per_epoch = 0.0
        loss_add_count = 0.0
        loss_per_epoch_dict = defaultdict(float)

        if rank == 0:
            train_start_time_per_epoch = time.time()

        pbar = tqdm(train_dataloader, desc=f"Train Epoch {epoch}...", disable=(rank != 0))
        sampler.set_epoch(epoch) if use_ddp else None
        for step, data in enumerate(pbar):
            images, ids = data['images'].to(module.device), data['ids']

            out = module(images)

            loss, aux_loss = out['loss'], out['aux_loss']

            if rank == 0 and (torch.isnan(loss + aux_loss) or torch.isinf(loss + aux_loss)):
                # Stop if loss is nan
                print('************Training stopped due to inf/nan loss.************')
                sys.exit(-1)

            for loss_name, loss_value in out['loss_dict'].items():
                loss_per_epoch_dict[loss_name] += loss_value.item()

            total_loss_per_epoch += loss.item()
            loss_add_count += 1.0

            optimizer.zero_grad()
            optimizer.step(loss, aux_loss)
            module.entropy_bottleneck.update()

            avr_loss = total_loss_per_epoch / loss_add_count  # Aux loss  is not included
            if rank == 0:
                pbar.set_description(f"Training Epoch {epoch}, Loss = {round(avr_loss, 5)}")

        dist.barrier() if use_ddp else None

        if rank == 0:
            loss_per_epoch_dict = dict(
                (loss_name, loss / loss_add_count) for loss_name, loss in loss_per_epoch_dict.items())
            training_consumed_sec += (time.time() - train_start_time_per_epoch)

            loss_keys = list(loss_per_epoch_dict.keys())
            main_loss_per_epoch_dict = {k: loss_per_epoch_dict[k] for k in loss_keys if k != 'aux_loss'}
            aux_loss_per_epoch_dict = {'aux_loss': loss_per_epoch_dict['aux_loss']} if 'aux_loss' in loss_keys else None

            writer.add_scalars('train/overall', {'loss': total_loss_per_epoch / loss_add_count}, epoch)
            writer.add_scalars('train/main_loss', main_loss_per_epoch_dict, epoch)
            writer.add_scalars('train/aux_loss', aux_loss_per_epoch_dict, epoch) if 'aux_loss' in loss_keys else None

            for i, param in enumerate(optimizer.main_optimizer.param_groups):
                writer.add_scalars('train/lr', {f'param{i}': param['lr']}, epoch)

            ''' Evaluate '''
            module.train(False)

            with torch.no_grad():
                eval_dir = _base_log_dir.format('temp')
                for quality_level in range(1, 9):
                    # save_flag = True if quality_level == 8 else False
                    save_flag = False
                    eval_image(module, test_dataloader, eval_dir, epoch, quality_level, tensorboard_dir, save_flag)

            save_dir = os.path.join(save_path, 'Train_record', model_exp_name, f"Param{postfix}"+".pth")
            torch.save(module.state_dict(), save_dir)
            module.train(True)

    writer.close()

    if rank == 0:
        result_list = str(datetime.timedelta(seconds=training_consumed_sec)).split(".")
        print("Training time :", result_list[0])

    dist.destroy_process_group() if use_ddp else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument('--model_name', type=str, default='SCR', help='Use model config file name')
    parser.add_argument('--exp_name', type=str, default='', help='postfix for save experiment')
    parser.add_argument('--train_config', type=str, default='Custom_v1', help='Use train config file name')
    parser.add_argument('--data_path', type=str, default='', help='Dataset directory')
    parser.add_argument('--save_path', type=str, default='', help='Checkpoints directory')

    args = parser.parse_args()

    # Run example
    main(args.model_name, args.exp_name, args.train_config, args.data_path, args.save_path)
