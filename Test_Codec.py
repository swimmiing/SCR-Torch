import torch
import yaml
import os
import argparse

from importlib import import_module
from TempDataset.Temp_Dataset import TempDataset
from Eval import eval_image
from modules.arg_utils import int_or_int_list_or_none, int_or_float
from typing import Union, List, Any


@torch.no_grad()
def main(
        model_name: str,
        exp_name: str,
        epochs: Union[int, List[Union[int, None]]],
        quality_level: float,
        data_path: Union[str, dict],
        save_path: str) -> None:
    """
    Main function for evaluating image compression models.

    Args:
        model_name (str): The name of the compression model, corresponding to the model config file in './config/model'.
        exp_name (str): The postfix for saving the experiment.
        epochs (Union[int, List[Union[int, None]]]): List of epochs to evaluate.
        quality_level (float): Quality level for adaptive compression. [1.0, 8.0]
        data_path (str): The directory for dataset.
        save_path (str): The directory for saving evaluation results.
    """
    USE_CUDA = torch.cuda.is_available()
    device = torch.device('cuda:0' if USE_CUDA else 'cpu')

    model_exp_name = f'{model_name}_{exp_name}' if exp_name != "" else model_name
    print(f"Experiment: {model_exp_name}")

    ''' Set logging dir '''
    _base_log_dir = os.path.join(save_path, 'Eval_log', '{}', model_exp_name)
    tensorboard_dir = os.path.join(save_path, 'Train_record', model_exp_name, "tensorboard")

    ''' Get model '''
    model_conf_file = f'./config/model/{model_name}.yaml'
    with open(model_conf_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        module_name = config['model']['module_name']
        class_name = config['model']['args']['class']

    model = getattr(import_module(f'modules.{module_name}.models'), class_name)(model_conf_file, device)
    model.to(device)

    for epoch in epochs:
        model.train(False)
        postfix = f'_{epoch}' if epoch is not None else ""
        model_dir = os.path.join(save_path, 'Train_record', model_exp_name, f"Param{postfix}.pth")
        model.load(model_dir)

        ''' Get dataloader '''
        test_dataset = TempDataset(data_path['temp'], 'temp_test', is_train=False)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1,
                                                         pin_memory=True, drop_last=False)
        # It doesn't have to be a batch size of 1.

        ''' Evaluate '''
        eval_dir = _base_log_dir.format('temp')
        eval_image(model, test_dataloader, eval_dir, epoch, quality_level, tensorboard_dir, True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='SCR', help='Use model config file name')
    parser.add_argument('--exp_name', type=str, default='released', help='postfix for save experiment')
    parser.add_argument('--epochs', type=int_or_int_list_or_none, default=[None], help='epochs ([None] for released)')
    parser.add_argument('--quality_level', type=float, default=8, help='Quality level')
    parser.add_argument('--data_path', type=str, default='', help='Dataset directory')
    parser.add_argument('--save_path', type=str, default='', help='Checkpoints directory')

    args = parser.parse_args()

    # Run example
    main(args.model_name, args.exp_name, args.epochs, args.quality_level, args.data_path, args.save_path)
