import torch
import os

from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from modules.eval_utils import ImgQEvaluator as Evaluator
from modules.torch_utils import save_img_from_tensor
from modules.eval_utils import save_string_dict, write_metric_log
from torch.utils.data import DataLoader
from typing import Optional, Union, Dict, Any


@torch.no_grad()
def eval_image(
    model: torch.nn.Module,
    test_dataloader: DataLoader,
    result_dir: str,
    epoch: Optional[int] = None,
    quality_level: Union[int, float] = 8,
    tensorboard_dir: Optional[str] = None,
    save_results: bool = False
) -> None:
    """
    Evaluate Image Compression.

    Args:
        model (torch.nn.Module): The image compression model to evaluate.
        test_dataloader (DataLoader): The data loader for the test dataset.
        result_dir (str): The directory where evaluation results will be saved.
        epoch (int, optional): The current epoch number (default: None).
        quality_level (int, or float): Quality level for adaptive compression. [1.0, 8.0] (default: 8)
        tensorboard_dir (str, optional): The directory for TensorBoard logs (default: None).
        save_results (bool, optional): Whether to save individual results (default: False).
    """
    if tensorboard_dir is not None:
        os.makedirs(tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(tensorboard_dir)

    split = test_dataloader.dataset.split

    # Directory for logging
    parent_dir = f'epoch{epoch}' if epoch is not None else ''
    quality_str = f'{float(quality_level):.1f}'.replace('.', '_')
    postfix = f'_q{quality_str}'
    viz_dir, log_dir = [os.path.join(result_dir, f'viz{postfix}'), os.path.join(result_dir, 'logs')]

    os.makedirs(viz_dir, exist_ok=True) if save_results else None
    os.makedirs(log_dir, exist_ok=True)

    # Image Compression Evaluator
    compression_evaluator = Evaluator()

    for step, data in enumerate(tqdm(test_dataloader, desc=f"Evaluate {split} dataset...")):
        images, ids = data['images'].to(model.device), data['ids']

        # Inference
        out = model(images, quality_level=quality_level)

        recon_images, strings = out['recon_images'], list(out['strings'].values())
        num_pixels = images.shape[0] * images.shape[-2] * images.shape[-1]

        # Update evaluator
        batch_metric_dict = compression_evaluator.evaluate_batch(recon_images, images, strings, num_pixels)

        # Save result
        if save_results:
            save_img_from_tensor(recon_images, os.path.join(viz_dir, 'recon', parent_dir), ids)
            save_string_dict(out['strings'], os.path.join(viz_dir, 'string', parent_dir), ids)

        # Save single evaluation result
        if test_dataloader.batch_size == 1 and save_results:
            msg_postfix = f"[Epoch {epoch}, Q_lv {quality_level}]" if epoch is not None else f"[Q_lv {quality_level}]"
            os.makedirs(os.path.join(viz_dir, 'test', parent_dir), exist_ok=True)
            rst_path = os.path.join(viz_dir, 'test', parent_dir, f'{ids[0]}.txt')
            write_metric_log(model.__class__.__name__, batch_metric_dict, rst_path, msg_postfix)

    metric_dict = compression_evaluator.finalize()

    # Save result
    msg_postfix = f"[Epoch {epoch}, Q_lv {quality_level}]" if epoch is not None else f"[Q_lv {quality_level}]"
    rst_path = os.path.join(log_dir, f'test_logs_{parent_dir}.txt')

    # Write and print text log
    msg = write_metric_log(model.__class__.__name__, metric_dict, rst_path, msg_postfix, mode='a')
    print(msg)

    if tensorboard_dir is not None:
        for k, v in metric_dict.items():
            writer.add_scalars(f'test/{k}(Q_lv {float(quality_level):.1f}))', {k: v}, epoch)

    if tensorboard_dir is not None:
        writer.close()
