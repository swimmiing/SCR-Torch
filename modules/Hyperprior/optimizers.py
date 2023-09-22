import torch
import yaml
from typing import Optional
from contextlib import nullcontext
from torch.optim.optimizer import Optimizer


class CustomOptimizer:
    def __init__(self, model: torch.nn.Module, conf_file: str):
        """
        Initialize a CustomOptimizer.

        Args:
            model (torch.nn.Module): The image compression model to evaluate.
            conf_file (str): Path to the configuration file in YAML format.

        The configuration file should specify optimizer and scheduler settings.
        """
        # Get configuration
        with open(conf_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            opt_module = config['optimizer']['name']
            main_opt_arg = config['optimizer']['arg']

            sch_module = config['scheduler']['name']
            self.use_scheduler = True if sch_module else False
            sch_arg = config['scheduler']['arg']

            aux_opt_arg = dict(main_opt_arg)
            aux_opt_arg['lr'] *= config['optimizer']['aux_lr_scale']

            for i, item in enumerate(sch_arg['milestones']):
                if isinstance(item, float):
                    sch_arg['milestones'][i] = int((1 - item) * config['epoch'])

        # Define main optimizer
        self.main_optimizer = getattr(torch.optim, opt_module)(model.parameters(), **main_opt_arg)

        # Define auxiliary optimizer with scaled learning rate
        self.aux_optimizer = getattr(torch.optim, opt_module)(model.parameters(), **aux_opt_arg)

        # Define main scheduler and share it with the auxiliary optimizer if available
        if self.use_scheduler:
            main_scheduler = getattr(torch.optim.lr_scheduler, sch_module)(self.main_optimizer, **sch_arg)
            self.main_scheduler = main_scheduler
            self.aux_scheduler = main_scheduler

    def zero_grad(self):
        """
        Zero the gradients of both main and auxiliary optimizers.
        """
        self.main_optimizer.zero_grad()
        self.aux_optimizer.zero_grad()

    def step(self, main_loss: torch.Tensor, aux_loss: Optional[torch.Tensor] = None):
        """
        Perform optimization steps for both main and auxiliary losses.

        Args:
            main_loss (torch.Tensor): The main loss value.
            aux_loss (torch.Tensor): The auxiliary loss value (optional).
        """
        # Backward and update main optimizer
        main_loss.backward()
        self.main_optimizer.step()

        # Backward and update auxiliary optimizer if aux_loss is provided
        if aux_loss is not None:
            aux_loss.backward()
            self.aux_optimizer.step()

        # Step the schedulers if they are available
        if self.use_scheduler:
            self.main_scheduler.step()
