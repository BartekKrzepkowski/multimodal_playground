import logging
from typing import Dict

import torch

from src.visualization.wandb_logger import WandbLogger
from src.utils.utils_general import save_checkpoint, save_training_artefacts, load_model
from src.utils.utils_metrics import accuracy, get_val_score, mAP, accuracy_for_binary
from src.utils.utils_trainer import update_metrics, adjust_to_log
from src.utils.utils_visualize import matplotlib_scatters_training, log_to_console, show_and_save_grid

class Trainer:
    """
    Manages the **training process** of a machine learning model: initialization, training, validation, testing,
    checkpoint saving, and metric logging.

    Attributes:
        model (torch.nn.Module): The model to train.
        criterion (Dict): Dictionary of loss functions for training and evaluation.
        loaders (Dict): Data loaders for each phase ('train', 'val', 'test').
        optim (torch.optim.Optimizer): Optimizer.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        extra_modules (Dict): Additional modules/utilities.
        device (str): Device ('cpu' or 'cuda').
    """
    def __init__(self, model, criterion, loaders, optim, lr_scheduler, extra_modules, device):

        """
        Initializes all core components needed for training.

        Args:
            model (torch.nn.Module): **Model** to be trained.
            criterion (Dict): **Loss functions** for training and evaluation.
            loaders (Dict): **Data loaders** for each phase ('train', 'val', 'test').
            optim (torch.optim.Optimizer): **Optimizer**.
            lr_scheduler (torch.optim.lr_scheduler._LRScheduler): **Learning rate scheduler**.
            extra_modules (Dict): **Additional modules** or utilities.
            device (str): **Device** to use ('cpu' or 'cuda').
        """
        self.model = model
        self.criterion = criterion
        self.loaders = loaders
        self.optim = optim
        self.lr_scheduler = lr_scheduler
        self.device = device

        self.logger = None
        self.base_path = None
        self.base_save_path = None
        self.global_step = None

        self.extra_modules = extra_modules


    def at_exp_start(self, config):
        """
        Initializes the **experiment** – creates paths, logger, and sets up logging configuration.

        Args:
            config: **Configuration** object with experiment parameters.
        """

        # ════════════════════════ prepare paths ════════════════════════ #

        self.extra_modules['path_manager'].create_directories()
        self.base_path = self.extra_modules['path_manager'].root_logs
        self.base_save_path = self.extra_modules['path_manager'].root_logs_checkpoints
        config.logger_params['log_dir'] = f'{self.base_path}/{config.logger_params["logger_name"]}'
        self.extra_modules['early_stopping'].ckpt = f"{self.base_save_path}/best_model.pth" \
            if self.extra_modules['early_stopping'].ckpt is None \
            else self.extra_modules['early_stopping'].ckpt

        # ════════════════════════ prepare logging ════════════════════════ #

        self.logger = WandbLogger(config.logger_params, exp_name=config.trainer_params['exp_name'])
        
        self.logger.log_model(self.model, self.criterion, log=None)

        # for phase, loader in self.loaders.items():    #POPRAW
        #     show_and_save_grid(loader.dataset, save_path=f"{self.base_path}/{phase}_dataset.png")


    def train_model(self, config):
        """
        The **main training loop**.
        Handles training, validation, testing, checkpoint saving, metric logging, and early stopping.

        Args:
            config: **Configuration** object with all training parameters.
        """
        logging.info('Training started.')

        self.at_exp_start(config)
        
        # 1) Przygotowanie metryk
        epochs_metrics = {}
        
        for epoch in range(config.trainer_params['n_epochs']):
            epochs_metrics['epoch'] = epoch # czy musze to zapisywać?

            # 1) Zapisz stanu modelu i optymalizatora
            if (epoch > 0) and (config.trainer_params['save_checkpoint_modulo'] != 0) and (epoch % config.trainer_params['save_checkpoint_modulo'] == 0):
                save_checkpoint(
                    self.model,
                    self.optim,
                    epochs_metrics['epoch'],
                    save_path=f"{self.base_save_path}/checkpoint_epoch_{epochs_metrics['epoch']}.pth"
                )
            
            # 2) Training phase
            self.model.train()
            self.run_phase(epochs_metrics, phase='train', config=config)
            
            # 3) Validation phase
            with torch.no_grad():
                self.model.eval()
                score_val = self.run_phase(epochs_metrics, phase='val', config=config) # lepsza nazwa niż val_f1? desired_trigger?

                # scheduler step
                if self.lr_scheduler is not None:
                    if config.optim_scheduler_params['scheduler_name'] == 'reduce_on_plateau':
                        self.lr_scheduler.step(epochs_metrics['val_losses'][-1])
                    elif config.optim_scheduler_params['scheduler_name'] == 'cosine':   # w złym miejscu
                        self.lr_scheduler.step()
                    else:
                        raise ValueError(f"Unknown scheduler name: {config.optim_scheduler_params['scheduler_name']}")

            # 4) check if early stopping is triggered
            if self.extra_modules['early_stopping'](score_val, self.model, self.optim, epoch):
                logging.info(f"Early stopping triggered at epoch {epoch}.")
                break

            # 5) Logowanie metryk
            self.at_epoch_end(epoch, epochs_metrics)


        if self.extra_modules['early_stopping'].ckpt is not None:
            logging.info(f"Loading best model, given metric: {self.extra_modules['early_stopping'].mode} {self.extra_modules['early_stopping'].best_score:.4f} \
                         at epoch {self.extra_modules['early_stopping'].best_epoch}.")
            self.model = load_model(self.model, self.extra_modules['early_stopping'].ckpt)

        # self.run_phase(epochs_metrics, phase='test', config=config)
        # self.at_epoch_end(epoch, epochs_metrics)
        
        # 4) Logowanie metryk do konsoli
        log_to_console(epochs_metrics)
        
        self.at_exp_end(config, epochs_metrics)
    

    def run_phase(self, epochs_metrics, phase, config):
        """
        Runs **one phase** of training/validation/testing: iterates over batches, collects metrics, logs selected data.

        Args:
            epochs_metrics (dict): **Dictionary** with epoch metrics.
            phase (str): **Phase name** ('train', 'val', 'test').
            config: **Configuration** object.

        Returns:
            float or None: f1 score for validation phase, otherwise None.
        """
        logging.info(f'Epoch: {epochs_metrics["epoch"]}, Phase: {phase}.')
        
        running_metrics = {
            f"{phase}_{metric_name}": []
            for metric_name in ('losses', 'accs')
        }
        running_metrics['batch_sizes'] = []

        batches_per_epoch = len(self.loaders[phase])
        logging.info(f'Batches per epoch: {batches_per_epoch}.')
        self.global_step = epochs_metrics['epoch'] * batches_per_epoch  # czy musi tu być self.?
        config.trainer_params['running_window_start'] = batches_per_epoch // 1 # OGARNIJ TĄ STAŁĄ
        
        for i, data in enumerate(self.loaders[phase]):
            y_pred, y_true = self.infer_from_data(data, config=config)
            running_metrics = self.gather_batch_metrics(phase, running_metrics, y_pred, y_true, config)

            # ════════════════════════ logging (running) ════════════════════════ #

            if (i + 1) % config.trainer_params['running_window_start'] == 0: # lepsza nazwa na log_multi? słowniek multi = {'log'...}?
                # prepare logs for logging
                running_logs = adjust_to_log(running_metrics, scope='running', window_start=config.trainer_params['running_window_start'])
                self.log(
                    running_logs,
                    phase,
                    scope='running',
                    step=self.global_step
                )

            self.global_step += 1
        
        update_metrics(epochs_metrics, running_metrics)

        val_score = get_val_score(epochs_metrics, phase, self.extra_modules['early_stopping'].score_name)

        return val_score if phase == 'val' else None
        

    def log(self, scope_logs: Dict, phase: str, scope: str, step: int):
        """
        **Logs** selected metrics and data to the logger and progress bar.

        Args:
            scope_logs (Dict): **Metrics/data** to log.
            phase (str): **Phase name** ('train', 'val', 'test').
            scope (str): **Logging scope** ('running', 'epoch').
            step (int): **Step** (iteration number).
        """
        scope_logs[f'steps/{phase}_{scope}'] = step
        self.logger.log_scalars(scope_logs, step)
        # progress_bar.set_postfix(evaluators_log)

        if self.lr_scheduler is not None and phase == 'train' and scope == 'running':
            self.logger.log_scalars({f'lr_scheduler': self.lr_scheduler.get_last_lr()[0]}, step)


    def at_epoch_end(self, epoch, epochs_metrics):
        """
        Actions performed at the **end of each epoch** – logs metrics, prints results to console.

        Args:
            epoch (int): **Epoch number**.
            epochs_metrics (dict): **Dictionary** with epoch metrics.
        """
        logging.info(f'Epoch {epochs_metrics["epoch"]} finished.')

        # ════════════════════════ logging (epoch) ════════════════════════ #

        epoch_logs = adjust_to_log(epochs_metrics, scope='epoch', window_start=0)
        self.log(
            epoch_logs,
            phase='test',
            scope='epoch',
            step=epoch
        )
        # 5) Logowanie metryk do konsoli
        log_to_console(epochs_metrics)    
        
    
    def at_exp_end(self, config, epochs_metrics):
        """
        Actions after the **entire experiment is finished** – saves training artifacts, final checkpoint, closes the logger.

        Args:
            config: **Configuration** object.
            epochs_metrics (dict): **Training metrics**.
        """
        logging.info('Training finished.')
        save_training_artefacts(
            config,
            epochs_metrics,
            save_path=f"{self.base_save_path}/training_artefacts.pth"
        )
        save_checkpoint(
            self.model,
            self.optim,
            epochs_metrics['epoch'],
            save_path=f"{self.base_save_path}/epoch_{epochs_metrics['epoch']}.pth"
        )
        self.logger.close()
        # matplotlib_scatters_training(epochs_metrics, save_path=f"{self.base_path}/metrics.pdf")


    def infer_from_data(self, data, config):
        """
        **Performs inference** (prediction) on a single batch, moving data to the correct device.

        Args:
            data: **Data batch** (x, y).
            device: **Device** ('cpu' or 'cuda').

        Returns:
            tuple: (y_pred, y_true) – model predictions and ground truth labels.
        """
        device = config.trainer_params['device']
        x_true1, x_true2, y_true = data
        x_true1, x_true2, y_true = x_true1.to(device), x_true2, y_true.to(device)
        # x_true1, x_true2, y_true = x_true1.to(device), x_true2.to(device), y_true.to(device)
        forward_params = {
            'x1': x_true1,
            'x2': x_true2,
            'enable_left_branch': config.trainer_params['enable_left_branch'],
            'enable_right_branch': config.trainer_params['enable_right_branch']
        }
        y_pred = self.model(**forward_params)
        return y_pred, y_true
    
    
    def gather_batch_metrics(self, phase, running_metrics, y_pred, y_true, config):
        """
        Calculates **all metrics** for a single batch (loss, accuracy, precision, recall, f1, etc.)
        and prepares them for logging.

        Args:
            phase (str): **Phase name** ('train', 'val', 'test').
            running_metrics (dict): **Metrics dictionary** to update.
            y_pred: **Predictions**.
            y_true: **True labels**.

        Returns:
            dict: Updated metrics dictionary.
        """
        y_true = y_true.unsqueeze(-1).float() if y_true.dim() == 1 else y_true  # for binary classification

        loss_list = self.criterion(y_pred, y_true)
  
        loss = loss_list.mean()
            
        if phase == 'train':
            loss.backward()
            if config.trainer_params['gradient_clipping_max_norm'] > 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=config.trainer_params['gradient_clipping_max_norm'])
            self.optim.step()
            self.optim.zero_grad(set_to_none=True)
            # tutaj per-batch scheduler.step()
            self.run_extra_modules(config)
        
        # acc = accuracy(y_pred, y_true)
        acc = accuracy_for_binary(y_pred, y_true)
        batch_size = y_true.shape[0]

        # ════════════════════════ gathering scalars to logging ════════════════════════ #
        
        running_metrics['batch_sizes'].append(batch_size)
        running_metrics[f'{phase}_losses'].append(loss.item() * batch_size)
        running_metrics[f'{phase}_accs'].append(acc * batch_size)

        return running_metrics
    

    def run_extra_modules(self, config):
        """
        Runs additional modules (e.g., simplicity meter) if they are available.

        Args:
            phase (str): **Phase name** ('train', 'val', 'test').
            y_pred: **Predictions**.
            y_true: **True labels**.
        """
        if 'simplicity_meter' in self.extra_modules and self.global_step % config.mods['simplicity_meter'] == 0:
            self.extra_modules['simplicity_meter'].model_report(self.model, self.logger, self.global_step, scope='periodic', phase='train')
        
        # Add other extra modules here if needed