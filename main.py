import logging
import os

import torch

from src.trainer.trainer import Trainer
from src.utils.utils_data import prepare_loaders
from src.utils.utils_model import prepare_model
from src.utils.utils_optim import prepare_optim_and_scheduler
from src.utils.utils_criterion import prepare_criterion
from src.modules.early_stopping import EarlyStopping
from src.modules.paths_manager import PathsManager
from src.modules.simplicity_meter import SimplicityMeter


from src.utils.utils_general import set_seed, yield_hyperparameters


device = "cuda" if torch.cuda.is_available() else "cpu"     


def main(config):
     # ════════════════════════ prepare seed ════════════════════════ #


    set_seed(config.trainer_params['seed'])
    logging.info('Random seed prepared.')


    # ════════════════════════ prepare loaders ════════════════════════ #


    loaders = prepare_loaders(config.data_params)
    logging.info('Loaders prepared.')
    

    # ════════════════════════ prepare model ════════════════════════ #


    model = prepare_model(config.model_params).to(device)
    logging.info('Model prepared.')
    
    
    # ════════════════════════ prepare optimizer & scheduler ════════════════════════ #


    optim, lr_scheduler = prepare_optim_and_scheduler(model, config.optim_scheduler_params)
    logging.info('Optimizer and scheduler prepared.')
    
    
    # ════════════════════════ prepare criterion ════════════════════════ #

    # weights = torch.tensor([1.0, 3168/384], device=device)r
    criterion = prepare_criterion(config.criterion_params)
    logging.info('Criterion prepared.')


    # ════════════════════════ prepare extra modules ════════════════════════ #
    

    extra_modules = {}
    
    path_manager = PathsManager(
        custom_root=config.trainer_params['custom_root'],
        experiment_name=config.trainer_params['exp_name']
    )
    early_stopping = EarlyStopping(**config.extra_modules_params['early_stopping'])
    # simplicity_meter = SimplicityMeter(**config.extra_modules_params['simplicity_meter'])

    # config.mods['simplicity_meter'] = len(loaders['train']) // 2 # how often to log simplicity metrics, every 1/2 epochs

    extra_modules['path_manager'] = path_manager
    extra_modules['early_stopping'] = early_stopping
    # extra_modules['simplicity_meter'] = simplicity_meter
    logging.info('Extra modules prepared.')


    # ════════════════════════ prepare trainer ════════════════════════ #
    

    params_trainer = {
        'model': model,
        'criterion': criterion,
        'loaders': loaders,
        'optim': optim,
        'lr_scheduler': lr_scheduler,
        'device': device,
        'extra_modules': extra_modules,
    }
    trainer = Trainer(**params_trainer)
    logging.info('Trainer prepared.')


    # ════════════════════════ train model ════════════════════════ #


    trainer.train_model(config)
    logging.info('Training finished.')


if __name__ == "__main__":
    logging.basicConfig(
            format=(
                '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] ' '%(message)s'
            ),
            level=logging.INFO,
            handlers=[logging.StreamHandler()],
            force=True,
        )
    

    # ════════════════════════ hyperparameters ════════════════════════ #


    hyperparameters = {
        'optim_name': ['adamw'],
        'lr': [1e-4, 3e-4],
        'weight_decay': [1e-4],
        'scheduler_name': [None],
        'hidden_dim': [256],
    }

    scheduler_params = {
        None: None,
        'cosine': {'T_max': 200, 'eta_min': 1e-3, 'verbose': True},
        'reduce_on_plateau': {'mode': 'min', 'factor': 0.5, 'patience': 10, 'verbose': True},
    }


    # ═══════════════════════ run experiments ════════════════════════ #


    for (optim_name, lr, weight_decay, scheduler_name, hidden_dim) in yield_hyperparameters(hyperparameters):

        class Config:
            """
            Holds configuration dictionaries for training, data loading, model, optimizer/scheduler, logging, and extra modules.
            """
            trainer_params = {
                'device': device,
                'seed': 83,
                'n_epochs': 400,
                'exp_name': f'mm_imbd_{lr=}_{weight_decay=}_{optim_name=}_{hidden_dim=}_bce_resnet34_exe', # nazwa eksperymentu, która będzie użyta do tworzenia folderów i logowania
                'custom_root': os.environ['REPORTS_DIR'],     # custom root directory for saving logs and checkpoints
                'load_checkpoint_path': None,    # saving checkpoint of model and optimizer
                'save_checkpoint_modulo': 50,  # how many epochs to save the model,
                'gradient_clipping_max_norm': 5.0,  # maximum norm for gradient clipping
                'enable_left_branch': True,  # whether to enable the left branch of the model
                'enable_right_branch': True,  # whether to enable the right branch of the model
            }
            data_params = {
                'dataset_name' : 'fhmd',
                'dataset_params': {},
                'loader_params': {'batch_size': 125, 'pin_memory': True, 'num_workers': 12}
            }
            model_params = {
                'model_name': 'fusion_mlp_exe',
                'model_params': {
                    'img_dim': 512,  # dimension of image features
                    'txt_dim': 768,  # dimension of text features
                    'proj_dim': 256,  # dimension of projection layer
                    'res_hidden_dim': 512,  # hidden dimension for residual MLP
                    'fusion_hidden_dim': hidden_dim,  # hidden dimension for fusion MLP
                    'num_labels': 1,  # number of labels for classification
                    'dropout': 0.2,  # dropout rate
                    # 'sigma': 0.0,  # noise level for augmentation
                    'additional_params': {
                        'encoder1_params': {
                            'model_name': 'resnet_encoder_pretrained',
                            'model_params': {'resnet_version': 'resnet34', 'pretrained': True},
                            'checkpoint_path': None,
                            'freeze_backbone': False,
                        },
                        'encoder2_params': {
                            'model_name': 'text_encoder_pretrained',
                            'model_params': {'model_name': "distilbert-base-uncased", 'max_length': 90},
                            'checkpoint_path': None,
                            'freeze_backbone': False,
                        }
                    }
                },
                # 'model_params': {
                #     'resnet_type': 'resnet18',
                #     'num_classes': 100,
                #     'img_height': 32,  # height of input images
                #     'img_width': 32,   # width of input images
                #     'fusion': 'concat',  # how to fuse the two branches, options: '
                #     'pretrained': False
                #     },
                'checkpoint_path': None, # path to load model checkpoint
                'init': None,
                'freeze_backbone': False,  # whether to freeze the backbone of the model
            }
            optim_scheduler_params = {
                'optim_name': optim_name,
                'optim_params': {'lr': lr, 'weight_decay': weight_decay},
                'checkpoint_path': None,  # path to load optimizer state
                'scheduler_name': scheduler_name,
                'scheduler_params': scheduler_params[scheduler_name],
            }
            criterion_params = {
                'criterion_name': 'bce_with_logits',
                'criterion_params': {'pos_weight': torch.tensor([5450 / 3050]).to(device)},
                # 'criterion_params': {'reduction': 'none', 'label_smoothing': 0.1},
            }
            logger_params = {
                'logger_name': 'wandb',
                'entity': os.environ['WANDB_ENTITY'],
                'project_name': 'mm_playground',
                'mode': 'online',   # używając tego określ również czy logować info na dysk
                # 'hyperparameters': h_params_overall,
            }
            extra_modules_params = {
                'early_stopping': {
                    'patience': 40,  # how many epochs without improvement to wait before stopping
                    'mode': 'max',
                    'delta': 1e-4,
                    'enable_early_stopping': False,  # whether to enable early stopping
                    'score_name': 'accs',  # name of the metric to monitor for early stopping, in plural form
                },
                'simplicity_meter': {
                    'max_rank': 1024,  # maximum rank for the SVD approximation
                }
            }
            mods = {
                'simplicity_meter': 1,  # how often to log simplicity metrics
            }
            
        config = Config()
        main(config)
