import numpy as np
import wandb
from omegaconf import OmegaConf
import math
import random,torch,time,os
from torch.optim import lr_scheduler
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning.loggers as pl_loggers
import cv2,rawpy

##########################
# Generic util functions #
##########################

@rank_zero_only
def init_path_and_expname(cfg):
    # set date_time_model
    if cfg.load.ckpt_path != None and cfg.load.load_state:
        if 'backups' in cfg.load.ckpt_path:
            cfg.path.date_time_model = cfg.load.ckpt_path.split('/')[6]
        else:
            cfg.path.date_time_model = cfg.load.ckpt_path.split('/')[2]
    else:
        cfg.path.date_time_model = time.strftime(cfg.path.time_format,time.localtime(time.time())) + '_' + \
                                    cfg.model.name + '_' + cfg.model.ver
    
    # set path
    cfg.path.log_path = '../logs'
    cfg.path.ckpt_path = os.path.join(cfg.path.ckpt_root, cfg.path.date_time_model)
    cfg.path.result_path = os.path.join(cfg.path.result_root,cfg.path.date_time_model)
    
    # make directories
    if cfg.mode == 'train':
        os.makedirs(cfg.path.log_path, exist_ok=True)
    os.makedirs(cfg.path.ckpt_path, exist_ok=True)
    os.makedirs(cfg.path.result_path, exist_ok=True)

    # set experiment name
    cfg.exp_name = f'{cfg.servername}/{cfg.path.date_time_model}'

def get_optimizer(opt_mode, net_params, **opt_params):
    if opt_mode == 'adam':
        return torch.optim.Adam(net_params, **opt_params)

def get_scheduler(sched_mode, optimizer, **sched_params):
    if sched_mode == 'StepLR':
        return lr_scheduler.StepLR(optimizer, **sched_params)
    elif sched_mode == 'CosineAnnealingLR':
        return lr_scheduler.CosineAnnealingLR(optimizer, **sched_params)
    elif sched_mode == 'ReduceLROnPlateau':
        return lr_scheduler.ReduceLROnPlateau(optimizer, **sched_params)    
    else:
        raise NotImplementedError
    
def get_trainer_args(cfg):
    trainer_args = {}

    # env
    trainer_args['devices'] = cfg.devices if cfg.mode == 'train' else 1
    trainer_args['accelerator'] = "auto"

    # training
    trainer_args['max_epochs'] = cfg.train.end_epoch
    trainer_args['check_val_every_n_epoch'] = cfg.train.check_val_every_n_epoch

    # logging
    trainer_args['logger'] = get_logger(cfg)
    trainer_args['log_every_n_steps'] = cfg.logger.log_every_n_steps
    
    # callbacks
    trainer_args['callbacks'] = get_callbacks(cfg)

    return trainer_args
    
def get_callbacks(cfg):
    # return custom callbacks
    callbacks = []

    # model checkpoint callbacks using criterion
    for criteria in cfg.saver.monitor_keys:
        key, criteria = criteria.split('/')
        if criteria == 'l':
            mode = 'min'
        elif criteria == 'h':
            mode = 'max'
        else:
            raise NotImplementedError

        callbacks.append(
            ModelCheckpoint(
                dirpath=cfg.path.ckpt_path,
                filename=f'best_{key}_{{epoch:04d}}',
                monitor=key,
                mode=mode,
                save_top_k=1,
                save_last=True,
            )
        )
    
    # model checkpoint callback for every n epoch
    callbacks.append(
        ModelCheckpoint(
            dirpath=cfg.path.ckpt_path,
            filename=f'{{epoch:04d}}',
            every_n_epochs=cfg.saver.save_every_n_epoch
        )
    )

    # you can add early stopping callback here (https://lightning.ai/docs/pytorch/stable/common/early_stopping.html)
    
    return callbacks

def get_logger(cfg):
    # return custom logger
    logger = None

    if cfg.logger.use_wandb:
        logger = pl_loggers.WandbLogger(
            project=cfg.project_name,
            name=cfg.exp_name,
            save_dir='../logs/',
        )
        # update wandb config
        # logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))

    else:
        logger = pl_loggers.TensorBoardLogger(
            save_dir=cfg.path.log_path,
            name=cfg.exp_name,
        )

    return logger