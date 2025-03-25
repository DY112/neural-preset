import pytorch_lightning as pl
import torch,wandb,random
import torch.nn as nn
import os.path as osp
import numpy as np
from omegaconf import OmegaConf
from utils.setup import get_optimizer, get_scheduler

class Solver(pl.LightningModule):
    def __init__(self,net,loss,**cfg):
        super().__init__()
        self.save_hyperparameters(ignore=['net','loss'])
        self.cfg = OmegaConf.create(cfg)
        self.net = net
        self.loss = loss
    
    def training_step(self, batch, batch_idx):
        x,y = batch
        batch_dict = {'x':x,'y':y}

        # forward
        ret_dict = self.net(x)

        # loss
        loss_dict = self.loss(ret_dict,batch_dict,'train')

        # log
        self.log_dict(
            loss_dict,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            rank_zero_only=True,
        )
        
        return loss_dict['train-total_loss']
    
    def validation_step(self, batch, batch_idx):
        x,y = batch
        batch_dict = {'x':x,'y':y}

        # forward
        ret_dict = self.net(x)

        # loss
        loss_dict = self.loss(ret_dict,batch_dict,'val')

        # log
        self.log_dict(
            loss_dict,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            rank_zero_only=True,
        )
    
    def test_step(self, batch, batch_idx):
        self.validation_step(batch,batch_idx)

    def configure_optimizers(self):
        # get optimizer and scheduler
        optimizer_mode = self.cfg.train.optimizer.mode
        optimizer = get_optimizer(opt_mode=optimizer_mode,
                                  net_params=self.net.parameters(),
                                  **(self.cfg.train.optimizer[optimizer_mode]))
        scheduler_mode = self.cfg.train.scheduler.mode
        scheduler = get_scheduler(sched_mode=scheduler_mode,
                                  optimizer=optimizer,
                                  **(self.cfg.train.scheduler[scheduler_mode]))
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': self.cfg.train.check_val_every_n_epoch,
                'monitor': self.cfg.train.scheduler.monitor,
            }
        }