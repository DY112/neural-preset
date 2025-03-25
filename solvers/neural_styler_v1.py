import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
import numpy as np
from omegaconf import OmegaConf
from utils.setup import get_optimizer, get_scheduler
import os
import cv2

class Solver(pl.LightningModule):
    def __init__(self, net, criterion, cfg):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        
        # Initialize network, criterion, and config
        self.net = net
        self.criterion = criterion
        self.cfg = cfg
        
        # Initialize loss functions
        self.l2 = nn.MSELoss()
        self.l1 = nn.L1Loss()
        
        # Initialize best value & model dict
        self.best_metric_dict = {}
        self.best_model = {}
        for key_criteria in self.cfg.saver.monitor_keys:
            key, criteria = key_criteria.split('/')
            if criteria == 'l':
                self.best_metric_dict[key] = 987654321
            elif criteria == 'h':
                self.best_metric_dict[key] = -987654321
            self.best_model[key] = None

    def training_step(self, batch, batch_idx):
        img_i, img_j = batch["img_i"], batch["img_j"]
        
        # Stack images for batch processing
        stacked_content = torch.cat([img_i, img_j], dim=0)
        stacked_style = torch.cat([img_j, img_i], dim=0)
        
        # Forward pass
        stacked_z, stacked_result = self.net(stacked_content, stacked_style)
        z_i, z_j = torch.split(stacked_z, [img_i.shape[0], img_j.shape[0]], dim=0)
        y_j, y_i = torch.split(stacked_result, [img_i.shape[0], img_j.shape[0]], dim=0)
        
        # Calculate losses
        consistency_loss = self.l2(z_i, z_j)
        reconstruction_loss = self.l1(y_i, img_i) + self.l1(y_j, img_j)
        total_loss = reconstruction_loss + self.cfg.criterion.lambda_consistency * consistency_loss
        
        # Log metrics
        self.log_dict({
            'train-consistency_loss': consistency_loss,
            'train-reconstruction_loss': reconstruction_loss,
            'train-total_loss': total_loss
        }, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, rank_zero_only=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        img_i, img_j = batch["img_i"], batch["img_j"]
        
        # Stack images for batch processing
        stacked_content = torch.cat([img_i, img_j], dim=0)
        stacked_style = torch.cat([img_j, img_i], dim=0)
        
        # Forward pass
        stacked_z, stacked_result = self.net(stacked_content, stacked_style)
        z_i, z_j = torch.split(stacked_z, [img_i.shape[0], img_j.shape[0]], dim=0)
        y_j, y_i = torch.split(stacked_result, [img_i.shape[0], img_j.shape[0]], dim=0)
        
        # Calculate losses
        consistency_loss = self.l2(z_i, z_j)
        reconstruction_loss = self.l1(y_i, img_i) + self.l1(y_j, img_j)
        total_loss = reconstruction_loss + self.cfg.criterion.lambda_consistency * consistency_loss
        
        # Log metrics
        self.log_dict({
            'val-consistency_loss': consistency_loss,
            'val-reconstruction_loss': reconstruction_loss,
            'val-total_loss': total_loss
        }, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True, rank_zero_only=True)
        
        # Visualize results
        grid_img = self.make_grid_image(img_i, img_j, z_i, z_j, y_i, y_j)
        self.visualize_result(img=grid_img, phase='val')
        
        return total_loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        # Get optimizer
        optimizer_mode = self.cfg.train.optimizer.mode
        optimizer = get_optimizer(
            opt_mode=optimizer_mode,
            net_params=self.net.parameters(),
            **(self.cfg.train.optimizer[optimizer_mode])
        )
        
        # Get scheduler
        scheduler_mode = self.cfg.train.scheduler.mode
        scheduler = get_scheduler(
            sched_mode=scheduler_mode,
            optimizer=optimizer,
            **(self.cfg.train.scheduler[scheduler_mode])
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': self.cfg.train.check_val_every_n_epoch,
                'monitor': self.cfg.train.scheduler.monitor,
            }
        }

    def make_grid_image(self, img_i, img_j, z_i, z_j, y_i, y_j):
        # detach tensor from GPU, and convert to numpy
        img_i = img_i.detach().cpu().numpy().transpose(1, 2, 0)
        img_j = img_j.detach().cpu().numpy().transpose(1, 2, 0)
        z_i = z_i.detach().cpu().numpy().transpose(1, 2, 0)
        z_j = z_j.detach().cpu().numpy().transpose(1, 2, 0)
        y_i = y_i.detach().cpu().numpy().transpose(1, 2, 0)
        y_j = y_j.detach().cpu().numpy().transpose(1, 2, 0)

        # clip to [0, 1]
        img_i = np.clip(img_i * 255, 0, 255)
        img_j = np.clip(img_j * 255, 0, 255)
        z_i = np.clip(z_i * 255, 0, 255)
        z_j = np.clip(z_j * 255, 0, 255)
        y_i = np.clip(y_i * 255, 0, 255)
        y_j = np.clip(y_j * 255, 0, 255)

        # convert to uint8
        img_i = img_i.astype(np.uint8)
        img_j = img_j.astype(np.uint8)
        z_i = z_i.astype(np.uint8)
        z_j = z_j.astype(np.uint8)
        y_i = y_i.astype(np.uint8)
        y_j = y_j.astype(np.uint8)

        # make grid image
        #####################
        # input_i, z_i, y_j #
        # input_j, z_j, y_i #
        #####################
        first_row = np.concatenate((img_i, z_i, y_j), axis=1)
        second_row = np.concatenate((img_j, z_j, y_i), axis=1)
        grid_image = np.concatenate((first_row, second_row), axis=0)

        return grid_image

    def visualize_result(self, img, phase):
        # save image to ssd visualization folder
        save_dir = os.path.join(self.cfg.path.result_path, phase)
        os.makedirs(save_dir, exist_ok=True)
        
        # save image with epoch number
        save_path = os.path.join(save_dir, f'epoch_{self.current_epoch:04d}.png')
        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        # save image to wandb
        if self.cfg.logger.use_wandb:
            self.logger.experiment.log({
                f'{phase}/visualization': wandb.Image(img)
            }, step=self.current_epoch)
        