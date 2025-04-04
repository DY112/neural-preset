import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
import numpy as np
import os
import cv2
from omegaconf import OmegaConf
from utils.setup import get_optimizer, get_scheduler

class Solver(pl.LightningModule):
    """Neural Styler Solver using PyTorch Lightning.
    
    This solver implements the training and validation logic for the neural style transfer model.
    It handles both local visualization and wandb logging.
    """
    
    def __init__(self, net, criterion, cfg):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        
        # Initialize components
        self.net = net
        self.criterion = criterion
        self.cfg = cfg
        
        # Initialize loss functions
        self.l2 = nn.MSELoss()
        self.l1 = nn.L1Loss()
    
    def _process_batch(self, batch):
        """Process a batch of images and return model outputs."""
        img_i, img_j = batch["img_i"], batch["img_j"]
        
        # Stack images for batch processing
        stacked_content = torch.cat([img_i, img_j], dim=0)
        stacked_style = torch.cat([img_j, img_i], dim=0)
        
        # Forward pass
        stacked_z, stacked_result = self.net(stacked_content, stacked_style)
        z_i, z_j = torch.split(stacked_z, [img_i.shape[0], img_j.shape[0]], dim=0)
        y_j, y_i = torch.split(stacked_result, [img_i.shape[0], img_j.shape[0]], dim=0)
        
        return img_i, img_j, z_i, z_j, y_i, y_j
    
    def _compute_losses(self, z_i, z_j, y_i, y_j, img_i, img_j):
        """Compute all losses for the model outputs."""
        consistency_loss = self.l2(z_i, z_j)
        reconstruction_loss = self.l1(y_i, img_i) + self.l1(y_j, img_j)
        total_loss = reconstruction_loss + self.cfg.criterion.lambda_consistency * consistency_loss
        
        return {
            'consistency_loss': consistency_loss,
            'reconstruction_loss': reconstruction_loss,
            'total_loss': total_loss
        }
    
    def _log_metrics(self, losses, phase):
        """Log metrics to wandb and progress bar."""
        # Create separate log dictionaries for train and val
        if phase == 'train':
            log_dict = {
                'train/losses/consistency_loss': losses['consistency_loss'],
                'train/losses/reconstruction_loss': losses['reconstruction_loss'],
                'train/losses/total_loss': losses['total_loss']
            }
        else:  # val
            log_dict = {
                'val/losses/consistency_loss': losses['consistency_loss'],
                'val/losses/reconstruction_loss': losses['reconstruction_loss'],
                'val/losses/total_loss': losses['total_loss']
            }
        
        # Log using PyTorch Lightning's log_dict
        self.log_dict(
            log_dict,
            on_step=(phase == 'train'),
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            rank_zero_only=True
        )
    
    def training_step(self, batch, batch_idx):
        # Process batch and compute losses
        img_i, img_j, z_i, z_j, y_i, y_j = self._process_batch(batch)
        losses = self._compute_losses(z_i, z_j, y_i, y_j, img_i, img_j)
        
        # Log metrics
        self._log_metrics(losses, 'train')
        
        return losses['total_loss']
    
    def validation_step(self, batch, batch_idx):
        # Process batch and compute losses
        img_i, img_j, z_i, z_j, y_i, y_j = self._process_batch(batch)
        losses = self._compute_losses(z_i, z_j, y_i, y_j, img_i, img_j)
        
        # Log metrics
        self._log_metrics(losses, 'val')
        
        # Visualize results
        phase = 'test' if self.trainer.testing else 'val'
        
        if phase == 'test':
            # Test mode: visualize all images in the batch
            for idx in range(img_i.shape[0]):
                grid_img = self.make_grid_image(
                    img_i[idx], img_j[idx],
                    z_i[idx], z_j[idx],
                    y_i[idx], y_j[idx]
                )
                fname_i  = batch["fname_i"][idx].split('.')[0]
                fname_j  = batch["fname_j"][idx].split('.')[0]
                fname = f'{fname_i}_{fname_j}'
                self.visualize_result(img=grid_img, phase=phase, fname=fname)
                
        elif batch_idx == 0:  # Validation mode: only visualize first batch
            rand_idx = np.random.randint(0, img_i.shape[0])
            grid_img = self.make_grid_image(
                img_i[rand_idx], img_j[rand_idx],
                z_i[rand_idx], z_j[rand_idx],
                y_i[rand_idx], y_j[rand_idx]
            )
            self.visualize_result(img=grid_img, phase=phase)
        
        return losses['total_loss']
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = get_optimizer(
            opt_mode=self.cfg.train.optimizer.mode,
            net_params=self.net.parameters(),
            **(self.cfg.train.optimizer[self.cfg.train.optimizer.mode])
        )
        
        scheduler = get_scheduler(
            sched_mode=self.cfg.train.scheduler.mode,
            optimizer=optimizer,
            **(self.cfg.train.scheduler[self.cfg.train.scheduler.mode])
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
        """Create a grid image from model outputs for visualization."""
        # Convert tensors to numpy arrays
        images = [img_i, img_j, z_i, z_j, y_i, y_j]
        images = [img.detach().cpu().numpy().transpose(1, 2, 0) for img in images]
        
        # Normalize and convert to uint8
        images = [np.clip(img * 255, 0, 255).astype(np.uint8) for img in images]
        
        # Create grid layout
        # [input_i, z_i, y_j]
        # [input_j, z_j, y_i]
        first_row = np.concatenate((images[0], images[2], images[5]), axis=1)
        second_row = np.concatenate((images[1], images[3], images[4]), axis=1)
        
        return np.concatenate((first_row, second_row), axis=0)
    
    def visualize_result(self, img, phase, fname=None, batch_idx=None, img_idx=None):
        """Save visualization results locally and to wandb."""
        # Save locally
        save_dir = os.path.join(self.cfg.path.result_path, phase)
        os.makedirs(save_dir, exist_ok=True)
        if fname is None:
            if batch_idx is None or img_idx is None:
                save_path = os.path.join(save_dir, f'epoch_{self.current_epoch:04d}.png')
            else:
                save_path = os.path.join(save_dir, f'batch_{batch_idx}_img_{img_idx}.png')
        else:
            save_path = os.path.join(save_dir, f'{fname}.png')
        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        # Log to wandb with separate panels for train and val
        if self.cfg.logger.use_wandb:
            self.logger.experiment.log({
                f'{phase}/visualization': wandb.Image(img)
            })
        