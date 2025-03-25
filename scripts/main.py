import sys,os,random,torch,importlib,time
import torch.nn as nn
import wandb
import torch.distributed as dist
from omegaconf import OmegaConf
from pytorch_lightning.utilities.rank_zero import rank_zero_only

sys.path.insert(1,os.path.abspath('..'))
sys.path.insert(1,os.path.abspath('../../'))
from utils.setup import init_path_and_expname,get_callbacks,get_logger,get_trainer_args
from datasets.unified_loader import get_loader
from criterions.criterion import MasterCriterion
import pytorch_lightning as pl

if __name__ == '__main__':
    # import default config file
    cfg = OmegaConf.merge(OmegaConf.load(f'../configs/default.yaml'),OmegaConf.load('../configs/env.yaml'))
    # read from command line
    cfg_cmd = OmegaConf.from_cli()
    # merge model specific config file
    if "model" in cfg_cmd  and 'name' in cfg_cmd.model:
        cfg = OmegaConf.merge(cfg,OmegaConf.load(f'../configs/{cfg_cmd.model.name}.yaml'))
    else:
        cfg = OmegaConf.merge(cfg,OmegaConf.load(f'../configs/{cfg.model.name}.yaml'))
    # merge cfg from command line
    cfg = OmegaConf.merge(cfg,cfg_cmd)

    # Path and exp_name configuration
    init_path_and_expname(cfg)      # This function is only done in master process
    pl.seed_everything(cfg.seed)

    # Dataloader
    dataloader = {
        'train': get_loader(cfg,'train') if cfg.mode == 'train' else None,
        'valid': get_loader(cfg,'valid') if cfg.mode == 'train' else None,
        'test' : get_loader(cfg,'test') if cfg.mode == 'test' else None
    }
    
    # Dynamic Model module import
    network_mod = importlib.import_module(f'models.{cfg.model.name}_{cfg.model.ver}')
    network_class = getattr(network_mod,cfg.model.name)
    network = network_class(cfg)
    
    # Loss
    loss = MasterCriterion(cfg)

    # Dynamic Solver module import
    solver_mod = importlib.import_module(f'solvers.{cfg.model.name}_{cfg.model.solver}')
    solver_class = getattr(solver_mod,'Solver')
    solver = solver_class(net=network,
                          loss=loss,
                          **(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)))
    
    # Load Network if ckpt_path is given
    if cfg.load.ckpt_path is not None:
        solver = solver.load_from_checkpoint(cfg.load.ckpt_path, net=network, loss=loss)
    
    # Init trainer
    trainer_args = get_trainer_args(cfg)
    trainer = pl.Trainer(**trainer_args)

    if cfg.mode == 'train':
        trainer.fit(
            model=solver,
            train_dataloaders=dataloader['train'],
            val_dataloaders=dataloader['valid'],
            ckpt_path=cfg.load.ckpt_path if cfg.load.load_state else None
        )
    
    elif cfg.mode == 'test':
        trainer.test(
            model=solver,
            dataloaders=dataloader['test']
        )