from datasets.coco import get_loader as get_coco_loader
from datasets.test_data import get_loader as get_test_loader

def get_loader(cfg, phase):
    if cfg.mode == 'train' and cfg.data.name == 'coco':
        return get_coco_loader(cfg, phase)
    elif cfg.mode == 'test':
        return get_test_loader(cfg, phase)
    else:
        raise NotImplementedError(f"Dataset {cfg.data.name} is not implemented")