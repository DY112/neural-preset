from datasets.coco import get_loader as get_coco_loader

def get_loader(cfg, phase):
    if cfg.data.name == 'coco':
        return get_coco_loader(cfg, phase)
    else:
        raise NotImplementedError