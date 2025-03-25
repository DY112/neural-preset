from datasets.example_dataset import get_loader as get_example_loader
from datasets.mnist import get_loader as get_mnist_loader

def get_loader(cfg, phase):
    if cfg.data.name == 'example_dataset':
        return get_example_loader(cfg, phase)
    elif cfg.data.name == 'mnist':
        return get_mnist_loader(cfg, phase)
    else:
        raise NotImplementedError