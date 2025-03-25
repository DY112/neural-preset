import torch
from torchvision import datasets, transforms

def get_loader(cfg, phase):
    if phase == 'train':
        dataset = datasets.MNIST(root=cfg.data.root, train=True, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))
                                 ]))
    else:
        dataset = datasets.MNIST(root=cfg.data.root, train=False, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))
                                 ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.data.batch_size, shuffle=True if phase == 'train' else False, num_workers=cfg.data.num_workers)
    return dataloader