import torch
from torch.utils import data

class ExampleDataset(data.Dataset):
    def __init__(self, cfg, phase):
        super(ExampleDataset, self).__init__()
        self.cfg = cfg
        self.phase = phase
        self.data = self._load_data()
        self.transform = self._get_transform()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        batch_dict = {}
        batch_dict['x'] = torch.randn(10)
        batch_dict['y'] = torch.randn(10)

        if self.transform is None:
            return batch_dict
        else:
            return self.transform(batch_dict)

    def _load_data(self):
        return range(10000)

    def _get_transform(self):
        # Get transform
        pass

def get_loader(cfg, phase):
    """
    Args:
        cfg: config file
        phase: ['train', 'valid', 'test']
    """

    dataset = ExampleDataset(cfg, phase)
    batch_size = cfg.data.batch_size
    num_workers = cfg.data.num_workers

    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if phase == "train" else False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader