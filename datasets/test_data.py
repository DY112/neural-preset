import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import itertools

class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        # load image files
        self.image_files = [f for f in os.listdir(root_dir) 
                          if f.endswith('.jpg') and f[:-4].isdigit()]
        self.image_files.sort(key=lambda x: int(x[:-4]))
        
        # generate all possible image pairs
        self.pairs = list(itertools.permutations(range(len(self.image_files)), 2))
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        
        # load image
        img_i_path = os.path.join(self.root_dir, self.image_files[i])
        img_j_path = os.path.join(self.root_dir, self.image_files[j])
        
        img_i = Image.open(img_i_path).convert('RGB')
        img_j = Image.open(img_j_path).convert('RGB')
        
        # apply transform
        if self.transform:
            img_i = self.transform(img_i)
            img_j = self.transform(img_j)
        
        return {
            'img_i': img_i,
            'img_j': img_j
        }

def get_loader(cfg, phase):
    if phase != 'test':
        raise ValueError("TestDataset is only available for test phase")
    
    dataset = TestDataset(root_dir=cfg.test.root)
    loader = DataLoader(
        dataset,
        batch_size=cfg.test.batch_size,
        shuffle=False,
        num_workers=cfg.test.num_workers,
        pin_memory=True
    )
    return loader 