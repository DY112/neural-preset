from PIL import Image
from pillow_lut import load_cube_file
import torch, glob
import os.path as osp
import numpy as np
from omegaconf import OmegaConf
import torchvision.transforms as T
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.distributed import DistributedSampler

class COCO(Dataset):
    def __init__(self,cfg,split,transform=None):
        self.cfg = cfg
        self.root = cfg.data.root
        self.split = split  # train / valid / test
        self.lut_root = cfg.data.lut_root

        self.transform = transform
        self.totensor = T.ToTensor()
        
        # prepare data list
        self.data = []
        self.data += glob.glob(osp.join(self.root, split, '*.jpg'))

        # prepare LUTs
        self.lut = glob.glob(self.lut_root + '/*.cube')
        self.lut = [load_cube_file(lut) for lut in self.lut]
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # return two type of augmented image, img_i & img_j
        item_dict = {}
        
        img = Image.open(self.data[idx]).convert('RGB')
        img = img.resize((self.cfg.data.size,self.cfg.data.size),Image.BICUBIC)

        # get two random LUTs
        random_lut_idx = np.random.randint(0,len(self.lut),2)
        
        lut_i = self.lut[random_lut_idx[0]]
        lut_j = self.lut[random_lut_idx[1]]

        # apply LUTs
        img_i = img.filter(lut_i)
        img_j = img.filter(lut_j)
        
        if self.transform is not None:
            img_i = self.transform(img_i)
            img_j = self.transform(img_j)

        img_i = self.totensor(img_i)
        img_j = self.totensor(img_j)

        item_dict['img_i'] = img_i
        item_dict['img_j'] = img_j

        return item_dict

def get_loader(cfg,phase):
    # Dataset
    dataset = COCO(cfg=cfg, split=phase)
    batch_size = cfg.data.batch_size
    num_workers = cfg.data.num_workers

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if phase == "train" else False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader

if __name__=='__main__':
    cfg = OmegaConf.merge(OmegaConf.load('../configs/default.yaml'),OmegaConf.load('../configs/neural_styler.yaml'),OmegaConf.from_cli())
    test_data = COCO(
        cfg=cfg,
        split='train'
    )
    test_loader = DataLoader(test_data,batch_size=10,shuffle=True)
    print("The number of data: ",len(test_data))
    print("The number of batch: ",len(test_loader))

    for item_dict in test_loader:
        print(item_dict["img_i"].shape)
        print(item_dict["img_j"].shape)
        input()