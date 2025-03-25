import torch, os, sys
import torch.nn as nn
sys.path.insert(1,os.path.abspath('..'))
from models.efficientnet import EfficientNet

class neural_styler(nn.Module):
    def __init__(self, cfg):
        super(neural_styler, self).__init__()
        self.cfg = cfg
        self.k = cfg.model.k
        self.style_encoder = EfficientNet.from_name(f'{cfg.model.style_encoder}', num_classes=(self.k ** 2)*2)
        self.transform_p = nn.Parameter(torch.rand(3, self.k))
        self.transform_q = nn.Parameter(torch.rand(self.k, 3))

    def get_r_and_d(self, style):
        """
        Args:
            style: style image (B x 3 x 256 x 256)
        """
        tmp_matrix = self.style_encoder(style)  # B x (k^2)*2
        
        # split tmp_matrix into r and d (B x k x k matrix each)
        r = tmp_matrix[:, :self.k**2].reshape(-1, self.k, self.k)
        d = tmp_matrix[:, self.k**2:].reshape(-1, self.k, self.k)

        return r, d
        

    def forward(self, content, style):
        """
        Args:
            content: content image (B x 3 x H x W)
            style: style image (B x 3 x 256 x 256)
        """

        # batch-specific style whitening/coloring matrix (B x k x k matrix)
        r_c, d_c = self.get_r_and_d(content)
        r_s, d_s = self.get_r_and_d(style)

        # calculate whitening transform matrix & coloring transform matrix
        transform_z = torch.matmul(torch.matmul(self.transform_p, d_c), self.transform_q)   # B x 3 x 3
        transform_y = torch.matmul(torch.matmul(self.transform_p, r_s), self.transform_q)   # B x 3 x 3

        # apply whitening transform matrix & coloring transform matrix to content image
        whitened_content = torch.bmm(transform_z, content.reshape(-1, 3, content.shape[2]*content.shape[3]))  # B x 3 x (H*W)
        colored_content = torch.bmm(transform_y, whitened_content).reshape(-1, 3, content.shape[2], content.shape[3])  # B x 3 x H x W

        return whitened_content.reshape(-1, 3, content.shape[2], content.shape[3]), colored_content