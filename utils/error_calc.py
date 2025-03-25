import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim

def get_MAE(pred,gt,tensor_type,camera=None,mask=None,reduce_batch=True,get_errormap=False):
    """
    pred : (b,c,w,h)
    gt : (b,c,w,h)
    """
    if tensor_type == "rgb":
        if camera == 'galaxy':
            pred = torch.clamp(pred, 0, 1023)
            gt = torch.clamp(gt, 0, 1023)
        elif camera == 'sony' or camera == 'nikon':
            pred = torch.clamp(pred, 0, 16383)
            gt = torch.clamp(gt, 0, 16383)
    elif tensor_type == "illum":
        if pred.size(1) == 2:   # illum type : [R,B]
            ones = torch.ones_like(pred[:,0:1,:,:])
            pred = torch.cat([pred[:,:1,:,:],ones,pred[:,1:,:,:]],dim=1)
        if gt.size(1) == 2:
            ones = torch.ones_like(gt[:,0:1,:,:])
            gt = torch.cat([gt[:,:1,:,:],ones,gt[:,1:,:,:]],dim=1)
    else:
        raise NotImplementedError

    cos_similarity = F.cosine_similarity(pred+1e-4,gt+1e-4,dim=1)
    cos_similarity = torch.clamp(cos_similarity, -1, 1)
    rad = torch.acos(cos_similarity)
    ang_error = torch.rad2deg(rad)      # [B,W,H] - 1 ang error value per pixel

    if mask is not None:
        # mask ang_error where mask is 0
        mask = mask.squeeze(1)
        ang_error = ang_error * mask
    
    if reduce_batch:
        mean_angular_error = ang_error.sum() / mask.sum()
    else:
        mean_angular_error = ang_error.sum(dim=(1,2)) / mask.sum(dim=(1,2))
    
    if get_errormap:
        # return errormap & MAE tuple
        return ang_error, mean_angular_error
    else:
        return mean_angular_error

def get_PSNR(pred, gt, white_level):
    """
    pred & gt   : (b,c,h,w) numpy array 3 channel RGB
    returns     : average PSNR of two images
    """
    if white_level != None:
        pred = torch.clamp(pred,0,white_level)
        gt = torch.clamp(gt,0,white_level)

    mse = torch.mean((pred-gt)**2)
    psnr = 20 * torch.log10(white_level / torch.sqrt(mse))

    # pred_np = pred.cpu().numpy()
    # gt_np = gt.cpu().numpy()

    # psnr_cv = cv2.PSNR(pred_np,gt_np,white_level)

    return psnr

def get_SSIM(pred, GT, white_level):
    """
    pred & GT   : (h,w,c) numpy array 3 channel RGB

    returns     : average PSNR of two images
    """
    # if pred & GT is tortch tensor, convert to numpy array
    if type(pred) == torch.Tensor:
        pred = pred.cpu().numpy()
    if type(GT) == torch.Tensor:
        GT = GT.cpu().numpy()

    if white_level != None:
        pred = np.clip(pred, 0, white_level)
        GT = np.clip(GT, 0, white_level)

    return ssim(pred, GT, multichannel=True, data_range=white_level)