import torch
import torch.nn as nn
import numpy as np

from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn


def PSNR(X, Y):
    '''
    X and Y's shape: C x W x H
    X and Y are np.array
    '''
    X = np.round(X * 255)
    Y = np.round(Y * 255)
    band_num = X.shape[0]
    psnr = 0
    for i in range(band_num):
        psnr += 10 * np.log10(255**2 / np.mean((X[i, :, :] - Y[i, :, :])**2))
    psnr = psnr / band_num
    return psnr

def PSNR_gpu(X, Y):
    '''
    X and Y's shape: C x W x H
    X and Y are torch.tensor
    '''
    X = torch.round(X * 255)
    Y = torch.round(Y * 255)
    band_num = X.shape[0]
    psnr = 0
    for i in range(band_num):
        psnr += 10 * torch.log10(255**2 / torch.mean((X[i, :, :] - Y[i, :, :])**2))
    psnr = psnr / band_num
    return psnr

def SSIM(x,y):
    '''
    X and Y's shape: C x W x H
    X and Y are np.array
    '''
    ssim_val = 0.
    x = np.round(255.*x).astype(np.uint8)
    y = np.round(255.*y).astype(np.uint8)
    for i in range(x.shape[0]):
        ssim_val = ssim_val + ssim_fn(x[i, :, :], y[i, :, :])
    ssim_val = ssim_val / x.shape[0]
    return ssim_val  

def SAM(x,y):
    '''
    X and Y's shape: C x W x H
    X and Y are np.array
    '''
    x = x.transpose(1, 2, 0)
    y = y.transpose(1, 2, 0)
    HH,WW,CC = x.shape
    x = x.reshape(HH*WW,CC)
    y = y.reshape(HH*WW,CC)
    sam = np.sum(x*y,axis=-1) / np.sqrt( (np.sum(x*x,axis=-1)+1e-6) * (np.sum(y*y,axis=-1)+1e-6) )
    sam = np.clip(sam, 0., 1.)
    sam = np.arccos(sam)
    sam = np.mean(sam)
    sam = 180*sam/np.pi
    return sam

def ERGAS(img_fake, img_real, scale=1):
    '''
    X and Y's shape: C x W x H
    X and Y are np.array
    '''
    img_fake = img_fake.transpose(1, 2, 0)
    img_real = img_real.transpose(1, 2 ,0)
    """ERGAS for 2D (H, W) or 3D (H, W, C) image; uint or float [0, 1].
    scale = spatial resolution of HRMS / spatial resolution of MUL, default 4."""
    if not img_fake.shape == img_real.shape:
        raise ValueError('Input images must have the same dimensions.')
    img_fake_ = img_fake.astype(np.float64)
    img_real_ = img_real.astype(np.float64)
    if img_fake_.ndim == 2:
        mean_real = img_real_.mean()
        mse = np.mean((img_fake_ - img_real_)**2)
        return 100 / scale * np.sqrt(mse / (mean_real**2 + np.finfo(np.float64).eps))
    elif img_fake_.ndim == 3:
        means_real = img_real_.reshape(-1, img_real_.shape[2]).mean(axis=0)
        mses = ((img_fake_ - img_real_)**2).reshape(-1, img_fake_.shape[2]).mean(axis=0)
        return 100 / scale * np.sqrt((mses / (means_real**2 + np.finfo(np.float64).eps)).mean())
    else:
        raise ValueError('Wrong input image dimensions.')
    
def FBC(img_out, GT, R=0, mode='torch'):
    '''
    Frequency-Band Correspondence Metric
    H = fft(f) / fft(y)
    Shi, Zenglin, et al. "On measuring and controlling the spectral bias of the deep image prior." International Journal of Computer Vision 130.4 (2022): 885-908.
    '''
    if mode == 'torch':
        _, rows, cols = GT.shape
        crow,ccol = int(rows/2), int(cols/2)
        mask = torch.zeros_like(GT)
        mask[crow-(R+1)*20:crow+(R+1)*20, ccol-(R+1)*20:ccol+(R+1)*20] = 1
        mask[crow-R*20:crow+R*20, ccol-R*20:ccol+R*20] = 0
        return torch.abs(torch.mean(torch.fft.fftshift(torch.fft.fftn(img_out))*mask/torch.fft.fftshift(torch.fft.fftn(GT))))
    elif mode == 'np':
        _, rows, cols = GT.shape
        crow,ccol = int(rows/2), int(cols/2)
        mask = np.zeros_like(GT)
        mask[crow-(R+1)*20:crow+(R+1)*20, ccol-(R+1)*20:ccol+(R+1)*20] = 1
        mask[crow-R*20:crow+R*20, ccol-R*20:ccol+R*20] = 0
        return np.abs(np.mean(np.fft.fftn(img_out) / np.fft.fftn(GT)))
    
def AUC(test_target, output, seg_idx=None):
    C, H, W = test_target.shape
    y = test_target.reshape(C, H*W)
    x = output.reshape(C, H*W)
    x = np.round(x*255)
    idx = np.argsort(x, axis=-1)
    range_ = np.arange(H*W) + 1
    M = np.sum(y==1, axis=-1, keepdims=True)
    N = H*W - M
    res = 0
    cnt = C
    if seg_idx is None:
        for i in range(C):
            if M[i] == 0:
                cnt -= 1
                continue
            res += (np.sum(range_*(y[i][idx[i]]==1)) - (M[i]+1)*M[i]/2)/(M[i]*N[i])
        return float(res / cnt)
    else:
        res += (np.sum(range_*(y[seg_idx][idx[seg_idx]]==1)) - (M[seg_idx]+1)*M[seg_idx]/2)/(M[seg_idx]*N[seg_idx])
        return float(res)
    
def AUC_gpu(test_target, output, seg_idx=None):
    C, H, W = test_target.shape
    y = test_target.reshape(C, H*W)
    x = output.reshape(C, H*W)
    x = torch.round(x*255)
    idx = torch.argsort(x, axis=-1)
    range_ = torch.arange(H*W) + 1
    range_ = range_.to(test_target.device)
    M = torch.sum(y==1, axis=-1, keepdims=True)
    N = H*W - M
    res = 0
    cnt = C
    if seg_idx is None:
        for i in range(C):
            if M[i] == 0:
                cnt -= 1
                continue
            res += (torch.sum(range_*(y[i][idx[i]]==1)) - (M[i]+1)*M[i]/2)/(M[i]*N[i])
        return float(res / cnt)
    else:
        res += (torch.sum(range_*(y[seg_idx][idx[seg_idx]]==1)) - (M[seg_idx]+1)*M[seg_idx]/2)/(M[seg_idx]*N[seg_idx])
        return float(res)

def _error(actual: np.ndarray, predicted: np.ndarray):
    """ Simple error """
    return actual - predicted

def mse(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Squared Error """
    return np.mean(np.square(_error(actual, predicted)))

def rmse(actual: np.ndarray, predicted: np.ndarray):
    """ Root Mean Squared Error """
    return np.sqrt(mse(actual, predicted))

def nrmse(actual: np.ndarray, predicted: np.ndarray):
    """ Normalized Root Mean Squared Error """
    return rmse(actual, predicted) / (actual.max() - actual.min())


