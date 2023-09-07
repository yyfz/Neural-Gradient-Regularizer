import scipy.io as scio
from .noise_utils import *
import torch
import numpy as np
import os 
from scipy.io import * 

def prepare_true_data_from_mat(path, channels=None, key=None):
    noisy = load_from_mat(path, key)
    noisy = noisy.transpose(2, 0, 1) / 255
    noisy = torch.from_numpy(crop(noisy, 32))
    if channels == None:
        channels = int(np.ceil(noisy.shape[0] / 32) * 32)
    if noisy.shape[0] < channels:
        temp = torch.zeros([channels, noisy.shape[1], noisy.shape[2]])
        temp[:noisy.shape[0], :, :] = noisy
        for i in range(channels - noisy.shape[0]):
            temp[noisy.shape[0] + i, :, :] = noisy[noisy.shape[0] - 1 - i, :, :]
        noisy = temp
    else:
        noisy = noisy[0:channels, :, :]
    return noisy

def load_from_mat(path=r'./data/WDC.mat', key=None):
    # matdata = scio.loadmat(path)
    # GT = matdata['GT']
    # input = matdata['Input']
    # GT = GT.transpose(2, 0, 1)
    # input = input.transpose(2, 0, 1)
    # input = crop(input, 32)
    # GT = crop(GT, 32)

    # channels = int(np.ceil(GT.shape[0] / 32) * 32)

    # if GT.shape[0] < channels:
    #     temp = np.zeros((channels, GT.shape[1], GT.shape[2]))
    #     temp[:GT.shape[0], :, :] = GT
    #     temp[GT.shape[0]:, :, :] = GT[GT.shape[0] - 1, :, :]
    #     GT = temp / 255
    # if input.shape[0] < channels:
    #     temp = np.zeros((channels, input.shape[1], input.shape[2]))
    #     temp[:input.shape[0], :, :] = input
    #     temp[input.shape[0]:, :, :] = input[input.shape[0] - 1, :, :]
    #     input = temp / 255
    
    # return torch.from_numpy(GT), torch.from_numpy(input)
    matdata = scio.loadmat(path)
    if key != None:
        return matdata[key]
    else:
        return matdata['data']
        

def load_from_pth(path):
    return torch.load(path)

def prepare_mask(image):
    mask = torch.ones(image.shape)
    for i in range(mask.shape[2]):
        for j in range(mask.shape[0]):
            if image[j,:,i].sum() == 0:
                mask[j,:,i] = 0
    for i in range(mask.shape[1]):
        for j in range(mask.shape[0]):
            if image[j, i, :].sum() == 0:
                mask[j, i, :] = 0
    return mask

def crop(img, d):
    size = (
        img.shape[0],
        img.shape[1] - img.shape[1] % d,
        img.shape[2] - img.shape[2] % d)
    
    img = img[
        :, 
        int((img.shape[1] - size[1])/2):int((img.shape[1] + size[1])/2),
        int((img.shape[2] - size[2])/2):int((img.shape[2] + size[2])/2)]
    
    return img

def prepare_data_for_denoising(path, channels=None, key=None, noise_case=1):
    img_var = torch.from_numpy(load_from_mat(path, key))
    if torch.max(img_var) > 1:
        img_var = img_var / 255
    img_var = img_var.permute(2, 0, 1)
    img_var = crop(img_var, 32)
    if channels == None:
        channels = int(np.ceil(img_var.shape[0] / 32) * 32)
    if img_var.shape[0] < channels:
        temp = torch.zeros([channels, img_var.shape[1], img_var.shape[2]])
        temp[:img_var.shape[0], :, :] = img_var
        for i in range(channels - img_var.shape[0]):
            temp[img_var.shape[0] - 1 + i, :, :] = img_var[img_var.shape[0] - 1 - i, :, :]
        img_var = temp
    else:
        img_var = img_var[0:channels, :, :]

    img_noise_var = img_var.clone()
    img_noise_var = torch.clip(add_noise(img_noise_var, noise_case), 0, 1)
    
    return img_noise_var, img_var 

def prepare_data_for_inpainting(path, channels=None, key=None, ratio=0.9):
    img_var = torch.from_numpy(load_from_mat(path, key))
    if torch.max(img_var) > 1:
        img_var = img_var / 255
    img_var = img_var.permute(2, 0, 1)
    img_var = crop(img_var, 32)
    if channels == None:
        channels = int(np.ceil(img_var.shape[0] / 32) * 32)
    if img_var.shape[0] < channels:
        temp = torch.zeros([channels, img_var.shape[1], img_var.shape[2]])
        temp[:img_var.shape[0], :, :] = img_var
        for i in range(channels - img_var.shape[0]):
            temp[img_var.shape[0] - 1 + i, :, :] = img_var[img_var.shape[0] - 1 - i, :, :]
        img_var = temp
    else:
        img_var = img_var[0:channels, :, :]
    
    nband, hh, ww = img_var.shape
    mask = torch.zeros(img_var.shape)
    assert ratio > 0 and ratio < 1, "ratio should be in (0, 1)"
    element_num = nband * hh * ww
    obs_num = int(element_num * (1 - ratio))
    obs_ind = np.unravel_index(np.random.choice(element_num, obs_num, replace=False), shape=[nband, hh, ww], order='F')
    mask[obs_ind] = 1

    img_noisy_var = img_var * mask

    return img_noisy_var, img_var, mask