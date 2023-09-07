'''
Multichannel data inpainting
'''
import numpy as np
from NGR import NGR
from scipy.io import loadmat

### import config file
from config.ngr_inpainting_mc import *

### set data path
## video
data_name = 'claire'                                  # expected result: PSNR=41.05, SSIM=0.990
gt_path = f'./data/{data_name}_gt.mat'
obs_path = f'./data/{data_name}_sr0.15.mat'
## HSI
# data_name = 'PA'                                        # expected result: PSNR=33.72, SSIM=0.961
# gt_path = f'./data/{data_name}_gt.mat'
# obs_path = f'./data/{data_name}_deadline.mat' 

### save path
save_path = './result'

gpu_num = 1


if __name__ == '__main__':
    ### prepare data
    gt = loadmat(gt_path)['gt'].transpose(2, 0, 1)
    obs = loadmat(obs_path)['obs'].transpose(2, 0, 1)
    mask = np.ones_like(obs); mask[obs == 0] = 0

    ### init NGR
    ngr = NGR(obs.astype(np.float32), 
              gt.astype(np.float32),
              mask=mask,
              seed=seed,
              params=hyperparameters, 
              save_path=save_path, 
              show_image=show_image, 
              gpu_num=gpu_num, 
              iterations=epoches, 
              show_every=show_every, 
              lr=lr,
              weight_decay=weight_decay, 
              exp_weight=exp_weight, 
              smoothing=smoothing,
              data_name=data_name, 
              task=task, 
              input_type=input_type, 
              need_noise_reg=need_noise_reg, 
              reg_noise_std=reg_noise_std, 
              input_depth=input_depth,
              metrics=metrics)

    ### train
    ngr.train()

    ### save result
    ngr.save()