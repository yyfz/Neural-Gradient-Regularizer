'''
Multi-temporal MSIs decloud
'''
import numpy as np
from NGR import NGR
from scipy.io import loadmat

### import config file
from config.ngr_decloud import *

### set data path
data_name = 'Uzbekistan17'                               # expected result: PSNR=35.98, SSIM=0.933
gt_path = f'./data/{data_name}_gt.mat'
obs_path = f'./data/{data_name}_large.mat' 

### save path
save_path = './result'

gpu_num = 0

if __name__ == '__main__':
    ### prepare data
    gt = loadmat(gt_path)['gt'].transpose(2, 0, 1)
    mat = loadmat(obs_path)
    obs = mat['obs'].transpose(2, 0, 1)
    mask = mat['mask'].transpose(2, 0, 1)
    cloud_shape = mat['shape'][0]

    ### init NGR
    ngr = NGR(obs.astype(np.float32), 
              gt[:cloud_shape[2], :, :],
              mask=mask,
              seed=seed,
              cloud_shape=cloud_shape,
              params=hyperparameters, 
              save_path=save_path, 
              show_image=show_image, 
              smoothing=smoothing,
              gpu_num=gpu_num, 
              iterations=epoches, 
              show_every=show_every, 
              lr=lr,
              weight_decay=weight_decay, 
              exp_weight=exp_weight, 
              data_name=data_name, 
              task=task, input_type=input_type, 
              need_noise_reg=need_noise_reg, 
              reg_noise_std=reg_noise_std, 
              input_depth=input_depth,
              metrics=metrics)

    ### train
    ngr.train()

    ### save result
    ngr.save()