'''
RGB data inpainting
'''
from PIL import Image
import numpy as np
from NGR import NGR

### import config file
from config.ngr_inpainting_rgb import *

### set data path
data_name = '148089'                    # expected result: PSNR=23.98, SSIM=0.765
gt_path = f'./data/{data_name}.png'
obs_path = f'./data/{data_name}_sr0.1.png'

### save path
save_path = './result'

gpu_num = 0


if __name__ == '__main__':
    ### prepare data
    gt = np.array(Image.open(gt_path)).transpose(2, 0, 1)/255
    obs = np.array(Image.open(obs_path)).transpose(2, 0, 1)/255
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
              smoothing=smoothing,
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
    ngr.save(save_type='png')