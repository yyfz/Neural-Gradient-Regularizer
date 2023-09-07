'''
These are recommended parameters for HSI denoising.
'''

### Task
task = 'denoising'
noise_type = 'M'    # Mixed noise
# noise_type = 'G'    # Gaussian noise

### Hyperparameters, mu_max, mu12_max and mu3_max can be tuned
if noise_type == 'M':
    hyperparameters = dict(
        rho = 1.2,
        lmd = 3, 
        mu_max = 0.16,
        mu12_max = 0.16,
        mu3_max = 0.6,
        alpha = 1,
        beta = 2,
        gamma = 150
    ) 
if noise_type == 'G':
    hyperparameters = dict(
        rho = 1.05,
        lmd = 4, 
        mu_max = 0.16,
        mu12_max = 0.16,
        mu3_max = 0.4,
        alpha = 1,
        beta = 150,
        gamma = 150
    ) 


### Optimizer
lr = 0.01
weight_decay = 5

### Train config
epoches = 4000
need_noise_reg = True
reg_noise_std = 0.03
exp_weight = 0.99
smoothing = True
input_type = 'random'
input_depth = 'bands'


### Display config
metrics = ['psnr', 'ssim', 'sam', 'ergas']
show_every = 200
show_image = False

seed = 2333