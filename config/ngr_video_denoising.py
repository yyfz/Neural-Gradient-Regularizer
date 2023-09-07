'''
These are recommended parameters for video denoising.
'''

### Task
task = 'denoising'

### Hyperparameters, mu_max, mu12_max and mu3_max can be tuned
hyperparameters = dict(
    rho = 1.05,
    lmd = 2, 
    mu_max = 0.16,
    mu12_max = 0.16,
    mu3_max = 0.34,
    alpha = 0.02,
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
noise_type = 'G'    # Gaussian noise


### Display config
metrics = ['psnr', 'ssim']
show_every = 200
show_image = False

seed = 2333