'''
These are recommended parameters for multichannel (MC) data inpainting.
'''

### Task
task = 'inpainting'

### Hyperparameters, lmd and mu_max can be tuned
hyperparameters = dict(
    lmd = 5,
    mu_max = 50,
    alpha = 0.1,
    rho = 1.2
)

### Optimizer
lr = 0.01
weight_decay = 0

### Train config
epoches = 15000
need_noise_reg = True
reg_noise_std = 0.03
exp_weight = 0.99
smoothing = False
input_type = 'random'
input_depth = 'bands'


### Display config
metrics = ['psnr', 'ssim', 'sam', 'ergas']
show_every = 200
show_image = False

seed = 2333