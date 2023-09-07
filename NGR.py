from models.BasicModel import BasicModel
import torch
import numpy as np
from torch import nn
from utils.common_utils import gradient, soft_thresholding, psf2otf
from scipy.sparse.linalg import svds
from models.skip import skip


class NGR_net(nn.Module):
    def __init__(self, input_depth, output_depth, pad='reflection', tt=5, task='inpainting'):
        super().__init__()
        if task in ['inpainting', 'decloud']:
            upsample = 'nearest'
        else:
            upsample = 'bilinear'

        self.net1 = skip(input_depth, output_depth,  
            num_channels_down = [128]*tt,
            num_channels_up =   [128]*tt,
            num_channels_skip =    [4]*tt,  
            filter_size_up = 3, filter_size_down = 3,  filter_skip_size=1,
            upsample_mode=upsample, 
            need1x1_up=True,    # True for rgb, False for HSI (need1x1_up and need_sigmoid)
            need_sigmoid=False, need_bias=True, pad=pad, act_fun='LeakyReLU')
        self.net2 = skip(input_depth, output_depth,  
            num_channels_down = [128]*tt,
            num_channels_up =   [128]*tt,
            num_channels_skip =    [4]*tt,  
            filter_size_up = 3, filter_size_down = 3,  filter_skip_size=1,
            upsample_mode=upsample, 
            need1x1_up=True,
            need_sigmoid=False, need_bias=True, pad=pad, act_fun='LeakyReLU')
        self.net3 = skip(input_depth, output_depth,  
            num_channels_down = [128]*tt,
            num_channels_up =   [128]*tt,
            num_channels_skip =    [4]*tt,  
            filter_size_up = 3, filter_size_down = 3,  filter_skip_size=1,
            upsample_mode=upsample, 
            need1x1_up=True,
            need_sigmoid=False, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.tanh = nn.Tanh()


    def forward(self, Y):
        _, _, H, W = Y.shape
        G = []
        # G.append(self.tanh(self.net1(Y)[0, :])) # for video denoising
        # G.append(self.tanh(self.net2(Y)[0, :]))
        # G.append(self.tanh(self.net3(Y)[0, :]))
        G.append(self.net1(Y)[0, :])      # for inpainting
        G.append(self.net2(Y)[0, :])
        G.append(self.net3(Y)[0, :])

        return G   

class NGR(BasicModel):
    def __init__(self, obs, gt, **kwargs):
        super().__init__(obs, gt, **kwargs)
        self.method_name = 'NGR'
        params = kwargs.get('params')

        C, H, W = self.noisy.shape
        self.lmd = params['lmd']

        _, norm_two, _ = svds(obs.reshape(C, H*W), 1)
        norm_inf = np.linalg.norm(obs.flatten(), np.inf) / (self.lmd+1e-8)
        dual_norm = max(norm_two, norm_inf)
        self.mu = float(1. / dual_norm) * params['alpha']
        self.rho = params['rho']

        self.Gamma = self.noisy / float(dual_norm)
        self.T = np.ones(self.noisy.shape)*(np.abs(psf2otf(np.array([[1, -1]]), (H, W)))**2).reshape(1, H, W)
        self.T += np.ones(self.noisy.shape)*(np.abs(psf2otf(np.array([[1, -1]]).T, (H, W)))**2).reshape(1, H, W)
        self.T1 = np.ones(self.noisy.shape)*(np.abs(psf2otf(np.array([[1, -1]]).T, (C, W)))**2).reshape(C, 1, W)
        self.T = torch.from_numpy(self.T).cuda(self.gpu_num)
        self.T1 = torch.from_numpy(self.T1).cuda(self.gpu_num)
        if self.task in ['denoising']:
            self.G1 = gradient(self.noisy, 0)
            self.G2 = gradient(self.noisy, 1)
            self.G3 = gradient(self.noisy, 2)
            self.mu12 = self.mu * params['beta']
            self.mu3 = self.mu * params['gamma']
            self.max_mu12 = params['mu12_max']
            self.max_mu3 = params['mu3_max']
            self.mu_max = params['mu_max']
            self.Gamma1 = self.Gamma.clone()
            self.Gamma2 = self.Gamma.clone()
            self.Gamma3 = self.Gamma.clone()
            self.S = torch.zeros_like(self.noisy)
            self.out1 = self.noisy.clone()
            self.closure = self.closure_denoising
        elif self.task in ['inpainting', 'decloud']:
            self.omega = torch.ones_like(self.noisy)
            self.omega[self.noisy == 0] = 0
            self.K = torch.zeros_like(self.noisy)
            self.out1 = torch.rand_like(self.noisy)
            self.out1 = self.omega * self.noisy + (1 - self.omega) * self.out1
            self.closure = self.closure_inpainting
            self.mu_max = params['mu_max']
            if self.task == 'decloud':
                self.cloud_shape = kwargs.get('cloud_shape')
                self.closure = self.closure_decloud
            
        self.mode = kwargs.get('mode', 'M')

    def set_net(self):
        self.net = NGR_net(self.input_depth, self.noisy.shape[0]).cuda(self.gpu_num)

        
    def closure_denoising(self):
        if self.need_noise_reg is True:
            self.net_input = self.net_input_saved + torch.randn_like(self.net_input) * self.reg_noise_std

        F1, F2, F3 = self.net(self.net_input)

        ## updating Theta
        total_loss = torch.norm(F1-self.G1, 2)**2+torch.norm(F2-self.G2, 2)**2+torch.norm(F3-self.G3, 2)**2
        total_loss += self.mu12/2*torch.norm(gradient(self.out1, 0)-F1+self.Gamma1/self.mu12, 2)**2
        total_loss += self.mu12/2*torch.norm(gradient(self.out1, 1)-F2+self.Gamma2/self.mu12, 2)**2
        total_loss += self.mu3/2*torch.norm(gradient(self.out1, 2)-F3+self.Gamma3/self.mu3, 2)**2
        total_loss.backward()
        F1 = F1.detach(); F2 = F2.detach(); F3 = F3.detach()

        ## updating X
        numer = gradient(self.mu12*F1-self.Gamma1, 0, transpose=True)+gradient(self.mu12*F2-self.Gamma2, 1, transpose=True)+gradient(self.mu3*F3-self.Gamma3, 2, transpose=True)
        numer += self.mu*(self.noisy-self.S)+self.Gamma
        self.out1 = torch.real(torch.fft.ifftn(torch.fft.fftn(numer)/(self.mu+self.T*self.mu12+self.T1*self.mu3)))
        
        ## updating S
        if self.mode == 'M':
            self.S = soft_thresholding(self.noisy - self.out1 + self.Gamma / self.mu, self.lmd / self.mu)
        elif self.mode == 'G':
            self.S = (self.Gamma + self.mu * (self.noisy - self.out1)) / (2 * self.lmd + self.mu)

        ## updating multiplier
        self.Gamma1 += self.mu12 * (gradient(self.out1, 0) - F1)
        self.Gamma2 += self.mu12 * (gradient(self.out1, 1) - F2)
        self.Gamma3 += self.mu3 * (gradient(self.out1, 2) - F3)
        self.Gamma += self.mu * (self.noisy - self.out1 - self.S)
        self.mu = min(self.mu_max, self.rho * self.mu)
        self.mu12 = min(self.max_mu12, self.rho * self.mu12)
        self.mu3 = min(self.max_mu3, self.rho * self.mu3)

        if self.smoothing is True:
            if self.out is None:
                self.out = self.out1
            self.out = self.out * self.exp_weight + self.out1 * (1 - self.exp_weight)
        else:
            self.out = self.out1


    def closure_inpainting(self):
        if self.need_noise_reg is True:
            self.net_input = self.net_input_saved + torch.randn_like(self.net_input) * self.reg_noise_std
      
        F1, F2, F3 = self.net(self.net_input)

        ## updating Theta
        total_loss = 0.5*torch.norm(gradient(self.out1, 0)-F1, 2)**2 + 0.5*torch.norm(gradient(self.out1, 1)-F2, 2)**2 + self.lmd * 0.5*torch.norm(gradient(self.out1, 2)-F3, 2)**2
        total_loss.backward()
        F1 = F1.detach(); F2 = F2.detach(); F3 = F3.detach()

        ## updating X
        numer = gradient(F1, 0, transpose=True)+gradient(F2, 1, transpose=True)+gradient(self.lmd*F3, 2, transpose=True)
        numer += self.mu*(self.noisy-self.K)+self.Gamma
        self.out1 = torch.real(torch.fft.ifftn(torch.fft.fftn(numer)/(self.mu+self.T+self.lmd*self.T1)))
        
        ## updating K
        self.K = self.noisy - self.out1 + self.Gamma / self.mu
        self.K = self.K * (1 - self.omega)

        ## updating multiplier
        self.Gamma += self.mu * (self.noisy - self.out1 - self.K)
        self.mu = min(self.mu_max, self.rho * self.mu)

        if self.smoothing is True:
            if self.out is None:
                self.out = self.out1
            self.out = self.out * self.exp_weight + self.out1 * (1 - self.exp_weight)
            self.out = self.omega * self.noisy + (1 - self.omega) * self.out
        else:
            self.out = self.omega * self.noisy + (1 - self.omega) * self.out1

    def closure_decloud(self):
        if self.need_noise_reg is True:
            self.net_input = self.net_input_saved + torch.randn_like(self.net_input) * self.reg_noise_std
        
        F1, F2, F3 = self.net(self.net_input)
        
        ## updating Theta
        total_loss = 0.5*torch.norm(gradient(self.out1, 0)-F1, 2)**2 + 0.5*torch.norm(gradient(self.out1, 1)-F2, 2)**2 + self.lmd * 0.5*torch.norm(gradient(self.out1, 2)-F3, 2)**2
        total_loss.backward()
        F1 = F1.detach(); F2 = F2.detach(); F3 = F3.detach()

        ## updating X
        numer = gradient(F1, 0, transpose=True)+gradient(F2, 1, transpose=True)+gradient(self.lmd*F3, 2, transpose=True)
        numer += self.mu*(self.noisy-self.K)+self.Gamma
        self.out1 = torch.real(torch.fft.ifftn(torch.fft.fftn(numer)/(self.mu+self.T+self.lmd*self.T1)))
        
        ## updating K
        self.K = self.noisy - self.out1 + self.Gamma / self.mu
        self.K = self.K * (1 - self.omega)

        ## updating multiplier
        self.Gamma += self.mu * (self.noisy - self.out1 - self.K)
        self.mu = min(self.mu_max, self.rho * self.mu)

        nband = self.cloud_shape[2]
        out = self.out1[:nband, :, :]
        noisy = self.noisy[:nband, :, :]
        mask = self.mask[:nband, :, :]

        if self.smoothing is True:
            if self.out is None:
                self.out = out
            self.out = self.out * self.exp_weight + out * (1 - self.exp_weight)
            self.out = mask * noisy + (1 - mask) * self.out
        else:
            self.out = mask * noisy + (1 - mask) * out
