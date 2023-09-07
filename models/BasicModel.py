import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import os
from utils.metrics import *
from scipy.io import savemat
from .common import set_input_inpainting
import numpy as np
import time
from utils.common_utils import setup_seed
from PIL import Image

class BasicModel:
    def __init__(self, obs, gt=None, **kwargs):
        setup_seed(kwargs.get('seed', 2333))
        self.noisy = torch.from_numpy(obs).cuda(kwargs.get('gpu_num', 0))
        if gt is not None:
            self.GT = torch.from_numpy(gt).cuda(kwargs.get('gpu_num', 0))
        self.notes = kwargs.get('notes')
        self.input_type = kwargs.get('input_type', 'random')
        self.lr = kwargs.get('lr', 0.01)
        self.smoothing = kwargs.get('smoothing', True)
        self.exp_weight = kwargs.get('exp_weight', 0.99)
        self.learn_with_input = kwargs.get('learn_with_input', False)
        self.need_noise_reg = kwargs.get('need_noise_reg', True)
        self.reg_noise_std = kwargs.get('reg_noise_std', 0.03)
        self.show_every = kwargs.get('show_every', 200)
        self.show_image = kwargs.get('show_image', False)
        self.iterations = kwargs.get('iterations', 6000)
        self.need_mask = kwargs.get('need_mask', False)
        self.gpu_num = kwargs.get('gpu_num', 0)
        if kwargs.get('mask') is not None:
            self.mask = torch.from_numpy(kwargs.get('mask')).cuda(self.gpu_num)
        torch.cuda.set_device(self.gpu_num)
        self.data_name = kwargs.get('data_name')
        self.method_name = kwargs.get('method_name')
        self.noise_type = kwargs.get('noise_type')
        self.task = kwargs.get('task', 'denoising')
        save_path = kwargs.get('save_path')
        if save_path is not None:
            save_path = os.path.join(save_path, self.task) if self.task is not None else save_path
            save_path = os.path.join(save_path, self.data_name) if self.data_name is not None else save_path
            save_path = os.path.join(save_path, self.method_name) if self.method_name is not None else save_path
            os.makedirs(save_path, exist_ok=True)
            self.save_path = save_path
        else:
            self.save_path = None
        self.net = None
        self.display_band_list = kwargs.get('display_band_list', [0, 1, 2])
        self.display_type = kwargs.get('display_type', 'gray')
        self.weight_decay = kwargs.get('weight_decay')
        self.flag = False
        self.stop = False
        if self.task == 'extraction':
            self.seg_index = kwargs.get('seg_index')
            self.metrics = ['auc']
        else:
            self.metrics = kwargs.get('metrics', ['psnr', 'ssim'])
        if self.input_type == 'noisy':
            self.input_depth = self.noisy.shape[0]
        else:
            if kwargs.get('input_depth') == 'bands':
                self.input_depth = self.noisy.shape[0]
            else:
                self.input_depth = kwargs.get('input_depth', 32)
        
    def print_start_info(self):
        log = f"{'-' * 18} CONFIG {'-' * 18}\n"
        if self.data_name is not None:
            log += f'Data name: {self.data_name}\n'
        if self.task is not None:
            log += f'Task: {self.task}\n'
        if self.noise_type is not None:
            log += f'Degradation type: {self.noise_type}\n'
        if self.method_name is not None:
            log += f'Method: {self.method_name}\n'
        log += f'Learning rate: {self.lr}\n'
        if self.weight_decay is not None:
            log += f'Weight decay: {self.weight_decay}\n'
        log += f'Epoches: {self.iterations}\nInput type: {self.input_type}\n'
        if self.input_type == 'random':
            log += f'Input depth: {self.input_depth}\n'
        log += f'Smoothing: {self.smoothing}\nNoise regularization: {self.need_noise_reg}\n'
        if self.need_noise_reg:
            log += f'Std of noise regularization: {self.reg_noise_std}\n'
        log += f'GPU number: {self.gpu_num}\n'
        
        n_params = self.get_parameter_number()
        log += 'Network parameters: %d\n'%(n_params['Total'])
        log += 'Trainable network parameters: %d\n'%(n_params['Trainable'])
        log += '-' * 45 + '\n'
        self.start_time = time.time()
        log += f'#### Run at {time.ctime(self.start_time)} ####\n'
        if self.notes is not None:
            log += f"{'-' * 19} NOTES {'-' * 19}\n"
            log += self.notes
            log += '\n'
        log += '-' * 45 + '\n'
        if self.save_path is not None:
            print(log, file=self.file, end='')
        print(log, end='')

    def print_end_info(self):
        self.end_time = time.time()
        log = '-' * 45 + '\n'
        log += f'#### End at {time.ctime(self.end_time)} ####\n'
        log += '-' * 45 + '\n'
        log += f'Elapsed time: {self.end_time - self.start_time: .2f} s\n'
        if self.GT is not None:
            self.best_metrics = self.evaluate(self.best_out.cpu().numpy(), self.GT.cpu().numpy(), metrics=self.metrics)
            self.best_metrics['iter'] = self.best_iter
            log += f'Best output was at epoch {self.best_iter}:\n'
            if 'psnr' in self.best_metrics:
                psnr = self.best_metrics['psnr']
                log += f'PSNR: {psnr: .6f}\n'
            if 'ssim' in self.best_metrics:
                ssim = self.best_metrics['ssim']
                log += f'SSIM: {ssim: .6f}\n'
            if 'sam' in self.best_metrics:
                sam = self.best_metrics['sam']
                log += f'SAM: {sam: .6f}\n'
            if 'ergas' in self.best_metrics:
                ergas = self.best_metrics['ergas']
                log += f'ERGAS: {ergas: .6f}\n'
            if 'nrmse' in self.best_metrics:
                nrmse = self.best_metrics['nrmse']
                log += f'NRMSE: {nrmse: .6f}\n'
            if 'auc' in self.best_metrics:
                auc = self.best_metrics['auc']
                log += f'AUC: {auc: .6f}\n'
        log += '-' * 45 + '\n'
        if self.save_path is not None:
            print(log, file=self.file, end='')
        print(log, end='')       

    def init_input(self):
        if self.input_type == 'random':
            if self.task == 'super-resolution':
                self.net_input = torch.rand((self.input_depth, self.noisy.shape[1]*self.factor, self.noisy.shape[2]*self.factor)) * 0.1
            else:
                self.net_input = torch.rand((self.input_depth, self.noisy.shape[1], self.noisy.shape[2])) * 0.1
        elif self.input_type == 'noisy':
            if self.task == 'super-resolution':
                self.net_input = nn.UpsamplingBilinear2d(scale_factor=self.factor)(self.noisy.clone()[None, :])[0, :]
            else:
                self.net_input = self.noisy.clone()
        elif self.input_type == 'custom':
            self.net_input = self.set_input()
        elif self.input_type == 'inpainting_noisy':
            self.net_input = torch.from_numpy(set_input_inpainting(self.noisy.cpu().numpy())).cuda(self.gpu_num)
        # if isinstance(self.net_input, list):
        #     self.net_input = [x[None, :].cuda(self.gpu_num) for x in self.net_input]
        # else:
        self.net_input = self.net_input[None, :].cuda(self.gpu_num)

        self.net_input_saved = self.net_input.clone()

        if self.need_mask is True:
            if not hasattr(self, 'mask'):
                self.mask = torch.ones(self.noisy.shape).cuda(self.gpu_num)
                B, M, N = self.noisy.shape
                if self.task == 'denoising':
                    for i in range(N):
                        for j in range(B):
                            if self.noisy[j,:,i].sum() == 0:
                                self.mask[j,:,i] = 0
                    for i in range(M):
                        for j in range(B):
                            if self.noisy[j,i,:].sum() == 0:
                                self.mask[j,i,:] = 0
                else:
                    self.mask[self.noisy == 0] = 0

    def get_parameter_number(self):
        total_num = sum(p.numel() for p in self.net.parameters())
        trainable_num = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    def get_params(self):
        if self.net is None:
            self.set_net()
        self.params = [x for x in self.net.parameters()]
        if self.learn_with_input is True:
            self.net_input.requires_grad = True
            self.params += [self.net_input]

    def optimize(self):
        if self.task == 'extraction':
            self.best_auc = -2e9
        else:
            self.best_psnr = -2e9
            psnr_noisy_last = 0
        self.out = None
        self.best_out = None
        last_net = None

        if self.save_path is not None:
            if self.noise_type is not None:
                file_name = f'{self.noise_type}.txt'
            else:
                file_name = f'log.txt'
            self.file = open(self.save_path+'/'+file_name, 'w+')
        if self.weight_decay is not None:
            optimizer = torch.optim.Adam(self.params, self.lr, weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.Adam(self.params, self.lr)

        self.print_start_info()

        for iter in range(1, self.iterations + 1):
            self.iter = iter
            optimizer.zero_grad()

            self.closure()

            self.out = self.out.clamp(0, 1)

            if self.GT is not None:
                if self.task == 'extraction':
                    auc = AUC_gpu(self.GT, torch.abs(self.out))
                    if auc > self.best_auc:
                        self.best_auc = auc
                        self.best_out = self.out.clamp(0, 1)
                        self.best_iter = iter
                else:
                    psnr = PSNR_gpu(self.out, self.GT)
                    if psnr > self.best_psnr:
                        self.best_psnr = psnr
                        self.best_out = self.out.clamp(0, 1)
                        self.best_iter = iter

            if iter % self.show_every == 0 or iter == 1:
                if self.GT is not None:
                    if self.task == 'extraction':
                        self.print_info({'auc': auc}, {'auc': self.best_auc}, iter)
                    else:
                        metrics = self.evaluate(self.out.cpu().numpy(), self.GT.cpu().numpy(), metrics=self.metrics)
                        best_metrics = self.evaluate(self.best_out.cpu().numpy(), self.GT.cpu().numpy(), metrics=self.metrics)
                        self.print_info(metrics, best_metrics, iter)
                else:
                    print(f'ITER: [{iter}/{self.iterations}]')
                
                if self.show_image is True:
                    self.show_process()

                if self.task != 'extraction':
                    if psnr - psnr_noisy_last < -5:
                        print('Falling back to previous checkpoint.')
                        for new_param, net_param in zip(last_net, self.net.parameters()):
                            net_param.data.copy_(new_param.cuda())
                    else:
                        last_net = [x.detach().cpu() for x in self.net.parameters()]
                        psnr_noisy_last = psnr

            torch.cuda.empty_cache()

            if self.stop is True:
                break

            optimizer.step()
        
        self.print_end_info()
        
        if self.save_path is not None and self.GT is not None:
            self.file.close()

    def train(self):
        self.init_input()
        self.get_params()
        self.optimize()

    def evaluate(self, X=None, Y=None, metrics=['psnr', 'ssim', 'sam', 'ergas', 'nrmse']):
        if X is None and Y is None:
            best_out = self.best_out.cpu().numpy()
            GT = self.GT.cpu().numpy()
            res = {}
            if 'psnr' in metrics:
                res['psnr'] = PSNR(best_out, GT)
            if 'ssim' in metrics:
                res['ssim'] = SSIM(best_out, GT)
            if 'sam' in metrics:
                res['sam'] = SAM(best_out, GT)
            if 'ergas' in metrics:
                res['ergas'] = ERGAS(best_out, GT)
            if 'nrmse' in metrics:
                res['nrmse'] = nrmse(GT, best_out)      
            if 'auc' in metrics:
                res['auc'] = AUC(GT, best_out)                                                                     
            return res
        else:
            res = {}
            if 'psnr' in metrics:
                res['psnr'] = PSNR(X, Y)
            if 'ssim' in metrics:
                res['ssim'] = SSIM(X, Y)
            if 'sam' in metrics:
                res['sam'] = SAM(X, Y)
            if 'ergas' in metrics:
                res['ergas'] = ERGAS(X, Y)
            if 'nrmse' in metrics:
                res['nrmse'] = nrmse(Y, X)
            if 'auc' in metrics:
                res['auc'] = AUC(Y, np.abs(X))
            return res
    
    def print_info(self, metrics, best_metrics, iter):
        log = f''
        if self.task is not None:
            log += f'TASK: {self.task}\t'
        if self.data_name is not None:
            log += f'DATA: {self.data_name}\t'
        if self.method_name is not None:
            log += f'METHOD: {self.method_name}\t'
        if self.noise_type is not None:
            log += f'DEGRADATION: {self.noise_type}\t'

        log += f'ITER: [{iter}/{self.iterations}]\t'
        if 'psnr' in metrics:
            psnr = metrics['psnr']; best_psnr = best_metrics['psnr']
            log += f'PSNR: [{psnr:.6f}/{best_psnr:.6f}]\t'
        if 'ssim' in metrics:                
            ssim = metrics['ssim']; best_ssim = best_metrics['ssim']
            log += f'SSIM: [{ssim:.6f}/{best_ssim:.6f}]\t'
        if 'sam' in metrics:   
            sam = metrics['sam']; best_sam = best_metrics['sam']
            log += f'SAM: [{sam:.6f}/{best_sam:.6f}]\t'
        if 'ergas' in metrics: 
            ergas = metrics['ergas']; best_ergas = best_metrics['ergas']
            log += f'ERGAS: [{ergas:.6f}/{best_ergas:.6f}]\t'
        if 'nrmse' in metrics:
            nrmse = metrics['nrmse']; best_nrmse = best_metrics['nrmse']
            log += f'NRMSE: [{nrmse:.6f}/{best_nrmse:.6f}]\t'
        if 'auc' in metrics:
            auc = metrics['auc']; best_auc = best_metrics['auc']
            log += f'AUC: [{auc:.6f}/{best_auc:.6f}]\t'   

        log = log[:-1] + '\n'          
        if self.save_path is not None:
            print(log, file=self.file, end='')
        print(log, end='')

    def show_process(self):
        plt.figure(figsize=(12, 6))
        if self.display_type == 'gray':
            band = self.display_band_list[0]
            if self.GT is None:
                ax1 = plt.subplot(1, 2, 1)
                ax1.imshow(self.noisy[band, :, :].cpu(), cmap='gray')
                ax2 = plt.subplot(1, 2, 2)
                ax2.imshow(self.out[band, :, :].cpu(), cmap='gray')
            else:
                ax1 = plt.subplot(1, 3, 1)
                ax1.imshow(self.noisy[band, :, :].cpu(), cmap='gray')
                ax2 = plt.subplot(1, 3, 2)
                ax2.imshow(self.out[band, :, :].cpu(), cmap='gray')
                ax2 = plt.subplot(1, 3, 3)
                ax2.imshow(self.GT[band, :, :].cpu(), cmap='gray')
        elif self.display_type == 'pseudo':
            band_list = self.display_band_list
            out = torch.clamp(self.out.cpu(), 0, 1)
            noisy = torch.clamp(self.noisy.cpu(), 0, 1)
            if self.GT == None:
                ax1 = plt.subplot(1, 2, 1)
                ax2 = plt.subplot(1, 2, 2)
                ax1.imshow(torch.stack((noisy[band_list[0], :, :].cpu(), noisy[band_list[1], :, :].cpu(), noisy[band_list[2], :, :].cpu()), 2))
                ax2.imshow(torch.stack((out[band_list[0], :, :].cpu(), out[band_list[1], :, :].cpu(), out[band_list[2], :, :].cpu()), 2))
            else:
                GT = self.GT.cpu()
                ax1 = plt.subplot(1, 3, 1)
                ax2 = plt.subplot(1, 3, 2)
                ax3 = plt.subplot(1, 3, 3)
                ax1.imshow(torch.stack((noisy[band_list[0], :, :].cpu(), noisy[band_list[1], :, :].cpu(), noisy[band_list[2], :, :].cpu()), 2))
                ax2.imshow(torch.stack((out[band_list[0], :, :].cpu(), out[band_list[1], :, :].cpu(), out[band_list[2], :, :].cpu()), 2))
                ax3.imshow(torch.stack((GT[band_list[0], :, :].cpu(), GT[band_list[1], :, :].cpu(), GT[band_list[2], :, :].cpu()), 2))
        plt.show()
            
    def save(self, name=None, save_type='mat'):
        if self.save_path is not None:
            if name is None:
                name = f''
                if self.noise_type is not None:
                    name = f'{self.noise_type}'
                else:
                    name = f'out'
            
            if self.task == 'extraction':
                savemat(self.save_path+'/L_'+name, {'data': self.L.cpu().numpy()})

            if self.GT is not None:
                if save_type == 'mat':
                    savemat(self.save_path+'/'+name+'.mat', {'data': self.best_out.cpu().numpy(), 'metrics': self.best_metrics})
                elif save_type == 'png':
                    Image.fromarray(np.uint8(self.best_out.cpu().numpy().transpose(1, 2, 0)*255)).save(self.save_path+'/'+name+'.png')
            else:
                if save_type == 'mat':
                    savemat(self.save_path+'/'+name+'.mat', {'data': self.out.cpu().numpy()})
                elif save_type == 'png':
                    Image.fromarray(np.uint8(self.best_out.cpu().numpy().transpose(1, 2, 0)*255)).save(self.save_path+'/'+name+'.png')