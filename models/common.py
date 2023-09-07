import torch
import torch.nn as nn
import numpy as np
from .downsampler import Downsampler

def add_module(self, module):
    self.add_module(str(len(self) + 1), module)
    
torch.nn.Module.add = add_module

class Concat(nn.Module):
    def __init__(self, dim, *args):
        super(Concat, self).__init__()
        self.dim = dim

        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input):
        inputs = []
        for module in self._modules.values():
            inputs.append(module(input))

        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]        

        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(np.array(inputs_shapes3) == min(inputs_shapes3)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs: 
                diff2 = (inp.size(2) - target_shape2) // 2 
                diff3 = (inp.size(3) - target_shape3) // 2 
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])

        return torch.cat(inputs_, dim=self.dim)

    def __len__(self):
        return len(self._modules)


class GenNoise(nn.Module):
    def __init__(self, dim2):
        super(GenNoise, self).__init__()
        self.dim2 = dim2

    def forward(self, input):
        a = list(input.size())
        a[1] = self.dim2
        # print (input.data.type())

        b = torch.zeros(a).type_as(input.data)
        b.normal_()

        x = torch.autograd.Variable(b)

        return x


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
        self.s = nn.Sigmoid()

    def forward(self, x):
        return x * self.s(x)


def act(act_fun = 'LeakyReLU'):
    '''
        Either string defining an activation function or module (e.g. nn.ReLU)
    '''
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun == 'Swish':
            return Swish()
        elif act_fun == 'ELU':
            return nn.ELU()
        elif act_fun == 'none':
            return nn.Sequential()
        else:
            assert False
    else:
        return act_fun()


def bn(num_features):
    return nn.BatchNorm2d(num_features)


def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':

        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        elif downsample_mode  in ['lanczos2', 'lanczos3']:
            downsampler = Downsampler(n_planes=out_f, factor=stride, kernel_type=downsample_mode, phase=0.5, preserve_size=True)
        else:
            assert False

        stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0
  
    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)


    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
    return nn.Sequential(*layers)



def set_input_inpainting(img_degraded):
    img_degraded = img_degraded.transpose(1, 2, 0)
    mask = np.ones_like(img_degraded); mask[img_degraded == 0] = 0
    height, width, n_channel = img_degraded.shape[0], img_degraded.shape[1], img_degraded.shape[-1]
    # mask = self.mask # [hh,ww,in_c]
    # img_degraded = img_degraded.copy()
    
    valid_loc = np.sum(mask,axis=-1) == n_channel           # where all channels are not missing
    semivalid_loc = np.logical_and( (~ valid_loc) , np.sum(mask,axis=-1)!=0 )   # where some channels are missing but not all channels are missing
    
    # process semivalid values
    denom = np.sum(mask, -1)
    denom[denom==0] = 1
    semivalid_value = np.sum(img_degraded,-1) / denom             # averging img_degraded where has value
    semivalid_value = np.stack([semivalid_value]*n_channel, -1)     # H x W x C shape averged img_degraded
    
    semivalid_loc = np.stack([semivalid_loc]*n_channel, -1)==1      # H x W x C shape semivalid_loc
    semivalid_loc = np.logical_and(semivalid_loc , np.logical_not(mask) )   # location where img_degraded does not have value and some channels have
    img_degraded[semivalid_loc] = semivalid_value[semivalid_loc]       # change the place where is missing to the averging value

    # process invalid values
    init_invalid = (~ valid_loc) & (np.sum(mask,-1)==0)         # where all channels are missing
    invalid = init_invalid 
    process_invalid = 1
    win_size = 3
    count = 0
    while process_invalid:
        index,indey = np.nonzero(invalid==1)            # index of pixel where all channels are missing
        for k in range(len(index)):
            i = index[k]
            j = indey[k]
            x1 = min(max(i-win_size,0),height-1)
            x2 = min(max(i+win_size,0),height-1)
            y1 = min(max(j-win_size,0),width-1)
            y2 = min(max(j+win_size,0),width-1)
            # local_area = img_degraded[x1:x2, y1:y2, :]
            local_area = img_degraded[x1:x2, y1:y2, ...]
            local_area = np.reshape(local_area, [local_area.shape[0]*local_area.shape[1], *local_area.shape[2:]])
            # img_degraded[i,j,:]=np.median(local_area,0)
            img_degraded[i,j,...]=np.median(local_area,0)

        invalid = np.logical_and(np.sum(img_degraded,-1)<=0.012 , init_invalid)
        count = count +1
        if np.sum(invalid)==0 or count>=50:
            process_invalid = 0
        else:
            process_invalid = 1
        if (count+1)%10==0:
            win_size = win_size+1
    
    return img_degraded.transpose(2, 0, 1)


# from scipy.io import loadmat
# noisy = loadmat('data/suzie_R0.9')['data']
# Input = set_input_inpainting(noisy)
# import matplotlib.pyplot as plt
# plt.imshow(Input[35], cmap='gray')
# plt.show()