import torch
import numpy as np
import os
import random

def setup_seed(manual_seed=2022):
    os.environ['PYTHONHASHSEED'] = str(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)  # if you are using multi-GPU
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

def reconstruct_from_gradient(gradient, edge, dim, mode='torch'):
    if mode == 'np':
        out = np.zeros_like(gradient).astype(edge.dtype)
    elif mode == 'torch':
        out = torch.zeros_like(gradient).to(gradient.device)
    C, H, W = gradient.shape
    if dim == 1 or dim == 'W':
        out[:, :, 0] = edge
        for i in range(1, W):
            out[:, :, i] = out[:, :, i-1] + gradient[:, :, i-1]
    elif dim == 2 or dim == 'C':
        out[0, :, :] = edge
        for i in range(1, H):
            out[i, :, :] = out[i-1, :, :] + gradient[i-1, :, :]
    elif dim == 0 or dim == 'H':
        out[:, 0, :] = edge
        for i in range(1, H):
            out[:, i, :] = out[:, i-1, :] + gradient[:, i-1, :]
    
    return out
        

def f_unfold(tensor, mode=0):
    return torch.reshape(torch.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))

def soft_thresholding(w, gamma, mode='torch'):
    if mode == 'torch':
        return torch.sign(w) * (torch.abs(w) - gamma) * (torch.abs(w) - gamma > 0)
    if mode == 'np':
        return np.sign(w) * (np.abs(w) - gamma) * (np.abs(w) - gamma > 0)

def gradient(X, dim, padding=True, mode='torch', transpose=False):
    assert len(X.shape) == 3, "shape of X should be b x m x n"
    C, H, W = X.shape;
    if dim == 0 or dim == 'H':
        if padding is False:
            return X[:, 1:, :] - X[:, :-1, :]
        if mode == 'np' or mode == 'numpy':
            G = np.zeros((C, H, W)).astype(X.dtype)
        elif mode == 'torch':
            G = torch.zeros((C, H, W)).to(X.device)
        if transpose is True:
            G[:, 1:, :] = X[:, :-1, :] - X[:, 1:, :]
            G[:, 0, :] = X[:, -1, :] - X[:, 0, :]
        else:
            G[:, :-1, :] = X[:, 1:, :] - X[:, :-1, :]
            G[:, -1, :] = X[:, 0, :] - X[:, -1, :]
        return G
    elif dim == 1 or dim == 'W':
        if padding is False:
            return X[:, :, 1:] - X[:, :, :-1]
        if mode == 'np' or mode == 'numpy':
            G = np.zeros((C, H, W)).astype(X.dtype)
        elif mode == 'torch':
            G = torch.zeros((C, H, W)).to(X.device)
        if transpose is True:
            G[:, :, 1:] = X[:, :, :-1] - X[:, :, 1:]
            G[:, :, 0] = X[:, :, -1] - X[:, :, 0]
        else:
            G[:, :, :-1] = X[:, :, 1:] - X[:, :, :-1]
            G[:, :, -1] = X[:, :, 0] - X[:, :, -1]            
        return G
    elif dim == 2 or dim == 'C':
        if padding is False:
            return X[1:, :, :] - X[:-1, :, :]
        if mode == 'np' or mode == 'numpy':
            G = np.zeros((C, H, W)).astype(X.dtype)
        elif mode == 'torch':
            G = torch.zeros((C, H, W)).to(X.device)
        if transpose is True:
            G[1:, :, :] = X[:-1, :, :] - X[1:, :, :]
            G[0, :, :] = X[-1, :, :] - X[0, :, :]
        else:
            G[:-1, :, :] = X[1:, :, :] - X[:-1, :, :]
            G[-1, :, :] = X[0, :, :] - X[-1, :, :]                 
        return G
    else:
        assert False, "no dims refer to " + str(dim)


def TV(X, p=1, alpha=0.5):
    if p == 'l1-2':
        ani = TV(X, p=1)
        iso = torch.sum(torch.sqrt(gradient(X, 'H')**2 + gradient(X, 'W')**2))
        return ani - alpha * iso
    else:
        return torch.norm(gradient(X, 'H'), p) + torch.norm(gradient(X, 'W'), p)

def SSTV(X, p=1, alpha=0.5):
    if p == 'l1-2':
        ani = SSTV(X, p=1)
        G_z = gradient(X, 'z')
        iso = torch.sum(torch.sqrt(gradient(G_z, 'H')**2 + gradient(G_z, 'W')**2))
        return ani - alpha * iso
    else:
        G_z = gradient(X, 'z')
        return torch.norm(gradient(G_z, 'H'), p) + torch.norm(gradient(G_z, 'W'), p)
    
def zero_pad(image, shape, position='corner', mode='np'):
    """
    Extends image to a certain size with zeros
    Parameters
    ----------
    image: real 2d `numpy.ndarray`
        input_image image
    shape: tuple of int
        Desired output shape of the image
    position : str, optional
        The position of the input_image image in the output one:
            * 'corner'
                top-left corner (default)
            * 'center'
                centered
    Returns
    -------
    padded_img: real `numpy.ndarray`
        The zero-padded image
    """
    if mode == 'np':
        shape = np.asarray(shape, dtype=int)
        imshape = np.asarray(image.shape, dtype=int)

        if np.alltrue(imshape == shape):
            return image

        if np.any(shape <= 0):
            raise ValueError("ZERO_PAD: null or negative shape given")

        dshape = shape - imshape
        if np.any(dshape < 0):
            raise ValueError("ZERO_PAD: target size smaller than source one")

        pad_img = np.zeros(shape, dtype=image.dtype)

        idx, idy = np.indices(imshape)

        if position == 'center':
            if np.any(dshape % 2 != 0):
                raise ValueError("ZERO_PAD: source and target shapes "
                                "have different parity.")
            offx, offy = dshape // 2
        else:
            offx, offy = (0, 0)
        pad_img[idx + offx, idy + offy] = image
    
    elif mode == 'torch':
        shape = torch.tensor(shape, dtype=int).to(image.device)
        imshape = torch.tensor(image.shape, dtype=int).to(image.device)

        if torch.all(imshape == shape):
            return image

        if torch.any(shape <= 0):
            raise ValueError("ZERO_PAD: null or negative shape given")

        dshape = shape - imshape
        if torch.any(dshape < 0):
            raise ValueError("ZERO_PAD: target size smaller than source one")

        pad_img = torch.zeros((shape[0], shape[1]), dtype=image.dtype).to(image.device)

        idx, idy = torch.meshgrid(torch.range(0, imshape[0]-1), torch.range(0, imshape[1]-1))

        if position == 'center':
            if torch.any(dshape % 2 != 0):
                raise ValueError("ZERO_PAD: source and target shapes "
                                "have different parity.")
            offx, offy = dshape // 2
        else:
            offx, offy = (0., 0.)

        pad_img[(idx + offx).to(torch.long), (idy + offy).to(torch.long)] = image

    return pad_img

def psf2otf(psf, shape, mode='np'):
    """
    Convert point-spread function to optical transfer function.
    Compute the Fast Fourier Transform (FFT) of the point-spread
    function (PSF) array and creates the optical transfer function (OTF)
    array that is not influenced by the PSF off-centering.
    By default, the OTF array is the same size as the PSF array.
    To ensure that the OTF is not altered due to PSF off-centering, PSF2OTF
    post-pads the PSF array (down or to the right) with zeros to match
    dimensions specified in OUTSIZE, then circularly shifts the values of
    the PSF array up (or to the left) until the central pixel reaches (1,1)
    position.
    Parameters
    ----------
    psf : `numpy.ndarray`
        PSF array
    shape : int
        Output shape of the OTF array
    Returns
    -------
    otf : `numpy.ndarray`
        OTF array
    Notes
    -----
    Adapted from MATLAB psf2otf function
    """
    if mode == 'np':
        if np.all(psf == 0):
            return np.zeros_like(psf, mode=mode)
        inshape = psf.shape
        
        psf = zero_pad(psf, shape, position='corner', mode='np')
        for axis, axis_size in enumerate(inshape):
            psf = np.roll(psf, -int(axis_size / 2), axis=axis)
 
        otf = np.fft.fft2(psf)
        
        n_ops = np.sum(psf.size * np.log2(psf.shape))
        otf = np.real_if_close(otf, tol=n_ops)
        return otf  
    
    
    if mode == 'torch':
        if torch.all(psf == 0):
            return torch.zeros_like(psf, mode=mode).to(psf.device)
        
        inshape = psf.shape
    # Pad the PSF to outsize
        psf = zero_pad(psf, shape, position='corner', mode='torch')

    # Circularly shift OTF so that the 'center' of the PSF is
    # [0,0] element of the array

        for axis, axis_size in enumerate(inshape):
            psf = torch.roll(psf, -int(axis_size / 2), dims=axis)        

    # Compute the OTF
        otf = torch.fft.fft2(psf)

    # Estimate the rough number of operations involved in the FFT
    # and discard the PSF imaginary part if within roundoff error
    # roundoff error  = machine epsilon = sys.float_info.epsilon
    # or np.finfo().eps

        size = psf.shape[0] * psf.shape[1]
        n_ops = torch.sum(size * torch.log2(torch.tensor(psf.shape)))
        if torch.all(torch.real(otf) < n_ops):
            otf = torch.real(otf)

        return otf  
    
def filter(X, fft_kernel, mode='torch'):
    if mode == 'torch':
        return torch.real(torch.fft.ifftn(torch.fft.fftn(X)*fft_kernel))
    if mode == 'np':
        return np.real(np.fft.ifftn(np.fft.fftn(X)*fft_kernel))
    
def nearest_upsample(X, factor, mode='torch'):
    C, H, W = X.shape
    if mode == 'torch':
        out = torch.zeros((C, H*factor, W*factor)).to(X.device)
    elif mode == 'np':
        out = np.zeros((C, H*factor, W*factor))
    out[:, ::factor, ::factor] = X
    return out

def gkern(l=5, sig=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)