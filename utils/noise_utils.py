import torch
from skimage.util import random_noise
import math
import numpy as np

def degrad_by_dedalines(X, q=35, proportion=0.2):
    assert len(X.shape) == 3, "shape of X should be b x m x n"
    b, _, n = X.shape
    ddb = np.random.permutation(b)
    ddb = ddb[1:math.ceil(proportion * b)]
    for i in range(b):
        if i in ddb:
            linenum = q
            linewidth = np.ones(linenum).squeeze()
            lineloc = np.random.randint(0, n - int(max(linewidth)), (1, linenum)).squeeze()
            lineindex = []
            for k in range(linenum):
                lineindex += [x for x in range(lineloc[k], int(lineloc[k] + linewidth[k]))]
            X[i, :, lineindex] = 0

    return X

def degrad_by_stripes(X, q=35, proportion=0.2):
    assert len(X.shape) == 3, "shape of X should be b x m x n"
    b, _, n = X.shape
    stb = np.random.permutation(b)
    stb = stb[1:math.ceil(proportion * b)]
    for i in range(b):
        if i in stb:
            linenum = q
            lineloc = np.random.randint(0, n, (1, linenum)).squeeze()
            t = np.random.rand(linenum) * 0.5 - 0.25
            X[i, :, lineloc] -= t.reshape(-1, 1)

    return X

    

def add_noise(img, case=1):
    X = img.copy()
    if case == 1:
        sigma = 0.2
        X += np.random.randn(*X.shape) * sigma
    elif case == 2:
        sigma_down = 0.1
        sigma_up = 0.4
        assert len(X.shape) == 3, "shape of X should be b x m x n"
        for i in range(X.shape[0]):
            sigma = sigma_down + np.random.rand(1) * (sigma_up - sigma_down)
            X[i, :, :] += np.random.randn(X.shape[1], X.shape[2]) * sigma
    elif case == 3:
        sigma_down = 0.1
        sigma_up = 0.4
        p = 0.1
        for i in range(X.shape[0]):
            sigma = sigma_down + np.random.rand(1) * (sigma_up - sigma_down)
            X[i, :, :] += np.random.randn(X.shape[1], X.shape[2]) * sigma
        X = random_noise(X, mode='s&p', salt_vs_pepper=p)
    elif case == 4:
        sigma_down = 0.1
        sigma_up = 0.4
        p = 0.1
        q = 35
        proportion = 0.2
        for i in range(X.shape[0]):
            sigma = sigma_down + np.random.rand(1) * (sigma_up - sigma_down)
            X[i, :, :] += np.random.randn(X.shape[1], X.shape[2]) * sigma
        X = random_noise(X, mode='s&p', salt_vs_pepper=p)
        X = degrad_by_dedalines(X, q, proportion)
    elif case == 5:
        sigma_down = 0.1
        sigma_up = 0.4
        p = 0.1
        q = 35
        proportion = 0.2
        for i in range(X.shape[0]):
            sigma = sigma_down + np.random.rand(1) * (sigma_up - sigma_down)
            X[i, :, :] += np.random.randn(X.shape[1], X.shape[2]) * sigma
        X = random_noise(X, mode='s&p', salt_vs_pepper=p)
        X = degrad_by_dedalines(X, q, proportion)
        X = degrad_by_stripes(X, q, proportion)
    elif case == 6:
        sigma_down = 0.1
        sigma_up = 0.4
        p = 0.15
        q = 35
        proportion = 0.3
        for i in range(X.shape[0]):
            sigma = sigma_down + np.random.rand(1) * (sigma_up - sigma_down)
            X[i, :, :] += np.random.randn(X.shape[1], X.shape[2]) * sigma
        X = random_noise(X, mode='s&p', salt_vs_pepper=p)
        X = degrad_by_dedalines(X, q, proportion)
        X = degrad_by_stripes(X, q, proportion)
    elif case == 7:
        sigma_down = 0.1
        sigma_up = 0.4
        p = 0.2
        q = 35
        proportion = 0.4
        for i in range(X.shape[0]):
            sigma = sigma_down + np.random.rand(1) * (sigma_up - sigma_down)
            X[i, :, :] += np.random.randn(X.shape[1], X.shape[2]) * sigma
        X = random_noise(X, mode='s&p', salt_vs_pepper=p)
        X = degrad_by_dedalines(X, q, proportion)
        X = degrad_by_stripes(X, q, proportion)
    elif case == 8:
        sigma_down = 0.1
        sigma_up = 0.4
        p = 0.25
        q = 35
        proportion = 0.5
        for i in range(X.shape[0]):
            sigma = sigma_down + np.random.rand(1) * (sigma_up - sigma_down)
            X[i, :, :] += np.random.randn(X.shape[1], X.shape[2]) * sigma
        X = random_noise(X, mode='s&p', salt_vs_pepper=p)
        X = degrad_by_dedalines(X, q, proportion)
        X = degrad_by_stripes(X, q, proportion)
    elif case == 9:
        sigma_down = 0.05
        sigma_up = 0.4
        p = 0.25
        q = 35
        proportion = 0.5
        for i in range(X.shape[0]):
            sigma = sigma_down + np.random.rand(1) * (sigma_up - sigma_down)
            X[i, :, :] += np.random.randn(X.shape[1], X.shape[2]) * sigma
        X = random_noise(X, mode='s&p', salt_vs_pepper=p)
        X = degrad_by_dedalines(X, q, proportion)
        X = degrad_by_stripes(X, q, proportion)
    elif case == 10:
        sigma = 0.3
        X += np.random.randn(*X.shape) * sigma
    else:
        assert False, "no such case"

    return X.clip(0, 1)
    