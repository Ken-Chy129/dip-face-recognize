import random

import numpy as np

from numpy import array, clip, uint8
from numpy.random import uniform, rayleigh


# 添加高斯噪声
def add_gauss_noise(image, sigma=40):
    # 获得r,g,b三个图像通道
    r = image[:, :, 0].flatten()
    g = image[:, :, 1].flatten()
    b = image[:, :, 2].flatten()

    # 遍历像素随机生成高斯噪声
    # 对每个通道增加噪声
    for i in range(image.shape[0] * image.shape[1]):
        pr = int(r[i]) + random.gauss(0, sigma)
        pg = int(g[i]) + random.gauss(0, sigma)
        pb = int(b[i]) + random.gauss(0, sigma)
        if pr < 0:
            pr = 0
        if pr > 255:
            pr = 255
        if pg < 0:
            pg = 0
        if pg > 255:
            pg = 255
        if pb < 0:
            pb = 0
        if pb > 255:
            pb = 255
        r[i] = pr
        g[i] = pg
        b[i] = pb

    image[:, :, 0] = r.reshape([image.shape[0], image.shape[1]])
    image[:, :, 1] = g.reshape([image.shape[0], image.shape[1]])
    image[:, :, 2] = b.reshape([image.shape[0], image.shape[1]])
    return image


# 添加椒盐噪声，arg为信噪比(不增加噪声的像素点/总像素点)
def add_salt_noise(image, arg=0.9):
    image_ = image.copy()
    for h in range(image_.shape[0]):
        for w in range(image_.shape[1]):
            if np.random.random(1) > arg:
                image_[h, w] = np.random.randint(2) * 255
    return image_


# 添加伽马噪声，var为方差
def add_gamma_noise(image, var=0.1):
    image = array(image/255, dtype=float)
    # 2个服从指数分布的噪声叠加
    noise = np.random.gamma(2, var ** 0.5, image.shape)
    image = image + noise
    image = clip(image, 0, 1)
    image = uint8(image * 255)
    return image


# 添加均匀噪声，high为上界，low为下界
def add_uniform_noise(image, high=1.0, low=0.0):
    image = array(image/255, dtype=float)
    noise = uniform(low, high, image.shape)
    image = image + noise
    image = clip(image, 0, 1)
    image = uint8(image * 255)
    return image


# 添加瑞利噪声， var为方差
def add_rayleigh_noise(image, var=0.1):
    image = array(image/255, dtype=float)
    noise = rayleigh(var ** 0.5, image.shape)
    image = image + noise
    image = clip(image, 0, 1)
    image = uint8(image * 255)
    return image
