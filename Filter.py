import time

import cv2
import numpy as np


# 双边滤波，size为核大小
def bilateral_filter(image, size=3, sigma_s=20, sigma_c=20):
    h, w, c = image.shape  # 获取图像信息
    image = cv2.copyMakeBorder(image, size, size, size, size, cv2.BORDER_REPLICATE)  # 扩展边界
    pic_ = image.copy()
    time_start = time.time()
    for i in range(size, size + h):  # 双边滤波
        for j in range(size, size + w):
            for k in range(c):
                tol_up = 0.0
                tol_down = 0.0
                for m in range(i - size, i + size + 1):
                    for n in range(j - size, j + size + 1):
                        temp = get_g(image, i, j, m, n, k, sigma_s, sigma_c)
                        tol_up += temp * pic_[m, n, k]
                        tol_down += temp
                value = min(255.0, tol_up / tol_down)  # 防止值溢出
                pic_[i, j, k] = value
    time_end = time.time()
    print("花费时间:", time_end - time_start)
    return pic_[size: size + h, size: size + w]  # 裁剪回原图


def get_g(image, i, j, m, n, k, sigma_s, sigma_c):
    gs = np.exp(-((i - m) ** 2 + (j - n) ** 2) / (2 * sigma_s ** 2))
    gr = np.exp(-(float(image[i, j, k]) - float(image[m, n, k])) ** 2 / (2 * sigma_c ** 2))
    return gs * gr


# 中值滤波，size为核大小
def median_filter(image, size=3):
    h, w, c = image.shape
    image = cv2.copyMakeBorder(image, size, size, size, size, cv2.BORDER_REPLICATE)  # 扩展边界
    pic_ = image.copy()
    for i in range(size, size + h):
        for j in range(size, size + w):
            for k in range(c):
                pic_[i, j, k] = np.median(image[i-size: i+size, j-size: j+size, k])  # 调用np.median求取中值
    return pic_


# 均值滤波，size为核大小
def average_filter(image, size=3):
    h, w, c = image.shape
    image = cv2.copyMakeBorder(image, size, size, size, size, cv2.BORDER_REPLICATE)  # 扩展边界
    pic_ = image.copy()
    for i in range(size, size + h):
        for j in range(size, size + w):
            for k in range(c):
                pic_[i, j, k] = np.mean(image[i-size: i+size, j-size: j+size, k])  # 调用np.median求取均值
    return pic_


# 高通滤波, r为通带半径
def high_pass_filter(image, r=40):
    image_ = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    dft = np.fft.fft2(image_)
    # 将频域从左上角移动到中间
    dft_shift = np.fft.fftshift(dft)

    h, w = dft_shift.shape[0], dft_shift.shape[1]
    mh, mw = int(h/2), int(w/2)  # 中心位置
    dft_shift[mh-r: mh+r, mw-r: mw+r] = 0

    # 傅里叶逆变换
    i_dft_shift = np.fft.ifftshift(dft_shift)
    i_image = np.fft.ifft2(i_dft_shift)
    i_image = np.uint8(np.abs(i_image))
    return i_image


# 低通滤波器
def low_pass_filter(img, size=60):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 傅里叶变换
    img_dft = np.fft.fft2(img)
    # 将频域从左上角移动到中间
    dft_shift = np.fft.fftshift(img_dft)
    # 高通滤波
    h, w = dft_shift.shape[0], dft_shift.shape[1]
    mh, mw = int(h/2), int(w/2)
    mask = np.zeros(dft_shift.shape, dtype=np.uint8)
    mask[mh-int(size/2): mh+int(size / 2), mw-int(size/2): mw+int(size/2)] = 1
    dft_shift = dft_shift * mask

    # 将频域从中间移动到左上角
    i_dft_shift = np.fft.ifftshift(dft_shift)
    i_img = np.fft.ifft2(i_dft_shift)
    i_img = np.uint8(np.abs(i_img))
    return i_img
