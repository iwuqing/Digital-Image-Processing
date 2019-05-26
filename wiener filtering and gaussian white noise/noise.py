import numpy as np
import random


def gaussian_white_noise(intput_signal, mu, sigma):
    '''
    加性高斯白噪声（适用于灰度图）
    :param intput_signal: 输入图像
    :param mu: 均值
    :param sigma: 标准差
    :return:
    '''
    intput_signal_cp = np.copy(intput_signal)  # 输入图像的副本

    m, n = intput_signal_cp.shape   # 输入图像尺寸（行、列）

    # 添加高斯白噪声
    for i in range(m):
        for j in range(n):
            intput_signal_cp[i, j] = intput_signal_cp[i, j] + random.gauss(mu, sigma)


    return intput_signal_cp
