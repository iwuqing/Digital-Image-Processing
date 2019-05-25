import numpy as np


def hist_equalization(intput_signal):
    '''
    直方图均衡（适用于灰度图）
    :param intput_signal:   输入图像
    :return:    直方图均衡化后的输出图像
    '''

    output_signal = np.copy(intput_signal)   # 输出图像，初始化为输入

    intput_signal_cp = np.copy(intput_signal) # 输入图像的副本

    m, n = intput_signal_cp.shape # 输入图像的尺寸（行、列）

    pixel_total_num = m * n  # 输入图像的像素点总数

    p_r = []   # 输入图像的概率密度分布
    p_s = []   # 输出图像的概率密度分布

    # 求输入图像的概率密度分布函数
    for i in range(256):
        p_r.append(np.sum(intput_signal_cp == i) / pixel_total_num)

    # 求输出图像的概率密度分布函数
    single_pixel_class_probobility_t = 0  # 临时存储某一灰度级的概率
    for i in range(256):
        single_pixel_class_probobility_t = single_pixel_class_probobility_t + p_r[i]
        p_s.append(single_pixel_class_probobility_t)

    # 求解变换后的输出图像
    for i in range(256):
        output_signal[np.where(intput_signal_cp == i)] = 255 * p_s[i]

    return output_signal