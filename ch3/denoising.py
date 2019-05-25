import numpy as np
import random

def median_filtering(input_signal):
    '''
    中值滤波（适用于灰度图）
    如：
            - + -
            + * +
            - + -
    * 为噪点，滤波方法为：取4个+的中位数（若某+不在输入信号范围内，则随机添加0或255）
    :param input_signal: 输入信号矩阵（2D）
    :return: 滤波后的信号
    '''
    salt_pepper = [0, 255]

    m, n = input_signal.shape  # 获取输入图片的尺寸（行和列）

    input_signal_cp = input_signal.copy()   # 输入信号的副本

    nosiy_data_around = []  # 存放噪点上下左右的数据点
    # 遍历滤波
    for i in range(m):
        for j in range(n):
            # 当灰度值为0或255时，则认为该数据点为椒盐噪点
            if input_signal_cp[i, j] == 255 or input_signal_cp[i, j] == 0:
                # 每次无效数据点（即不再范围内）为4，每有一个在范围内，即-1
                invalid_data_per = 4
                if i + 1 < n:
                    nosiy_data_around.append(input_signal_cp[i + 1, j])
                    invalid_data_per = invalid_data_per - 1
                if i - 1 >= 0:
                    nosiy_data_around.append(input_signal_cp[i - 1, j])
                    invalid_data_per = invalid_data_per - 1
                if j + 1 < m:
                    nosiy_data_around.append(input_signal_cp[i, j + 1])
                    invalid_data_per = invalid_data_per - 1
                if j - 1 >= 0:
                    nosiy_data_around.append(input_signal_cp[i, j - 1])
                    invalid_data_per = invalid_data_per - 1
                else:
                    if invalid_data_per > 0:
                        # 根据无效数据点的个数，随机添加0或255
                        for k in range(invalid_data_per):
                            nosiy_data_around.append(salt_pepper[random.randint(0, 1)])
                # 取中位数
                input_signal_cp[i, j] = np.median(nosiy_data_around)

                # 该噪点的周围数据数组清空，为下一个噪点周围数据存在做准备
                nosiy_data_around = []

    return input_signal_cp

