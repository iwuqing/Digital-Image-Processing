import numpy as np
import math

def nearest_neighbor(input_signal, zoom_multiples):
    '''
    最近邻插值（适用于灰度图）
    :param input_signal: 输入图像
    :param zoom_multiples:  缩放倍数
    :return: 缩放后的图像
    '''
    input_signal_cp = np.copy(input_signal)   # 输入图像的副本

    input_row, input_col = input_signal_cp.shape # 输入图像的尺寸（行、列）

    # 输出图像的尺寸
    output_row = int(input_row * zoom_multiples)
    output_col = int(input_col * zoom_multiples)

    output_signal = np.zeros((output_row, output_col)) # 输出图片

    for i in range(output_row):
        for j in range(output_col):
            # 输出图片中坐标 （i，j）对应至输入图片中的（m，n）
            m = round(i / output_row * input_row)
            n = round(j / output_col * input_col)
            # 防止四舍五入后越界
            if m >= input_row:
                m = input_row - 1
            if n >= input_col:
                n = input_col - 1
            # 插值
            output_signal[i, j] = input_signal_cp[m, n]

    return output_signal


def double_linear(input_signal, zoom_multiples):
    '''
    双线性插值
    :param input_signal: 输入图像
    :param zoom_multiples: 放大倍数
    :return: 双线性插值后的图像
    '''
    input_signal_cp = np.copy(input_signal)   # 输入图像的副本

    input_row, input_col = input_signal_cp.shape # 输入图像的尺寸（行、列）

    # 输出图像的尺寸
    output_row = int(input_row * zoom_multiples)
    output_col = int(input_col * zoom_multiples)

    output_signal = np.zeros((output_row, output_col)) # 输出图片

    for i in range(output_row):
        for j in range(output_col):
            # 输出图片中坐标 （i，j）对应至输入图片中的最近的四个点点（x1，y1）（x2, y2），（x3， y3），(x4，y4)的均值
            temp_x = i / output_row * input_row
            temp_y = j / output_col * input_col

            x1 = int(temp_x)
            y1 = int(temp_y)

            x2 = x1
            y2 = y1 + 1

            x3 = x1 + 1
            y3 = y1

            x4 = x1 + 1
            y4 = y1 + 1

            u = temp_x - x1
            v = temp_y - y1

            # 防止越界
            if x4 >= input_row:
                x4 = input_row - 1
                x2 = x4
                x1 = x4 - 1
                x3 = x4 - 1
            if y4 >= input_col:
                y4 = input_col - 1
                y3 = y4
                y1 = y4 - 1
                y2 = y4 - 1

            # 插值
            output_signal[i, j] = (1-u)*(1-v)*int(input_signal_cp[x1, y1]) + (1-u)*v*int(input_signal_cp[x2, y2]) + u*(1-v)*int(input_signal_cp[x3, y3]) + u*v*int(input_signal_cp[x4, y4])
    return output_signal