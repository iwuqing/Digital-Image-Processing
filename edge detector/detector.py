import numpy as np

def point(intput_signal, threshold):
    '''
    点检测（使用于灰度图）
    :param intput_signal:   输入图像
    :param threshold:   拉普拉斯算子阈值
    :return:    点图像
    '''
    intput_signal_cp = np.copy(intput_signal)   # 输入图像的副本

    m, n = intput_signal_cp.shape   # 输入图像的尺寸（行、列）

    output_signal = np.zeros((m, n))  # 检测点的输出

    # 空间滤波模板
    filtering_template = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ])

    point_matrix = np.zeros((3, 3)) # 矩阵灰度值，用于计算拉普拉斯算子

    for i in range(m):
        for j in range(n):
            # 该像素点灰度值
            point_matrix[1, 1] = intput_signal_cp[i, j]

            # 该像素点周围的点的灰度值，如若已至边界则视为0
            if i - 1 >= 0:
                point_matrix[0, 1] = intput_signal_cp[i-1, j]
            if i + 1 < m:
                point_matrix[2, 1] = intput_signal_cp[i+1, j]
            if j - 1 >= 0:
                point_matrix[1, 0] = intput_signal_cp[i, j-1]
            if j + 1 < n:
                point_matrix[1, 2] = intput_signal_cp[i, j+1]

            temp = abs(np.sum(point_matrix * filtering_template))    # 拉普拉斯算子值计算

            if temp >= threshold:
                output_signal[i, j] = 255  # 拉普拉斯算子大于阈值，则为亮点
            else:
                output_signal[i, j] = 0   #否则为暗点

            point_matrix = np.zeros((3, 3))  # 重置，为下一个点做准备

    return output_signal