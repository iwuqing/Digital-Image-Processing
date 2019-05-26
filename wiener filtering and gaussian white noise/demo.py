from matplotlib import pyplot as plt
import skimage as sk
import noise
import numpy as np
import filtering
import math
import scipy.signal


if __name__ == '__main__':

    image = sk.data.moon()

    # 抖动噪声
    shake_noise = np.eye(20) / 5

    # 噪声与源图像卷积
    image_add_shake = scipy.signal.convolve2d(image, shake_noise, mode = 'same')

    # 增加加性高斯白噪声
    image_add_shake_and_gwn = noise.gaussian_white_noise(image_add_shake, 0, 10)

    # 求解退化函数h
    image_fft = np.fft.fft2(image)
    image_add_noise_fft = np.fft.fft2(image_add_shake)
    h = np.fft.ifft2(image_add_noise_fft / image_fft)

    # 逆滤波
    image_processed_inverse_filtering = filtering.inverse_filtering(image_add_shake_and_gwn, h)

    # 维纳滤波
    K = 0.03
    image_processed_wiener_filtering = filtering.wiener_filtering(image_add_shake_and_gwn, h, K)

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.title("Source Image")
    plt.imshow(image, cmap="gray")

    plt.subplot(2, 2, 2)
    plt.title("Image through H and Gaussian white noise(mu=0, sigma=10)")
    plt.imshow(image_add_shake_and_gwn, cmap="gray")

    plt.subplot(2, 2, 3)
    plt.title("Image processed with Inverse filtering")
    plt.imshow(image_processed_inverse_filtering, cmap="gray")

    plt.subplot(2, 2, 4)
    plt.title("Image processed with Wiener filtering: K=" + str(K))
    plt.imshow(image_processed_wiener_filtering, cmap="gray")
    plt.show()