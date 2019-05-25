import skimage as sk
from matplotlib import pyplot as plt
import numpy as np
import enhancement

if __name__ == '__main__':

    imgae = sk.data.camera()
    imgae_hist = enhancement.hist_equalization(imgae)

    # 作灰度图对比
    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.title("Source Image")
    plt.imshow(imgae, cmap="gray")

    plt.subplot(1, 2, 2)
    plt.title("Image processed by Histogram equalization")
    plt.imshow(imgae_hist, cmap="gray")

    # 作直方图对比
    plt.figure(2)
    plt.subplot(1, 2, 1)
    plt.title("Histogram of the source image")
    plt.hist(np.array(imgae).flatten(), bins=256)

    plt.subplot(1, 2, 2)
    plt.title("Histogram of the image processed by Histogram equalization")
    plt.hist(np.array(imgae_hist).flatten(), bins=256)
    plt.show()
