import detector
import skimage as sk
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':

    image = sk.data.coins()

    threshold = 70
    image_point = detector.point(image, threshold)

    plt.subplot(1, 2, 1)
    plt.title("Source Image")
    plt.imshow(image, cmap="gray")

    plt.subplot(1, 2, 2)
    plt.title("Image with point detector: threshold=" + str(threshold))
    plt.imshow(image_point, cmap="gray")

    plt.show()
