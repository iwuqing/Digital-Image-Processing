import skimage as sk
import interpolation
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':

    image = sk.data.coins()

    zoom_multiples = 0.5
    image_processed_with_nearest_neighbor = interpolation.nearest_neighbor(image, zoom_multiples)

    image_processed_with_linear = interpolation.double_linear(image, zoom_multiples)

    plt.imsave("image.png", image)
    plt.imsave("image_processed_with_nearest_neighbor.png", image_processed_with_nearest_neighbor)
    plt.imsave("image_processed_with_linear.png", image_processed_with_linear)

