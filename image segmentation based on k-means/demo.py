from skimage import data
from skimage import io
from matplotlib import pyplot as plt
import segamentation

if __name__ == '__main__':


    image = data.coins()
    k = 3
    threshold = 1
    labels = segamentation.k_means(image, k, threshold)

    plt.subplot(1, 2, 1)
    plt.title("Soucre Image")
    plt.imshow(image,cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("Segamenting Image with k-means\n" + "k=" + str(k) + "  threshold=" + str(threshold))
    plt.imshow(labels/3)
    plt.show()