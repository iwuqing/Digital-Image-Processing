import skimage as sk
import matplotlib.pyplot as plt
import noise
import denoising

if __name__ == '__main__':

    noisy_probability = 0.1  # 椒盐噪声概率

    # 加椒盐噪声
    camera_processed_with_salt_pepper, actual_noisy_data_num, theory_noisy_data_num = noise.salt_pepper(sk.data.camera(), noisy_probability)

    # 中值滤波处理
    camera_median = denoising.median_filtering(camera_processed_with_salt_pepper)

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.title("Source Image")
    plt.imshow(sk.data.camera(), cmap="gray")
    plt.subplot(1, 3, 2)
    plt.title("Image processed by Salt Pepper Noise: " + str(noisy_probability))
    plt.imshow(camera_processed_with_salt_pepper, cmap="gray")
    plt.subplot(1, 3, 3)
    plt.title("Image Processed by Median Filtering")
    plt.imshow(camera_median, cmap="gray")
    plt.show()