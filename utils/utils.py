import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_img_and_mask(img, mask):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def add_gaussian_noise(image, stddev=0.05):
    """
    Add Gaussian noise to an image tensor.

    Args:
        image (torch.Tensor): Input image tensor (batch_size x channels x height x width).
        stddev (float): Standard deviation of the Gaussian noise.

    Returns:
        torch.Tensor: Noisy image tensor with values clamped between 0 and 1.
    """
    # Generate random noise with the same size as the image
    noise = torch.randn_like(image) * stddev

    # Add the noise to the image
    noisy_image = image + noise

    # Clip the pixel values to be in the range [0, 1]
    noisy_image = torch.clamp(noisy_image, 0, 1)

    return noisy_image