import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as dist
from skimage.util import random_noise


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

def add_gaussian_noise(image, mean=0, var=0.05**2):
    gauss_img = torch.tensor(random_noise(image, mode='gaussian', mean=mean, var=var, clip=True))
    return gauss_img

def add_salt_pepper_noise(image):
    s_and_p = torch.tensor(random_noise(image, mode='s&p', salt_vs_pepper=0.5, clip=True))
    return s_and_p

# mean = 0.003에 대한 검증 필요
def add_poisson_noise(image):
    # poisson_img = torch.tensor(random_noise(image, mode='poisson', mean=0.003, clip=True))
    poisson_img = torch.tensor(random_noise(image, mode='poisson', clip=True))
    return poisson_img