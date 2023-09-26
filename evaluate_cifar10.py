import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np

@torch.inference_mode()
def evaluate(net, dataloader, device, amp, noise_func):
    net.eval()
    num_val_batches = len(dataloader)
    Psnr = []

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            images, noisy_images = batch[0], noise_func(batch[0].cpu().numpy())
            # move images and labels to correct device and type
            noisy_images = noisy_images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            images = images.to(device=device, dtype=torch.float32)

            # predict the mask
            pred_images = net(noisy_images)
            # skimage.metrics.peak_signal_noise_ratio(image_true, image_test, *, data_range=None)
            Psnr.append(psnr(images.cpu().numpy(), pred_images.cpu().numpy()))

    net.train()
    return round(np.mean(Psnr), 4)

@torch.inference_mode()
def evaluate_noising(net, dataloader, device, amp, noise_func):
    net.eval()
    num_val_batches = len(dataloader)
    Psnr = []

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            images, noisy_images = batch[0], noise_func(batch[0].cpu().numpy())
            # move images and labels to correct device and type
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            noisy_images = noisy_images.to(device=device, dtype=torch.float32)

            # predict the mask
            pred_images = net(images)
            # skimage.metrics.peak_signal_noise_ratio(image_true, image_test, *, data_range=None)
            Psnr.append(psnr(noisy_images.cpu().numpy(), pred_images.cpu().numpy()))

    net.train()
    return round(np.mean(Psnr), 4)
