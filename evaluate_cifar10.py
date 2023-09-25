import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from utils.dice_score import multiclass_dice_coeff, dice_coeff
from utils.utils import add_gaussian_noise
import numpy as np

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    Psnr = []

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            noisy_images, images = add_gaussian_noise(batch[0]), batch[0]

            # move images and labels to correct device and type
            noisy_images = noisy_images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            images = images.to(device=device, dtype=torch.float32)

            # predict the mask
            pred_images = net(noisy_images)

            Psnr.append(psnr(images.cpu().numpy(), pred_images.cpu().numpy()))

    net.train()
    return round(np.mean(Psnr), 4)

