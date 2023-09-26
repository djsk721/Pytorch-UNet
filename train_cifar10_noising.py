import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate_cifar10 import evaluate_noising
from unet import UNet
from utils.load_cifar10 import make_cifar10
from utils.utils import add_gaussian_noise, add_poisson_noise, add_salt_pepper_noise
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss


# 1. psnr 측정 https://scikit-image.org/docs/stable/api/skimage.metrics.html scikit-image 공식 doc
# 2. dropout(dropout 적용한 U-Net https://www.kaggle.com/code/phoenigs/u-net-dropout-augmentation-stratification)
# 3. noise 잘못줌 skimage.util random noise를 통한 학습
def train_model(
        model,
        device,
        epochs: int = 1,
        noise_type: str = "gaussian",
        batch_size: int = 16,
        learning_rate: float = 1e-3,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        gradient_clipping: float = 1.0,
        dir_checkpoint: str = "outputs"
):
    # 1. Create dataset
    train_set, train_loader, val_set, val_loader = make_cifar10(batch_size)

    # 2. Split into train / validation partitions
    n_val = len(val_set)
    n_train = len(train_set)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )


    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    # optimizer = optim.SGD(model.parameters(), momentum=0.9,
    #                         lr=learning_rate, weight_decay=weight_decay)
    # optimizer = optim.RMSprop(model.parameters(),
    #                           lr=learning_rate, weight_decay=weight_decay)
    optimizer = optim.Adam(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.MSELoss()
    global_step = 0
    if noise_type == "gaussian":
        noise_func = add_gaussian_noise
    elif noise_type == "salt_pepper":
        noise_func = add_salt_pepper_noise
    else:
        noise_func = add_poisson_noise

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        noise_type:      {noise_type}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
        optimizer:       {optimizer}
    ''')
    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        psnr_score = []
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, noisy_images = batch[0], noise_func(batch[0].cpu().numpy())
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                noisy_images = noisy_images.to(device=device, dtype=torch.float32)

                # 3 channel
                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    image_pred = model(images)
                    loss = criterion(image_pred, noisy_images)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate_noising(model, val_loader, device, amp, noise_func)
                        psnr_score.append(val_score)
                        scheduler.step(val_score)

                        logging.info('Validation PSNR score: {}'.format(round(sum(psnr_score) / len(psnr_score), 4)))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'Validation PSNR': val_score,
                                'images': wandb.Image(images.cpu()),
                                'masks': {
                                    'true': wandb.Image(noisy_images.float().cpu()),
                                    'pred': wandb.Image(image_pred.float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            logging.info('error of logging to wandb!!')                            
                            pass
        # Mean of PSNR
        experiment.log({
            'Mean PSNR': round(sum(psnr_score) / len(psnr_score), 4)
        })

        experiment.finish()
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            # state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(Path(dir_checkpoint) / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-3,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=3, help='Number of classes')
    parser.add_argument('--dir', '-d', dest='dir_checkpoint', type=str, default="outputs", help='saving model directory')
    parser.add_argument('--noise_type', '-n', dest='noise_type', type=str, default="gaussian", help='decide noist type')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            noise_type=args.noise_type,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            # amp=args.amp,
            dir_checkpoint=args.dir_checkpoint
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            noise_type=args.noise_type,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            amp=args.amp,
            dir_checkpoint=args.dir_checkpoint
        )
