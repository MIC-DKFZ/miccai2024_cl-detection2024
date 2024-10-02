# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: Apache-2.0

"""
Project: CL-Detection2024 Challenge Baseline
============================================

This script implements cephalometric landmark detection on X-Ray images using a UNet-based heatmap approach.
It includes functionality for model training and validation.

Email: zhanghongyuan2017@email.szu.edu.cn
"""

import os
import tqdm
import torch
import argparse
import numpy as np
import random
from types import SimpleNamespace
import wandb
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf, DictConfig
from PIL import Image
import matplotlib.pyplot as plt

from torchvision.utils import save_image
from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import warnings

warnings.filterwarnings("ignore")

from utils.transforms import Rescale, ToTensor, RandomHorizontalFlip, RandomRotation, GaussianNoise, ZScoreNormalization, Grayscale, RandomTranslation, ElasticTransform
from utils.dataset import CephXrayDataset
from utils.model import load_model
from utils.losses import load_loss

from utils.cldetection_utils import check_and_make_dir

# use hydra for managing configs
import hydra
from hydra.core.hydra_config import HydraConfig
import yaml
import sys
import subprocess


def get_workers_for_current_node() -> int:
    hostname = subprocess.getoutput(["hostname"])  # type: ignore

    if hostname in ["your-worker-nodes"]:
        return 16
    else:
        return 2
    # else:
    #    raise NotImplementedError()
    
def update_reduction_factor(sigma_reduction_factor):
    sigma_reduction_factor += sigma_reduction_factor*0.002
    return min(1.0, sigma_reduction_factor)
    
def update_sigma(sigma, sigma_reduction_factor):
    sigma_reduction_factor = update_reduction_factor(sigma_reduction_factor)
    return sigma * sigma_reduction_factor


def normalize_image_for_visualization(image):
    """
    Normalize a Z-score normalized image to the [0, 255] range for visualization.
    """
    image_min = image.min()
    image_max = image.max()
    image_normalized = 255 * (image - image_min) / (image_max - image_min)
    return image_normalized.astype('uint8')

def log_images_in_training(image, heatmap, epoch):    
    image = image.cpu().numpy().astype('float32').transpose(1, 2, 0)
    heatmap = heatmap.cpu().numpy().transpose(1, 2, 0)

    image_normalized = normalize_image_for_visualization(image)
    image_normalized = np.stack((image_normalized[:,:,0], image_normalized[:,:,0], image_normalized[:,:,0]), axis=-1)

    heatmap_combined = np.max(heatmap, axis=-1)  # Assuming multi-channel landmarks
    heatmap_normalized = (heatmap_combined * 255).astype('uint8')

    image_pil = Image.fromarray(image_normalized)
    heatmap_pil = Image.fromarray(heatmap_normalized).convert("RGB")

    overlay_image = Image.blend(image_pil, heatmap_pil, alpha=0.5)
    wandb.log({f"Heatmap_overlay": wandb.Image(overlay_image, caption=f"Heatmap_epoch_{epoch}")})
     

def save_augmented_images(dataset, name="train", output_dir="/path/to/project/2024_MICCAI24_CL-Detection2024_challenge/augm/", num_images=10):
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_images):
        sample = dataset[i]
        image, landmarks = sample

        # Convert image tensor to NumPy array if necessary
        if hasattr(image, 'numpy'):
            image = image.numpy().astype('float32').transpose(1, 2, 0)
        if hasattr(landmarks, 'numpy'):
            landmarks = landmarks.numpy().transpose(1, 2, 0)

        # Normalize the Z-scored image to [0, 255] for visualization
        image_normalized = normalize_image_for_visualization(image)
        image_normalized = np.stack((image_normalized[:,:,0], image_normalized[:,:,0], image_normalized[:,:,0]), axis=-1)
        # Convert landmarks to a displayable range (optional if needed)
        landmarks_combined = np.max(landmarks, axis=-1)  # Assuming multi-channel landmarks
        print(np.amax(landmarks_combined))
        print(np.amin(landmarks_combined))
        landmarks_normalized = (landmarks_combined * 255).astype('uint8')

        # Convert to PIL images
        image_pil = Image.fromarray(image_normalized)
        landmarks_pil = Image.fromarray(landmarks_normalized).convert("RGB")

        # Overlay the landmarks onto the image
        overlay_image = Image.blend(image_pil, landmarks_pil, alpha=0.5)
        overlay_save_path = os.path.join(output_dir, f"{name}_augmented_{i}_with_landmarks.png")
        image_save_path = os.path.join(output_dir, f"{name}_augmented_{i}.png")
        overlay_image.save(overlay_save_path)
        image_pil.save(image_save_path)

        print(f"Saved overlay image: {overlay_save_path}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


@hydra.main(config_path="configs", config_name="config_local")
def main_hydra(config):
    """
    Main function for model training and validation using hydra configs.py.
    :param config: Configuration object containing various configuration parameters.
    """
    # print the config
    print("Config:")
    print(OmegaConf.to_yaml(config))
    print("Config done.")

    # Convert the config to a dictionary
    config_dict = OmegaConf.to_container(config, resolve=True)

    # Flatten the dictionary
    flat_config = flatten_dict(config_dict)

    # Initialize wandb with the flattened config
    wandb.init(
        project="CL-Detection",
        config=flat_config
    )

    # Logging with Tensorboard
    if HydraConfig.get().mode == hydra.types.RunMode.RUN:
        config.save_model_dir = HydraConfig.get().run.dir
    else:
        config.save_model_dir = os.path.join(
            HydraConfig.get().sweep.dir, HydraConfig.get().sweep.subdir
        )

    train(config)


def train(config):
    """
    Main function for model training and validation.
    """

    # Set the seed
    seed = 13
    set_seed(seed)

    # GPU device | GPU设备
    gpu_id = config.cuda_id
    print(config)
    num_workers = get_workers_for_current_node()

    device = torch.device(
        "cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu"
    )

    sigma = config.sigma
    sigma_reduction_factor = config.sigma_reduction_factor
    to_tensor_transform = ToTensor(sigma)
    # Train and validation dataset | 训练和验证数据集
    if config.transform_name == "spatial_augmentation":
        train_transform = transforms.Compose([
            Rescale(output_size=(config.image_height, config.image_width)),
            Grayscale(), 
            #RandomHorizontalFlip(p=0.3),  # Horizontal flipping might not be anatomically valid for X-rays
            RandomRotation(10),
            #ElasticTransform(alpha=1.0, sigma=0.2, p=1),
            to_tensor_transform,
            RandomTranslation(translate_range=(0, 0.05)),
        ])
    else:
        raise ValueError(f"Unknown transform_name: {config.transform_name}")

    if config.image_transform_name == "random_noise":
        train_image_transform = transforms.Compose([
            v2.RandomApply([GaussianNoise(noise_variance=(0, 0.05))], p=0.2),  # Slightly reduced noise variance
            v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.9, 1.0))], p=0.2),  # Smaller kernel size for subtle blurring
            v2.RandomApply([v2.ColorJitter(brightness=0.1, contrast=0.1)], p=0.2),  # Reduced jitter for more realistic variations
            ZScoreNormalization(),
        ])

    else:
        train_image_transform = None
        print("No image specifc transform applied.")

    valid_transform = transforms.Compose([
        Rescale(output_size=(config.image_height, config.image_width)),
        Grayscale(),  
        to_tensor_transform,
        ])
    
    valid_image_transform = transforms.Compose([
        ZScoreNormalization(),
        ])

    train_dataset = CephXrayDataset(
        csv_file_path=config.train_csv_path,
        transform=train_transform,
        image_transform=train_image_transform,
        **config.dataset_kwargs
        )
    
    valid_dataset = CephXrayDataset(
        csv_file_path=config.valid_csv_path, 
        transform=valid_transform, 
        image_transform=valid_image_transform,
        **config.val_dataset_kwargs
        )
    
    save_augmented_images(train_dataset)
    save_augmented_images(valid_dataset, name="val")

    # Train and validation data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size_valid,
        shuffle=False,
        num_workers=num_workers,
        )

    # Training praparation
    model = load_model(model_name=config.model_name, **config.get('model_kwargs', {}))
    print(model)
    model = model.to(device)
    if os.path.exists(config.pretrained_model_weights):
        model.load_state_dict(torch.load(config.pretrained_model_weights))
        print("Loaded weights from " + config.pretrained_model_weights)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, betas=(config.beta1, config.beta2)
    )
    
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma
    )

    loss_fn = load_loss(loss_name=config.loss_name)

    best_loss = float('inf') 
    num_epoch_no_improvement = 0  
    check_and_make_dir(config.save_model_dir) 

    print("Starting model training...")

    for epoch in range(config.train_max_epoch):
        
        train_losses = []  
        valid_losses = [] 
        logged_first_image = False

        model.train()
        for image, heatmap in tqdm.tqdm(train_loader):
            image, heatmap = image.float().to(device), heatmap.float().to(device)

            output = model(image)
            loss = loss_fn(output, heatmap)
            train_losses.append(loss.item())

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if not logged_first_image:
            #     log_images_in_training(image[0], heatmap[0], epoch)
            #     logged_first_image = True

        train_loss = np.mean(train_losses)
        print("Train epoch [{:<4d}/{:<4d}], Loss: {:.6f}".format(epoch + 1, config.train_max_epoch, train_loss))

        # Save model checkpoint
        if epoch % config.save_model_step == 0:
            checkpoint_path = os.path.join(config.save_model_dir, "checkpoint_epoch_%s.pt" % epoch)
            torch.save(model.state_dict(), checkpoint_path)
            print("Saving checkpoint model ", checkpoint_path)

        # Validate model, save best model 
        with torch.no_grad():
            model.eval()
            print("Validating....")
            for image, heatmap in tqdm.tqdm(valid_loader):
                image, heatmap = image.float().to(device), heatmap.float().to(device)

                output = model(image)
                
                loss = loss_fn(output, heatmap)
                valid_losses.append(loss.item())
        
        valid_loss = np.mean(valid_losses)
        print("Validation loss:  {:.6f}".format(valid_loss))

        wandb.log({"train_loss": train_loss, "val_loss": valid_loss, "lr": optimizer.param_groups[0]['lr'], "sigma": sigma})

        # Update lR scheduler and sigma
        scheduler.step(epoch)
        # dynamic sigma
        if config.get("dynamic_sigma", False) and epoch % 2 == 0:
            sigma = update_sigma(sigma, sigma_reduction_factor)  # Replace with your logic to compute the new sigma value
            sigma_reduction_factor = update_reduction_factor(sigma_reduction_factor)
            to_tensor_transform.set_sigma(sigma) 

         # Best model
        if valid_loss < best_loss:
            print("Validation loss decreases from {:.6f} to {:.6f}".format(best_loss, valid_loss))
            best_loss = valid_loss
            torch.save(model.state_dict(),os.path.join(config.save_model_dir, "best_model.pt"))
            print("Saving best model ",os.path.join(config.save_model_dir, "best_model.pt"))

    # Save the last model
    torch.save(model.state_dict(), os.path.join(config.save_model_dir, f"final_model.pt"),)



if __name__ == "__main__":

    wandb.login(key='your-wandb-key')

    # use hydra for managing configs
    # Remove `--configtype` from `sys.argv` to avoid confusion for Hydra
    main_hydra()
