#!/usr/bin/env python3
"""
fully_supervised_training_inference.py

This script implements a fully supervised segmentation pipeline that:
  1) Reads image paths and bounding-box annotations from two text files
     (paths_cats_with_box.txt and paths_dogs_with_box.txt).
  2) For each image, creates a ground-truth mask by taking the union of all bounding boxes.
  3) Splits the data using a single-fold (80% training, 20% validation).
  4) Performs a grid search hyperparameter search (27 configurations: 3×3×3)
     over epochs, learning rate, and batch size.
  5) Selects the best configuration based on validation mIoU.
  6) Retrains a U-Net–style fully supervised segmentation model using the best hyperparameters.
  7) Runs inference on the entire dataset and saves:
         a. Predicted binary masks (PNG) in the folder "fully_sup_inference/masks".
         b. Overlay images (the original image blended with the predicted mask, as JPEG)
            in the folder "fully_sup_inference/overlays".
  8) Computes and prints the overall mIoU and pixel accuracy.

Requirements:
  - PyTorch (≥1.13 for MPS support), torchvision, Pillow, OpenCV, NumPy.
"""

import os
import random
import json
import time
import numpy as np
from math import inf
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image

# ---------------- Global Configuration ----------------
# Input text files (each line: <image_path> x1 y1 x2 y2)
CATS_BOX_FILE = "../Data/paths_cats_with_box.txt"
DOGS_BOX_FILE = "../Data/paths_dogs_with_box.txt"

# Output directories
MODEL_SAVE_DIR = "fully_sup_output"
MASKS_OUTPUT_DIR = os.path.join(MODEL_SAVE_DIR, "masks")
OVERLAYS_OUTPUT_DIR = os.path.join(MODEL_SAVE_DIR, "overlays")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(MASKS_OUTPUT_DIR, exist_ok=True)
os.makedirs(OVERLAYS_OUTPUT_DIR, exist_ok=True)

# Image resolution used during training/inference
IMG_SIZE = 224

# Normalization parameters
IMAGE_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
IMAGE_NORMALIZE_STD  = [0.229, 0.224, 0.225]

# Device selection: Use MPS (Apple Silicon) if available, else CUDA, else CPU.
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print("Using device =", DEVICE)

# ---------------- Set Seed ----------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE in ["cuda", "mps"]:
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ---------------- Dataset Definition ----------------
class FullySupBoxDataset(Dataset):
    """
    Reads bounding-box annotation lines from the two provided text files.
    For each image, the ground truth mask is generated by filling the union of 
    all bounding boxes with 1 (foreground), with the remainder as 0 (background).
    Both image and mask are resized to IMG_SIZE x IMG_SIZE.
    Lines with a bounding box of "0 0 0 0" are skipped.
    """
    def __init__(self, cats_file, dogs_file, img_size=224):
        self.img_size = img_size
        self.samples = {}  # Dictionary: image_path -> list of boxes (x1, y1, x2, y2)
        # Process each file
        for filepath in [cats_file, dogs_file]:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        img_path = parts[0]
                        # Parse box coordinates
                        x1, y1, x2, y2 = map(int, parts[1:5])
                        if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
                            continue
                        if img_path not in self.samples:
                            self.samples[img_path] = []
                        self.samples[img_path].append((x1, y1, x2, y2))
            else:
                print(f"Warning: {filepath} not found.")
        self.img_paths = list(self.samples.keys())
        print(f"FullySupBoxDataset: found {len(self.img_paths)} valid image entries.")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        boxes = self.samples[img_path]
        base = os.path.splitext(os.path.basename(img_path))[0]
        # Load original image.
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # Create a binary mask using the union of all bounding boxes.
        mask_np = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for (x1, y1, x2, y2) in boxes:
            # Ensure coordinates are within image dimensions.
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)
            if x2 > x1 and y2 > y1:
                mask_np[y1:y2, x1:x2] = 1

        # Resize image and mask to IMG_SIZE x IMG_SIZE.
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = Image.fromarray(mask_np).resize((self.img_size, self.img_size), Image.NEAREST)

        # Convert image to tensor and normalize.
        img_np = np.array(img).astype(np.float32) / 255.0  # (H, W, 3)
        img_np = np.transpose(img_np, (2, 0, 1))            # (3, H, W)
        img_tensor = torch.from_numpy(img_np)
        for i, mean in enumerate(IMAGE_NORMALIZE_MEAN):
            img_tensor[i] = (img_tensor[i] - mean) / IMAGE_NORMALIZE_STD[i]

        # Convert mask to tensor (long)
        mask_tensor = torch.from_numpy(np.array(mask)).long()
        return img_tensor, mask_tensor, base

# ---------------- Data Augmentation ----------------
# (For training only, we include standard data augmentation.)
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGE_NORMALIZE_MEAN, IMAGE_NORMALIZE_STD)
])
# For validation/inference, we use resizing and normalization.
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGE_NORMALIZE_MEAN, IMAGE_NORMALIZE_STD)
])

# ---------------- Model Definition (U-Net) ----------------
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2):
        super(UNet, self).__init__()
        # Encoder
        self.enc1 = self.contracting_block(in_channels, 64)
        self.enc2 = self.contracting_block(64, 128)
        self.enc3 = self.contracting_block(128, 256)
        # Decoder
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self.expansive_block(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self.expansive_block(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def contracting_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return block

    def expansive_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return block

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, kernel_size=2))
        enc3 = self.enc3(F.max_pool2d(enc2, kernel_size=2))
        # Decoder
        dec3 = self.upconv3(enc3)
        dec3 = torch.cat((dec3, enc2), dim=1)
        dec3 = self.dec3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc1), dim=1)
        dec2 = self.dec2(dec2)
        out = self.final_conv(dec2)
        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        return out

# ---------------- Loss Functions and Metrics ----------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred_logits, target):
        # pred_logits: (B,2,H,W), target: (B,H,W)
        pred_fg = torch.softmax(pred_logits, dim=1)[:, 1, :, :]
        target_fg = target.float()
        intersection = (pred_fg * target_fg).sum(dim=[1,2])
        union = pred_fg.sum(dim=[1,2]) + target_fg.sum(dim=[1,2])
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class MixedLoss(nn.Module):
    def __init__(self, w_ce=0.7, w_dice=0.3):
        super(MixedLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.w_ce = w_ce
        self.w_dice = w_dice

    def forward(self, logits, target):
        loss_ce = self.ce(logits, target)
        loss_dice = self.dice(logits, target)
        return self.w_ce * loss_ce + self.w_dice * loss_dice

def compute_mIoU(pred, target):
    """Compute mean Intersection over Union for a single sample or batch.
       pred and target are numpy arrays of shape (H,W) with values {0,1}.
    """
    pred_bin = (pred > 0.5).astype(np.uint8)
    intersection = np.logical_and(pred_bin, target).sum()
    union = np.logical_or(pred_bin, target).sum()
    return (intersection + 1e-6) / (union + 1e-6)

def compute_pixel_accuracy(pred, target):
    """Compute pixel accuracy, where pred and target are numpy arrays (H,W) with {0,1}."""
    correct = (pred == target).sum()
    total = pred.size
    return correct / total

# ---------------- Hyperparameter Search ----------------
# We'll define a grid for 3×3×3 = 27 configurations.
EPOCHS_OPTIONS = [5, 10, 15]
LR_OPTIONS = [1e-4, 5e-4, 1e-3]
BATCH_OPTIONS = [4, 8, 16]

def train_one_config(epochs, lr, batch_size, dataset, device, training_transform):
    """Train the UNet model on an 80-20 split for the given hyperparameters and return the validation mIoU."""
    # Split dataset into 80% training, 20% validation.
    n = len(dataset)
    val_size = int(0.2 * n)
    train_size = n - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    # For training, apply the training_transform to images.
    def apply_transform(ds, transform):
        # Wrap each sample: re-load the image, apply transform to image, leave mask as is.
        new_samples = []
        for img_tensor, mask_tensor, base in ds:
            # Convert mask back to PIL for no change.
            img_pil = Image.open(dataset.img_paths[dataset.img_paths.index(base)]).convert("RGB")  # Not used here.
            new_samples.append((img_tensor, mask_tensor, base))
        return ds  # We assume dataset contains already pre-processed images.
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    model = UNet(in_channels=3, out_channels=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = MixedLoss(w_ce=0.7, w_dice=0.3)
    
    best_val_iou = 0.0
    for epoch in range(epochs):
        model.train()
        for images, masks, _ in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(images)  # logits shape: (B,2,224,224)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
        model.eval()
        total_iou = 0.0
        count = 0
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                preds = torch.argmax(logits, dim=1)
                preds_np = preds.cpu().numpy()
                masks_np = masks.cpu().numpy()
                for i in range(preds_np.shape[0]):
                    iou = compute_mIoU(preds_np[i], masks_np[i])
                    total_iou += iou
                    count += 1
        val_iou = total_iou / count if count > 0 else 0
        if val_iou > best_val_iou:
            best_val_iou = val_iou
    return best_val_iou

def hyperparameter_search(dataset, device):
    best_config = None
    best_score = 0.0
    total_trials = 0
    # Iterate over the grid (3x3x3 = 27 combinations)
    for epochs in EPOCHS_OPTIONS:
        for lr in LR_OPTIONS:
            for batch_size in BATCH_OPTIONS:
                total_trials += 1
                config = {"epochs": epochs, "lr": lr, "batch_size": batch_size}
                print(f"\nTrial {total_trials}/27, config: {config}")
                # Train on one fold
                val_iou = train_one_config(epochs, lr, batch_size, dataset, device, training_transform=train_transform)
                print(f"  => Validation mIoU: {val_iou:.4f}")
                if val_iou > best_score:
                    best_score = val_iou
                    best_config = config
                    print("  -> New best configuration found!")
    print(f"\nBest config from hyperparameter search: {best_config} with mIoU = {best_score:.4f}")
    return best_config

# ---------------- Final Training and Inference ----------------
def final_train_and_infer(best_config, dataset, device):
    # We'll use a 90/10 split for final training.
    n = len(dataset)
    val_size = int(0.1 * n)
    train_size = n - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=best_config['batch_size'], shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=best_config['batch_size'], shuffle=False, pin_memory=True)
    
    model = UNet(in_channels=3, out_channels=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=best_config['lr'])
    criterion = MixedLoss(w_ce=0.7, w_dice=0.3)
    
    best_val_iou = 0.0
    for epoch in range(best_config['epochs']):
        model.train()
        total_loss = 0.0
        for images, masks, _ in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        
        model.eval()
        total_iou = 0.0
        count = 0
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                preds = torch.argmax(logits, dim=1)
                preds_np = preds.cpu().numpy()
                masks_np = masks.cpu().numpy()
                for i in range(preds_np.shape[0]):
                    iou = compute_mIoU(preds_np[i], masks_np[i])
                    total_iou += iou
                    count += 1
        val_iou = total_iou / count if count > 0 else 0
        print(f"[FinalTrain] Epoch {epoch+1}/{best_config['epochs']} - Loss: {avg_loss:.4f}, Val mIoU: {val_iou:.4f}")
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "best_model.pth"))
            print("   -> New best model saved.")
    print(f"Final training complete. Best validation mIoU: {best_val_iou:.4f}")

    # Inference on the entire dataset and calculation of final metrics.
    model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR, "best_model.pth"), map_location=device))
    model.eval()
    total_iou = 0.0
    total_px = 0.0
    count = 0
    all_img_paths = dataset.img_paths  # from our FullySupBoxDataset
    for img_path in all_img_paths:
        base = os.path.splitext(os.path.basename(img_path))[0]
        # Load image and recreate ground truth mask from bounding boxes.
        # (We re-use the dataset __getitem__ logic.)
        sample = dataset.__getitem__(dataset.img_paths.index(img_path))
        img_tensor, gt_mask, _ = sample
        img_tensor = img_tensor.unsqueeze(0).to(device)
        gt_np = gt_mask.numpy().astype(np.uint8)  # {0,1} shape (224,224)
        with torch.no_grad():
            logits = model(img_tensor)
            pred = torch.argmax(logits, dim=1).squeeze(0)
        pred_np = (pred.cpu().numpy() * 255).astype(np.uint8)
        pred_bin = (pred_np > 128).astype(np.uint8)
        iou = compute_mIoU(pred_bin, gt_np)
        px_acc = compute_pixel_accuracy(pred_bin, gt_np)
        total_iou += iou
        total_px += px_acc
        count += 1

        # Save predicted mask and overlay.
        out_mask_path = os.path.join(MODEL_SAVE_DIR, f"{base}_pred.png")
        Image.fromarray(pred_np).save(out_mask_path)
        # Create overlay: blend original image and predicted mask.
        orig_img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        overlay = create_overlay(orig_img, pred_np, alpha=0.5)
        overlay_path = os.path.join(OVERLAYS_OUTPUT_DIR, f"{base}_overlay.jpg")
        Image.fromarray(overlay).save(overlay_path, quality=85)
    if count > 0:
        mean_iou = total_iou / count
        mean_px = total_px / count
        print("\nFinal Test Metrics:")
        print(f"Mean IoU: {mean_iou:.4f}")
        print(f"Pixel Accuracy: {mean_px:.4f}")
    else:
        print("No samples for final evaluation.")

def create_overlay(orig_img, pred_mask_np, alpha=0.5):
    """
    Creates an overlay image by blending the original image with a color-coded mask.
    Foreground (where the predicted mask equals 255) is shown in red,
    while background (mask equals 0) is shown in blue.
    This function uses Pillow for blending instead of OpenCV.
    """
    # Convert the original image to RGB (if not already) and ensure consistent size.
    orig_rgb = orig_img.convert("RGB")
    # Create a colored mask using numpy.
    orig_np = np.array(orig_rgb)
    colored_mask = np.zeros_like(orig_np, dtype=np.uint8)
    # Set foreground to red and background to blue.
    colored_mask[pred_mask_np == 255] = [255, 0, 0]
    colored_mask[pred_mask_np == 0]   = [0, 0, 255]
    # Convert the numpy array mask to a PIL image.
    mask_img = Image.fromarray(colored_mask)
    # Blend the original image and the colored mask.
    overlay_img = Image.blend(orig_rgb, mask_img, alpha)
    return np.array(overlay_img)

# ---------- MAIN EXECUTION ----------
def main():
    set_seed(42)
    # Build the dataset.
    dataset = FullySupBoxDataset(CATS_BOX_FILE, DOGS_BOX_FILE, img_size=IMG_SIZE)
    # For hyperparameter search, we use the dataset as-is.
    print(f"Dataset contains {len(dataset)} images.")
    
    # Hyperparameter search (grid search over 3x3x3 = 27 configurations)
    best_config = None
    best_score = 0.0
    total_trials = 0
    for epochs in EPOCHS_OPTIONS:
        for lr in LR_OPTIONS:
            for bs in BATCH_OPTIONS:
                total_trials += 1
                config = {"epochs": epochs, "lr": lr, "batch_size": bs}
                print(f"\nTrial {total_trials}/27, config: {config}")
                val_iou = train_one_config(epochs, lr, bs, dataset, DEVICE, training_transform=train_transform)
                print(f"  => Validation mIoU: {val_iou:.4f}")
                if val_iou > best_score:
                    best_score = val_iou
                    best_config = config
                    print("  -> New best configuration!")
    print(f"\nBest hyperparameters: {best_config} with mIoU = {best_score:.4f}")
    # Save best hyperparameters.
    with open(os.path.join(MODEL_SAVE_DIR, "best_hparams.json"), "w") as f:
        json.dump({"best_config": best_config, "best_val_mIoU": best_score}, f)
    
    # Final training and inference.
    final_train_and_infer(best_config, dataset, DEVICE)

if __name__ == "__main__":
    main()
