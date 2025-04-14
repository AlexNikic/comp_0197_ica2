#!/usr/bin/env python3
"""
deeplabv3_training_offline_aug.py

A script that:
 1) Loads each original image + pseudo-mask,
 2) Applies data augmentation exactly once (random scaling, flips, color jitter),
 3) Saves the augmented image + mask in "offline_aug",
 4) Defines a dataset that loads these pre-augmented files,
 5) Performs hyperparameter random search (N trials) on a single fold,
 6) Saves the best config and final model,
 7) Trains with the best config on 90% of the offline dataset,
 8) Generates final predictions (binary masks + color-overlaid images).

Requirements:
  - PyTorch, torchvision
  - Pillow
"""

import os
import random
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from math import inf
from glob import glob
from PIL import Image

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.models.segmentation import deeplabv3_resnet50

# ---------- Global Config ----------
IMAGE_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
IMAGE_NORMALIZE_STD  = [0.229, 0.224, 0.225]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths for your data
CATS_FILE = "../Data/paths_cats_no_box.txt"
DOGS_FILE = "../Data/paths_dogs_no_box.txt"
PSEUDO_MASK_DIR = "better_boxsup"

# Output directories
OUTPUT_DIR = "deeplab_output"
OVERLAY_DIR = "deeplab_with_overlay"
AUGMENT_DIR = "offline_aug"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OVERLAY_DIR, exist_ok=True)
os.makedirs(AUGMENT_DIR, exist_ok=True)

# ---------- Set random seed ----------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------- Data Augmentation for Offline ----------
class RandomScaleCrop(object):
    def __call__(self, img, mask):
        scale = random.uniform(0.8,1.2)
        w,h = img.size
        new_w, new_h = int(w*scale), int(h*scale)
        img = img.resize((new_w,new_h), Image.BILINEAR)
        mask= mask.resize((new_w,new_h), Image.NEAREST)
        crop_w,crop_h= 224,224
        if new_w<crop_w or new_h<crop_h:
            pad_w= max(0,crop_w-new_w)
            pad_h= max(0,crop_h-new_h)
            img = TF.pad(img,(0,0,pad_w,pad_h))
            mask= TF.pad(mask,(0,0,pad_w,pad_h))
            new_w, new_h= img.size
        x= random.randint(0, new_w-crop_w)
        y= random.randint(0, new_h-crop_h)
        img = img.crop((x,y,x+crop_w,y+crop_h))
        mask=mask.crop((x,y,x+crop_w,y+crop_h))
        return img, mask

class OfflineAugTransform:
    """
    Offline transform: random scale/crop, random hflip, color jitter.
    We'll keep the mask 0..255, ensuring it's binary (0 or 255) at the end.
    """
    def __init__(self):
        self.scale_crop = RandomScaleCrop()
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    def __call__(self, img, mask):
        img, mask = self.scale_crop(img, mask)
        # random flip
        if random.random()<0.5:
            img= TF.hflip(img)
            mask= TF.hflip(mask)
        # color jitter
        img= self.color_jitter(img)
        return img, mask

def offline_augment_once():
    """
    Goes through all cat/dog images + pseudo-masks, 
    applies one random transform,
    saves them as 224×224 .png in offline_aug/images + offline_aug/masks.
    """
    image_paths= []
    for path_file in [CATS_FILE, DOGS_FILE]:
        if os.path.exists(path_file):
            with open(path_file,'r') as f:
                for line in f:
                    p= line.strip()
                    if p:
                        image_paths.append(p)
        else:
            print(f"Warning: {path_file} does not exist.")

    pairs=[]
    for ip in image_paths:
        base= os.path.splitext(os.path.basename(ip))[0]
        mp= os.path.join(PSEUDO_MASK_DIR, base+"_mask.png")
        if os.path.exists(mp):
            pairs.append((ip, mp))
        else:
            print(f"No mask for {ip}")
    print(f"OfflineAug: found {len(pairs)} pairs total.")

    images_dir= os.path.join(AUGMENT_DIR,"images")
    masks_dir = os.path.join(AUGMENT_DIR,"masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    aug= OfflineAugTransform()
    for (imgp, maskp) in pairs:
        base= os.path.splitext(os.path.basename(imgp))[0]
        img= Image.open(imgp).convert("RGB")
        msk= Image.open(maskp).convert("L")
        # apply once
        img_aug, msk_aug= aug(img, msk)
        # ensure mask is binary 0/255
        arr= np.array(msk_aug)
        bin_arr= (arr>128).astype(np.uint8)*255
        msk_aug_pil= Image.fromarray(bin_arr)

        # save
        img_out= os.path.join(images_dir,  base+".png")
        msk_out= os.path.join(masks_dir,  base+".png")
        img_aug.save(img_out)
        msk_aug_pil.save(msk_out)
    print(f"Done offline augmentation => {images_dir}, {masks_dir}")

# ---------- OfflineAugDataset ----------
class OfflineAugDataset(Dataset):
    """
    Each item in offline_aug/images has a matching item in offline_aug/masks
    with the same filename. We load them, convert to tensor, interpret mask 0..255 => 0..1.
    """
    def __init__(self, images_dir, masks_dir):
        super().__init__()
        self.images_dir= images_dir
        self.masks_dir = masks_dir
        self.files=[]
        for f in os.listdir(images_dir):
            if f.lower().endswith(".png"):
                base= os.path.splitext(f)[0]
                if os.path.exists(os.path.join(masks_dir,f)):
                    self.files.append(base)
        print(f"OfflineAugDataset: found {len(self.files)} items.")
        self.transform_img= transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_NORMALIZE_MEAN, IMAGE_NORMALIZE_STD)
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        base= self.files[idx]
        imgp= os.path.join(self.images_dir, base+".png")
        mskp= os.path.join(self.masks_dir, base+".png")
        img= Image.open(imgp).convert("RGB")
        msk= Image.open(mskp).convert("L")
        img_t= self.transform_img(img)
        arr= np.array(msk)
        bin_arr= (arr>128).astype(np.uint8)
        msk_t= torch.from_numpy(bin_arr).long()
        return img_t, msk_t, base

# ---------- Mixed Loss + mIoU ----------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    def forward(self, pred_logits, target):
        pred_fg= torch.softmax(pred_logits, dim=1)[:,1,:,:]
        target_fg= (target==1).float()
        intersection= (pred_fg*target_fg).sum(dim=[1,2])
        union= pred_fg.sum(dim=[1,2]) + target_fg.sum(dim=[1,2])
        dice= (2*intersection + self.smooth)/(union + self.smooth)
        return 1 - dice.mean()

class MixedLoss(nn.Module):
    def __init__(self, w_ce=0.7, w_dice=0.3):
        super().__init__()
        self.ce= nn.CrossEntropyLoss()
        self.dice= DiceLoss()
        self.w_ce= w_ce
        self.w_dice= w_dice
    def forward(self, logits, target):
        loss_ce= self.ce(logits, target)
        loss_dice= self.dice(logits, target)
        return self.w_ce*loss_ce + self.w_dice*loss_dice

def compute_mIoU(pred, target):
    pred= (pred>0.5).long()
    intersection= (pred & target).float().sum()
    union= (pred | target).float().sum()
    return (intersection+1e-6)/(union+1e-6)

# ---------- HPC: single-fold training ----------
def train_one_config(epochs, lr, batch_size, dataset, device):
    dataset_size= len(dataset)
    val_size= int(0.2*dataset_size)
    train_size= dataset_size - val_size
    if train_size<=0:
        print("Error: dataset too small for 80-20 split.")
        return 0.0
    
    train_ds, val_ds= random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_loader= DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    val_loader  = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    model= deeplabv3_resnet50(num_classes=2).to(device)
    optimizer= optim.Adam(model.parameters(), lr=lr)
    criterion= MixedLoss(w_ce=0.7, w_dice=0.3)

    best_iou=0.0
    for ep in range(epochs):
        model.train()
        for images,masks,names in train_loader:
            images,masks= images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits= model(images)["out"]
            loss= criterion(logits, masks)
            loss.backward()
            optimizer.step()
        # validation
        model.eval()
        total_iou=0.0
        with torch.no_grad():
            for images,masks,names in val_loader:
                images,masks= images.to(device), masks.to(device)
                logits= model(images)["out"]
                preds= torch.argmax(logits, dim=1)
                iou_= compute_mIoU(preds, masks)
                total_iou+= iou_.item()
        val_iou= total_iou/len(val_loader)
        if val_iou>best_iou:
            best_iou= val_iou
    return best_iou

def random_search_offline(dataset, device, n_search=5):
    best_config=None
    best_score=0.0
    for i in range(n_search):
        ep= random.randint(5,10)
        lr= 10**random.uniform(-4,-3)
        bs= random.choice([4,8,16])
        config= dict(epochs=ep, lr=lr, batch_size=bs)
        print(f"\nTrial {i+1}/{n_search}: {config}")
        val_iou= train_one_config(ep, lr, bs, dataset, device)
        print(f"  => val_iou={val_iou:.4f}")
        if val_iou> best_score:
            best_score= val_iou
            best_config= config
            print("   new best HPC!")
    return best_config, best_score

# ---------- Final training + predictions ----------
def final_train_and_predict(best_config, dataset, device):
    # 90/10
    n= len(dataset)
    val_size= int(0.1*n)
    train_size= n-val_size
    train_ds, val_ds= random_split(dataset, [train_size,val_size], generator=torch.Generator().manual_seed(42))
    train_loader= DataLoader(train_ds, batch_size=best_config['batch_size'], shuffle=True, pin_memory=True, drop_last=True)
    val_loader  = DataLoader(val_ds, batch_size=best_config['batch_size'], shuffle=False, pin_memory=True)

    model= deeplabv3_resnet50(num_classes=2).to(device)
    optimizer= optim.Adam(model.parameters(), lr=best_config['lr'])
    criterion= MixedLoss(w_ce=0.7,w_dice=0.3)

    best_iou=0.0
    for ep in range(best_config['epochs']):
        model.train()
        total_loss=0.0
        for images,masks,names in train_loader:
            images,masks= images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits= model(images)["out"]
            loss= criterion(logits, masks)
            loss.backward()
            optimizer.step()
            total_loss+= loss.item()
        avg_loss= total_loss/len(train_loader)

        # val
        model.eval()
        total_iou=0.0
        with torch.no_grad():
            for images,masks,names in val_loader:
                images,masks= images.to(device), masks.to(device)
                logits= model(images)["out"]
                preds= torch.argmax(logits, dim=1)
                iou_= compute_mIoU(preds, masks)
                total_iou+= iou_.item()
        val_iou= total_iou/len(val_loader)
        print(f"[FinalTrain] ep {ep+1}/{best_config['epochs']}: loss={avg_loss:.4f}, val_iou={val_iou:.4f}")
        if val_iou>best_iou:
            best_iou= val_iou
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_deeplabv3_model.pth"))
            print("   new best final model!")
    print(f"Final training done. best_iou={best_iou:.4f}")

    # predictions => use offline data
    checkpoint= os.path.join(OUTPUT_DIR, "best_deeplabv3_model.pth")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    transform_img= transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_NORMALIZE_MEAN, IMAGE_NORMALIZE_STD)
    ])
    overlay_dir= OVERLAY_DIR
    os.makedirs(overlay_dir, exist_ok=True)

    # We'll produce overlays for each item in the offline dataset
    for idx in range(len(dataset)):
        img_t, msk_t, base = dataset[idx]
        # For overlay, reload the augmented image from disk
        # If you prefer original images, you'd open them from the original path instead
        imgp= os.path.join(AUGMENT_DIR,"images", base+".png")
        orig_img= Image.open(imgp).convert("RGB")

        with torch.no_grad():
            inp= img_t.unsqueeze(0).to(device)
            out= model(inp)["out"]
            pred= torch.argmax(out, dim=1).squeeze(0)
        mask_np= (pred.cpu().byte().numpy()*255)
        raw_mask_path= os.path.join(OUTPUT_DIR, f"{base}_mask.png")
        Image.fromarray(mask_np).save(raw_mask_path)

        overlay_np= create_overlay(orig_img, mask_np, alpha=0.5)
        overlay_pil= Image.fromarray(overlay_np)
        overlay_pil.save(os.path.join(overlay_dir, f"{base}_overlay.jpg"), quality=85)

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

def main():
    set_seed(42)

    # Step1: offline augmentation if not done
    if not os.path.exists(os.path.join(AUGMENT_DIR,"images")):
        print("Performing offline augmentation once...")
        offline_augment_once()

    # Step2: define dataset from offline_aug
    images_dir= os.path.join(AUGMENT_DIR,"images")
    masks_dir = os.path.join(AUGMENT_DIR,"masks")
    offline_ds= OfflineAugDataset(images_dir, masks_dir)

    # Step3: HPC random search
    n_search=5
    best_config=None
    best_score=0.0
    if len(offline_ds)==0:
        print("Error: OfflineAugDataset is empty!")
        return
    for i in range(n_search):
        ep= random.randint(5,10)
        lr= 10**random.uniform(-4,-3)
        bs= random.choice([4,8,16])
        config= dict(epochs=ep, lr=lr, batch_size=bs)
        print(f"\nTrial {i+1}/{n_search}: {config}")
        val_iou= train_one_config(ep, lr, bs, offline_ds, DEVICE)
        print(f"  => val_iou={val_iou:.4f}")
        if val_iou> best_score:
            best_score= val_iou
            best_config= config
            print("   new best HPC!")

    if best_config is None:
        # fallback
        best_config= dict(epochs=8, lr=1e-4, batch_size=8)
        best_score= 0.0
    # save HPC
    with open(os.path.join(OUTPUT_DIR,"best_hparams.json"),"w") as f:
        json.dump({"best_config":best_config,"best_val_iou":best_score}, f)
    print(f"\nBEST HPC => {best_config}, val_iou={best_score:.4f}")

    # Step4: final train & predict
    final_train_and_predict(best_config, offline_ds, DEVICE)
    print("DONE.")

if __name__=="__main__":
    main()
