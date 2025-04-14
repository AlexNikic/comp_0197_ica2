#!/usr/bin/env python3
"""
deeplabv3_inference_boxfiles_mps.py

This script loads a saved DeepLabv3 best model from "deeplab_output/best_deeplabv3_model.pth"
and runs inference on images listed in the bounding-box text files:
    paths_cats_with_box.txt
    paths_dogs_with_box.txt
It skips any lines where the bounding box coordinates are "0 0 0 0".

For each valid line (formatted as):
    image_path x1 y1 x2 y2
the following steps are performed:
  - Load the original image.
  - Load the corresponding pseudo-mask from "./better_boxsup/<base>_mask.png".
  - Run the image through the model to generate a predicted segmentation mask.
  - Compare the predicted mask with the pseudo-mask to compute mIoU and pixel accuracy.
  - Save the predicted mask in "inference_boxfiles_masks".
  - If enabled, create an overlay image (with red for foreground and blue for background)
    and save it in "inference_boxfiles_overlays".

The script attempts to use the MPS backend (for Apple Silicon) if available, falling back to CUDA or CPU.
"""

import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image

# ------------------ CONFIGURATION ------------------
CATS_BOX_FILE   = "../Data/paths_cats_with_box.txt"
DOGS_BOX_FILE   = "../Data/paths_dogs_with_box.txt"
PSEUDO_MASK_DIR = "./better_boxsup"

BEST_MODEL_PATH = "deeplab_output/best_deeplabv3_model.pth"

OUT_MASKS_DIR   = "inference_boxfiles_masks"
os.makedirs(OUT_MASKS_DIR, exist_ok=True)

SAVE_OVERLAYS   = True
OVERLAY_DIR     = "inference_boxfiles_overlays"
if SAVE_OVERLAYS:
    os.makedirs(OVERLAY_DIR, exist_ok=True)

# ------------------ Device Selection ------------------
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print(f"Using device={DEVICE}")

IMAGE_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
IMAGE_NORMALIZE_STD  = [0.229, 0.224, 0.225]

# ------------------ Helper Metrics ------------------
def compute_mIoU_and_pixel_accuracy(pred_bin, gt_bin):
    """
    Computes the Intersection over Union (IoU) and pixel accuracy.
    
    Parameters:
      pred_bin (ndarray): Binary prediction mask (shape: H x W, values 0 or 1)
      gt_bin (ndarray): Binary ground truth mask (shape: H x W, values 0 or 1)
    
    Returns:
      iou (float): Intersection over Union.
      pix_acc (float): Pixel accuracy.
    """
    intersection = (pred_bin & gt_bin).sum()
    union = (pred_bin | gt_bin).sum()
    iou = (intersection + 1e-6) / (union + 1e-6)

    correct = (pred_bin == gt_bin).sum()
    total = pred_bin.size
    pix_acc = correct / total
    return iou, pix_acc

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
# ------------------ Dataset Definition ------------------
class BoxFilesDataset(Dataset):
    """
    Reads bounding-box text files (from cats and dogs). Each line is of the form:
         image_path x1 y1 x2 y2
    Lines where the bounding box is "0 0 0 0" are skipped.
    For each valid image, it loads the image and its corresponding pseudo-mask from:
         ./better_boxsup/<base>_mask.png
    """
    def __init__(self, cats_file, dogs_file, pseudo_mask_dir):
        super().__init__()
        self.pseudo_mask_dir = pseudo_mask_dir
        self.transform_img = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_NORMALIZE_MEAN, IMAGE_NORMALIZE_STD)
        ])
        self.image_mask_pairs = []
        for f in [cats_file, dogs_file]:
            if os.path.exists(f):
                with open(f, 'r') as file:
                    for line in file:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) < 5:
                            continue
                        img_path = parts[0]
                        x1, y1, x2, y2 = map(int, parts[1:5])
                        # Skip lines with no valid bounding box:
                        if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
                            continue
                        # Get the base name (string) properly:
                        base = os.path.splitext(os.path.basename(img_path))[0]
                        mask_path = os.path.join(pseudo_mask_dir, base + "_mask.png")
                        if os.path.exists(mask_path):
                            self.image_mask_pairs.append((img_path, mask_path))
                        else:
                            print(f"Warning: no pseudo-mask for {img_path}, skipping.")
            else:
                print(f"Warning: bounding-box file {f} not found.")
        print(f"BoxFilesDataset: found {len(self.image_mask_pairs)} valid lines with non-zero boxes and existing pseudo-masks.")

    def __len__(self):
        return len(self.image_mask_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.image_mask_pairs[idx]
        # IMPORTANT: extract the base name as a string
        base = os.path.splitext(os.path.basename(img_path))[0]
        pil_img = Image.open(img_path).convert("RGB")
        img_t = self.transform_img(pil_img)

        mask_pil = Image.open(mask_path).convert("L")
        mask_pil = mask_pil.resize((224, 224), resample=Image.NEAREST)
        mask_arr = np.array(mask_pil)
        # Convert to binary {0,1}
        mask_bin = (mask_arr > 128).astype(np.uint8)
        mask_t = torch.from_numpy(mask_bin).long()
        return img_t, mask_t, base

# ------------------ MAIN INFERENCE ------------------
def main():
    # Create the dataset
    dataset = BoxFilesDataset(CATS_BOX_FILE, DOGS_BOX_FILE, PSEUDO_MASK_DIR)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    if not os.path.exists(BEST_MODEL_PATH):
        print(f"Error: cannot find best model at {BEST_MODEL_PATH}")
        return

    # Load model
    model = deeplabv3_resnet50(num_classes=2)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"Loaded model from {BEST_MODEL_PATH}, found {len(dataset)} samples for inference.")

    total_iou, total_px, count = 0.0, 0.0, 0

    for idx, (img_t, pseudo_gt_t, base) in enumerate(loader):
        # img_t: [1,3,224,224], pseudo_gt_t: [1,224,224]
        img_t = img_t.to(DEVICE)
        pseudo_np = pseudo_gt_t.squeeze(0).numpy()  # shape (224,224) with values {0,1}

        if isinstance(base, tuple):
            base = base[0]

        # Run the model
        logits = model(img_t)["out"]  # [1,2,224,224]
        pred = torch.argmax(logits, dim=1).squeeze(0)  # (224,224), {0,1}
        pred_np = (pred.cpu().numpy() * 255).astype(np.uint8)  # Convert to {0,255}

        # Compute metrics
        pred_bin = (pred_np > 128).astype(np.uint8)  # Convert prediction to binary {0,1}
        iou, pix_acc = compute_mIoU_and_pixel_accuracy(pred_bin, pseudo_np)
        total_iou += iou
        total_px += pix_acc
        count += 1

        # Save predicted mask
        out_mask_name = os.path.join(OUT_MASKS_DIR, base + "_pred.png")
        Image.fromarray(pred_np).save(out_mask_name)

        # Optional overlay saving
        if SAVE_OVERLAYS:
            # Reload original image for overlay from the dataset's image path
            orig_img_path = dataset.image_mask_pairs[idx][0]
            orig_img_pil = Image.open(orig_img_path).convert("RGB")
            overlay_np = create_overlay(orig_img_pil, pred_np, alpha=0.5)
            overlay_pil = Image.fromarray(overlay_np)
            out_ov_name = os.path.join(OVERLAY_DIR, base + "_overlay.jpg")
            overlay_pil.save(out_ov_name, quality=85)

    if count > 0:
        mean_iou = total_iou / count
        mean_px = total_px / count
        print(f"\n--- Final Metrics over {count} images ---")
        print(f"Mean IoU:       {mean_iou:.4f}")
        print(f"Pixel Accuracy: {mean_px:.4f}")
    else:
        print("No valid samples to compute metrics.")

if __name__ == "__main__":
    main()
