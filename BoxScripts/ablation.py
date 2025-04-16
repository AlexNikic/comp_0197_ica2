#!/usr/bin/env python3
"""
ablation_experiment_scratch_rescaled.py

This script performs an ablation study of our weakly‑supervised segmentation pipeline
running from scratch (i.e., it generates refined pseudo‑masks on‑the‑fly based on bounding‑box
annotations). For each sample (loaded from paths_cats_with_box.txt and paths_dogs_with_box.txt):

  - Loads the original image.
  - **Scales** the bounding box to 224×224 coordinates (to match our model input).
  - Generates a refined pseudo‑mask using the advanced BoxSup pipeline (region proposals + edge maps).
  - Optionally applies CRF or morphological ops or edge weighting, based on the experiment config.
  - Runs inference with a fully‑supervised DeepLabv3 model (loaded from disk).
  - Computes mean IoU and pixel accuracy between the model prediction and the refined pseudo‑mask.

At the end, the script prints a table summarizing the results for each experiment.
"""

import os
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.transforms import functional as TF
from PIL import Image
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

# ------------------ Device Configuration ------------------
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print(f"Using device: {DEVICE}")

# ------------------ Global Constants ------------------
IMAGE_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
IMAGE_NORMALIZE_STD  = [0.229, 0.224, 0.225]

# Constant advanced BoxSup parameters (from best found configuration)
ADV_PARAMS_CONST = {
    "num_iters": 3,
    "alpha": 0.42,
    "overlap_thresh": 0.3,
    "reg_lambda": 0.017
}

# Experiment variations for ablation study
EXPERIMENT_CONFIGS = {
    "None": {
        "apply_crf": False,
        "do_morph": False,
        "edge_weighting": False
    },
    "Apply_CRF": {
        "apply_crf": True,
        "do_morph": False,
        "edge_weighting": False
    },
    "Do_Morph": {
        "apply_crf": False,
        "do_morph": True,
        "edge_weighting": False
    },
    "Edge_Weight": {
        "apply_crf": False,
        "do_morph": False,
        "edge_weighting": True
    },
    "CRF_and_Morph": {
        "apply_crf": True,
        "do_morph": True,
        "edge_weighting": False
    },
    "CRF_and_Edge": {
        "apply_crf": True,
        "do_morph": False,
        "edge_weighting": True
    },
    "Morph_and_Edge": {
        "apply_crf": False,
        "do_morph": True,
        "edge_weighting": True
    },
    "All": {
        "apply_crf": True,
        "do_morph": True,
        "edge_weighting": True
    }
}

# Data files and directories
CATS_BOX_FILE   = "../Data/paths_cats_with_box.txt"
DOGS_BOX_FILE   = "../Data/paths_dogs_with_box.txt"
BEST_MODEL_PATH = "deeplab_output/best_deeplabv3_model.pth"

# ------------------ Utility Functions ------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE in ["cuda", "mps"]:
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------ Metrics ------------------
def compute_mIoU_and_pixel_accuracy(pred_bin, gt_bin):
    """
    Compute the Intersection over Union (IoU) and pixel accuracy between
    two binary numpy arrays (H,W) with values in {0,1}.
    """
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    iou = (intersection + 1e-6) / (union + 1e-6)
    pixel_acc = (pred_bin == gt_bin).sum() / (gt_bin.size)
    return iou, pixel_acc

# ------------------ Dataset: BoxFilesDatasetScratch (with bounding box scaling) ------------------
class BoxFilesDatasetScratch(Dataset):
    """
    Reads bounding-box text files (cats & dogs) where each line is:
         image_path x1 y1 x2 y2
    Skips lines where the bounding box is "0 0 0 0".
    
    Returns a tuple (image_tensor, base, bbox_224, img_path, gt_np), where:
       - image_tensor is the normalized image (3,224,224),
       - base is the base filename (string),
       - bbox_224 is a list of 4 ints [x1, y1, x2, y2] in the 224×224 space,
       - img_path is the original image path (for reloading if needed).
       - gt_np is the ground truth labels
    """
    def __init__(self, cats_file, dogs_file, gt_mask_dir = 'ground_trut_masks'):
        self.items = []
        self.gt_mask_dir = gt_mask_dir
        for file_path in [cats_file, dogs_file]:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) < 5:
                            continue
                        img_path = parts[0]
                        x1, y1, x2, y2 = map(int, parts[1:5])
                        if x1==0 and y1==0 and x2==0 and y2==0:
                            continue
                        self.items.append((img_path, [x1, y1, x2, y2]))
            else:
                print(f"Warning: {file_path} not found.")
        print(f"BoxFilesDatasetScratch: found {len(self.items)} valid samples.")

        # We'll use the same transform for the image
        # but we need to scale the bounding box ourselves.
        self.transform = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(IMAGE_NORMALIZE_MEAN, IMAGE_NORMALIZE_STD)
        ])
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        img_path, bbox = self.items[idx]
        base = os.path.splitext(os.path.basename(img_path))[0]

        # Load original image to get its size (for bounding-box scaling).
        pil_img = Image.open(img_path).convert("RGB")
        W_orig, H_orig = pil_img.size  # (width, height)

        # Compute scale factors to 224x224
        sx = 224.0 / W_orig
        sy = 224.0 / H_orig

        # Scale bounding box to 224x224 coords
        x1, y1, x2, y2 = bbox
        x1_224 = int(round(x1 * sx))
        x2_224 = int(round(x2 * sx))
        y1_224 = int(round(y1 * sy))
        y2_224 = int(round(y2 * sy))
        scaled_bbox = [x1_224, y1_224, x2_224, y2_224]

        # Transform the image to 224x224 for model input
        img_t = self.transform(pil_img)

        #Load ground truth mask
        gt_mask_path = os.path.join(self.gt_mask_dir, base + ".png")
        gt_mask = Image.open(gt_mask_path).convert("L")
        gt_mask = TF.resize(gt_mask, [224, 224], interpolation=TF.InterpolationMode.NEAREST)
        gt_np = torch.from_numpy((np.array(gt_mask) >= 128).astype(np.uint8))
      
        # Return scaled bbox, not the original
        return img_t, base, scaled_bbox, img_path, gt_np

# ------------------ Helper Functions for Edges, Proposals, etc. ------------------
def compute_boundary(mask, kernel_size=3):
    mask = mask.float().unsqueeze(0).unsqueeze(0)
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=mask.device)
    dilated = F.conv2d(mask, kernel, padding=kernel_size//2)
    dilated = (dilated > 0).float()
    eroded = F.conv2d(mask, kernel, padding=kernel_size//2)
    eroded = (eroded >= kernel.numel()).float()
    boundary = dilated - eroded
    return (boundary > 0).float().squeeze(0).squeeze(0)

def boundary_alignment_weight(seg_mask, edge_map):
    boundary = compute_boundary(seg_mask, kernel_size=3)
    edge_map = edge_map.to(boundary.device)
    if boundary.sum() < 1:
        return 0.0
    weight = (boundary * edge_map).sum() / boundary.sum()
    return weight.item()

def get_region_proposals_slic(pil_img, n_segments=500, compactness=20):
    from skimage.segmentation import slic
    np_img = np.array(pil_img)
    segments = slic(np_img, n_segments=n_segments, compactness=compactness, start_label=1)
    proposals = []
    H, W = segments.shape
    for label in np.unique(segments):
        mask = (segments == label).astype(np.uint8)
        if mask.sum() < 50:
            continue
        ys, xs = np.where(mask)
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()), int(ys.max())
        seg_mask = torch.from_numpy(mask).float()
        proposals.append((seg_mask, (x1, y1, x2, y2)))
    return proposals

def compute_edge_map(pil_img, sigma=1.5):
    from skimage.feature import canny
    gray = np.array(pil_img.convert("L"))
    edges = canny(gray, sigma=sigma, low_threshold=0.1, high_threshold=0.3)
    return torch.from_numpy(edges.astype(np.float32))

def union_of_bboxes(boxes, H, W):
    # boxes is [x1, y1, x2, y2] in 224×224 space
    x1, y1, x2, y2 = boxes
    # Sort/clamp so we don't get negative widths
    x1, x2 = sorted([max(0, x1), min(W-1, x2)])
    y1, y2 = sorted([max(0, y1), min(H-1, y2)])
    if x2 < x1 or y2 < y1:
        return torch.zeros((H, W), dtype=torch.float32)
    mask = torch.zeros((H, W), dtype=torch.float32)
    mask[y1:y2, x1:x2] = 1
    return mask

def apply_crf(img_tensor, mask_tensor, n_iter=10):
    C, H, W = img_tensor.shape
    img_cpu = img_tensor.detach().cpu()
    mask_cpu = mask_tensor.detach().cpu()
    prob_fg = mask_cpu.numpy()
    prob_bg = 1 - prob_fg
    prob_2 = np.stack([prob_bg, prob_fg], axis=0)

    unary = unary_from_softmax(prob_2)
    mean = torch.tensor(IMAGE_NORMALIZE_MEAN).view(3,1,1)
    std  = torch.tensor(IMAGE_NORMALIZE_STD).view(3,1,1)
    img_raw = img_cpu * std + mean
    img_np = (img_raw.clamp(0,1) * 255).byte().permute(1,2,0).numpy()
    img_np = np.ascontiguousarray(img_np)

    d = dcrf.DenseCRF2D(W, H, 2)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=10, srgb=20, rgbim=img_np, compat=10)

    Q = d.inference(n_iter)
    Q = np.array(Q).reshape((2, H, W))
    refined_fg = Q[1,:,:]
    denom = Q[0,:,:] + Q[1,:,:] + 1e-8
    refined_fg = refined_fg / denom
    refined_mask = (refined_fg >= 0.5).astype(np.float32)
    return torch.from_numpy(refined_mask)

def morphological_close(mask, kernel_size=3):
    mask_4d = mask.unsqueeze(0).unsqueeze(0)
    kernel = torch.ones((1,1,kernel_size,kernel_size), dtype=mask.dtype, device=mask.device)
    dilated = F.conv2d(mask_4d, kernel, padding=kernel_size//2)
    dilated = (dilated > 0).float()

    inv_dilated = 1 - dilated
    eroded = F.conv2d(inv_dilated, kernel, padding=kernel_size//2)
    eroded = (eroded == 0).float()
    return eroded.squeeze(0).squeeze(0)

# ------------------ Advanced BoxSup Processing ------------------
def advanced_boxsup(img_tensor, pil_img, bounding_box, config, device):
    """
    Generates a refined pseudo-mask using the advanced BoxSup pipeline.

    bounding_box here is already scaled to the same resolution as img_tensor (224×224).
    """
    # Merge constant parameters with experimental booleans
    exp_config = ADV_PARAMS_CONST.copy()
    exp_config.update(config)
    
    _, H, W = img_tensor.shape
    base_mask = union_of_bboxes(bounding_box, H, W).to(device)

    # Edge map if edge_weighting is True
    if exp_config["edge_weighting"]:
        edge_map = compute_edge_map(pil_img, sigma=1.5)
        edge_map = T.Resize((H, W), interpolation=T.InterpolationMode.NEAREST)(
            edge_map.unsqueeze(0)
        ).squeeze(0)
    else:
        edge_map = None

    # Generate region proposals from SLIC in original PIL image
    # (Here, we do it on the PIL image at original size, which is typical,
    #  but we should also *resize that PIL image to 224×224* for consistent proposals.
    #  For example:)
    resized_pil = pil_img.resize((W, H), Image.BILINEAR)
    proposals = get_region_proposals_slic(resized_pil, n_segments=500, compactness=20)

    filtered_props = []
    for seg_mask, seg_bbox in proposals:
        x1, y1, x2, y2 = seg_bbox
        # clamp them in [0, W-1] x [0, H-1]
        x1, x2 = sorted([max(0, x1), min(W-1, x2)])
        y1, y2 = sorted([max(0, y1), min(H-1, y2)])
        prop_mask = torch.zeros((H, W), dtype=torch.float32, device=device)
        prop_mask[y1:y2, x1:x2] = 1
        candidate_area = prop_mask.sum().item()
        if candidate_area == 0:
            continue
        # Overlap with base_mask
        inclusion_ratio = (prop_mask * base_mask).sum().item() / (candidate_area + 1e-8)
        if inclusion_ratio >= exp_config["overlap_thresh"]:
            if exp_config["edge_weighting"] and edge_map is not None:
                weight = boundary_alignment_weight(prop_mask, edge_map)
                prop_mask = prop_mask * (0.3 + 0.7 * weight)
            filtered_props.append(prop_mask)

    if len(filtered_props) == 0:
        # Fallback if we got nothing
        filtered_props = [base_mask]

    current_mask = base_mask.clone()
    for it in range(exp_config["num_iters"]):
        valid_props = [p for p in filtered_props if (p * current_mask).sum().item() > 10]
        if valid_props:
            proposal_avg = torch.stack(valid_props).mean(0)
            # A simple update rule, somewhat ad hoc
            current_mask = 0.25 * current_mask + 0.75 * (
                proposal_avg - 0.015*(proposal_avg - current_mask)
            )

    refined_mask = current_mask

    if exp_config["apply_crf"]:
        refined_mask = apply_crf(img_tensor, refined_mask, n_iter=10).to(device)

    if exp_config["do_morph"]:
        refined_mask = morphological_close(refined_mask, kernel_size=3)

    return refined_mask

# ------------------ Ablation Study Experiment ------------------
def run_ablation_experiment(dataset, model, device):
    """
    For each experimental configuration, run inference on the dataset.
    For each sample:
      - Generate a refined pseudo-mask using advanced_boxsup (from scratch).
      - Run the fully‑supervised DeepLabv3 model to predict a segmentation mask.
      - Compute mIoU and pixel accuracy between the model prediction and the refined pseudo‑mask.
    Returns a dictionary: experiment -> (mean IoU, mean pixel accuracy).
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    total_results = {}

    for exp_name, exp_config in EXPERIMENT_CONFIGS.items():
        print(f"\nRunning experiment: {exp_name}")
        total_iou = 0.0
        total_px = 0.0
        count = 0

        for img_t, base, bbox_224, img_path, gt_np in loader:
            # Move input to device
            img_t = img_t.to(device)
            
            # Run model prediction
            with torch.no_grad():
                logits = model(img_t)["out"]
            pred = torch.argmax(logits, dim=1).squeeze(0)
            pred_np = (pred.cpu().numpy() > 0.5).astype(np.uint8)

            # Compute metrics
            gt_np = gt_np.numpy()
            iou_val, px_acc = compute_mIoU_and_pixel_accuracy(pred_np, gt_np)
            total_iou += iou_val
            total_px += px_acc
            count += 1

        mean_iou = total_iou / count if count > 0 else 0.0
        mean_px  = total_px / count if count > 0 else 0.0
        total_results[exp_name] = (mean_iou, mean_px)
        print(f"Experiment {exp_name}: Mean IoU = {mean_iou:.4f}, Pixel Acc. = {mean_px:.4f}")

    return total_results

def print_results_table(results):
    print("\nAblation Study Results:")
    print("{:<20} {:<15} {:<15}".format("Experiment", "Mean IoU", "Pixel Accuracy"))
    print("-" * 50)
    for exp, (iou, acc) in results.items():
        print("{:<20} {:<15.4f} {:<15.4f}".format(exp, iou, acc))

# ------------------ Main Execution ------------------
def main():
    set_seed(42)
    # Build dataset from bounding box files (running pipeline from scratch).
    dataset = BoxFilesDatasetScratch(CATS_BOX_FILE, DOGS_BOX_FILE)
    if len(dataset) == 0:
        print("No valid samples found. Exiting.")
        return
    print(f"Total samples in dataset: {len(dataset)}")
    
    # Load the fully supervised DeepLabv3 model.
    if not os.path.exists(BEST_MODEL_PATH):
        print(f"Error: Best model not found at {BEST_MODEL_PATH}.")
        return
    model = deeplabv3_resnet50(num_classes=2)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"Loaded best DeepLabv3 model from {BEST_MODEL_PATH} on device {DEVICE}.")

    # Run ablation experiments.
    results = run_ablation_experiment(dataset, model, DEVICE)
    print_results_table(results)

if __name__ == "__main__":
    main()
