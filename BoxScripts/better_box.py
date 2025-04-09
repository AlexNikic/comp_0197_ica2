#!/usr/bin/env python3
"""
better_box.py

Final version of an improved BoxSup-like pipeline with debug statements,
integrated edge detection (via Canny), SLIC-based region proposals,
and an alternative metric for filtering proposals based on an inclusion ratio.
This script:
  1) Parses bounding-box data from text files,
  2) Uses SLIC to generate pixel-level region proposals,
  3) Computes an edge map with Canny and weights proposals based on boundary alignment,
  4) Iteratively refines pseudo-masks using merging, CRF, and optional morphological closing,
  5) Performs a random hyperparameter search (5 trials),
  6) Saves the best configuration and final pseudo-masks into designated directories.

Requirements:
  - PyTorch and torchvision
  - scikit-image (pip install scikit-image)
  - pydensecrf (pip install pydensecrf)
  - scikit-learn
"""

import os
import time
import random
import json
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# For region proposals and edge detection
from skimage.segmentation import slic
from skimage.feature import canny

# For CRF
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
from sklearn.model_selection import KFold

##############################
# Debug Configuration
##############################
DEBUG = False         # Set to True to enable debug prints
DEBUG_IMAGE_LIMIT = 3  # Show detailed debug for first few images

##############################
# Global Configuration
##############################
IMAGE_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
IMAGE_NORMALIZE_STD  = [0.229, 0.224, 0.225]

N_FOLDS = 1         # Single validation split.
N_SEARCH = 5        # Number of random hyperparameter trials.
OUT_MASK_DIR = "better_boxsup"   # Directory to save final pseudo-masks.
OUT_MODEL_DIR = "boxsup_better_checkpoint"
os.makedirs(OUT_MASK_DIR, exist_ok=True)
os.makedirs(OUT_MODEL_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

########################################
# 1) Bounding Box Parsing Functions
########################################
def parse_paths_file(txt_file):
    """
    Parse lines of the form:
       ../Data/with_box/cats/Abyssinian/Abyssinian_1.jpg x1 y1 x2 y2
    Returns a dictionary mapping each image path to a list of bounding boxes.
    Boxes with "0 0 0 0" are skipped.
    """
    data_dict = {}
    with open(txt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            img_path = parts[0]
            x1, y1, x2, y2 = map(int, parts[1:5])
            if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
                continue
            data_dict.setdefault(img_path, []).append([x1, y1, x2, y2])
    return data_dict

def combine_cat_dog_dicts(cat_dict, dog_dict):
    """
    Merge the bounding box dictionaries from cats and dogs.
    """
    combined = {}
    for k, v in cat_dict.items():
        combined[k] = v[:]
    for k, v in dog_dict.items():
        if k not in combined:
            combined[k] = []
        combined[k].extend(v)
    return combined

########################################
# 2) Helper Functions for Edge and Boundary
########################################
def compute_boundary(mask, kernel_size=3):
    """
    Compute the boundary of a binary mask.
    Uses dilation minus erosion.
    Input: mask (torch.Tensor of shape (H,W), float 0/1)
    Returns: boundary (torch.Tensor of shape (H,W), float 0/1)
    """
    mask = mask.float().unsqueeze(0).unsqueeze(0)
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=mask.device)
    dilated = F.conv2d(mask, kernel, padding=kernel_size//2)
    dilated = (dilated > 0).float()
    eroded = F.conv2d(mask, kernel, padding=kernel_size//2)
    eroded = (eroded >= kernel.numel()).float()
    boundary = dilated - eroded
    return (boundary > 0).float().squeeze(0).squeeze(0)

def boundary_alignment_weight(seg_mask, edge_map):
    """
    Compute a weight measuring how well the boundary of seg_mask aligns with edge_map.
    Ensures both tensors are on the same device.
    Returns a float in [0,1].
    """
    boundary = compute_boundary(seg_mask, kernel_size=3)
    edge_map = edge_map.to(boundary.device)
    if boundary.sum() < 1:
        return 0.0
    weight = (boundary * edge_map).sum() / boundary.sum()
    return weight.item()

########################################
# 3) Region Proposals and Edge Detection
########################################
def get_region_proposals_slic(pil_img, n_segments=500, compactness=20):
    """
    Generate region proposals using SLIC superpixels.
    Returns a list of (seg_mask, bbox) pairs.
    seg_mask: torch.Tensor (H,W) in {0,1} (float)
    bbox: Tuple (x1, y1, x2, y2) of the segment.
    """
    if DEBUG:
        print(f"\n[DEBUG] Generating SLIC proposals with n_segments={n_segments}, compactness={compactness}...")
        t0 = time.time()
    np_img = np.array(pil_img)
    segments = slic(np_img, n_segments=n_segments, compactness=compactness, start_label=1)
    if DEBUG:
        print(f"[DEBUG] SLIC produced {len(np.unique(segments))} superpixels in {time.time()-t0:.2f}s")
    proposals = []
    H, W = segments.shape
    for label in np.unique(segments):
        mask = (segments == label).astype(np.uint8)
        if mask.sum() < 50:
            if DEBUG and label < 5:
                print(f"[DEBUG] Skipping small superpixel {label} (area={mask.sum()})")
            continue
        ys, xs = np.where(mask)
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        seg_mask = torch.from_numpy(mask).float()
        proposals.append((seg_mask, (x1, y1, x2, y2)))
    if DEBUG:
        print(f"[DEBUG] Kept {len(proposals)} valid proposals")
    return proposals

def compute_edge_map(pil_img, sigma=1.5):
    """
    Compute an edge map using the Canny detector.
    Returns a torch.Tensor of shape (H,W) with values in [0,1].
    """
    if DEBUG:
        print(f"\n[DEBUG] Computing edge map with sigma={sigma}...")
    gray = np.array(pil_img.convert("L"))
    edges = canny(gray, sigma=sigma, low_threshold=0.1, high_threshold=0.3)
    edge_density = edges.mean() * 100
    if DEBUG:
        print(f"[DEBUG] Edge density: {edge_density:.2f}% ({edges.sum()} edge pixels)")
    return torch.from_numpy(edges.astype(np.float32))

########################################
# 4) Merging, CRF, and Morphological Ops
########################################
def union_of_bboxes(boxes, H, W):
    mask = torch.zeros((H, W), dtype=torch.float32)
    for (x1, y1, x2, y2) in boxes:
        if x2 < x1 or y2 < y1:
            continue
        x1, x2 = sorted([max(0, x1), min(W-1, x2)])
        y1, y2 = sorted([max(0, y1), min(H-1, y2)])
        mask[y1:y2, x1:x2] = 1
    return mask

def mask_l2_regularization(old_mask, new_mask, reg_lambda=0.01):
    diff = new_mask - old_mask
    return new_mask - reg_lambda * diff

def apply_crf(img_tensor, mask_tensor, n_iter=10):
    """
    Run DenseCRF on CPU and return refined mask.
    """
    img_cpu = img_tensor.detach().cpu()
    mask_cpu = mask_tensor.detach().cpu()
    C, H, W = img_cpu.shape
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
    kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=mask.dtype, device=mask.device)
    dilated = F.conv2d(mask_4d, kernel, padding=kernel_size//2)
    dilated = (dilated > 0).float()
    inv_dilated = 1 - dilated
    eroded = F.conv2d(inv_dilated, kernel, padding=kernel_size//2)
    eroded = (eroded == 0).float()
    return eroded.squeeze(0).squeeze(0)

########################################
# 5) Advanced BoxSup Processing with Edge Weighting
########################################
def advanced_boxsup(img_tensor, pil_img, bounding_boxes, region_prop_list, device=DEVICE,
                    num_iters=3, alpha=0.3, overlap_thresh=0.2,
                    apply_crf_flag=True, reg_lambda=0.01, do_morph=False,
                    edge_weighting=True):
    """
    Advanced BoxSup processing:
      1) Build a base mask as the union of bounding boxes.
      2) Filter region proposals.
         (Instead of IoU, compute inclusion ratio: fraction of candidate area inside the box.)
         If edge_weighting is enabled, weight each candidate by its alignment with the edge map.
      3) Iteratively merge proposals using L2 regularization.
      4) Refine with CRF.
      5) Optionally apply morphological closing.
    Returns a refined pseudo-mask (tensor of shape (H,W)).
    """
    _, H, W = img_tensor.shape
    base_mask = union_of_bboxes(bounding_boxes, H, W).to(device)
    if DEBUG:
        print(f"[DEBUG] Base mask coverage: {base_mask.mean().item()*100:.2f}%")
    
    # Compute edge map from original image.
    if edge_weighting:
        edge_map = compute_edge_map(pil_img, sigma=1.5)
        edge_map = T.Resize((H, W), interpolation=T.InterpolationMode.NEAREST)(edge_map.unsqueeze(0)).squeeze(0)
    else:
        edge_map = None
    
    # Filter proposals using inclusion ratio.
    proposals = []
    for i, (seg_mask, seg_bbox) in enumerate(region_prop_list):
        rx1, ry1, rx2, ry2 = seg_bbox
        rx1, rx2 = sorted([rx1, rx2])
        ry1, ry2 = sorted([ry1, ry2])
        if rx2 < 0 or rx1 >= W or ry2 < 0 or ry1 >= H:
            continue
        rx1, rx2 = max(0, rx1), min(W-1, rx2)
        ry1, ry2 = max(0, ry1), min(H-1, ry2)
        if rx2 < rx1 or ry2 < ry1:
            continue
        prop_mask = torch.zeros((H, W), dtype=torch.float32, device=device)
        prop_mask[ry1:ry2, rx1:rx2] = 1
        # Use inclusion ratio: fraction of candidate area that lies inside base_mask.
        candidate_area = prop_mask.sum().item()
        if candidate_area == 0:
            continue
        inclusion_ratio = (prop_mask * base_mask).sum().item() / (candidate_area + 1e-8)
        if inclusion_ratio >= overlap_thresh:
            if edge_weighting and edge_map is not None:
                weight = boundary_alignment_weight(prop_mask, edge_map)
                if DEBUG and i < 5:
                    print(f"[DEBUG] Proposal {i} edge weight: {weight:.2f}")
                prop_mask = prop_mask * (0.3 + 0.7 * weight)
            proposals.append(prop_mask)
    
    if DEBUG:
        print(f"[DEBUG] {len(proposals)} proposals kept after filtering.")
        if len(proposals) == 0:
            print("[WARNING] No proposals remained!")
    
    # Iterative merging on GPU.
    current_mask = base_mask.clone()
    for it in range(num_iters):
        t0 = time.time()
        valid_props = [p for p in proposals if (p * current_mask).sum().item() > 10]
        if DEBUG:
            prev_cov = current_mask.mean().item()
            print(f"[DEBUG] Iteration {it+1}: {len(valid_props)} valid proposals, coverage: {prev_cov*100:.1f}%")
        if valid_props:
            proposal_avg = torch.stack(valid_props).mean(0)
            current_mask = 0.25 * current_mask + 0.75 * (proposal_avg - 0.015 * (proposal_avg - current_mask))
        if DEBUG:
            new_cov = current_mask.mean().item()
            print(f"[DEBUG] Iteration {it+1}: New coverage: {new_cov*100:.1f}%, time={time.time()-t0:.2f}s")
    
    refined_mask = current_mask
    if apply_crf_flag:
        if DEBUG:
            print("[DEBUG] Running CRF refinement...")
        refined_mask = apply_crf(img_tensor, refined_mask, n_iter=10).to(device)
        if DEBUG:
            print(f"[DEBUG] Post-CRF coverage: {refined_mask.mean().item()*100:.2f}%")
    
    if do_morph:
        if DEBUG:
            print("[DEBUG] Applying morphological closing...")
        refined_mask = morphological_close(refined_mask, kernel_size=3)
    
    return refined_mask

########################################
# 6) Pseudo-IoU Calculation
########################################
def compute_pseudo_iou(pred_mask, bounding_boxes):
    if pred_mask.dim() < 2:
        return 0
    H, W = pred_mask.shape
    union_box = torch.zeros((H, W), dtype=torch.float32, device=pred_mask.device)
    for (x1, y1, x2, y2) in bounding_boxes:
        x1, x2 = sorted([max(0, x1), min(W-1, x2)])
        y1, y2 = sorted([max(0, y1), min(H-1, y2)])
        union_box[y1:y2, x1:x2] = 1
    inter = (pred_mask * union_box).sum().item()
    union_area = (pred_mask + union_box).clamp_(0, 1).sum().item()
    if union_area == 0:
        return 1 if inter == 0 else 0
    return inter / union_area

########################################
# 7) Random Hyperparameter Search
########################################
def evaluate_boxsup(items, config, device):
    """
    For each image (img_path, bounding_boxes) in items, generate proposals using SLIC,
    run advanced_boxsup to produce a pseudo-mask, and compute pseudo-IoU.
    """
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(IMAGE_NORMALIZE_MEAN, IMAGE_NORMALIZE_STD)
    ])
    ious = []
    for (img_path, boxes) in items:
        pil_img = Image.open(img_path).convert('RGB')
        proposals = get_region_proposals_slic(pil_img, n_segments=500, compactness=20)
        img_tensor = transform(pil_img).to(device)
        pmask = advanced_boxsup(
            img_tensor, pil_img, boxes, proposals,
            num_iters=config['num_iters'],
            alpha=config['alpha'],
            overlap_thresh=config['overlap_thresh'],
            apply_crf_flag=config['apply_crf'],
            reg_lambda=config['reg_lambda'],
            do_morph=config['do_morph'],
            edge_weighting=config.get('edge_weighting', True),
            device=device
        )
        iou_val = compute_pseudo_iou(pmask, boxes)
        ious.append(iou_val)
        if DEBUG and len(ious) <= DEBUG_IMAGE_LIMIT:
            print(f"[DEBUG] {os.path.basename(img_path)}: pseudo-IoU = {iou_val:.4f}")
    return float(np.mean(ious))

def random_search_5fold(all_items, n_folds=1, n_search=5, device='cuda'):
    random.shuffle(all_items)
    if n_folds < 2:
        folds = [([], all_items)]
    else:
        fold_size = len(all_items) // n_folds
        folds = []
        for i in range(n_folds):
            start = i * fold_size
            end = (i + 1) * fold_size if i < n_folds - 1 else len(all_items)
            val_part = all_items[start:end]
            train_part = all_items[:start] + all_items[end:]
            folds.append((train_part, val_part))
    
    best_config = None
    best_score = 0.0
    for trial in range(n_search):
        hp = {
            'num_iters': random.randint(3, 6),
            'alpha': round(random.uniform(0.2, 0.7), 2),
            'overlap_thresh': round(random.uniform(0.05, 0.3), 2),
            'apply_crf': random.choice([True, False]),
            'reg_lambda': round(random.uniform(0.005, 0.02), 3),
            'do_morph': random.choice([True, False]),
            'edge_weighting': random.choice([True, False])
        }
        print(f"\n=== Trial {trial+1}/{n_search}: {hp}")
        fold_scores = []
        for i, (_, val_fold) in enumerate(folds):
            score = evaluate_boxsup(val_fold, hp, device)
            fold_scores.append(score)
            print(f"   Fold {i}: pseudo-IoU = {score:.4f}")
        avg_score = float(np.mean(fold_scores))
        print(f" => Trial average IoU = {avg_score:.4f}")
        if avg_score > best_score:
            best_score = avg_score
            best_config = hp
            print("   * New best configuration found *")
    return best_config, best_score

########################################
# 8) Produce Final Masks
########################################
def produce_final_masks(items, config, device, out_dir=OUT_MASK_DIR):
    os.makedirs(out_dir, exist_ok=True)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(IMAGE_NORMALIZE_MEAN, IMAGE_NORMALIZE_STD)
    ])
    for (img_path, boxes) in items:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        pil_img = Image.open(img_path).convert('RGB')
        proposals = get_region_proposals_slic(pil_img, n_segments=500, compactness=20)
        img_tensor = transform(pil_img).to(device)
        pmask = advanced_boxsup(
            img_tensor, pil_img, boxes, proposals,
            num_iters=config['num_iters'],
            alpha=config['alpha'],
            overlap_thresh=config['overlap_thresh'],
            apply_crf_flag=config['apply_crf'],
            reg_lambda=config['reg_lambda'],
            do_morph=config['do_morph'],
            edge_weighting=config.get('edge_weighting', True),
            device=device
        )
        pm = (pmask.detach().cpu().numpy() * 255).astype(np.uint8)
        out_file = os.path.join(out_dir, f"{base_name}_mask.png")
        Image.fromarray(pm).save(out_file)
        if DEBUG:
            print(f"[DEBUG] Saved final mask for {base_name} to: {out_file}")

########################################
# 9) MAIN EXECUTION
########################################
def main():
    # Define paths for bounding box text files.
    cat_txt = "../Data/paths_cats_with_box.txt"
    dog_txt = "../Data/paths_dogs_with_box.txt"
    
    # Parse and combine
    cat_dict = parse_paths_file(cat_txt)
    dog_dict = parse_paths_file(dog_txt)
    combined_dict = combine_cat_dog_dicts(cat_dict, dog_dict)
    
    # Build list of (img_path, bounding_boxes)
    all_items = []
    for img_path, boxes in combined_dict.items():
        if boxes:
            all_items.append((img_path, boxes))
    
    #print(f"[DEBUG] Using device={DEVICE}, total images={len(all_items)}")
    #best_config, best_score = random_search_5fold(all_items, n_folds=N_FOLDS, n_search=N_SEARCH, device=DEVICE)
    #print(f"\nBEST config: {best_config}, best pseudo-IoU = {best_score:.4f}")
    
    #best_config_path = os.path.join(OUT_MODEL_DIR, "boxsup_best_config.json")
    #with open(best_config_path, 'w') as f:
    #    json.dump(best_config, f)
    #print(f"[DEBUG] Saved best configuration to {best_config_path}")
    
    best_config = {"num_iters": 3, "alpha": 0.42, "overlap_thresh": 0.3, "apply_crf": False, "reg_lambda": 0.017, "do_morph": True, "edge_weighting": True}
    produce_final_masks(all_items, best_config, device=DEVICE, out_dir=OUT_MASK_DIR)
    print(f"[DEBUG] Final pseudo-masks saved in {OUT_MASK_DIR}. DONE.")

if __name__=="__main__":
    start_time = time.time()
    main()
    print(f"\nTotal runtime: {time.time()-start_time:.2f} seconds")
