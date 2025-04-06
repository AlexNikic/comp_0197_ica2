"""
advanced_boxsup_pipeline.py

A single script that:
 1) Reads bounding-box data from text files for cats & dogs,
 2) Combines them into a single data structure,
 3) Splits data for 5-fold cross validation,
 4) Performs a simplified advanced 'BoxSup' approach with:
    - Selective search region proposals,
    - Iterative merging & bounding box constraints,
    - Mask-level L2 regularization,
    - CRF post-processing,
 5) Conducts hyperparameter random search,
 6) Saves best hyperparameters as a 'model',
 7) Produces final pseudo-masks for the entire dataset.

These pseudo-masks can later be used to train DeepLabv3+.

Note: If an image’s bounding box coordinates are "0 0 0 0", that image is skipped.

DEPENDENCIES:
  pip install scikit-image pydensecrf scikit-learn

"""

import os
import math
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import selectivesearch
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
from sklearn.model_selection import KFold

##############################
# 1) GLOBAL CONFIGURATION
##############################
IMAGE_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
IMAGE_NORMALIZE_STD  = [0.229, 0.224, 0.225]

N_FOLDS = 1
N_SEARCH = 10  # Number of random hyperparameter trials
OUT_MASK_DIR = "output_masks_boxsup"
OUT_MODEL_DIR = "boxsup_checkpoint"
os.makedirs(OUT_MASK_DIR, exist_ok=True)
os.makedirs(OUT_MODEL_DIR, exist_ok=True)

##############################
# 2) BOUNDING BOX PARSING
##############################
def parse_paths_file(txt_file):
    """
    Parse a text file with lines in the following format:
      ../Data/with_box/cats/Abyssinian/Abyssinian_1.jpg x1 y1 x2 y2

    Returns:
      A dictionary mapping each image path to a list of bounding boxes.
      Bounding boxes with coordinates (0, 0, 0, 0) are skipped.
    """
    data_dict = {}
    with open(txt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue  # Skip lines without full coordinates
            img_path = parts[0]
            # Convert coordinates to integers
            x1, y1, x2, y2 = map(int, parts[1:5])
            # Skip boxes that are all zeros (invalid/no coordinates)
            if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
                continue
            if img_path not in data_dict:
                data_dict[img_path] = []
            data_dict[img_path].append([x1, y1, x2, y2])
    return data_dict

def combine_cat_dog_dicts(cat_dict, dog_dict):
    """
    Combine the bounding box dictionaries for cats and dogs into a single dictionary.
    If an image appears in both, merge the bounding box lists.
    """
    combined = {}
    for img_path, boxes in cat_dict.items():
        combined[img_path] = boxes[:]  # Create a copy of the list
    for img_path, boxes in dog_dict.items():
        if img_path not in combined:
            combined[img_path] = []
        combined[img_path].extend(boxes)
    return combined

##############################
# 3) DATASET & DATALOADER
##############################
class BoxImageDataset(Dataset):
    """
    Custom Dataset for loading images and their associated bounding boxes.
    Assumes images are already resized; only normalization is performed.
    """
    def __init__(self, items):
        """
        Args:
            items (list): List of tuples (img_path, boxes)
        """
        super().__init__()
        self.items = items
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(IMAGE_NORMALIZE_MEAN, IMAGE_NORMALIZE_STD)
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, boxes = self.items[idx]
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        return img_tensor, boxes, img_path

def collate_fn(batch):
    """
    Custom collate function for the DataLoader.
    """
    imgs, all_boxes, paths = [], [], []
    for img, boxes, path in batch:
        imgs.append(img)
        all_boxes.append(boxes)
        paths.append(path)
    imgs = torch.stack(imgs, dim=0)
    return imgs, all_boxes, paths

##############################
# 4) ADVANCED BOX SUP FUNCTIONS
##############################
def run_selective_search(pil_img):
    """
    Run selective search on the input PIL image to generate region proposals.
    Returns a list of proposals as tuples (x1, y1, x2, y2).
    """
    np_img = np.array(pil_img)  # Convert image to NumPy array (H,W,3)
    # Run selective search with moderate settings: scale=1.0, min_size=20
    _, proposals = selectivesearch.selective_search(np_img, scale=1.0, min_size=20)
    boxes = set()
    for r in proposals:
        # Each proposal's 'rect' is in the format (y, x, height, width)
        y, x, h, w = r['rect']
        if h == 0 or w == 0:
            continue
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        boxes.add((x1, y1, x2, y2))
    return list(boxes)


def union_of_bboxes(boxes, H, W):
    """
    Create a union mask from a list of bounding boxes.
    Args:
        boxes (list): List of bounding boxes (x1, y1, x2, y2).
        H (int): Height of the image.
        W (int): Width of the image.
    Returns:
        A torch tensor mask with ones in regions covered by any bounding box.
    """
    mask = torch.zeros((H, W), dtype=torch.float32)
    for (x1, y1, x2, y2) in boxes:
        # Skip invalid boxes
        if x2 < x1 or y2 < y1:
            continue
        # Ensure coordinates are within image bounds
        x1, x2 = sorted([max(0, x1), min(W - 1, x2)])
        y1, y2 = sorted([max(0, y1), min(H - 1, y2)])
        mask[y1:y2, x1:x2] = 1
    return mask

def mask_l2_regularization(old_mask, new_mask, reg_lambda=0.01):
    """
    Apply L2 regularization to smooth the new mask relative to the old mask.
    Args:
        old_mask (torch.Tensor): The original mask.
        new_mask (torch.Tensor): The new mask to be regularized.
        reg_lambda (float): Regularization strength.
    Returns:
        The regularized new mask.
    """
    diff = new_mask - old_mask
    return new_mask - reg_lambda * diff

def apply_crf(img_tensor, mask_tensor, n_iter=5):
    """
    Refine a mask using a 2-class DenseCRF.
    Args:
        img_tensor (torch.Tensor): The normalized image tensor of shape (C,H,W).
        mask_tensor (torch.Tensor): The soft mask (foreground probability) of shape (H,W).
        n_iter (int): Number of CRF inference iterations.
    Returns:
        A refined binary mask (torch.Tensor) with values 0 or 1.
    """
    C, H, W = img_tensor.shape
    # Convert the mask to foreground and background probabilities.
    prob_fg = mask_tensor.cpu().numpy()
    prob_bg = 1 - prob_fg
    prob_2 = np.stack([prob_bg, prob_fg], axis=0)  # Shape: (2,H,W)
    unary = unary_from_softmax(prob_2)
    
    # Reconstruct raw image tensor (reverse normalization)
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(3, 1, 1)
    img_tensor_raw = img_tensor * std + mean
    
    # Convert image tensor to a NumPy array (H,W,3) for CRF.
    img_np = (img_tensor_raw.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    # Make sure the numpy array is C-contiguous.
    img_np = np.ascontiguousarray(img_np)
    
    d = dcrf.DenseCRF2D(W, H, 2)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=10, srgb=13, rgbim=img_np, compat=10)
    
    Q = d.inference(n_iter)
    Q = np.array(Q).reshape((2, H, W))
    refined_fg = Q[1, :, :]
    denom = Q[0, :, :] + Q[1, :, :] + 1e-8
    refined_fg = refined_fg / denom
    refined_mask = (refined_fg >= 0.5).astype(np.float32)
    return torch.from_numpy(refined_mask)

def advanced_boxsup(img_tensor, bounding_boxes, region_prop_list, 
                    num_iters=3, alpha=0.6, overlap_thresh=0.2,
                    apply_crf_flag=True, reg_lambda=0.01):
    """
    Apply an advanced BoxSup approach to generate a pseudo-mask.
    Steps:
      1) Compute the union of the provided bounding boxes.
      2) Filter region proposals based on overlap (IoU) with the union mask.
      3) Iteratively merge proposals using L2 regularization.
      4) Optionally refine the result with a CRF.
    
    Args:
        img_tensor (torch.Tensor): Normalized image tensor (C,H,W).
        bounding_boxes (list): List of bounding boxes [(x1,y1,x2,y2), ...].
        region_prop_list (list): List of region proposals (x1,y1,x2,y2).
        num_iters (int): Number of iterative merging iterations.
        alpha (float): Blending factor for updating the mask.
        overlap_thresh (float): Minimum IoU threshold to consider a proposal.
        apply_crf_flag (bool): Whether to apply CRF post-processing.
        reg_lambda (float): Regularization strength for L2 reg.
    
    Returns:
        A pseudo-mask (torch.Tensor) refined to 0/1 values.
    """
    _, H, W = img_tensor.shape
    # Step 1: Create a base mask from the union of bounding boxes.
    combined_mask = union_of_bboxes(bounding_boxes, H, W).float()
    # box_union will be used to compute IoU with proposals.
    box_union = combined_mask.clone()

    # Step 2: Filter region proposals based on IoU with the union mask.
    proposals = []
    for (rx1, ry1, rx2, ry2) in region_prop_list:
        rx1, rx2 = sorted([rx1, rx2])
        ry1, ry2 = sorted([ry1, ry2])
        # Skip proposals completely outside the image.
        if rx2 < 0 or rx1 >= W or ry2 < 0 or ry1 >= H:
            continue
        rx1 = max(0, rx1); rx2 = min(W - 1, rx2)
        ry1 = max(0, ry1); ry2 = min(H - 1, ry2)
        if rx2 < rx1 or ry2 < ry1:
            continue
        mask_ = torch.zeros((H, W), dtype=torch.float32)
        mask_[ry1:ry2, rx1:rx2] = 1
        # Compute IoU with the union mask.
        inter = (mask_ * box_union).sum().item()
        union = (mask_ + box_union).clamp_(0, 1).sum().item()
        if union > 0:
            iou = inter / union
            if iou >= overlap_thresh:
                proposals.append(mask_)
    
    # Step 3: Iterative merging of proposals with L2 regularization.
    for _ in range(num_iters):
        new_mask_accum = torch.zeros((H, W), dtype=torch.float32)
        count_accum = torch.zeros((H, W), dtype=torch.float32)
        for pmask in proposals:
            # Only consider proposals with a minimal overlap.
            if (pmask * combined_mask).sum().item() > 10:
                new_mask_accum += pmask
                count_accum += (pmask > 0)
        merged = torch.zeros((H, W), dtype=torch.float32)
        valid = (count_accum > 0)
        merged[valid] = new_mask_accum[valid] / count_accum[valid]
        # Apply L2 regularization to smooth the update.
        merged_reg = mask_l2_regularization(combined_mask, merged, reg_lambda=reg_lambda)
        # Blend the previous mask with the new regularized mask.
        combined_mask = alpha * combined_mask + (1 - alpha) * merged_reg
        combined_mask.clamp_(0, 1)

    # Step 4: Optionally refine the mask with CRF.
    if apply_crf_flag:
        refined_mask = apply_crf(img_tensor, combined_mask, n_iter=5)
        return refined_mask
    else:
        return combined_mask

##############################
# 5) CROSS-VALIDATION & HYPERPARAMETER SEARCH
##############################
def compute_pseudo_iou(pred_mask, bounding_boxes):
    """
    Compute a pseudo Intersection-over-Union (IoU) between the predicted mask
    and the union of the bounding boxes.
    This is a proxy metric since we lack ground truth segmentation.
    """
    if pred_mask.dim() < 2:
        return 0
    H, W = pred_mask.shape
    union_box = torch.zeros((H, W), dtype=torch.float32)
    for (x1, y1, x2, y2) in bounding_boxes:
        x1, x2 = sorted([max(0, x1), min(W - 1, x2)])
        y1, y2 = sorted([max(0, y1), min(H - 1, y2)])
        union_box[y1:y2, x1:x2] = 1
    inter = (pred_mask * union_box).sum().item()
    union = (pred_mask + union_box).clamp_(0, 1).sum().item()
    if union == 0:
        return 1 if inter == 0 else 0
    return inter / union

def evaluate_boxsup(items, config, device='cpu'):
    """
    Evaluate the advanced BoxSup approach on a set of images by computing
    the average pseudo-IoU between the generated pseudo-mask and the union
    of the provided bounding boxes.
    """
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(IMAGE_NORMALIZE_MEAN, IMAGE_NORMALIZE_STD)
    ])
    ious = []
    for (img_path, boxes) in items:
        # Load image and apply transformation.
        pil_img = Image.open(img_path).convert('RGB')
        img_tensor = transform(pil_img).to(device)
        # Generate region proposals using selective search.
        region_props = run_selective_search(pil_img)
        # Generate pseudo-mask using advanced BoxSup.
        pmask = advanced_boxsup(
            img_tensor, boxes, region_props,
            num_iters=config['num_iters'],
            alpha=config['alpha'],
            overlap_thresh=config['overlap_thresh'],
            apply_crf_flag=config['apply_crf'],
            reg_lambda=config['reg_lambda']
        )
        iou_ = compute_pseudo_iou(pmask, boxes)
        ious.append(iou_)
    return float(np.mean(ious))

def random_search_5fold(all_items, n_folds=5, n_search=5, device='cpu'):
    """
    Perform random hyperparameter search using 5-fold cross-validation.
    Returns the best configuration and its associated average IoU score.
    """
    # Shuffle items randomly
    random.shuffle(all_items)
    # Create folds manually
    folds = []
    fold_size = len(all_items) // n_folds
    for i in range(n_folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_folds - 1 else len(all_items)
        val_part = all_items[start:end]
        train_part = all_items[:start] + all_items[end:]
        folds.append((train_part, val_part))

    best_config = None
    best_score = 0.0

    for trial_i in range(n_search):
        # Randomly sample hyperparameters
        hp = {
            'num_iters': random.randint(2, 5),
            'alpha': round(random.uniform(0.5, 0.9), 2),
            'overlap_thresh': round(random.uniform(0.1, 0.3), 2),
            'apply_crf': random.choice([True, False]),
            'reg_lambda': round(random.uniform(0.005, 0.02), 3)
        }
        print(f"\n=== Trial {trial_i + 1}/{n_search}: {hp}")

        fold_scores = []
        for i, (_, valp) in enumerate(folds):
            # Evaluate on the validation partition only.
            sc = evaluate_boxsup(valp, hp, device=device)
            fold_scores.append(sc)
            print(f"   Fold {i}: IoU = {sc:.4f}")

        avg_sc = float(np.mean(fold_scores))
        print(f" => Trial average IoU = {avg_sc:.4f}")
        if avg_sc > best_score:
            best_score = avg_sc
            best_config = hp
            print("   * New best configuration found *")

    return best_config, best_score

def produce_final_masks(all_items, config, out_dir, device='cpu'):
    """
    Produce and save the final pseudo-masks for each image using the best
    hyperparameter configuration.
    Each mask is saved as a PNG file.
    """
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(IMAGE_NORMALIZE_MEAN, IMAGE_NORMALIZE_STD)
    ])
    for (img_path, boxes) in all_items:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        pil_img = Image.open(img_path).convert('RGB')
        img_tensor = transform(pil_img).to(device)
        # Generate region proposals and compute the pseudo-mask.
        region_props = run_selective_search(pil_img)
        pmask = advanced_boxsup(
            img_tensor, boxes, region_props,
            num_iters=config['num_iters'],
            alpha=config['alpha'],
            overlap_thresh=config['overlap_thresh'],
            apply_crf_flag=config['apply_crf'],
            reg_lambda=config['reg_lambda']
        )
        # Convert mask to 0-255 uint8 and save as PNG.
        pm = (pmask.cpu().numpy() * 255).astype(np.uint8)
        out_path = os.path.join(out_dir, f"{base_name}_mask.png")
        Image.fromarray(pm).save(out_path)

##############################
# 6) MAIN PIPELINE EXECUTION
##############################
def main():
    # Define paths for the bounding box text files.
    cat_txt = "../Data/paths_cats_with_box.txt"
    dog_txt = "../Data/paths_dogs_with_box.txt"

    # Parse the text files to obtain dictionaries mapping image paths to bounding boxes.
    cat_dict = parse_paths_file(cat_txt)
    dog_dict = parse_paths_file(dog_txt)
    combined_dict = combine_cat_dog_dicts(cat_dict, dog_dict)

    # Build a list of (img_path, bounding_boxes) but skip images without any valid boxes.
    all_items = []
    for img_path, boxes in combined_dict.items():
        if boxes:  # Only include images with at least one valid bounding box
            all_items.append((img_path, boxes))

    # Determine computing device.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Perform 5-fold cross validation with random hyperparameter search.
    best_config, best_score = random_search_5fold(all_items, n_folds=N_FOLDS, n_search=N_SEARCH, device=device)
    print(f"\nBEST config: {best_config}, best 5-fold average IoU = {best_score:.4f}")

    # Save the best hyperparameters as the "model".
    model_path = os.path.join(OUT_MODEL_DIR, "boxsup_best_config.json")
    with open(model_path, 'w') as f:
        json.dump(best_config, f)
    print(f"Saved best BoxSup configuration to {model_path}")

    # Produce final pseudo-masks for all images.
    produce_final_masks(all_items, best_config, OUT_MASK_DIR, device=device)
    print(f"Pseudo-masks saved in {OUT_MASK_DIR} for each image.")
    print("DONE. You can now train DeepLabv3+ using these pseudo masks.")

if __name__ == "__main__":
    main()
