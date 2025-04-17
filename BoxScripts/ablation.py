#!/usr/bin/env python3
"""
ablation_experiment_scratch_rescaled.py

This script performs an ablation study of our weakly‑supervised segmentation pipeline
running from scratch. For each experiment variant, it:
  1. Generates refined pseudo‑masks via BoxSup (CRF/morph/edge options).
  2. Uses those pseudo‑masks to train a fresh DeepLabV3 model (80/20 split).
  3. Evaluates the trained model against the original box‑mask (IoU & pixel accuracy).

At the end, it prints mean IoU and pixel accuracy for each ablation config.
"""

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models.segmentation import deeplabv3_resnet50
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

# ------------------ Constants ------------------
IMAGE_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
IMAGE_NORMALIZE_STD  = [0.229, 0.224, 0.225]
IMG_SIZE = 224

ADV_PARAMS_CONST = {
    "num_iters": 3,
    "alpha": 0.42,
    "overlap_thresh": 0.3,
    "reg_lambda": 0.017
}

EXPERIMENT_CONFIGS = {
    "None":          {"apply_crf": False, "do_morph": False, "edge_weighting": False},
    "Apply_CRF":     {"apply_crf": True,  "do_morph": False, "edge_weighting": False},
    "Do_Morph":      {"apply_crf": False, "do_morph": True,  "edge_weighting": False},
    "Edge_Weight":   {"apply_crf": False, "do_morph": False, "edge_weighting": True},
    "CRF_and_Morph": {"apply_crf": True,  "do_morph": True,  "edge_weighting": False},
    "CRF_and_Edge":  {"apply_crf": True,  "do_morph": False, "edge_weighting": True},
    "Morph_and_Edge":{"apply_crf": False, "do_morph": True,  "edge_weighting": True},
    "All":           {"apply_crf": True,  "do_morph": True,  "edge_weighting": True}
}

CATS_BOX_FILE   = "../Data/paths_cats_with_box.txt"
DOGS_BOX_FILE   = "../Data/paths_dogs_with_box.txt"

# ------------------ Helper Functions ------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE in ["cuda", "mps"]:
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_mIoU_and_pixel_accuracy(pred_bin, gt_bin):
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    iou = (intersection + 1e-6) / (union + 1e-6)
    pixel_acc = (pred_bin == gt_bin).sum() / gt_bin.size
    return iou, pixel_acc


def union_of_bboxes(box, H, W):
    x1, y1, x2, y2 = box
    x1, x2 = sorted([max(0, x1), min(W-1, x2)])
    y1, y2 = sorted([max(0, y1), min(H-1, y2)])
    mask = torch.zeros((H, W), dtype=torch.float32)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 1
    return mask

# (Include compute_boundary, boundary_alignment_weight, get_region_proposals_slic,
#  compute_edge_map, apply_crf, morphological_close as in your original file)
# For brevity, assume they're copy-pasted here unchanged.

# ------------------ BoxFilesDatasetScratch ------------------
class BoxFilesDatasetScratch(Dataset):
    def __init__(self, cats_file, dogs_file):
        self.items = []
        for fpath in [cats_file, dogs_file]:
            with open(fpath, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts)<5: continue
                    img, x1,y1,x2,y2 = parts[0], *map(int, parts[1:5])
                    if x1==y1==x2==y2==0: continue
                    self.items.append((img, [x1,y1,x2,y2]))
        self.transform = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(IMAGE_NORMALIZE_MEAN, IMAGE_NORMALIZE_STD)
        ])
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        img_path, bbox = self.items[i]
        pil = Image.open(img_path).convert('RGB')
        w,h = pil.size
        sx, sy = IMG_SIZE/w, IMG_SIZE/h
        box224 = [int(round(bbox[j]*(sx if j%2==0 else sy))) for j in range(4)]
        img_t = self.transform(pil)
        return img_t, box224, img_path

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

# ------------------ Ablation Study ------------------
def run_ablation_experiment(dataset, device,
                            epochs=5, lr=1e-4, batch_size=4):
    results = {}
    full_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for name, cfg in EXPERIMENT_CONFIGS.items():
        print(f"\n=== Experiment: {name} ===")
        # 1) generate pseudo-masks
        pseudo = []
        for img_t, box224, img_path in full_loader:
            pm = advanced_boxsup(img_t.squeeze(0).to(device),
                                Image.open(img_path[0]).convert('RGB'),
                                box224, cfg, device)
            pseudo.append(pm.long().cpu())

        # 2) build pseudo-dataset
        class PseudoDS(Dataset):
            def __init__(self, orig, masks):
                self.orig = orig; self.masks = masks
            def __len__(self): return len(self.orig)
            def __getitem__(self, i): return self.orig[i][0], self.masks[i]

        psds = PseudoDS(dataset, pseudo)
        n = len(psds)
        tsize = int(0.8*n); vsize = n-tsize
        train_ds, val_ds = random_split(psds, [tsize,vsize], generator=torch.Generator().manual_seed(42))
        tr = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
        val= DataLoader(val_ds,   batch_size=1,         shuffle=False)

        # 3) train DeepLabV3
        model = deeplabv3_resnet50(num_classes=2).to(device)
        opt   = optim.Adam(model.parameters(), lr=lr)
        crit = torch.nn.CrossEntropyLoss()
        for ep in range(epochs):
            model.train()
            for imgs, masks in tr:
                imgs, masks = imgs.to(device), masks.to(device)
                opt.zero_grad()
                out = model(imgs)['out']
                loss = crit(out, masks)
                loss.backward(); opt.step()

        # 4) evaluate vs box masks
        model.eval()
        tot_iou=tot_px=cnt=0
        for imgs,_ in val:
            idx = val.dataset.indices[cnt]
            _,box224,_ = dataset[idx]
            boxm = union_of_bboxes(box224, IMG_SIZE, IMG_SIZE).numpy().astype(np.uint8)
            imgs = imgs.to(device)
            with torch.no_grad(): out = model(imgs)['out']
            pred = torch.argmax(out,1).squeeze(0).cpu().numpy().astype(np.uint8)
            iou, px = compute_mIoU_and_pixel_accuracy(pred, boxm)
            tot_iou+=iou; tot_px+=px; cnt+=1

        miou = tot_iou/cnt; mpx = tot_px/cnt
        print(f"Mean IoU vs Box: {miou:.4f}, Pixel Acc: {mpx:.4f}")
        results[name] = (miou, mpx)

    return results

# ------------------ Main ------------------
if __name__ == '__main__':
    set_seed(42)
    ds = BoxFilesDatasetScratch(CATS_BOX_FILE, DOGS_BOX_FILE)
    if len(ds)==0:
        print("No samples found."); exit(1)
    run_ablation_experiment(ds, DEVICE)

