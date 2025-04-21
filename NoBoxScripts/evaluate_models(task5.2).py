import os
import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
import numpy as np

# === Load split ===
def load_eval_names(split_file):
    with open(split_file, "r") as f:
        return sorted([line.strip().split()[0] for line in f if line.strip()])

# === mIoU ===
def compute_mIoU_eval(pred, target):
    pred = torch.argmax(pred, dim=1).byte()
    target = target.byte()
    intersection = (pred & target).float().sum((1, 2))
    union = (pred | target).float().sum((1, 2))
    return (intersection + 1e-6) / (union + 1e-6)

# === pixel accuracy ===
def compute_pixel_accuracy(pred, target):
    pred = torch.argmax(pred, dim=1)
    correct = (pred == target).float().sum()
    total = torch.numel(target)
    return (correct / total).item()

# ==== Evaluation Function ====
def evaluate_model(model_path, image_dir, mask_dir, val_names, device):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = deeplabv3_resnet50(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    resize_mask = transforms.Resize((224, 224), interpolation=Image.NEAREST)

    total_iou = 0.0
    total_pa = 0.0
    total_samples = 0

    for name in val_names:
        # === Load image ===
        img_path = os.path.join(image_dir, name + ".jpg")
        mask_path = os.path.join(mask_dir, name + ".png")

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            continue

        # === Load image ===
        img = Image.open(img_path).convert("RGB")
        input_tensor = transform(img).unsqueeze(0).to(device)

        # === Load and preprocess mask ===
        mask = Image.open(mask_path).convert("L")
        mask_np = np.array(resize_mask(mask))

        # === Create binary mask and valid mask ===
        if "pseudo" in mask_dir.lower():
            binary_mask_np = (mask_np >= 128).astype(np.uint8)
            valid_mask_np = np.ones_like(mask_np, dtype=bool)
        else:  # Trimap
            binary_mask_np = (mask_np == 1).astype(np.uint8)
            valid_mask_np = np.ones_like(mask_np, dtype=bool)
            
        
        gt = torch.from_numpy(binary_mask_np).long().to(device)
        valid = torch.from_numpy(valid_mask_np).bool().to(device)

        with torch.no_grad():
            output = model(input_tensor)["out"]
            pred = torch.argmax(output, dim=1).squeeze(0)

        pred = pred[valid]
        gt = gt[valid]

        if gt.numel() == 0:
            continue

        intersection = ((pred == 1) & (gt == 1)).sum().item()
        union = ((pred == 1) | (gt == 1)).sum().item()
        correct = (pred == gt).sum().item()
        total = gt.numel()

        total_iou += intersection / (union + 1e-6)
        total_pa += correct / total
        total_samples += 1

    if total_samples == 0:
        return 0.0, 0.0

    mean_iou = float(total_iou / total_samples)
    mean_pa = float(total_pa / total_samples)

    return mean_iou, mean_pa




def evaluate_pseudo_with_dice(model_path, image_dir, mask_dir, val_names, device):
    model = deeplabv3_resnet50(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    resize_mask = transforms.Resize((224, 224), interpolation=Image.NEAREST)

    total_iou = 0.0
    total_acc = 0.0
    total_samples = 0

    for name in val_names:
        img_path = os.path.join(image_dir, name + ".jpg")
        mask_path = os.path.join(mask_dir, name + ".png")
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            continue

        # Load image
        img = Image.open(img_path).convert("RGB")
        input_tensor = transform(img).unsqueeze(0).to(device)

        # Load pseudo mask
        mask = Image.open(mask_path).convert("L")
        mask = resize_mask(mask)
        mask_np = np.array(mask)
        mask_bin = (mask_np >= 128).astype(np.uint8)
        mask_tensor = torch.from_numpy(mask_bin).long().to(device)

        with torch.no_grad():
            output = model(input_tensor)["out"]
            pred = torch.argmax(output, dim=1).squeeze(0)

        # Evaluate
        intersection = ((pred == 1) & (mask_tensor == 1)).sum().item()
        union = ((pred == 1) | (mask_tensor == 1)).sum().item()
        correct = (pred == mask_tensor).sum().item()
        total = mask_tensor.numel()

        if union == 0:
            continue

        total_iou += intersection / (union + 1e-6)
        total_acc += correct / total
        total_samples += 1

    if total_samples == 0:
        return 0.0, 0.0

    mean_iou = float(total_iou / total_samples)
    mean_acc = float(total_acc / total_samples)

    return mean_iou, mean_acc



# === result function ===
def main():
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    test_txt = "oxford-iiit-pet/annotations/test.txt"
    val_names = load_eval_names(test_txt)

    models = {
        "Pseudo Mask": ("deeplab_output/best_deeplabv3_pseudo.pth", "pseudo_mask"),
        "Pseudo + Dice Loss": ("best_deeplabv3_pseudo_with_dice.pth", "pseudo_mask"),
        "Pseudo + Aug + ComboLoss": ("best_deeplabv3_pseudo_with_aug_refined.pth", "pseudo_mask"),
        "Pseudo + threshold0.5": ("best_deeplab_pseudo_thr05.pth", "pseudo_mask"),
        "Pseudo + threshold0.8": ("best_deeplab_pseudo_thr08.pth", "pseudo_mask"),
        "Pseudo + ER": ("deeplab_er_and_ecr.pth", "pseudo_mask_er"),
    }

    print("=== Evaluation Results (Test Set) ===")
    print(f"📦 Total test samples: {len(val_names)}\n")
    
    results = {}

    for name, (path, mask_dir) in models.items():
        if "dice" in name.lower():
            miou, acc = evaluate_pseudo_with_dice(path, "oxford-iiit-pet/images", mask_dir, val_names, device)
        #elif "aug" in name.lower():
            #miou, acc = evaluate_model_with_aug(path, "oxford-iiit-pet/images", mask_dir, val_names, device)
        else:
            miou, acc = evaluate_model(path, "oxford-iiit-pet/images", mask_dir, val_names, device)


        results[name] = (miou, acc)
        print(f"{name:} mIoU: {miou:.4f}   |   Pixel Acc: {acc:.4f}")

    # Save results to file
    result_path = "evaluation_results5.2.txt"
    with open(result_path, "w") as f:
        f.write("=== Evaluation Results (Test Set) ===\n")
        f.write(f"Total test samples: {len(val_names)}\n\n")
        for name, (miou, acc) in results.items():
            result_line = f"{name:} mIoU: {miou:.4f}   |   Pixel Acc: {acc:.4f}"
            f.write(result_line + "\n")

    print(f"\n📄 Saved evaluation results to: {result_path}")

if __name__ == "__main__":
    main()
