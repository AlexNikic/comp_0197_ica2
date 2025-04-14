import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
import numpy as np

# ============================
# Set seeds
# ============================
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# ============================
# Dataset
# ============================
class PetSegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        image = self.transform(image)
        mask = TF.resize(mask, [224, 224], interpolation=Image.NEAREST)
        mask = TF.pil_to_tensor(mask).squeeze(0).long() // 255  # binary: 0 or 1
        return image, mask

# ============================
# mIoU Calculation
# ============================
def compute_mIoU(pred, target):
    pred = (pred > 0.5).long()
    intersection = (pred & target).float().sum()
    union = (pred | target).float().sum()
    return (intersection + 1e-6) / (union + 1e-6)

# ============================
# Load Official Trainval Names
# ============================
def load_official_trainval_names(txt_path):
    with open(txt_path, "r") as f:
        lines = f.readlines()
    return sorted([line.strip().split()[0] for line in lines if line.strip()])

# ============================
# Training Function
# ============================
def train_weaksup_model():
    set_seed()

    image_dir = "oxford-iiit-pet/images"
    mask_dir = "pseudo_mask"
    trainval_txt = "oxford-iiit-pet/annotations/trainval.txt"
    save_dir = "deeplab_output"
    model_path = os.path.join(save_dir, "best_deeplabv3_pseudo.pth")
    os.makedirs(save_dir, exist_ok=True)
    
    if os.path.exists(model_path):
        print("✅ Model already exists. Skipping training.")
        return
    
    # Load all image names (sorted for deterministic split)
    all_names = load_official_trainval_names(trainval_txt)
    split = int(0.8 * len(all_names))
    train_names = all_names[:split]
    val_names = all_names[split:]

    def get_paths(names):
        imgs = [os.path.join(image_dir, name + ".jpg") for name in names]
        masks = [os.path.join(mask_dir, name + ".png") for name in names]
        return imgs, masks

    train_imgs, train_masks = get_paths(train_names)
    val_imgs, val_masks = get_paths(val_names)

    train_loader = DataLoader(PetSegmentationDataset(train_imgs, train_masks), batch_size=8, shuffle=True)
    val_loader = DataLoader(PetSegmentationDataset(val_imgs, val_masks), batch_size=1, shuffle=False)

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model = deeplabv3_resnet50(num_classes=2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_mIoU = 0.0

    for epoch in range(1, 6):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)["out"]
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch} - Train Loss: {running_loss / len(train_loader):.4f}")

        # ====== Validation ======
        model.eval()
        total_iou = 0.0
        with torch.no_grad():
            for i, (image, mask) in enumerate(val_loader):
                image, mask = image.to(device), mask.to(device)
                output = model(image)["out"]
                pred = torch.argmax(output, dim=1)
                iou = compute_mIoU(pred, mask)
                total_iou += iou.item()

                # Save prediction image
                pred_mask = pred.squeeze(0).cpu().byte() * 255
                Image.fromarray(pred_mask.numpy()).save(os.path.join(save_dir, f"pred_{val_names[i]}.png"))

        mean_iou = total_iou / len(val_loader)
        print(f"Epoch {epoch} - Validation mIoU: {mean_iou:.4f}")

        # Save best model
        if mean_iou > best_mIoU:
            best_mIoU = mean_iou
            torch.save(model.state_dict(), os.path.join(save_dir, "best_deeplabv3_pseudo.pth"))
            print("✅ Best model saved.")
        
        print(f"Best mIoU so far: {best_mIoU:.4f}")

if __name__ == "__main__":
    train_weaksup_model()

# This script trains a DeepLabV3 model on a pet segmentation dataset using pseudo-labels.
# It includes dataset loading, training, validation, and saving the best model based on mIoU.
# The model is saved in the "deeplab_output" directory, and the predictions are saved as images.