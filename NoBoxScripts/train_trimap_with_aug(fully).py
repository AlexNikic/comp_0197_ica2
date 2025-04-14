import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np

# ======================
# set seeds
# ======================
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

# ======================
# 加Dataset using trainval.txt split
# ======================
def load_trainval_list(txt_path):
    with open(txt_path, "r") as f:
        return sorted([line.strip().split()[0] for line in f if line.strip()])

# ======================
# Enhanced custom datasets
# ======================
class AugmentedTrimapDataset(Dataset):
    def __init__(self, names, image_dir, mask_dir, augment=False):
        self.names = names
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.augment = augment
        self.resize = transforms.Resize((224, 224), interpolation=Image.NEAREST)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        image = Image.open(os.path.join(self.image_dir, name + ".jpg")).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, name + ".png"))

        # Convert trimap to binary: foreground=1, others=0
        mask_np = np.array(mask)
        mask_np = (mask_np == 1).astype(np.uint8)
        mask = Image.fromarray(mask_np * 255)

        if self.augment:
            if torch.rand(1).item() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if torch.rand(1).item() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            if torch.rand(1).item() > 0.5:
                image = TF.adjust_brightness(image, brightness_factor=1.2)
            if torch.rand(1).item() > 0.5:
                image = TF.adjust_contrast(image, contrast_factor=1.2)

        image = TF.resize(image, [224, 224])
        image = TF.to_tensor(image)

        mask = self.resize(mask)
        mask_np = np.array(mask)
        mask_tensor = torch.from_numpy((mask_np > 0).astype(np.uint8)).long()

        return image, mask_tensor

# ======================
# mIoU calculation
# ======================
def compute_mIoU(pred, target):
    pred = torch.argmax(pred, dim=1).byte()
    target = target.byte()
    intersection = (pred & target).float().sum()
    union = (pred | target).float().sum()
    return (intersection + 1e-6) / (union + 1e-6)

# ======================
# Training function
# ======================
def train_with_augmentation():
    set_seed()

    image_dir = "oxford-iiit-pet/images"
    mask_dir = "oxford-iiit-pet/annotations/trimaps"
    split_file = "oxford-iiit-pet/annotations/trainval.txt"
    model_path = "best_deeplab_with_aug.pth"

    if os.path.exists(model_path):
        print("✅ Model already exists. Skipping training.")
        return

    all_names = load_trainval_list(split_file)
    split_idx = int(0.8 * len(all_names))
    train_names = all_names[:split_idx]
    val_names = all_names[split_idx:]

    train_set = AugmentedTrimapDataset(train_names, image_dir, mask_dir, augment=True)
    val_set = AugmentedTrimapDataset(val_names, image_dir, mask_dir, augment=False)

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False)

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model = deeplabv3_resnet50(num_classes=2).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_mIoU = 0.0

    for epoch in range(1, 6):
        model.train()
        total_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)["out"]
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        ious = []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)["out"]
                iou = compute_mIoU(outputs, masks)
                ious.append(iou)
        miou = sum(ious) / len(ious)

        print(f"Epoch {epoch} - Train Loss: {total_loss / len(train_loader):.4f}")
        print(f"Epoch {epoch} - Validation mIoU: {miou:.4f}")
        
        if miou > best_mIoU:
            best_mIoU = miou
            torch.save(model.state_dict(), model_path)
            print("✅ Best model saved.")
        print(f"Best mIoU so far: {best_mIoU:.4f}")

if __name__ == "__main__":
    train_with_augmentation()
