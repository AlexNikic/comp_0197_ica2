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

# ========= Dice + BCE Loss =========
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.softmax(preds, dim=1)[:, 1, :, :]
        targets = targets.float()
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice

class ComboLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, preds, targets):
        return self.alpha * self.ce(preds, targets) + (1 - self.alpha) * self.dice(preds, targets)

# ========= Seed =========
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

# ========= Dataset =========
class PseudoMaskDatasetWithAug(Dataset):
    def __init__(self, names, image_dir, mask_dir, augment=False, mix_prob=0.3):
        self.names = names
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.augment = augment
        self.mix_prob = mix_prob
        self.resize = transforms.Resize((224, 224), interpolation=Image.NEAREST)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        image = Image.open(os.path.join(self.image_dir, name + ".jpg")).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, name + ".png")).convert("L")

        # ========== Optional Mix ==========
        if self.augment and torch.rand(1).item() < self.mix_prob:
            mix_idx = torch.randint(0, len(self.names), (1,)).item()
            mix_name = self.names[mix_idx]
            image2 = Image.open(os.path.join(self.image_dir, mix_name + ".jpg")).convert("RGB")
            mask2 = Image.open(os.path.join(self.mask_dir, mix_name + ".png")).convert("L")

            image = TF.resize(image, [224, 224])
            image2 = TF.resize(image2, [224, 224])
            mask = self.resize(mask)
            mask2 = self.resize(mask2)

            image = TF.to_tensor(image)
            image2 = TF.to_tensor(image2)

            lambda_ = torch.FloatTensor(1).uniform_(0.3, 0.7).item()
            image = lambda_ * image + (1 - lambda_) * image2

            mask_np = np.array(mask)
            mask2_np = np.array(mask2)
            mask_mix = ((lambda_ * (mask_np >= 128) + (1 - lambda_) * (mask2_np >= 128)) >= 0.5).astype(np.uint8)
            mask_tensor = torch.from_numpy(mask_mix).long()

            return image, mask_tensor

        # ========== Standard Preprocess ==========
        if self.augment:
            if torch.rand(1).item() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if torch.rand(1).item() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            if torch.rand(1).item() > 0.5:
                angle = torch.randint(-20, 20, (1,)).item()
                image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
            if torch.rand(1).item() > 0.5:
                image = TF.adjust_brightness(image, 0.8 + 0.4 * torch.rand(1).item())
            if torch.rand(1).item() > 0.5:
                image = TF.adjust_contrast(image, 0.8 + 0.4 * torch.rand(1).item())

        image = TF.resize(image, [224, 224])
        image = TF.to_tensor(image)
        mask = self.resize(mask)
        mask_np = np.array(mask)
        mask_bin = (mask_np >= 128).astype(np.uint8)
        mask_tensor = torch.from_numpy(mask_bin).long()

        return image, mask_tensor


# ========= mIoU =========
def compute_mIoU(pred, target):
    pred = torch.argmax(pred, dim=1).byte()
    target = target.byte()
    intersection = (pred & target).float().sum((1, 2))
    union = (pred | target).float().sum((1, 2))
    return (intersection + 1e-6) / (union + 1e-6)

# ========= Training =========
def train_pseudo_with_aug():
    set_seed()
    image_dir = "oxford-iiit-pet/images"
    mask_dir = "pseudo_mask"
    split_file = "oxford-iiit-pet/annotations/trainval.txt"
    save_path = "best_deeplabv3_pseudo_with_aug.pth"

    if os.path.exists(save_path):
        print("✅ Already trained. Skipping.")
        return

    with open(split_file, "r") as f:
        all_names = sorted([line.strip().split()[0] for line in f if line.strip()])

    split = int(0.8 * len(all_names))
    train_names = all_names[:split]
    val_names = all_names[split:]

    #train_set = PseudoMaskDatasetWithAug(train_names, image_dir, mask_dir, augment=True)
    train_set = PseudoMaskDatasetWithAug(train_names, image_dir, mask_dir, augment=True, mix_prob=0.7)

    val_set = PseudoMaskDatasetWithAug(val_names, image_dir, mask_dir, augment=False)

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = deeplabv3_resnet50(num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    criterion = ComboLoss(alpha=0.7)

    best_miou = 0.0
    patience = 6
    no_improve = 0

    for epoch in range(1, 51):
        model.train()
        total_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)["out"]
            loss = criterion(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        ious = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)["out"]
                ious.append(compute_mIoU(preds, masks))
        miou = torch.stack(ious).mean().item()

        print(f"Epoch {epoch} - Loss: {total_loss / len(train_loader):.4f} | mIoU: {miou:.4f}")
        
        scheduler.step(miou)
        print(f"🔧 Current LR: {scheduler.optimizer.param_groups[0]['lr']:.6f}")

        if miou > best_miou:
            best_miou = miou
            no_improve = 0
            torch.save(model.state_dict(), save_path)
            print("✅ Best model saved.")
        else:
            no_improve += 1

        if no_improve >= patience:
            print("⏹️ Early stopping due to no improvement.")
            break

        print(f"📊 Best mIoU so far: {best_miou:.4f}")

if __name__ == "__main__":
    train_pseudo_with_aug()

