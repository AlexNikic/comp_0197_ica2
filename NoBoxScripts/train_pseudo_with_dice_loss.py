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

# ======== Dice Loss =========
#class DiceLoss(nn.Module):
    #def __init__(self, smooth=1e-6):
        #super(DiceLoss, self).__init__()
        #self.smooth = smooth

    #def forward(self, preds, targets):
        #preds = torch.softmax(preds, dim=1)[:, 1, :, :]
        #targets = targets.float()
        #intersection = (preds * targets).sum()
        #union = preds.sum() + targets.sum()
        #dice = (2. * intersection + self.smooth) / (union + self.smooth)
        #return 1 - dice


class WeightedDiceLoss(nn.Module):
    def __init__(self, weight=None, smooth=1e-6):
        super(WeightedDiceLoss, self).__init__()
        self.weight = weight  # Tensor of shape [num_classes], e.g., [0.2, 0.8]
        self.smooth = smooth

    def forward(self, preds, targets):
        num_classes = preds.shape[1]
        preds = torch.softmax(preds, dim=1)
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        total_dice = 0.0
        for c in range(num_classes):
            pred_c = preds[:, c]
            target_c = targets_one_hot[:, c]
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            w = self.weight[c] if self.weight is not None else 1.0
            total_dice += w * (1 - dice)
        return total_dice / num_classes

# ======== Set seed =========
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

# ======== Dataset =========
class PseudoMaskDataset(Dataset):
    def __init__(self, names, image_dir, mask_dir):
        self.names = names
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.resize = transforms.Resize((224, 224), interpolation=Image.NEAREST)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        image = Image.open(os.path.join(self.image_dir, name + ".jpg")).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, name + ".png")).convert("L")

        image = TF.resize(image, [224, 224])
        image = TF.to_tensor(image)

        mask = self.resize(mask)
        mask_np = np.array(mask)
        mask_bin = (mask_np >= 128).astype(np.uint8)
        mask_tensor = torch.from_numpy(mask_bin).long()

        return image, mask_tensor

# ======== mIoU =========
def compute_mIoU(pred, target):
    pred = torch.argmax(pred, dim=1).byte()
    target = target.byte()
    intersection = (pred & target).float().sum()
    union = (pred | target).float().sum()
    return (intersection + 1e-6) / (union + 1e-6)

# ======== Training =========
def train_pseudo_with_weighted_dice():
#def train_pseudo_with_dice_loss():
    set_seed()

    image_dir = "oxford-iiit-pet/images"
    mask_dir = "pseudo_mask"
    split_file = "oxford-iiit-pet/annotations/trainval.txt"
    model_path = "best_deeplabv3_pseudo_with_dice.pth"

    if os.path.exists(model_path):
        print("✅ Already trained. Skipping.")
        return

    with open(split_file, "r") as f:
        all_names = sorted([line.strip().split()[0] for line in f if line.strip()])

    val_split = int(0.8 * len(all_names))
    train_names = all_names[:val_split]
    val_names = all_names[val_split:]

    train_set = PseudoMaskDataset(train_names, image_dir, mask_dir)
    val_set = PseudoMaskDataset(val_names, image_dir, mask_dir)

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = deeplabv3_resnet50(num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    #criterion = DiceLoss()
    # Class weights: [background, foreground]
    dice_loss = WeightedDiceLoss(weight=torch.tensor([0.3, 0.7]).to(device))

    best_miou = 0.0
    for epoch in range(1, 6):
        model.train()
        total_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)["out"]
            #loss = criterion(preds, masks)
            loss = dice_loss(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            ious = []
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)["out"]
                ious.append(compute_mIoU(preds, masks))
            miou = sum(ious) / len(ious)

        print(f"Epoch {epoch} - Loss: {total_loss / len(train_loader):.4f} | mIoU: {miou:.4f}")
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), model_path)
            print("✅ Best model saved.")
        print(f"📊 Best mIoU so far: {best_miou:.4f}")

if __name__ == "__main__":
    train_pseudo_with_weighted_dice()
