import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

# === Set seeds ===
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

# === Dataset using trainval.txt split ===   
class PetSegDataset(Dataset):
    def __init__(self, image_names, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_names = image_names
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.resize = transforms.Resize((224, 224), interpolation=Image.NEAREST)  # <- Resize masks

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        image = Image.open(os.path.join(self.image_dir, name + ".jpg")).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, name + ".png"))

        image = self.transform(image)

        # === Resize & Process mask ===
        mask = self.resize(mask)  # Make sure size is (224, 224)
        mask_np = np.array(mask)
        mask_np = (mask_np == 1).astype(np.uint8)  # foreground=1, rest=0
        mask_tensor = torch.from_numpy(mask_np).long()

        return image, mask_tensor


def compute_mIoU_sup(preds, targets):
    preds = torch.argmax(preds, dim=1)
    preds, targets = preds.byte(), targets.byte()
    intersection = (preds & targets).float().sum((1, 2))
    union = (preds | targets).float().sum((1, 2))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()

# ============================
# Load Official Trainval Names
# ============================
def load_official_trainval_names(txt_path):
    with open(txt_path, "r") as f:
        lines = f.readlines()
    return sorted([line.strip().split()[0] for line in lines if line.strip()])


def trimap_train():
    set_seed()

    # === Paths ===
    image_dir = "oxford-iiit-pet/images"
    mask_dir = "oxford-iiit-pet/annotations/trimaps"
    split_file = "oxford-iiit-pet/annotations/trainval.txt"
    save_path = "best_deeplab_trimap.pth"

    if os.path.exists(save_path):
        print("✅ Model already trained. Skipping training.")
        return

    all_names = load_official_trainval_names(split_file)

    
    # === Split into train and validation sets (80% train, 20% validation) ===
    total = len(all_names)
    val_count = int(0.2 * total)
    train_names = all_names[:total - val_count]
    val_names = all_names[total - val_count:]

    train_set = PetSegDataset(train_names, image_dir, mask_dir)
    val_set = PetSegDataset(val_names, image_dir, mask_dir)
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8)

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model = deeplabv3_resnet50(num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_miou = 0.0

    # Training loop
    print("Starting training...")
    for epoch in range(1, 6):
        model.train()
        total_loss = 0
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)["out"]
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if i % 10 == 0:
                print(f"  Batch {i}/{len(train_loader)} - Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            ious = []
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)["out"]
                ious.append(compute_mIoU_sup(outputs, masks))
            mean_iou = np.mean(ious)

        print(f"Epoch {epoch} - Train Loss: {total_loss/len(train_loader):.4f}")
        print(f"Epoch {epoch} - Validation mIoU: {mean_iou:.4f}")

        if mean_iou > best_miou:
            best_miou = mean_iou
            torch.save(model.state_dict(), save_path)
            print("✅ Best model saved.")
        print(f"Best mIoU so far: {best_miou:.4f}")

if __name__ == "__main__":
    trimap_train()
