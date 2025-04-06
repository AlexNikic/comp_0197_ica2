import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

class PetSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.transform = transform
        self.resize = transforms.Resize((224, 224), interpolation=Image.NEAREST)  # <- Resize masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace(".jpg", ".png"))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)

        # === Resize & Process mask ===
        mask = self.resize(mask)  # Make sure size is (224, 224)
        mask_np = np.array(mask)
        mask_np = (mask_np == 1).astype(np.uint8)  # foreground=1, rest=0
        mask_tensor = torch.from_numpy(mask_np).long()

        return image, mask_tensor


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

def compute_mIoU(preds, targets):
    preds = torch.argmax(preds, dim=1)
    preds, targets = preds.byte(), targets.byte()
    intersection = (preds & targets).float().sum((1, 2))
    union = (preds | targets).float().sum((1, 2))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = get_transform()
    dataset = PetSegDataset(
        image_dir="oxford-iiit-pet/images",
        mask_dir="oxford-iiit-pet/annotations/trimaps",
        transform=transform
    )

    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8)

    model = deeplabv3_resnet50(num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_miou = 0.0
    for epoch in range(10):
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
                ious.append(compute_mIoU(outputs, masks))
            mean_iou = np.mean(ious)

        print(f"Epoch {epoch+1} - Train Loss: {total_loss/len(train_loader):.4f}")
        print(f"Epoch {epoch+1} - Validation mIoU: {mean_iou:.4f}")

        if mean_iou > best_miou:
            best_miou = mean_iou
            torch.save(model.state_dict(), "best_deeplab_trimap.pth")
            print("✅ Best model saved.")

if __name__ == "__main__":
    train()
