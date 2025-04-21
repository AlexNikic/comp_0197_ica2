import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

# ============================
# Model: ResNetMultiTask
# ============================
class ResNetMultiTask(nn.Module):
    def __init__(self, num_breeds):
        super(ResNetMultiTask, self).__init__()
        self.backbone = resnet18(weights=None)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.fc_species = nn.Linear(num_features, 2)
        self.fc_breed = nn.Linear(num_features, num_breeds)

    def forward(self, x):
        features = self.backbone(x)
        out_species = self.fc_species(features)
        out_breed = self.fc_breed(features)
        return out_species, out_breed

# ============================
# Dataset
# ============================
class OxfordPetCAMDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label, os.path.basename(img_path)

# ============================
# CAM generation (tensor)
# ============================
def generate_cam_tensor(model, input_tensor, class_idx, device):
    model.eval()
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        conv_features = nn.Sequential(*list(model.backbone.children())[:-2])(input_tensor)
        weights = model.fc_breed.weight[class_idx].view(-1, 1, 1)
        cam = (conv_features.squeeze(0) * weights).sum(0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam

# ============================
# Equivariant Regularization Loss
# ============================
def equivariant_loss(model, image, class_idx, device):
    transform = transforms.RandomRotation(degrees=30)
    cam_orig = generate_cam_tensor(model, image, class_idx, device)

    image_trans = transform(image.squeeze(0)).unsqueeze(0)
    cam_trans = generate_cam_tensor(model, image_trans, class_idx, device)

    cam_orig_trans = TF.rotate(cam_orig.unsqueeze(0), angle=30)
    return F.mse_loss(cam_trans, cam_orig_trans.squeeze(0))

# ============================
# Training with ER
# ============================
def train_with_er(model, dataloader, device, epochs=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, labels, _ in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            _, out_breed = model(images)
            loss_cls = criterion(out_breed, labels)

            # Compute ER loss for each sample (simplified: only one sample)
            er_loss = equivariant_loss(model, images[0].unsqueeze(0), labels[0].item(), device)
            loss = loss_cls + 0.2 * er_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Loss: {total_loss/len(dataloader):.4f}")

# ============================
# CAM → Mask Saving
# ============================
#def save_mask_from_cam(cam_tensor, save_path, threshold=0.3):
def save_mask_from_cam(cam_tensor, save_path, threshold=0.5):
    cam = (cam_tensor >= threshold).float() * 255
    mask = Image.fromarray(cam.byte().cpu().numpy())
    mask.save(save_path)

# ============================
# Load list.txt and Start Training
# ============================
def parse_list_txt(list_path, image_dir):
    pairs = []
    with open(list_path, 'r') as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            image_name = parts[0] + ".jpg"
            label = int(parts[1]) - 1
            full_path = os.path.join(image_dir, image_name)
            pairs.append((full_path, label))
    return pairs

def main():
    print("🚀 SEAM_with_ER started.")
    image_dir = "oxford-iiit-pet/images"
    list_path = "oxford-iiit-pet/annotations/list.txt"
    mask_dir = "pseudo_mask_er"
    os.makedirs(mask_dir, exist_ok=True)

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model = ResNetMultiTask(num_breeds=37)

    print("📂 Loading dataset...")
    data_list = parse_list_txt(list_path, image_dir)
    dataset = OxfordPetCAMDataset(data_list)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Step 1: Train with ER
    print("🧠 Start training with ER...")
    train_with_er(model, dataloader, device)

    # Step 2: Generate refined CAM masks
    print("🧪 Generating CAM pseudo-masks...")
    model.eval()
    for image, label, name in dataset:
        input_tensor = image.unsqueeze(0).to(device)
        cam_tensor = generate_cam_tensor(model, input_tensor, label, device)
        save_path = os.path.join(mask_dir, name.replace(".jpg", ".png"))
        save_mask_from_cam(cam_tensor, save_path)

    print("✅ All pseudo-masks (with ER) saved in:", mask_dir)

if __name__ == "__main__":
    main()
