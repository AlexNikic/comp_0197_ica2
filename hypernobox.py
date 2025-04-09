import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# Define the fraction of data to load (between 0 and 1)
load_fraction = 0.001  # Load 10% of the dataset

#############################################
# Model: Multi-task ResNet for Species + Breed Classification
#############################################
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

#############################################
# Load list.txt with labels
#############################################
def parse_list_txt(list_path, image_dir):
    image_label_pairs = []
    with open(list_path, 'r') as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            image_name = parts[0] + ".jpg"
            label = int(parts[1]) - 1  # label range: [0, 36]
            full_path = os.path.join(image_dir, image_name)
            image_label_pairs.append((full_path, label))
    return image_label_pairs

#############################################
# Dataset for list.txt images
#############################################
class OxfordPetCAMDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label, os.path.basename(img_path)

#############################################
# Generate CAM (Class Activation Map)
#############################################
def generate_cam_pil(model, input_tensor, class_idx, device):
    model.eval()
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        layers = list(model.backbone.children())[:-2]
        conv_net = nn.Sequential(*layers).to(device)
        features = conv_net(input_tensor)
        weights = model.fc_breed.weight[class_idx].view(-1, 1, 1)
        cam = (features.squeeze(0) * weights).sum(0)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = (cam * 255).clamp(0, 255).byte()
        cam_img = transforms.ToPILImage()(cam.unsqueeze(0))
        cam_img = cam_img.resize((224, 224), Image.BILINEAR)
    return cam_img

#############################################
# Convert CAM to Binary Mask
#############################################
def save_mask_from_cam(cam_pil, save_path, threshold=0.3):
    cam_gray = cam_pil.convert("L")
    cam_tensor = transforms.ToTensor()(cam_gray).squeeze(0)
    binary_mask = (cam_tensor >= threshold).float()
    binary_mask = (binary_mask * 255).byte()
    mask_pil = transforms.ToPILImage()(binary_mask.unsqueeze(0))
    mask_pil.save(save_path)

#############################################
# Blend CAM Overlay for Visualization
#############################################
def overlay_cam_on_image(image_tensor, cam_pil):
    image_clamped = image_tensor.squeeze(0).clamp(0, 1)
    image_pil = transforms.ToPILImage()(image_clamped)
    cam_red = Image.merge("RGB", (cam_pil, Image.new("L", cam_pil.size), Image.new("L", cam_pil.size)))
    return Image.blend(image_pil, cam_red, alpha=0.5)

#############################################
# Find Unlabeled Images
#############################################
def find_unlabeled_images(image_dir, mask_dir):
    all_images = set(f.replace(".jpg", "") for f in os.listdir(image_dir) if f.endswith(".jpg"))
    all_masks = set(f.replace(".png", "") for f in os.listdir(mask_dir) if f.endswith(".png"))
    return sorted(list(all_images - all_masks))

#############################################
# Process Unlabeled Images (Not in list.txt)
#############################################
def process_unlabeled_images(model, image_dir, mask_dir, device, threshold=0.3):
    missing_images = find_unlabeled_images(image_dir, mask_dir)
    print(f"\n🟡 Found {len(missing_images)} unlabeled images to process...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    for name in missing_images:
        img_path = os.path.join(image_dir, name + ".jpg")
        if not os.path.exists(img_path):
            continue
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            _, breed_logits = model(input_tensor)
            pseudo_label = torch.argmax(breed_logits, dim=1).item()
        cam_pil = generate_cam_pil(model, input_tensor, pseudo_label, device)
        save_mask_from_cam(cam_pil, os.path.join(mask_dir, name + ".png"), threshold=threshold)
        print(f"✅ Generated pseudo-mask for: {name}")

#############################################
# Process Remaining Missing Masks (Fallback)
#############################################
def process_remaining_missing_masks(model, image_dir, mask_dir, device, threshold=0.3):
    all_images = set(f.replace(".jpg", "") for f in os.listdir(image_dir) if f.endswith(".jpg"))
    all_masks = set(f.replace(".png", "") for f in os.listdir(mask_dir) if f.endswith(".png"))
    missing = sorted(list(all_images - all_masks))
    print(f"\n🔍 Total images: {len(all_images)}")
    print(f"✅ Masks generated: {len(all_masks)}")
    print(f"❌ Missing masks: {len(missing)}")
    print(f"Missing image names: {missing}")
    if not missing:
        print("🎉 All images have corresponding pseudo-masks!")
        return
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    for name in missing:
        img_path = os.path.join(image_dir, name + ".jpg")
        if not os.path.exists(img_path):
            print(f"⚠️ Image not found: {img_path}")
            continue
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            _, breed_logits = model(input_tensor)
            pseudo_label = torch.argmax(breed_logits, dim=1).item()
        cam_pil = generate_cam_pil(model, input_tensor, pseudo_label, device)
        save_mask_from_cam(cam_pil, os.path.join(mask_dir, name + ".png"), threshold=threshold)
        print(f"✅ Recovered mask for: {name}")

#############################################
# Hyperparameter Tuning with Hyperopt
#############################################
def train_model(model, dataloader, device):
    # Implement your training logic here
    # This is just a placeholder
    return np.random.rand()  # Placeholder for training loss

def validate_model(model, dataloader, device):
    # Implement your validation logic here
    # This is just a placeholder
    return np.random.rand()  # Placeholder for validation loss

def objective(params):
    num_layers = params['num_layers'] + 2  # Increase by 2 to allow for the identity and final fully-connected layers reference
    freeze_layers = params['freeze_layers']

    # Create and configure the model
    model = ResNetMultiTask(num_breeds=37)
    model.backbone = resnet18(weights=None)

    # Modify number of layers
    layers = list(model.backbone.children())[:num_layers]
    model.backbone = nn.Sequential(*layers)

    # Freeze specified layers
    for layer in model.backbone[:freeze_layers]:
        for param in layer.parameters():
            param.requires_grad = False

    model.to(device)

    # Train the model
    train_loss = train_model(model, dataloader, device)
    val_loss = validate_model(model, dataloader, device)

    best_val_loss = val_loss

    return {'loss': best_val_loss, 'status': STATUS_OK}

# Define the hyperparameter space
space = {
    'num_layers': hp.randint('num_layers', 1, 16),  # Choose number of layers (1 to 16)
    'freeze_layers': hp.randint('freeze_layers', 0, 16),  # Choose number of layers to freeze (0 to 15)
}

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("cam_vis", exist_ok=True)
    os.makedirs("pseudo_mask", exist_ok=True)

    # Load labeled data from list.txt
    data_list = parse_list_txt("oxford-iiit-pet/annotations/list.txt", "oxford-iiit-pet/images")
    
    # Load only a fraction of the data
    load_size = int(len(data_list) * load_fraction)
    data_list = data_list[:load_size]
    
    dataset = OxfordPetCAMDataset(data_list)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Start hyperparameter tuning
    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)

    print(f"Best Hyperparameters: {best}")

    # Generate CAMs and masks for labeled images using the best found configuration
    model = ResNetMultiTask(num_breeds=37)
    model.to(device)

    for image, label, filename in DataLoader(dataset, batch_size=1):
        cam_pil = generate_cam_pil(model, image, label.item(), device)
        overlay_path = f"cam_vis/{filename[0]}_cam_overlay.jpg"
        overlay = overlay_cam_on_image(image, cam_pil)
        overlay.save(overlay_path)
        mask_path = f"pseudo_mask/{filename[0].replace('.jpg', '.png')}"
        save_mask_from_cam(cam_pil, mask_path, threshold=0.3)

    # Handle additional unlabeled images
    process_unlabeled_images(model, "oxford-iiit-pet/images", "pseudo_mask", device, threshold=0.3)
    process_remaining_missing_masks(model, "oxford-iiit-pet/images", "pseudo_mask", device, threshold=0.3)