# ===== SEAM Project Pipeline using PyTorch =====
# Full code from data loading, CAM generation (SEAM),
# pseudo-label creation, DeepLabv3+ training and final evaluation

import os
import subprocess
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Try to import hyperopt, install if not available
try:
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
except ImportError:
    print("hyperopt is not installed. Installing...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'hyperopt'])
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration for loading a fraction of the dataset
data_fraction = 0.001  # Adjust this value to change the fraction of data to load (0.0 to 1.0)

# === 1. Dataset Loader (Oxford Pet recursive by breed) ===
class PetDataset(Dataset):
    def __init__(self, img_dir, label_dict, transform=None, fraction=1.0):
        self.img_paths = []
        self.labels = []
        self.transform = transform
        
        # Load all images and labels first
        for label_name, label_idx in label_dict.items():
            class_folder = os.path.join(img_dir, label_name)
            for root, _, files in os.walk(class_folder):
                for file in files:
                    if file.endswith('.jpg'):
                        self.img_paths.append(os.path.join(root, file))
                        self.labels.append(label_idx)

        # Calculate the number of samples to load based on the fraction
        num_samples = int(len(self.img_paths) * fraction)

        # Select a subset of the dataset
        self.img_paths = self.img_paths[:num_samples]
        self.labels = self.labels[:num_samples]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, img_path

# === 2. CAM extraction from ResNet ===
def get_cam(feature_conv, weight_softmax, class_idx):
    bz, nc, h, w = feature_conv.shape
    cam = weight_softmax[class_idx].unsqueeze(1).unsqueeze(2) * feature_conv
    cam = cam.sum(dim=1).squeeze().detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    return cam

# === 3. Train classification model with SEAM (equivariance constraint) ===
def train_classifier_seam(model, dataloader, optimizer, criterion, num_epochs=3):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for images, labels, _ in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)
            flipped_images = torch.flip(images, dims=[3])

            outputs_orig = model(images)
            outputs_flip = model(flipped_images)

            loss_cls = criterion(outputs_orig, labels)
            # SEAM equivariance loss
            features_orig = model.layer4(model.layer3(model.layer2(model.layer1(model.conv1(images)))))
            features_flip = model.layer4(model.layer3(model.layer2(model.layer1(model.conv1(flipped_images)))))
            loss_eqv = torch.mean(torch.abs(torch.flip(features_orig, dims=[3]) - features_flip))

            loss = loss_cls + 0.1 * loss_eqv
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}")

# === 4. Pseudo mask generation ===
def generate_pseudo_masks(model, dataloader, class_names, output_dir):
    model.eval()
    finalconv_name = 'layer4'
    features_blobs = []
    def hook_feature(module, input, output):
        features_blobs.append(output)
    model._modules.get(finalconv_name).register_forward_hook(hook_feature)

    params = list(model.parameters())
    weight_softmax = params[-2].data

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("outputs/cams/", exist_ok=True)
    os.makedirs("outputs/binarized_masks/", exist_ok=True)

    with torch.no_grad():
        for images, labels, paths in tqdm(dataloader):
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(dim=1)
            CAMs = get_cam(features_blobs[-1], weight_softmax, preds[0])

            # Save CAM heatmap
            cam_img = (CAMs * 255).astype(np.uint8)
            cam_img = np.clip(cam_img, 0, 255)  # Ensure values are within [0, 255]
            cam_img_color = plt.get_cmap("jet")(cam_img / 255.0)[:, :, :3] * 255
            
            # Convert CAM image color to uint8
            cam_img_color = cam_img_color.astype(np.uint8)  # Convert to uint8 for PIL
            cam_img_color = cam_img_color.squeeze()  # Remove single-dimensional entries

            # Ensure it's in the correct shape
            if cam_img_color.ndim == 3 and cam_img_color.shape[2] == 3:  # Ensure we have 3 channels
                cam_img_pil = Image.fromarray(cam_img_color)  # Generate PIL image

                cam_out_path = os.path.join("outputs/cams/", os.path.basename(paths[0]).replace('.jpg', '_cam.jpg'))
                cam_img_pil.save(cam_out_path)

            # Save binarized pseudo mask
            mask = (CAMs > 0.3).astype(np.uint8)
            bin_out_path = os.path.join("outputs/binarized_masks/", os.path.basename(paths[0]).replace('.jpg', '.png'))
            Image.fromarray(mask * 255).save(bin_out_path)

            # Save main pseudo label
            out_path = os.path.join(output_dir, os.path.basename(paths[0]).replace('.jpg', '.png'))
            Image.fromarray(mask * 255).save(out_path)
            features_blobs.clear()

# === 5. DeepLabv3+ Training ===
from torchvision.models.segmentation import deeplabv3_resnet50
class PseudoSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_paths = []
        for root, _, files in os.walk(img_dir):
            for fname in files:
                if fname.endswith(".jpg"):
                    self.img_paths.append(os.path.join(root, fname))
        self.mask_dir = mask_dir
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        mask_path = os.path.join(self.mask_dir, os.path.basename(self.img_paths[idx]).replace(".jpg", ".png"))
        mask = Image.open(mask_path).convert("L")
        if self.transform:
            img = self.transform(img)
        mask = transforms.ToTensor()(mask).long().squeeze()
        return img, mask

    def __len__(self):
        return len(self.img_paths)

def train_deeplab(model, dataloader, epochs=3):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(epochs):
        total_loss = 0
        for imgs, masks in tqdm(dataloader):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)['out']
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}: Seg Loss = {total_loss:.4f}")

# === 6. Evaluation (mIoU and accuracy) ===
def evaluate(model, dataloader):
    model.eval()
    iou_total, pixel_correct, pixel_total = 0, 0, 0
    with torch.no_grad():
        for imgs, masks in tqdm(dataloader):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)['out'].argmax(1)
            intersection = ((preds == 1) & (masks == 1)).sum().item()
            union = ((preds == 1) | (masks == 1)).sum().item()
            correct = (preds == masks).sum().item()
            total = masks.numel()
            iou_total += intersection / (union + 1e-6)
            pixel_correct += correct
            pixel_total += total
    print(f"Mean IoU: {iou_total / len(dataloader):.4f}")
    print(f"Pixel Accuracy: {pixel_correct / pixel_total:.4f}")

# === 7. Hyperparameter Tuning with Hyperopt ===
def objective(params):
    num_layers = params['num_layers']  # Number of layers for ResNet
    freeze_layers = params['freeze_layers']  # Number of layers to freeze

    # Select ResNet based on num_layers
    if num_layers == 0:
        model = models.resnet18(weights='IMAGENET1K_V1')
    elif num_layers == 1:
        model = models.resnet34(weights='IMAGENET1K_V1')
    else:
        model = models.resnet50(weights='IMAGENET1K_V1')

    model.fc = nn.Linear(model.fc.in_features, 2)

    # Freezing layers
    for name, param in model.named_parameters():
        if "layer" in name:
            layer_num = int(name.split('.')[1])
            if layer_num < freeze_layers:
                param.requires_grad = False

    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Load dataset
    dataset = PetDataset(".", label_dict={"cats": 0, "dogs": 1}, transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]), fraction=data_fraction)  # Pass the fraction

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Training the model (Modify epochs as necessary)
    for epoch in range(3):  # Keep it small for tuning
        train_classifier_seam(model, dataloader, optimizer, criterion, num_epochs=1)
    
    # Evaluate the model on some validation set (mock this part for hyperopt)
    val_loss = 0  # Ideally, compute the actual validation loss here

    return {'loss': val_loss, 'status': STATUS_OK}

# Define the search space
space = {
    'num_layers': hp.choice('num_layers', [0, 1, 2]),  # 0: ResNet-18, 1: ResNet-34, 2: ResNet-50
    'freeze_layers': hp.randint('freeze_layers', 5),  # Randomly freeze 0-4 layers
}

# Run Hyperopt to find the best hyperparameters
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=10,  # Number of evaluations
            trials=Trials())

print("Best hyperparameters:", best)

# === 8. Run All ===
if __name__ == '__main__':
    # Adjust label_dict to match the directory structure
    label_dict = {"cats": 0, "dogs": 1}  # Assuming folders for cats and dogs
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = PetDataset(".", label_dict, transform, fraction=data_fraction)  # Pass the fraction
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = models.resnet50(weights='IMAGENET1K_V1')  # Updated to use weights argument
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    print("Training classifier with SEAM...")
    train_classifier_seam(model, dataloader, optimizer, criterion)

    print("Generating CAM and pseudo masks...")
    generate_pseudo_masks(model, dataloader, ["cat", "dog"], output_dir="outputs/masks/")

    print("Training DeepLabv3+ on pseudo masks...")
    seg_dataset = PseudoSegDataset(".", "outputs/masks/", transform)  # Same current directory
    seg_loader = DataLoader(seg_dataset, batch_size=4)
    deeplab = deeplabv3_resnet50(num_classes=2, pretrained=False).to(device)
    train_deeplab(deeplab, seg_loader)

    print("Evaluating model...")
    evaluate(deeplab, seg_loader)