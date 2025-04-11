import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF

class PetSegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)), 
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        image = self.transform(image)
        mask = TF.resize(mask, [128, 128], interpolation=Image.NEAREST)
        mask = TF.pil_to_tensor(mask).squeeze(0).long() // 255  # Binary: 0 or 1
        return image, mask

def compute_mIoU(pred, target):
    pred = (pred > 0.5).long()
    intersection = (pred & target).float().sum()
    union = (pred | target).float().sum()
    return (intersection + 1e-6) / (union + 1e-6)

def equivariant_loss(pred1, pred2, target, weight=1.0):
    criterion = nn.CrossEntropyLoss()
    
    # Loss for the original and augmented predictions
    loss1 = criterion(pred1, target)
    loss2 = criterion(pred2, target)

    # Equivariant Regularisation (comparing the predictions from similar augmentations)
    equivariant_reg_loss = torch.mean((pred1 - pred2) ** 2)

    # Combine the relevant components
    return loss1 + loss2 + weight * equivariant_reg_loss

def train_model(train_fraction=0.01, val_fraction=0.01):  
    image_dir = "images"
    mask_dir = "NoBoxScripts/pseudo_mask"
    save_dir = "deeplab_output"
    os.makedirs(save_dir, exist_ok=True)

    all_names = sorted([f.replace(".png", "") for f in os.listdir(mask_dir) if f.endswith(".png")])
    total_count = len(all_names)

    # Determine the split for training and validation
    train_count = int(train_fraction * total_count)
    val_count = int(val_fraction * total_count)

    if train_count + val_count > total_count:
        raise ValueError("The sum of training and validation fractions exceeds the total data count.")

    # Get training and validation names
    train_names = all_names[:train_count]
    val_names = all_names[train_count:train_count + val_count]

    print(f"Training on {len(train_names)} samples.")
    print(f"Validating on {len(val_names)} samples.")

    def get_paths(names):
        imgs = [os.path.join(image_dir, name + ".jpg") for name in names]
        masks = [os.path.join(mask_dir, name + ".png") for name in names]
        return imgs, masks

    train_imgs, train_masks = get_paths(train_names)
    val_imgs, val_masks = get_paths(val_names)

    train_loader = DataLoader(PetSegmentationDataset(train_imgs, train_masks), 
                              batch_size=8, 
                              shuffle=True, 
                              num_workers=4)

    val_loader = DataLoader(PetSegmentationDataset(val_imgs, val_masks), 
                            batch_size=1, 
                            shuffle=False, 
                            num_workers=4)

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = deeplabv3_resnet50(num_classes=2)
    model.to(device)
    optimiser = optim.Adam(model.parameters(), lr=1e-4)

    best_mIoU = 0.0

    for epoch in range(1, 11):  
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            # Augment images (horizontal flipping)
            if torch.rand(1).item() > 0.5:
                augmented_images = torch.flip(images, [3])  # Flipping horizontally
            else:
                augmented_images = images
            
            optimiser.zero_grad()

            # Predictions from both original and augmented images
            outputs = model(images)["out"]
            augmented_outputs = model(augmented_images)["out"]

            # Compute the loss with equivariant regularization
            loss = equivariant_loss(outputs, augmented_outputs, masks)

            loss.backward()
            optimiser.step()
            running_loss += loss.item()

        print(f"Epoch {epoch} - Train Loss: {running_loss / len(train_loader):.4f}")

        # ====== Validation ======
        model.eval()
        total_iou = 0.0
        total_accuracy = 0.0
        with torch.no_grad():
            for i, (image, mask) in enumerate(val_loader):
                image, mask = image.to(device), mask.to(device)
                output = model(image)["out"]
                pred = torch.argmax(output, dim=1)
                iou = compute_mIoU(pred, mask)
                total_iou += iou.item()

                # Calculate pixel accuracy
                correct_predictions = (pred == mask).float().sum()
                pixel_accuracy = correct_predictions / (mask.numel())
                total_accuracy += pixel_accuracy.item()

                pred_mask = pred.squeeze(0).cpu().byte() * 255
                Image.fromarray(pred_mask.numpy()).save(os.path.join(save_dir, f"pred_{val_names[i]}.png"))

        mean_iou = total_iou / len(val_loader)
        mean_accuracy = total_accuracy / len(val_loader)  # Average over all validation batches
        print(f"Epoch {epoch} - Validation mIoU: {mean_iou:.4f}, Pixel Accuracy: {mean_accuracy:.4f}")

        # Save best model
        if mean_iou > best_mIoU:
            best_mIoU = mean_iou
            torch.save(model.state_dict(), os.path.join(save_dir, "deeplab_no_ecr.pth"))
            print("✅ Best model saved.")

if __name__ == "__main__":
    train_model(train_fraction=0.8, val_fraction=0.2)  # The fraction of the data used for training and validation 