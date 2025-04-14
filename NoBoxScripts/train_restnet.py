import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from sklearn.model_selection import StratifiedKFold
from data_split import load_entire_data

# -----------------------------
# 1. PyTorch Dataset (Multi-Task)
# -----------------------------
class PetDataset(Dataset):
    def __init__(self, images, species_labels, breed_labels):
        """
        images: NumPy array of shape (N, 224, 224, 3), float32 in [0,1]
        species_labels: NumPy array of shape (N,) with 0 for cats, 1 for dogs
        breed_labels: NumPy array of shape (N,) with integer labels for each breed
        """
        self.images = images
        self.species_labels = species_labels
        self.breed_labels = breed_labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]  # shape (224,224,3)
        # Convert (H,W,C) -> (C,H,W)
        img = np.transpose(img, (2, 0, 1))
        img_torch = torch.from_numpy(img).float()
        species_label = torch.tensor(self.species_labels[idx], dtype=torch.long)
        breed_label = torch.tensor(self.breed_labels[idx], dtype=torch.long)
        return img_torch, species_label, breed_label

# -----------------------------
# 2. Multi-Task ResNet Model
# -----------------------------
class ResNetMultiTask(nn.Module):
    def __init__(self, num_breeds):
        super(ResNetMultiTask, self).__init__()
        self.backbone = resnet18(weights=None)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # remove default FC
        self.fc_species = nn.Linear(num_features, 2)      # for species (cats vs. dogs)
        self.fc_breed = nn.Linear(num_features, num_breeds) # for breed classification

    def forward(self, x):
        features = self.backbone(x)
        out_species = self.fc_species(features)
        out_breed = self.fc_breed(features)
        return out_species, out_breed

# -----------------------------
# 3. Train/Evaluate Functions for Multi-Task
# -----------------------------
def train_one_epoch(model, dataloader, criterion_species, criterion_breed, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_species = 0
    correct_breed = 0
    total = 0
    for imgs, species_labels, breed_labels in dataloader:
        imgs = imgs.to(device)
        species_labels = species_labels.to(device)
        breed_labels = breed_labels.to(device)

        optimizer.zero_grad()
        out_species, out_breed = model(imgs)
        loss_species = criterion_species(out_species, species_labels)
        loss_breed = criterion_breed(out_breed, breed_labels)
        loss = loss_species + loss_breed
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, pred_species = torch.max(out_species, 1)
        _, pred_breed = torch.max(out_breed, 1)
        total += imgs.size(0)
        correct_species += (pred_species == species_labels).sum().item()
        correct_breed += (pred_breed == breed_labels).sum().item()
    epoch_loss = running_loss / total
    epoch_species_acc = 100.0 * correct_species / total
    epoch_breed_acc = 100.0 * correct_breed / total
    return epoch_loss, epoch_species_acc, epoch_breed_acc

def evaluate(model, dataloader, criterion_species, criterion_breed, device):
    model.eval()
    running_loss = 0.0
    correct_species = 0
    correct_breed = 0
    total = 0
    with torch.no_grad():
        for imgs, species_labels, breed_labels in dataloader:
            imgs = imgs.to(device)
            species_labels = species_labels.to(device)
            breed_labels = breed_labels.to(device)

            out_species, out_breed = model(imgs)
            loss_species = criterion_species(out_species, species_labels)
            loss_breed = criterion_breed(out_breed, breed_labels)
            loss = loss_species + loss_breed

            running_loss += loss.item() * imgs.size(0)
            _, pred_species = torch.max(out_species, 1)
            _, pred_breed = torch.max(out_breed, 1)
            total += imgs.size(0)
            correct_species += (pred_species == species_labels).sum().item()
            correct_breed += (pred_breed == breed_labels).sum().item()
    epoch_loss = running_loss / total
    epoch_species_acc = 100.0 * correct_species / total
    epoch_breed_acc = 100.0 * correct_breed / total
    return epoch_loss, epoch_species_acc, epoch_breed_acc

# -----------------------------
# 4. Cross-Validation + Hyperparameter Search + Final Model Training
# -----------------------------
def run_experiments(box_type="no_box"):
    # Use MPS if available (for Apple Silicon), else CUDA, else CPU.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using device:", device)
    
    # Load entire dataset (images, species labels, breed labels)
    images_np, species_labels_np, breed_labels_np = load_entire_data(box_type=box_type, shuffle_data=True)
    print(f"Dataset loaded: {images_np.shape[0]} images.")
    unique_breeds = np.unique(breed_labels_np)
    num_breeds = len(unique_breeds)
    print(f"Found {num_breeds} unique breeds.")
    
    # Hyperparameter grid (including momentum and weight decay)
    param_grid = {
        "lr": [1e-3, 1e-4, 1e-5],
        "batch": [16, 32, 64, 128],
        "epochs": [3, 5, 10],
        "momentum": [0.85, 0.9, 0.95],
        "weight_decay": [1e-4, 1e-3, 1e-2]
    }
    
    # Stratified 5-fold CV based on species_labels (to preserve cats vs. dogs ratio)
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []
    for (lr, batch_size, epochs, momentum, weight_decay) in itertools.product(
            param_grid["lr"], param_grid["batch"], param_grid["epochs"],
            param_grid["momentum"], param_grid["weight_decay"]):
        print("\n============================")
        print(f"Testing LR={lr}, BATCH={batch_size}, EPOCHS={epochs}, "
              f"MOMENTUM={momentum}, WEIGHT_DECAY={weight_decay}")
        fold_species_accs = []
        fold_breed_accs = []
        fold_index = 1
        for train_idx, val_idx in skf.split(images_np, species_labels_np):
            X_train = images_np[train_idx]
            y_species_train = species_labels_np[train_idx]
            y_breed_train = breed_labels_np[train_idx]
            X_val = images_np[val_idx]
            y_species_val = species_labels_np[val_idx]
            y_breed_val = breed_labels_np[val_idx]
            
            train_dataset = PetDataset(X_train, y_species_train, y_breed_train)
            val_dataset = PetDataset(X_val, y_species_val, y_breed_val)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            model = ResNetMultiTask(num_breeds=num_breeds)
            model.to(device)
            criterion_species = nn.CrossEntropyLoss()
            criterion_breed = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
            
            for epoch in range(epochs):
                train_loss, train_species_acc, train_breed_acc = train_one_epoch(
                    model, train_loader, criterion_species, criterion_breed, optimizer, device)
                val_loss, val_species_acc, val_breed_acc = evaluate(
                    model, val_loader, criterion_species, criterion_breed, device)
                print(f"Fold {fold_index}, Epoch {epoch+1}/{epochs}: "
                      f"Train Loss={train_loss:.4f}, Species Acc={train_species_acc:.2f}%, Breed Acc={train_breed_acc:.2f}% | "
                      f"Val Loss={val_loss:.4f}, Species Acc={val_species_acc:.2f}%, Breed Acc={val_breed_acc:.2f}%")
            _, final_species_acc, final_breed_acc = evaluate(model, val_loader, criterion_species, criterion_breed, device)
            fold_species_accs.append(final_species_acc)
            fold_breed_accs.append(final_breed_acc)
            fold_index += 1
        mean_species_acc = np.mean(fold_species_accs)
        mean_breed_acc = np.mean(fold_breed_accs)
        print(f"Mean Val Species Acc = {mean_species_acc:.2f}%, Mean Val Breed Acc = {mean_breed_acc:.2f}%")
        results.append((lr, batch_size, epochs, momentum, weight_decay, mean_species_acc, mean_breed_acc))
    
    best = max(results, key=lambda x: x[5])
    print("\n==== SUMMARY OF RUNS ====")
    for (lr, bsz, ep, mom, wd, sp_acc, br_acc) in results:
        print(f"LR={lr}, BATCH={bsz}, EPOCHS={ep}, MOMENTUM={mom}, WEIGHT_DECAY={wd} => "
              f"Species Acc={sp_acc:.2f}%, Breed Acc={br_acc:.2f}%")
    print(f"BEST => LR={best[0]}, BATCH={best[1]}, EPOCHS={best[2]}, "
          f"MOMENTUM={best[3]}, WEIGHT_DECAY={best[4]}, Species Acc={best[5]:.2f}%, Breed Acc={best[6]:.2f}%")
    
    print("\nRetraining final model on entire dataset with best hyperparameters...")
    final_dataset = PetDataset(images_np, species_labels_np, breed_labels_np)
    final_loader = DataLoader(final_dataset, batch_size=best[1], shuffle=True)
    
    final_model = ResNetMultiTask(num_breeds=num_breeds)
    final_model.to(device)
    criterion_species = nn.CrossEntropyLoss()
    criterion_breed = nn.CrossEntropyLoss()
    optimizer = optim.SGD(final_model.parameters(), lr=best[0], momentum=best[3], weight_decay=best[4])
    
    for epoch in range(best[2]):
        train_loss, train_species_acc, train_breed_acc = train_one_epoch(
            final_model, final_loader, criterion_species, criterion_breed, optimizer, device)
        print(f"Final Model Epoch {epoch+1}/{best[2]}: Loss={train_loss:.4f}, "
              f"Species Acc={train_species_acc:.2f}%, Breed Acc={train_breed_acc:.2f}%")
    
    torch.save(final_model.state_dict(), "best_resnet_multitask_model.pth")
    print("Saved final model as best_resnet_multitask_model.pth")

if __name__ == "__main__":
    run_experiments("no_box")
    # To use the variant with drawn boxes, call:
    # run_experiments("with_box")
