import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

# Define data transformations (resize, convert to tensor, and normalize)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 (typical input size for pretrained models)
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Pre-trained values for normalization
])

# TODO: change these to test/train folders when done
train_dataset = datasets.ImageFolder("images2", transform=transform)
test_dataset = datasets.ImageFolder("images2", transform=transform)

# Load datasets into data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = models.alexnet(pretrained=True)

# Modify the final fully connected layer (classifier) to match the number of classes
num_ftrs = model.classifier[6].in_features  # Get the number of input features to the last fully connected layer
model.classifier[6] = nn.Linear(num_ftrs, len(train_dataset.classes))  # Update to match number of classes
model.to(device)

# Training loop
def train_model(model, num_epochs=1, lr=1e-3):
    print("*** Training ***")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward() 
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Function to evaluate the model
def evaluate_model(model, test_loader):
    print("*** Evaluating ***")
    model.eval()
    running_corrects = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            running_corrects += (predicted == labels).sum().item()

    accuracy = running_corrects / total
    print(f"Accuracy: {accuracy:.4f}")

# Train the model
train_model(model, num_epochs=10)

# Evaluate the model
evaluate_model(model, test_loader)

# Save the model
torch.save(model.state_dict(), "model_weights.pth")
