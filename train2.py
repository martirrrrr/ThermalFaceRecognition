import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections import Counter
import torchvision
import numpy as np

import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CustomResNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet, self).__init__()
        # Input
        self.in_channels = 64

        # Conv1 per estrazione pattern basilari
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # MaxPooling 3x3 per ridurre dim
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layer 1 di due blocchi residuali (2 conv. 3x3 + BatchNorm + Shortcut)
        self.layer1 = self._make_layer(64, 2, stride=1)
        # Layer 2 di due blocchi residuali
        self.layer2 = self._make_layer(128, 2, stride=2)
        # Layer 3 di due blocchi residuali
        self.layer3 = self._make_layer(256, 2, stride=2)
        # (estrae features sempre più astratte)
        # Adaptive avg pooling ()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Fully connected layer Nx142 numero di classi)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        # Lista di stride per blocchi
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.pool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        # Vettorializzazione
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

# === TRAINING & VALUTAZIONE ===
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct = 0, 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images) # fwd
        loss = criterion(outputs, labels) #loss
        loss.backward() #calcolo grad
        optimizer.step() #upd

        total_loss += loss.item() * images.size(0) #weighted loss
        total_correct += (outputs.argmax(1) == labels).sum().item()

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = total_correct / len(dataloader.dataset) * 100
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_correct = 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            total_correct += (outputs.argmax(1) == labels).sum().item()

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = total_correct / len(dataloader.dataset) * 100
    return avg_loss, accuracy

# === MAIN ===
if __name__ == '__main__':
    # Trasformazioni TRAINING
    transform_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), #BW (già in grayscale per GRAU, ma se passi Iron serve)
        transforms.Resize((112, 112)), # resizing
        transforms.RandomHorizontalFlip(), #data augm con flipping orizz
        transforms.RandomRotation(5), # data augm con rotazioni fino a +-5 degrees
        transforms.ToTensor(),  # tensorializzazione senza Normalize (transforms.Normalize(mean=[0.5], std=[0.5])  # se gray
    ])
    # Trasformazioni TEST E VAL
    transform_test = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), # BW
        transforms.Resize((112, 112)), # resizing
        transforms.ToTensor(), #tensorializzazione
        # NO data augmentation per gli altri set
    ])

    # Dataset split
    train_ds = datasets.ImageFolder('dataset_merged_split/train', transform=transform_train)
    val_ds = datasets.ImageFolder('dataset_merged_split/val', transform=transform_test)
    test_ds = datasets.ImageFolder('dataset_merged_split/test', transform=transform_test)

    # Distributions
    print("Classes:", train_ds.classes)
    print("Training distribution:", Counter(train_ds.targets))

    # LOADERS
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # Devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomResNet(num_classes=len(train_ds.classes)).to(device)

    # Loss + Optim
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    #show_some_images(train_ds)

    best_val_acc = 0
    # 20 epochs
    for epoch in range(1, 21):
        # Train
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        # Val
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}: Train Acc {train_acc:.2f}%, Val Acc {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Best model
            torch.save(model.state_dict(), 'best_thermal_model2.pth')
            print("Best model saved.")

    # Test
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\n🎯 Test Accuracy: {test_acc:.2f}%")

    #show_confusion(model, val_loader, device, train_ds.classes)