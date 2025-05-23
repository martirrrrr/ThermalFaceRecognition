import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from face_dataset import FaceDetectionDataset
from collections import Counter
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import json
import time
import matplotlib.pyplot as plt

# === RESNET CUSTOM ===
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
        return F.relu(out)

class CustomResNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        # Layer 4 not implemented (compact resnet)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        # Building of residual blocks (1-3) 
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
        out = torch.flatten(out, 1)
        return self.fc(out)

# === TRAINING ===
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct = 0, 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(dataloader.dataset), total_correct / len(dataloader.dataset) * 100

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
    return total_loss / len(dataloader.dataset), total_correct / len(dataloader.dataset) * 100

# === MAIN ===
if __name__ == '__main__':
    # Transforms + Normalization + Data Augmentation
    transform_train = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    transform_test = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # LabelEncoder globale
    all_labels = []
    for split in ['train', 'val', 'test']:
        with open(f'dataset_merged_split/{split}/{split}_bb.txt') as f:
            all_labels += [line.strip().split()[0].split("/")[-2] for line in f]
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)

    # Datasets + Loaders
    train_ds = FaceDetectionDataset('dataset_merged_split/train.jsonl', root_dir='dataset_merged_split/train', transform=transform_train, label_encoder=label_encoder)
    val_ds = FaceDetectionDataset('dataset_merged_split/val.jsonl', root_dir='dataset_merged_split/val', transform=transform_test, label_encoder=label_encoder)
    test_ds = FaceDetectionDataset('dataset_merged_split/test.jsonl', root_dir='dataset_merged_split/test', transform=transform_test, label_encoder=label_encoder)

    print("Classes:", list(label_encoder.classes_))
    print("Train distribution:", Counter([lbl for _, lbl in train_ds]))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomResNet(num_classes=len(label_encoder.classes_)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # Best model values
    best_val_acc = 0
    train_accuracies, val_accuracies = [], []
    train_losses =  []
    val_losses = []

    epoch_times = []
    train_times = []
    val_times = []

    # Training epochs=20
    for epoch in range(1, 21):
        start_epoch = time.time()

        start_train = time.time()
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        end_train = time.time()

        start_val = time.time()
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        end_val = time.time()

        end_epoch = time.time()

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        epoch_time = end_epoch - start_epoch
        train_time = end_train - start_train
        val_time = end_val - start_val

        epoch_times.append(epoch_time)
        train_times.append(train_time)
        val_times.append(val_time)

        print(f"Epoch {epoch}: Train Acc {train_acc:.2f}%, Val Acc {val_acc:.2f}%")
        print(f"Training time: {train_time:.2f}s | Validation time: {val_time:.2f}s | Epoch time: {epoch_time:.2f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model_2.pth')
            print("Best model saved.")

    start_test = time.time()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    end_test = time.time()
    test_time = end_test - start_test
    print(f"\n Test Accuracy: {test_acc:.2f}%")
    print(f" Test evaluation time: {test_time:.2f}s")

    # Save metrics
    metrics = {
        "train_acc": train_accuracies,
        "val_acc": val_accuracies,
        "train_loss": train_losses,
        "val_loss": val_losses
    }
    with open("training_metrics.json", "w") as f:
        json.dump(metrics, f)

    timing_metrics = {
        "epoch_times": epoch_times,
        "train_times": train_times,
        "val_times": val_times,
        "test_time": test_time
    }
    with open("timing_metrics.json", "w") as f:
        json.dump(timing_metrics, f)

    # Plotting
    epochs = list(range(1, len(train_accuracies) + 1))

    # Accuracy plot
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, val_accuracies, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy per Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("accuracy_plot.png")
    plt.show()

    # Loss plot
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_plot.png")
    plt.show()

