import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dual_loader import ThermalGrayTransform,DualInputDataset
from torchvision import transforms
from collections import Counter
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import json
import time
import matplotlib.pyplot as plt

"""
DUAL CUSTOM-RESNET FOR THERMAL FACE CLASSIFICATION
__________________________________________________________
Both thermal and visual inputs are processed independently with a 
lighter custom network based on ResNet for feature extraction.
We define two models trained on two different modalities dataset
exploiting the pairing of the images.
This model uses late fusion, meaning the two branches are merged 
only after feature extraction is complete.

STEPS:
*Face detection - Not implemented. Dataset provides bounding boxes coordinates
 for image pairs. The strategy simulates the face detection feeding to the network
 the cropped images of faces areas.
*Normalization + Feature extraction - Separately, using 3 residual blocks
 for each modality.
*Fusion - After the feature vectors from the thermal and RGB branches are concatenated,
  the resulting fused embedding captures complementary information from both modalities.
  This fused vector is passed through a fully connected layer which outputs class logits.
*Classification -Nnetwork is trained using a standard cross-entropy loss, comparing 
 predicted logits against the ground truth labels. 
 Prediction corresponds to the index of the highest logit via argmax.

"""


# === CUSTOM RESNET  ===
class CustomResNetEmbedding(nn.Module):
    def __init__(self, in_channels=1):
        super(CustomResNetEmbedding, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, out_channels, blocks, stride):
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
        return out

# === CLASSIFIER ===
class FusionClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FusionClassifier, self).__init__()
        # OUTPUT: CLASSIFICATION PREDICTION 0-141
        self.fc = nn.Linear(256 * 2, num_classes)

    def forward(self, feat_thermal, feat_rgb):
        # INPUT: THERMAL + RGB FEATURES
        x = torch.cat([feat_thermal, feat_rgb], dim=1)
        return self.fc(x)

# === BASIC BLOCK STRUCTURE ===
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

def check_pairs_integrity(dataset, label_encoder, n=5):
 # Sanity check for pairs extraction
    print(f"Checking {n} pairs in dataset...")
    for i in range(n):
        (x_thermal, x_rgb), label = dataset[i]
        sample = dataset.samples[i]
        thermal_path = sample['image']
        rgb_path = thermal_path.replace('dataset_merged_split', 'dataset_merged_split_rgb').replace('_1.png', '_3.png')

        label_str = label_encoder.inverse_transform([label])[0]

        print(f"Pair {i}:")
        print(f"  Thermal: {thermal_path}")
        print(f"  RGB:     {rgb_path}")
        print(f"  Label:   {label_str}")

    print("Pairs check done.")

def train(model_thermal, model_rgb, fusion_classifier, dataloader, optimizer, criterion, device):
    model_thermal.train()
    model_rgb.train()
    fusion_classifier.train()
    total_loss = 0
    total_correct = 0
    for (x_thermal, x_rgb), labels in dataloader:
        x_thermal, x_rgb, labels = x_thermal.to(device), x_rgb.to(device), labels.to(device)
        optimizer.zero_grad()
        feat_t = model_thermal(x_thermal)
        feat_r = model_rgb(x_rgb)
        outputs = fusion_classifier(feat_t, feat_r)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_thermal.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(dataloader.dataset), total_correct / len(dataloader.dataset) * 100

def evaluate(model_thermal, model_rgb, fusion_classifier, dataloader, criterion, device):
    model_thermal.eval()
    model_rgb.eval()
    fusion_classifier.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for (x_thermal, x_rgb), labels in dataloader:
            x_thermal, x_rgb, labels = x_thermal.to(device), x_rgb.to(device), labels.to(device)
            feat_t = model_thermal(x_thermal)
            feat_r = model_rgb(x_rgb)
            outputs = fusion_classifier(feat_t, feat_r)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * x_thermal.size(0)
            total_correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(dataloader.dataset), total_correct / len(dataloader.dataset) * 100


if __name__ == '__main__':

    transform_train = ThermalGrayTransform(train=True)
    transform_test = ThermalGrayTransform(train=False)

    # ------------------------ TRANSFORM ------------------------
    transform_rgb_train = transforms.Compose([
        transforms.Resize((112, 112)), # CROP
        transforms.RandomHorizontalFlip(), # AUGM
        transforms.RandomRotation(5), # AUGM
        transforms.Grayscale(num_output_channels=1), # GRAYSCALE
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]) # NORM
    ])

    transform_rgb_test = transforms.Compose([
        transforms.Resize((112, 112)), # CROP
        transforms.Grayscale(num_output_channels=1), # GRAYSCALE
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]) # NORM
    ])

    # ------------------------ SAMPLES LABELS ------------------------
    all_labels = []
    for split in ['train', 'val', 'test']:
        with open(f'dataset_merged_split/{split}/{split}_bb.txt') as f: # GROUND TRUTH FILE
            all_labels += [line.strip().split()[0].split("/")[-2] for line in f]
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)

    # ------------------------ DATALOADERS THERMAL+VISUAL ------------------------
    train_ds = DualInputDataset(
        'dataset_merged_split/train.jsonl',
        thermal_root='dataset_merged_split/train',
        rgb_root='dataset_merged_split_rgb/train',
        transform_thermal=transform_train,
        transform_rgb=transform_rgb_train,
        label_encoder=label_encoder
    )

    val_ds = DualInputDataset(
        'dataset_merged_split/val.jsonl',
        thermal_root='dataset_merged_split/val',
        rgb_root='dataset_merged_split_rgb/val',
        transform_thermal=transform_test,
        transform_rgb=transform_rgb_test,
        label_encoder=label_encoder
    )

    test_ds = DualInputDataset(
        'dataset_merged_split/test.jsonl',
        thermal_root='dataset_merged_split/test',
        rgb_root='dataset_merged_split_rgb/test',
        transform_thermal=transform_test,
        transform_rgb=transform_rgb_test,
        label_encoder=label_encoder
    )

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    print("Classes:", list(label_encoder.classes_))
    print("Train distribution:", Counter([lbl for _, lbl in train_ds]))

    # ------------------------ PAIRS SANITY CHECK ------------------------
    #check_pairs_integrity(train_ds, label_encoder)
    #check_pairs_integrity(val_ds, label_encoder)

    # ------------------------ DEVICE AND MODEL DETAILS ------------------------
    best_val_acc = 0
    train_accuracies, val_accuracies = [], []
    train_losses =  []
    val_losses = []

    epoch_times = []
    train_times = []
    val_times = []

    counter = 0
    patience = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_thermal = CustomResNetEmbedding(in_channels=1).to(device)
    model_rgb = CustomResNetEmbedding(in_channels=1).to(device)
    fusion_classifier = FusionClassifier(num_classes=len(label_encoder.classes_)).to(device)

    # ------------------------ OPTIMIZATION ------------------------
    criterion = nn.CrossEntropyLoss()
    params = list(model_thermal.parameters()) + list(model_rgb.parameters()) + list(fusion_classifier.parameters())
    optimizer = optim.Adam(params, lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    # ------------------------ TRAINING ------------------------
    for epoch in range(1, 21):
        start_epoch = time.time()

        start_train = time.time()
        train_loss, train_acc = train(model_thermal, model_rgb, fusion_classifier, train_loader, optimizer, criterion,
                                      device)
        end_train = time.time()

        start_val = time.time()
        val_loss, val_acc = evaluate(model_thermal, model_rgb, fusion_classifier, val_loader, criterion, device)
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

        scheduler.step(val_acc)

        # ------------------------ Best model saving + Early stopping  ------------------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_thermal_state_dict': model_thermal.state_dict(),
                'model_rgb_state_dict': model_rgb.state_dict(),
                'fusion_classifier_state_dict': fusion_classifier.state_dict()
            }, 'best_model_fusion.pth')
            print("Best model saved.")
        else:
            counter += 1
            if counter >= patience:
                print(f"⏹️ Early stopping triggered at epoch {epoch}")
                break

    # ------------------------ Epoch evaluation ------------------------
    start_test = time.time()
    test_loss, test_acc = evaluate(model_thermal, model_rgb, fusion_classifier, test_loader, criterion, device)
    end_test = time.time()
    test_time = end_test - start_test
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print(f"Test evaluation time: {test_time:.2f}s")
    start_test = time.time()
    test_loss, test_acc = evaluate(model_thermal, model_rgb, fusion_classifier, test_loader, criterion, device)
    end_test = time.time()
    test_time = end_test - start_test
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print(f"Test evaluation time: {test_time:.2f}s")

    # ------------------------ Save metrics ------------------------
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
    with open("timing_metrics_2.json", "w") as f:
        json.dump(timing_metrics, f)

    # ------------------------ Plots ------------------------
    epochs = list(range(1, len(train_accuracies) + 1))

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, val_accuracies, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy per Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("accuracy_plot_2.png")
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_plot_2.png")
    plt.show()
