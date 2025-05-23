import json
import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from face_dataset import FaceDetectionDataset
from train import CustomResNet  # import your trained model class
from sklearn.preprocessing import LabelEncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
MODEL_PATH = 'best_model.pth'

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

# Plot of confusion matrix - some arrangements are done for correct displaying of the matrix 142x142
def plot_compact_confusion_matrix(y_true, y_pred, class_names, normalize=True, figsize=(18, 14), dpi=100):
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)  # in caso di divisione per zero
        fmt = ".2f"
        title = "Normalized Confusion Matrix"
    else:
        fmt = "d"
        title = "Confusion Matrix"

    plt.figure(figsize=figsize, dpi=dpi)
    sns.heatmap(cm, cmap="Blues", xticklabels=False, yticklabels=False, cbar=True)

    plt.title(title)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.tight_layout()
    plt.show()

# Loading data (images, json)
def load_data():
    transform_test = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    all_labels = []
    with open("dataset_merged_split/test.jsonl") as f:
        for line in f:
            label = json.loads(line)["annotations"][0]["label"]
            all_labels.append(label)

    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)

    test_ds = FaceDetectionDataset(
        jsonl_file="dataset_merged_split/test.jsonl",
        root_dir="dataset_merged_split/test",
        transform=transform_test,
        label_encoder=label_encoder
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    return test_ds, test_loader, label_encoder

# Load model CustomResnet
def load_model(num_classes):
    model = CustomResNet(num_classes=num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# Predict (logits and softmax)
def get_predictions(model, test_loader):
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            probs = F.softmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

# CSV save of results
def save_results_with_filenames(test_ds, true_labels, preds, label_encoder, output_path="test_results.csv"):
    filenames = [os.path.basename(p) for (p, _, _) in test_ds.entries]
    decoded_gt = label_encoder.inverse_transform(true_labels)
    decoded_preds = label_encoder.inverse_transform(preds)

    df = pd.DataFrame({
        "filename": filenames,
        "ground_truth": decoded_gt,
        "prediction": decoded_preds
    })
    df.to_csv(output_path, index=False)
    print(f"Results saved to '{output_path}'")

# Display metrics values (accuracy, precision, recall, f1)
def print_metrics(true_labels, preds, label_encoder):
    decoded_gt = label_encoder.inverse_transform(true_labels)
    decoded_preds = label_encoder.inverse_transform(preds)

    print("Accuracy:", accuracy_score(decoded_gt, decoded_preds))
    print("Precision (macro):", precision_score(decoded_gt, decoded_preds, average='macro'))
    print("Recall (macro):", recall_score(decoded_gt, decoded_preds, average='macro'))
    print("F1 Score (macro):", f1_score(decoded_gt, decoded_preds, average='macro'))
    print("\nClassification Report:\n")
    print(classification_report(decoded_gt, decoded_preds, zero_division=0))

def plot_conf_matrix(true_labels, preds, class_names, normalize=False):
    cm = confusion_matrix(true_labels, preds)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='.2f' if normalize else 'd', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix' if normalize else 'Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Focus on misclassified samples of testing set
def show_misclassified(test_ds, true_labels, preds, class_names, max_images=5):
    wrong_idxs = [i for i, (t, p) in enumerate(zip(true_labels, preds)) if t != p]
    fig = plt.figure(figsize=(12, 6))
    for idx, wrong_i in enumerate(wrong_idxs[:max_images]):
        img, true_lbl = test_ds[wrong_i]
        pred_lbl = preds[wrong_i]
        ax = fig.add_subplot(1, max_images, idx + 1)
        img = img.squeeze(0).numpy()
        plt.imshow(img, cmap='gray')
        ax.set_title(f"True: {class_names[true_lbl]}\nPred: {class_names[pred_lbl]}")
        plt.axis('off')
    plt.show()

# Read CSV and pprint misclassified
def show_misclassified_from_csv(csv_path, max_samples=10):
    df = pd.read_csv(csv_path)
    if 'ground_truth' not in df.columns or 'prediction' not in df.columns:
        raise ValueError("CSV must contain 'ground_truth' and 'prediction' columns.")
    misclassified = df[df['ground_truth'] != df['prediction']]
    print(f"\nFound {len(misclassified)} misclassified samples. Showing up to {max_samples}:\n")
    #misclassified.to_csv("misclassified_samples.csv")
    print(misclassified.head(max_samples))

def main():
    test_ds, test_loader, label_encoder = load_data()
    model = load_model(num_classes=len(label_encoder.classes_))
    true_labels, preds, probs = get_predictions(model, test_loader)

    save_results_with_filenames(test_ds, true_labels, preds, label_encoder)
    print_metrics(true_labels, preds, label_encoder)
    plot_conf_matrix(true_labels, preds, label_encoder.classes_, normalize=True)
    show_misclassified(test_ds, true_labels, preds, label_encoder.classes_)
    plot_compact_confusion_matrix(true_labels, preds,label_encoder.classes_,normalize=True )
    show_misclassified_from_csv("test_results.csv", max_samples=67)

if __name__ == '__main__':
    main()
