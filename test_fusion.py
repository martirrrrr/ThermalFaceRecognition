from advanced_train import CustomResNetEmbedding, FusionClassifier, DualInputDataset
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
MODEL_PATH = 'best_model_fusion.pth'


def load_models(num_classes, device, model_path='best_model_fusion.pth'):
    model_thermal = CustomResNetEmbedding(in_channels=1).to(device)
    model_rgb = CustomResNetEmbedding(in_channels=1).to(device)
    fusion_classifier = FusionClassifier(num_classes=num_classes).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model_thermal.load_state_dict(checkpoint['model_thermal_state_dict'])
    model_rgb.load_state_dict(checkpoint['model_rgb_state_dict'])
    fusion_classifier.load_state_dict(checkpoint['fusion_classifier_state_dict'])

    model_thermal.eval()
    model_rgb.eval()
    fusion_classifier.eval()
    return model_thermal, model_rgb, fusion_classifier

def load_data():
    transform_thermal = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    transform_rgb = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.Grayscale(num_output_channels=1),  # se RGB -> grayscale
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

    test_ds = DualInputDataset(
        jsonl_path="dataset_merged_split/test.jsonl",
        thermal_root="dataset_merged_split/test",
        rgb_root="dataset_merged_split_rgb/test",
        transform_thermal=transform_thermal,
        transform_rgb=transform_rgb,
        label_encoder=label_encoder
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    return test_ds, test_loader, label_encoder

def get_predictions(model_thermal, model_rgb, fusion_classifier, test_loader, device):
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for (x_thermal, x_rgb), labels in test_loader:
            x_thermal, x_rgb, labels = x_thermal.to(device), x_rgb.to(device), labels.to(device)
            feat_t = model_thermal(x_thermal)
            feat_r = model_rgb(x_rgb)
            outputs = fusion_classifier(feat_t, feat_r)
            preds = torch.argmax(outputs, dim=1)
            probs = torch.softmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def show_misclassified(test_ds, true_labels, preds, class_names, max_images=5):
    wrong_idxs = [i for i, (t, p) in enumerate(zip(true_labels, preds)) if t != p]
    fig = plt.figure(figsize=(12, 6))
    for idx, wrong_i in enumerate(wrong_idxs[:max_images]):
        img, true_lbl = test_ds[wrong_i]
        pred_lbl = preds[wrong_i]
        ax = fig.add_subplot(1, max_images, idx + 1)
        (x_thermal, x_rgb), true_lbl = test_ds[wrong_i]
        pred_lbl = preds[wrong_i]
        ax = fig.add_subplot(1, max_images, idx + 1)

        img = x_thermal.squeeze(0).numpy()
        plt.imshow(img, cmap='gray')
        ax.set_title(f"True: {class_names[true_lbl]}\nPred: {class_names[pred_lbl]}")
        plt.axis('off')
    plt.show()

def show_misclassified_from_csv(csv_path, max_samples=10):
    df = pd.read_csv(csv_path)
    if 'ground_truth' not in df.columns or 'prediction' not in df.columns:
        raise ValueError("CSV must contain 'ground_truth' and 'prediction' columns.")
    misclassified = df[df['ground_truth'] != df['prediction']]
    print(f"\nFound {len(misclassified)} misclassified samples. Showing up to {max_samples}:\n")
    #misclassified.to_csv("misclassified_samples.csv")
    print(misclassified.head(max_samples))

def print_metrics(true_labels, preds, label_encoder):
    decoded_gt = label_encoder.inverse_transform(true_labels)
    decoded_preds = label_encoder.inverse_transform(preds)

    print("Accuracy:", accuracy_score(decoded_gt, decoded_preds))
    print("Precision (macro):", precision_score(decoded_gt, decoded_preds, average='macro'))
    print("Recall (macro):", recall_score(decoded_gt, decoded_preds, average='macro'))
    print("F1 Score (macro):", f1_score(decoded_gt, decoded_preds, average='macro'))
    print("\nClassification Report:\n")
    print(classification_report(decoded_gt, decoded_preds, zero_division=0))

def save_results_with_filenames(test_ds, true_labels, preds, label_encoder, output_path="test_results_fusion.csv"):
    filenames = [os.path.basename(sample['image']) for sample in test_ds.samples]
    decoded_gt = label_encoder.inverse_transform(true_labels)
    decoded_preds = label_encoder.inverse_transform(preds)

    df = pd.DataFrame({
        "filename": filenames,
        "ground_truth": decoded_gt,
        "prediction": decoded_preds
    })
    df.to_csv(output_path, index=False)
    print(f"Results saved to '{output_path}'")


def main():
    test_ds, test_loader, label_encoder = load_data()
    model_thermal, model_rgb, fusion_classifier = load_models(num_classes=len(label_encoder.classes_), device=DEVICE, model_path=MODEL_PATH)
    true_labels, preds, probs = get_predictions(model_thermal, model_rgb, fusion_classifier, test_loader, DEVICE)

    save_results_with_filenames(test_ds, true_labels, preds, label_encoder)
    print_metrics(true_labels, preds, label_encoder)
    #plot_conf_matrix(true_labels, preds, label_encoder.classes_, normalize=True)
    show_misclassified(test_ds, true_labels, preds, label_encoder.classes_)
    show_misclassified_from_csv("test_results_fusion.csv", max_samples=67)

if __name__ == "__main__":
    main()