import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import pandas as pd
import time
import os
import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

from train2 import CustomResNet  # o modifica in base al nome del file

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
MODEL_PATH = 'best_thermal_model2.pth'


def save_results_with_filenames(test_ds, true_labels, preds, output_path="test_results.csv"):
    """
    Salva CSV con colonne: filename, ground_truth, prediction

    Args:
        test_ds: dataset ImageFolder
        true_labels: lista/array delle etichette vere (indici)
        preds: lista/array delle predizioni (indici)
        output_path: percorso file CSV di output
    """
    # Estraggo solo il nome file (ultimo pezzo del path) da test_ds.samples
    filenames = [os.path.basename(sample[0]) for sample in test_ds.samples]

    assert len(filenames) == len(true_labels) == len(preds), "Dimensioni non corrispondono"

    df = pd.DataFrame({
        "filename": filenames,
        "ground_truth": true_labels,
        "prediction": preds
    })
    df.to_csv(output_path, index=False)
    print(f"Risultati salvati in '{output_path}'")

def load_data():
    transform_test = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
    ])
    test_ds = datasets.ImageFolder('dataset_merged_split/test', transform=transform_test)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    print("Classi test:", test_ds.classes)
    print("Distribuzione test:", Counter(test_ds.targets))
    return test_ds, test_loader


def load_model(num_classes):
    model = CustomResNet(num_classes=num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def get_predictions(model, test_loader):
    all_preds = []
    all_labels = []
    all_probs = []
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


def print_basic_metrics(true_labels, preds):
    accuracy = accuracy_score(true_labels, preds)
    precision_macro = precision_score(true_labels, preds, average='macro', zero_division=0)
    recall_macro = recall_score(true_labels, preds, average='macro', zero_division=0)
    f1_macro = f1_score(true_labels, preds, average='macro', zero_division=0)

    precision_micro = precision_score(true_labels, preds, average='micro', zero_division=0)
    recall_micro = recall_score(true_labels, preds, average='micro', zero_division=0)
    f1_micro = f1_score(true_labels, preds, average='micro', zero_division=0)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision (macro): {precision_macro:.4f}")
    print(f"Test Recall (macro): {recall_macro:.4f}")
    print(f"Test F1 Score (macro): {f1_macro:.4f}")
    print(f"Test Precision (micro): {precision_micro:.4f}")
    print(f"Test Recall (micro): {recall_micro:.4f}")
    print(f"Test F1 Score (micro): {f1_micro:.4f}")


def print_classification_report(true_labels, preds, class_names):
    print("\nClassification Report:")
    print(classification_report(true_labels, preds, target_names=class_names, zero_division=0))


def print_label_distribution(true_labels, preds, class_names):
    print("\nDistribuzione delle etichette vere:", Counter(true_labels))
    print("Distribuzione delle predizioni:", Counter(preds))
    print("\nSupport (numero di esempi per classe):")
    support = Counter(true_labels)
    for cls, count in support.items():
        print(f"{class_names[cls]}: {count}")


def save_predictions_csv(true_labels, preds, probs, class_names, filename="test_predictions.csv"):
    df = pd.DataFrame({
        "true_label": true_labels,
        "pred_label": preds,
    })

    for i, class_name in enumerate(class_names):
        df[f'prob_{class_name}'] = probs[:, i]

    df.to_csv(filename, index=False)
    print(f"\nPredizioni salvate in {os.path.abspath(filename)}")


def show_misclassified_images(test_ds, true_labels, preds, class_names, max_images=5):
    wrong_indices = [i for i, (t, p) in enumerate(zip(true_labels, preds)) if t != p]

    def imshow(img, title=None):
        img = img.squeeze(0)  # 1 channel
        plt.imshow(img, cmap='gray')
        if title:
            plt.title(title)
        plt.axis('off')

    fig = plt.figure(figsize=(12, 6))
    for idx, wrong_i in enumerate(wrong_indices[:max_images]):
        img, true_label = test_ds[wrong_i]
        pred_label = preds[wrong_i]

        ax = fig.add_subplot(1, max_images, idx + 1)
        imshow(img, title=f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}")
    plt.show()


def measure_inference_time(model, test_loader):
    model.eval()
    start = time.time()
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
    end = time.time()

    total_images = len(test_loader.dataset)
    time_per_image = (end - start) / total_images
    print(f"\nTempo medio inferenza per immagine: {time_per_image * 1000:.2f} ms")


def stampa_distribuzioni(y_true, y_pred):
    print("Distribuzione etichette vere:", Counter(y_true))
    print("Distribuzione predizioni:", Counter(y_pred))


def plot_confusion_matrix(y_true, y_pred, class_names=None, figsize=(12, 10), normalize=False):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    # Se non passo class_names, li calcolo dalle etichette vere + predette
    if class_names is None:
        class_names = sorted(list(set(y_true) | set(y_pred)))

    cm = confusion_matrix(y_true, y_pred, labels=class_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Matrice di confusione normalizzata'
    else:
        title = 'Matrice di confusione'

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=False, fmt='.2f' if normalize else 'd', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predizioni')
    plt.ylabel('Etichette vere')
    plt.title(title)
    plt.show()

def metriche_globali(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 score macro: {f1_macro:.4f}")
    print(f"F1 score micro: {f1_micro:.4f}")
    print(f"Precision macro: {precision_macro:.4f}")
    print(f"Recall macro: {recall_macro:.4f}")

def report_classi(y_true, y_pred):
    print("Report dettagliato per classe:")
    print(classification_report(y_true, y_pred, zero_division=0))

def plot_errore_distribution(y_true, y_pred):
    error_indices = np.where(y_true != y_pred)[0]
    error_counts = Counter(y_pred[error_indices])
    plt.figure(figsize=(12,6))
    plt.bar(error_counts.keys(), error_counts.values())
    plt.xlabel("Classe predetta (errata)")
    plt.ylabel("Numero di errori")
    plt.title("Distribuzione degli errori di classificazione per classe predetta")
    plt.show()


if __name__ == '__main__':
    test_ds, test_loader = load_data()
    model = load_model(num_classes=len(test_ds.classes))
    true_labels, preds, probs = get_predictions(model, test_loader)

    # Attiva/disattiva le funzioni qui:
    print_basic_metrics(true_labels, preds)
    print_classification_report(true_labels, preds, test_ds.classes)
    print_label_distribution(true_labels, preds, test_ds.classes)
    save_predictions_csv(true_labels, preds, probs, test_ds.classes)
    show_misclassified_images(test_ds, true_labels, preds, test_ds.classes)
    measure_inference_time(model, test_loader)
    print("---cose nuove---")
    # 1) Visualizza distribuzione dati
    stampa_distribuzioni(true_labels, preds)

    # 2) Matrice di confusione (non normalizzata)
    plot_confusion_matrix(true_labels, preds)

    # 3) Matrice di confusione normalizzata (valori da 0 a 1 per ogni riga)
    plot_confusion_matrix(true_labels, preds, normalize=True)

    # 4) Metriche globali di valutazione
    metriche_globali(true_labels, preds)

    # 5) Report dettagliato per classe (precision, recall, F1 per ogni classe)
    report_classi(true_labels, preds)

    # 6) Grafico distribuzione errori (classi più predette erroneamente)
    plot_errore_distribution(np.array(true_labels), np.array(preds))

    save_results_with_filenames(test_ds, true_labels, preds, output_path="test_results_with_filenames.csv")