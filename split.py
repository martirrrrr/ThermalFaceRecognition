import os
import shutil
import random

original_dir = "identities"
# Ignore augmented data on disk because data augm is performed in the training stage
output_base = "dataset_merged_split"

# 12 train - val 1 - test 5
split_ratio = {"train": 0.7, "val": 0.10, "test": 0.20}
random.seed(142)

for split in split_ratio:
    os.makedirs(os.path.join(output_base, split), exist_ok=True)

class_names = sorted(os.listdir(original_dir))

for cls_name in class_names:
    original_imgs = [os.path.join(original_dir, cls_name, f) for f in os.listdir(os.path.join(original_dir, cls_name))]
    #augmented_imgs = [os.path.join(augmented_dir, cls_name, f) for f in os.listdir(os.path.join(augmented_dir, cls_name))]

    all_imgs = original_imgs #+ augmented_imgs
    random.shuffle(all_imgs)

    n_total = len(all_imgs)
    n_train = int(n_total * split_ratio["train"])
    n_val = int(n_total * split_ratio["val"])
    n_test = n_total - n_train - n_val

    split_data = {
        "train": all_imgs[:n_train],
        "val": all_imgs[n_train:n_train + n_val],
        "test": all_imgs[n_train + n_val:]
    }

    for split, img_list in split_data.items():
        out_dir = os.path.join(output_base, split, cls_name)
        os.makedirs(out_dir, exist_ok=True)
        for img_path in img_list:
            fname = os.path.basename(img_path)
            dst = os.path.join(out_dir, fname)
            shutil.copy2(img_path, dst)

# === DOPO AVER COPIATO LE IMMAGINI, CREA I FILE train_bb.txt / val_bb.txt / test_bb.txt ===

bbox_file = "bounding_boxes.txt"
bbox_dict = {}

# Leggi bounding_boxes.txt in un dizionario {identities/...: linea completa}
with open(bbox_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 5:
            bbox_dict[parts[0]] = line.strip()

# Scrivi solo le immagini effettivamente presenti nei rispettivi split
for split in ["train", "val", "test"]:
    txt_path = f"dataset_merged_split/{split}/{split}_bb.txt"
    with open(txt_path, "w") as f_out:
        split_path = os.path.join(output_base, split)
        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            if not os.path.isdir(class_path):
                continue
            for img_name in os.listdir(class_path):
                relative_path = f"identities/{class_name}/{img_name}"  # chiave nel bbox_dict
                if relative_path in bbox_dict:
                    # percorso effettivo dove l’immagine è stata copiata
                    net_path = f"{output_base}/{split}/{class_name}/{img_name}"
                    # prendi solo le coordinate bbox (dalla seconda colonna in poi)
                    coords = " ".join(bbox_dict[relative_path].split()[1:])
                    f_out.write(f"{net_path} {coords}\n")


import json

def bbtxt_to_jsonl(bbtxt_path, jsonl_path):
    with open(bbtxt_path, 'r') as f_in, open(jsonl_path, 'w') as f_out:
        for line in f_in:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            image_path = parts[0]
            bbox = list(map(int, parts[1:5]))
            label = image_path.split(os.sep)[-2]  # nome cartella es: '113'

            data = {
                "image": image_path,
                "bbox": bbox,
                "annotations": [{"label": label}]
            }
            f_out.write(json.dumps(data) + "\n")

# Percorsi base
base_dir = "dataset_merged_split"

# Converti ogni file *_bb.txt in JSONL
for split in ["train", "val", "test"]:
    bbtxt_file = f"{base_dir}/{split}/{split}_bb.txt"
    jsonl_file = f"{base_dir}/{split}.jsonl"
    print(f"Converting {bbtxt_file} -> {jsonl_file}")
    bbtxt_to_jsonl(bbtxt_file, jsonl_file)
print("✅ JSONL files created.")
