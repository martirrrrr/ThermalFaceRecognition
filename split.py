import os
import shutil
import random

original_dir = "identities"
# Ignore augmented data on disk because data augm is performed in the training stage
#augmented_dir = "augmented"
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