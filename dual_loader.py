import torch
from torchvision import transforms
import json
from PIL import Image

# === THERMAL TRANSFORM ===
class ThermalGrayTransform:
    def __init__(self, train=True):
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
            transforms.RandomRotation(5) if train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __call__(self, img):
        img = img.convert("L")  # assicurati grayscale
        return self.transform(img)

# === DUAL DATALOADER CUSTOM ===
class DualInputDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path, thermal_root, rgb_root, transform_thermal, transform_rgb, label_encoder):
        with open(jsonl_path, 'r') as f:
            self.samples = [json.loads(line) for line in f]

        self.thermal_root = thermal_root
        self.rgb_root = rgb_root
        self.transform_thermal = transform_thermal
        self.transform_rgb = transform_rgb
        self.label_encoder = label_encoder
        self.skipped = 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label_str = sample['annotations'][0]['label']
        label = self.label_encoder.transform([label_str])[0]

        # EXTRACT THERMAL IMAGE
        thermal_path = sample['image']
        thermal_img_path = thermal_path

        # GRAYSCALE OF THERMAL  -  (SUITABLE FOR GRAYSCALE AND IRON THERMAL)
        thermal_img = Image.open(thermal_img_path).convert("L")
        x_thermal = self.transform_thermal(thermal_img)

        # FIND THE VISUAL PAIR
        rgb_path = thermal_path.replace('dataset_merged_split', 'dataset_merged_split_rgb').replace('_1.png', '_3.png')
        rgb_img_path = rgb_path

        # TRY-EXCEPT FOR PAIRING
        try:
            rgb_img = Image.open(rgb_img_path).convert("RGB")
        except FileNotFoundError:
            self.skipped += 1
            return self.__getitem__((idx + 1) % len(self))
        x_rgb = self.transform_rgb(rgb_img)

        return (x_thermal, x_rgb), label

    def get_skipped_count(self):
        # SKIP IF PAIRING FAILS
        return self.skipped
