import json
import os
from PIL import Image
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

class FaceDetectionDataset(Dataset):
    def __init__(self, jsonl_file, root_dir, transform=None, label_encoder=None):
        self.entries = []
        self.transform = transform
        self.label_encoder = label_encoder
        labels = []

        with open(jsonl_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                img_name = os.path.basename(data['image'])  # es: '112_2_2_6_161_24_1.png'
                person_id = img_name.split('_')[0]          # es: '112'
                full_path = os.path.join(root_dir, person_id, img_name)

                x, y, w, h = data['bbox']
                box = [x, y, w, h]

                label = person_id
                labels.append(label)
                self.entries.append((full_path, box, label))

        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(labels)

        # Encode all labels to integers
        self.entries = [(p, b, self.label_encoder.transform([l])[0]) for p, b, l in self.entries]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        img_path, bbox, label = self.entries[idx]
        image = Image.open(img_path).convert('L')

        x, y, w, h = map(int, bbox)
        cropped = image.crop((x, y, x + w, y + h))

        if self.transform:
            cropped = self.transform(cropped)

        return cropped, label
