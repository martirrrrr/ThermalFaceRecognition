import os
import json

input_jsonl = 'labels.jsonl'
output_jsonl = 'face_bboxes.jsonl'
identities_root = 'identities'

with open(input_jsonl, 'r') as infile, open(output_jsonl, 'w') as outfile:
    count_total = 0
    count_valid = 0

    for line in infile:
        count_total += 1
        data = json.loads(line)

        # Prendi filename originale es. images/113_1_2_2_156_102_1.png
        full_image_name = data['image']
        base_name = os.path.basename(full_image_name)  # es. 113_1_2_2_156_102_1.png

        # Estrai ID e nome reale
        parts = base_name.split('_')
        id_folder = parts[0]
        #actual_filename = '_'.join(parts[1:])  # es. 1_2_2_156_102_1.png

        image_path = os.path.join(identities_root, id_folder, base_name)

        if os.path.exists(image_path):
            if 'face' in data:
                bbox = data['face'][0] + data['face'][1]  # [[x_min, y_min], [x_max, y_max]] → [x_min, y_min, x_max, y_max]
                record = {
                    "image": image_path,
                    "bbox": bbox
                }
                outfile.write(json.dumps(record) + '\n')
                count_valid += 1
        else:
            print(f"[!] Img NOT found: {image_path}")

print(f"\n{count_valid}/{count_total} valid bounding boxes saved from samples.")
