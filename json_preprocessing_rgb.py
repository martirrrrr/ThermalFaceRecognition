import os
import json
import shutil

# RGB files 
rgb_source_folder = 'rgb/'

# Destination folder
rgb_target_root = 'dataset_merged_split_rgb'

# JSONL thermal files 
jsonl_files = {
    'train': 'dataset_merged_split/train.jsonl',
    'val': 'dataset_merged_split/val.jsonl',
    'test': 'dataset_merged_split/test.jsonl',
}


def organize_rgb():
    for split, jsonl_path in jsonl_files.items():
        with open(jsonl_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                term_img_path = entry['image']  # example: dataset_merged_split/train/135/135_2_2_1_202_109_1.png
                filename = os.path.basename(term_img_path)  # example: 135_2_2_1_202_109_1.png

                # Label ID = first part before the underscore (identity)
                label = filename.split('_')[0]

                # Replace _1 with _3 (extension names to distinguish rgb from thermal images by name)
                if filename.endswith('_1.png'):
                    rgb_filename = filename[:-6] + '_3.png'
                else:
                    print(f"Unexpected finename: {filename}")
                    continue

                rgb_src_path = os.path.join(rgb_source_folder, rgb_filename)
                if not os.path.isfile(rgb_src_path):
                    print(f"RGB pair not found: {rgb_src_path}")
                    continue

                # Dest. folder: dataset_merged_split_rgb/train/135/
                dst_folder = os.path.join(rgb_target_root, split, label)
                os.makedirs(dst_folder, exist_ok=True)

                dst_path = os.path.join(dst_folder, rgb_filename)

                # Copia il file
                shutil.copy2(rgb_src_path, dst_path)
                print(f"Copied {rgb_src_path} -> {dst_path}")

def create_rgb_jsonl_copy(input_jsonl_path):
    # Complete path
    base, ext = os.path.splitext(input_jsonl_path)
    output_jsonl_path = base + '_rgb' + ext

    with open(input_jsonl_path, 'r') as fin, open(output_jsonl_path, 'w') as fout:
        for line in fin:
            entry = json.loads(line)
            img_path = entry.get('image', '')
            # replace _1.png with _3.png in the path if it exists
            if img_path.endswith('_1.png'):
                img_path = img_path[:-6] + '_3.png'
                entry['image'] = img_path
            else:
                # Otherwise skip
                pass
            fout.write(json.dumps(entry) + '\n')
    print(f"Creato file aggiornato: {output_jsonl_path}")


if __name__ == "__main__":
    #organize_rgb()
    for split, filepath in jsonl_files.items():
        create_rgb_jsonl_copy(filepath)
