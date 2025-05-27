import os
import json
import shutil

# Cartella con tutti i file rgb sparsi
rgb_source_folder = 'rgb/'

# Cartella root dove creare la struttura ordinata
rgb_target_root = 'dataset_merged_split_rgb'

# JSONL termici da leggere
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
                term_img_path = entry['image']  # esempio: dataset_merged_split/train/135/135_2_2_1_202_109_1.png
                filename = os.path.basename(term_img_path)  # es 135_2_2_1_202_109_1.png

                # Prendi label/ID: parte prima del primo underscore
                label = filename.split('_')[0]

                # Costruisci nome RGB cambiando _1.png in _3.png
                if filename.endswith('_1.png'):
                    rgb_filename = filename[:-6] + '_3.png'
                else:
                    print(f"Filename inatteso: {filename}")
                    continue

                rgb_src_path = os.path.join(rgb_source_folder, rgb_filename)
                if not os.path.isfile(rgb_src_path):
                    print(f"File RGB non trovato: {rgb_src_path}")
                    continue

                # Cartella di destinazione es: dataset_merged_split_rgb/train/135/
                dst_folder = os.path.join(rgb_target_root, split, label)
                os.makedirs(dst_folder, exist_ok=True)

                dst_path = os.path.join(dst_folder, rgb_filename)

                # Copia il file
                shutil.copy2(rgb_src_path, dst_path)
                print(f"Copiato {rgb_src_path} -> {dst_path}")

def create_rgb_jsonl_copy(input_jsonl_path):
    # costruisci nome file output con _rgb prima dell'estensione
    base, ext = os.path.splitext(input_jsonl_path)
    output_jsonl_path = base + '_rgb' + ext

    with open(input_jsonl_path, 'r') as fin, open(output_jsonl_path, 'w') as fout:
        for line in fin:
            entry = json.loads(line)
            img_path = entry.get('image', '')
            # sostituisci _1.png con _3.png nel path
            if img_path.endswith('_1.png'):
                img_path = img_path[:-6] + '_3.png'
                entry['image'] = img_path
            else:
                # se non finisce con _1.png puoi decidere cosa fare (qui lascio invariato)
                pass
            fout.write(json.dumps(entry) + '\n')
    print(f"Creato file aggiornato: {output_jsonl_path}")


if __name__ == "__main__":
    #organize_rgb()
    for split, filepath in jsonl_files.items():
        create_rgb_jsonl_copy(filepath)
