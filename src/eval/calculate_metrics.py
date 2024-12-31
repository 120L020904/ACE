import json

from cleanfid import fid
import argparse
import os
import pandas as pd
from torch.utils.data import Dataset


class coco30k_dataset(Dataset):
    def __init__(self, csv_path):
        df_generate = pd.read_csv(args.csv_path)
        self.image_id = []
        for index, row in df_generate.iterrows():
            self.image_id.append(row.case_number)

    def __len__(self):
        return len(self.image_id)

    def __getitem__(self, index):
        image_id_tem = self.image_id[index % len(self.image_id)]
        data_path = "coco/train2017"
        data_path_tem = os.path.join(data_path, f"{image_id_tem}.jpg")
        return data_path_tem


parser = argparse.ArgumentParser()
parser.add_argument("--original", type=str, required=False)
parser.add_argument("--generated", type=str, required=True)
parser.add_argument("--save_dir", type=str, required=True)
parser.add_argument("--csv_path", type=str, required=True)
parser.add_argument("--model_method", type=str, required=False, default=None)
parser.add_argument("--image_concept_path", type=str, required=False, default=None)
parser.add_argument("--edited_concept_path", type=str, required=False)
args = parser.parse_args()

# concepts = [f for f in os.listdir(args.original) if not (f.startswith('.') or f.startswith("coco30k")) and os.path.isdir(os.path.join(args.original, f))]
concepts = []
image_concept_path = args.image_concept_path
if image_concept_path is not None:
    with open(image_concept_path, "r") as f:
        for line in f:
            concepts.append(line.strip())
else:
    concepts.append(None)
print(concepts)
generate_path = args.generated
# pandas dataframe

model_name = args.generated.split('/')[-1]
save_dir = args.save_dir

model_method = args.model_method

# concept-wise metrics
for image_concept in concepts:
    result_dict = {}
    generate_path_tem = generate_path.format(image_concept)
    save_dir_tem = save_dir.format(image_concept)
    os.makedirs(save_dir_tem, exist_ok=True)
    print(f"Concept: {image_concept}")
    metrics = fid.compute_fid(
        os.path.join(generate_path_tem, str(None)),
        "coco/train2017",
        )
    result_dict[image_concept] = metrics
    result_json_path = os.path.join(save_dir_tem,f"fid_{image_concept}.json")
    with open(result_json_path, "w") as fp:
        json.dump(result_dict, fp)