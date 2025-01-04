import json
import os
import sys
from argparse import ArgumentParser

import pandas as pd
from prettytable import PrettyTable
from torch.utils.data import Dataset
from tqdm import tqdm

import clip
sys.path.append("/home/yxwei/wangzihao/ACE/src/eval/evaluation")
from eval_util import clip_eval_by_image, create_meta_json
from evaluator import Evaluator


class CocoDataset(Dataset):
    def __init__(self, save_path):
        self.image_names = os.listdir(save_path)
        self.image_paths = []
        for image_name in self.image_names:
            image_path = os.path.join(save_path, image_name)
            self.image_paths.append(image_path)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_path = self.image_names[index % len(self.image_paths)]
        return image_path


class ClipEvaluator(Evaluator):
    """
    Evaluation for CLIP-protocol accepts `save_folder` as a *JSON file* with the following format:
    {
        CONCEPT_1: {
            TEMPLATE_IDX_1_1: [IMAGE_PATH_1_1_1, IMAGE_PATH_1_1_2, ...],
            TEMPLATE_IDX_1_2: [IMAGE_PATH_1_2_1, IMAGE_PATH_1_2_2, ...],
            ...
        },
        CONCEPT_2: {
            TEMPLATE_IDX_2_1: [IMAGE_PATH_2_1_1, IMAGE_PATH_2_1_2, ...],
            TEMPLATE_IDX_2_2: [IMAGE_PATH_2_2_1, IMAGE_PATH_2_2_2, ...],
            ...
        },
        ...
    }
    CONCEPT_i: str, the i-th concept to be evaluated.
    TEMPLATE_IDX_i_j: int, range(80), the j-th selected template for CONCEPT_i.
    IMAGE_PATH_i_j_k: str, the k-th image path for CONCEPT_i, TEMPLATE_IDX_i_j.
    """

    def __init__(
            self,
            save_folder: str = "benchmark/generated_imgs/",
            output_path: str = "benchmark/results/",
            given_concept: str = None,
            eval_with_template: bool = False,
            method: str = "",
            image_concept: str = None,
            prompt_path: str = None
    ):
        super().__init__(save_folder=save_folder, output_path=output_path)
        self.given_concept = given_concept
        self.image_concept = image_concept
        self.prompt_path = prompt_path
        self.method = method
        self.model, _ = clip.load("ViT-B/32", device="cuda")
        if self.method != "coco":
            if given_concept is None or self.method == "concept_relation":
                self.img_metadata = json.load(open(os.path.join(self.save_folder, "meta.json")))

            else:
                self.img_metadata = json.load(
                    open(os.path.join(self.save_folder, f"meta_{given_concept}_{image_concept}.json")))
        self.eval_with_template = eval_with_template

    def evaluation(self):
        all_scores = {}
        all_cers = {}
        for concept, data in self.img_metadata.items():
            print(f"Evaluating concept:", concept)
            scores = accs = 0.0
            num_all_images = 0
            for template_idx, image_paths in tqdm(data.items()):
                if self.method == "concept_relation":
                    target_prompt = self.given_concept
                else:
                    target_prompt = concept
                anchor_prompt = ""
                num_images = len(image_paths)
                score, acc = clip_eval_by_image(
                    image_paths,
                    [target_prompt] * num_images,
                    [anchor_prompt] * num_images,
                    model=self.model
                )
                scores += score * num_images
                accs += acc * num_images
                num_all_images += num_images
            scores /= num_all_images
            accs /= num_all_images
            all_scores[concept] = scores
            all_cers[concept] = 1 - accs

        table = PrettyTable()
        table.field_names = ["Concept", "CLIPScore", "CLIPErrorRate"]
        for concept, score in all_scores.items():
            table.add_row([concept, score, all_cers[concept]])
        print(table)

        save_name = f"evaluation_results_clip_{self.given_concept}_image_{self.image_concept}.json"
        print(os.path.join(self.output_path, save_name))
        with open(os.path.join(self.output_path, save_name), "w") as f:
            json.dump([all_scores, all_cers], f)

    def evaluation_30k(self):
        df = pd.read_csv(self.prompt_path)
        num_images = len(os.listdir(self.save_folder))
        scores = 0.0
        row_list = []
        for _, row in df.iterrows():
            row_list.append(row)
        for row in tqdm(row_list):
            target_prompt = row.prompt
            image_path = os.path.join(self.save_folder, f"{row.case_number}_0.png")
            score, _ = clip_eval_by_image(
                [image_path],
                [target_prompt],
                [""],
                model=self.model
            )
            scores += score
        mean_score = scores / num_images
        save_name = f"evaluation_results.json"
        with open(os.path.join(self.output_path, save_name), "w") as f:
            json.dump({"mean_score": mean_score}, f)


def eval_clip(args):
    csv_df = pd.read_csv(args.csv_path)
    if args.is_full:
        dir_path = os.path.join("evaluation-outputs", args.csv_name + args.add_name)
        dir_name_list = os.listdir(dir_path)
        for dir_name in dir_name_list:
            save_folder = os.path.join(dir_path, dir_name)
            create_meta_json(csv_df=csv_df,
                             save_folder=save_folder,
                             num_samples=args.num_samples,
                             concept=args.concept,
                             image_concept=args.image_concept)
            evaluator = ClipEvaluator(
                save_folder=save_folder,
                output_path=save_folder,
                given_concept=args.concept,
                image_concept=args.image_concept
            )
            json_path = os.path.join(save_folder, "evaluation_results(concept only).json")
            if not os.path.exists(json_path):
                try:
                    evaluator.evaluation()
                except Exception as Ex:
                    print(Ex)
    elif args.edit_prompt_path is not None and args.image_concept_path is None:
        print(args.edit_prompt_path)
        df = pd.read_csv(args.edit_prompt_path)
        for _, row in df.iterrows():
            output_path = os.path.join("evaluation-outputs",
                                       args.csv_name + args.add_name,
                                       args.model_name_format.format(row.prompt))
            create_meta_json(csv_df=csv_df,
                             save_folder=output_path,
                             num_samples=args.num_samples,
                             concept=row.concept,
                             is_nudity=args.is_nudity,
                             image_concept=args.image_concept)
            evaluator = ClipEvaluator(
                save_folder=output_path,
                output_path=output_path,
                given_concept=row.concept,
                image_concept=args.image_concept
            )
            evaluator.evaluation()

    elif args.edit_prompt_path is not None and args.image_concept_path is not None:
        print(args.edit_prompt_path)
        df = pd.read_csv(args.edit_prompt_path)
        image_concept_list = []
        if not args.is_specific:
            with open(args.image_concept_path, "r") as f:
                for line in f:
                    image_concept_list.append(line.strip())
        with open(args.image_concept_path, "r") as f:
            for line in f:
                image_concept = line.strip()
                for _, row in df.iterrows():
                    output_path = os.path.join("evaluation-outputs",
                                               args.csv_name + args.add_name,
                                               args.model_name_format.format(
                                                   image_concept, row.edit_prompt))
                    if args.is_edit:
                        json_path = f"evaluation_results_{row.concept}_{image_concept}.json"
                        try:
                            create_meta_json(csv_df=csv_df,
                                             save_folder=output_path,
                                             num_samples=args.num_samples,
                                             concept=row.concept,
                                             is_nudity=args.is_nudity,
                                             image_concept=image_concept)
                            evaluator = ClipEvaluator(
                                save_folder=output_path,
                                output_path=output_path,
                                given_concept=row.concept,
                                image_concept=image_concept
                            )
                            if not os.path.exists(os.path.join(output_path, json_path)):
                                try:
                                    evaluator.evaluation()
                                except Exception as Ex:
                                    print(Ex)
                        except Exception as Ex:
                            print(Ex)
                    else:
                        for eval_concept in image_concept_list:
                            json_path = f"evaluation_results_{eval_concept}.json"
                            try:
                                create_meta_json(csv_df=csv_df,
                                                 save_folder=output_path,
                                                 num_samples=args.num_samples,
                                                 concept=eval_concept,
                                                 is_nudity=args.is_nudity,
                                                 image_concept=eval_concept)
                                evaluator = ClipEvaluator(
                                    save_folder=output_path,
                                    output_path=output_path,
                                    given_concept=eval_concept,
                                    image_concept=eval_concept
                                )
                                try:
                                    evaluator.evaluation()
                                except Exception as Ex:
                                    print(Ex)
                            except Exception as Ex:
                                print(Ex)

    else:
        create_meta_json(csv_df=csv_df, save_folder=args.save_folder, num_samples=args.num_samples,
                         concept=args.concept, is_nudity=args.is_nudity, image_concept=args.image_concept)
        evaluator = ClipEvaluator(
            save_folder=args.save_folder, output_path=args.output_path, given_concept=args.concept,
            image_concept=args.image_concept
        )
        evaluator.evaluation()


def concept_relation(args):
    prompts_path = args.csv_path
    df = pd.read_csv(prompts_path)
    specific_concept_dict = {}
    image_concept_set = set()
    with open(args.image_concept_path, "r") as f:
        for line in f:
            image_concept_set.add(line.strip())
    for _, row in df.iterrows():
        if row.concept not in specific_concept_dict:
            specific_concept_dict[row.concept] = []
            specific_concept_dict[row.concept].append(row.case_number)
        else:
            specific_concept_dict[row.concept].append(row.case_number)
    num_samples = args.num_samples
    save_path = f"evaluation-outputs/{args.csv_name}/SD3"
    create_meta_json(csv_df=df, save_folder=save_path, num_samples=num_samples,
                     is_nudity=args.is_nudity)
    for image_concept in specific_concept_dict:
        if image_concept not in image_concept_set:
            continue

        evaluator = ClipEvaluator(
            save_folder=save_path, output_path=save_path, given_concept=image_concept, method=args.method
        )
        evaluator.evaluation()


def coco_prompt_relation(args):
    prompts_path = args.csv_path
    save_path = args.save_folder
    image_concept_path = args.image_concept_path
    image_concept_list = []
    if image_concept_path != None:
        with open(image_concept_path, "r") as f:
            for line in f:
                image_concept_list.append(line.strip())
    else:
        image_concept_list.append(None)
    output_path = args.output_path
    for image_concept in image_concept_list:
        evaluator = ClipEvaluator(
            prompt_path=prompts_path, save_folder=save_path.format(image_concept),
            output_path=output_path.format(image_concept), method=args.method
        )
        evaluator.evaluation_30k()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--csv_path",
        type=str,
        help="path to csv file that contains information of generated images"
    )
    parser.add_argument(
        "--csv_name",
        type=str,
        required=False,
        help="path to csv file that contains information of generated images"
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        help="path to json that contains metadata for generated images.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="path to save evaluation results.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        help="image number for each case",
    )
    parser.add_argument(
        "--is_full",
        action='store_true',
        help='evaluation every file',
        required=False,
        default=False
    )
    parser.add_argument(
        "--is_nudity",
        action='store_true',
        help='',
        required=False,
        default=False
    )
    parser.add_argument(
        "--concept",
        type=str,
        required=False,
        help="Specify the concept"
    )
    parser.add_argument(
        "--image_concept",
        type=str,
        required=False,
        help="Specify the concept"
    )
    parser.add_argument(
        "--image_concept_path",
        type=str,
        required=False,
        help="Specify the concept"
    )
    parser.add_argument(
        "--format",
        type=str,
        required=False,
        default=None,
        help="Specify the concept"
    )
    parser.add_argument(
        "--edit_prompt_path",
        type=str,
        required=False,
        default=None,
        help="Specify the concept"
    )
    parser.add_argument(
        "--add_name",
        type=str,
        required=False,
        default="",
        help="Specify the concept"
    )
    parser.add_argument(
        "--model_name_format",
        type=str,
        required=False,
        default="",
        help="Specify the concept"
    )
    parser.add_argument(
        "--method",
        type=str,
        required=False,
        default="",
        help="Specify the concept"
    )
    parser.add_argument(
        "--is_specific",
        action='store_true',
        help='',
        required=False,
        default=False
    )
    parser.add_argument(
        "--is_edit",
        action='store_true',
        help='',
        required=False,
        default=False
    )
    args = parser.parse_args()
    if args.method == "concept_relation":
        concept_relation(args)
    elif args.method == "coco":
        coco_prompt_relation(args)
    else:
        eval_clip(args)
