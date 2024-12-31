from __future__ import print_function

import argparse
import json
import os

import lpips
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image

# desired size of the output image
imsize = 64
loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    image = (image - 0.5) * 2
    return image.to(torch.float)


def lpips_evaluation(edited_path,
                     save_path,
                     prompts_path,
                     num_samples,
                     data_path,
                     specific_concept_set,
                     ):
    df_prompts = pd.read_csv(prompts_path)  # read the prompts csv to get correspoding case_number and prompts
    file_names = []
    df_prompts['lpips_loss'] = df_prompts['case_number'] * 0  # initialise lpips column in df
    meta_concept = {}
    for index, row in df_prompts.iterrows():
        if row.concept not in specific_concept_set:
            continue
        case_number = row.case_number
        save_path_tem = os.path.join(
            edited_path,
            row.concept,
            f"{case_number}" + "_{}.png",
        )
        data_path_tem = os.path.join(
            data_path,
            row.concept,
            f"{case_number}" + "_{}.png",
        )
        lpips_scores = []
        for i in range(0, num_samples):
            file = save_path_tem.format(i)
            file_ori = data_path_tem.format(i)
            # print(file)
            # read both the files (original image to compare with and the edited image)
            edited = image_loader(os.path.join(file))
            original = image_loader(os.path.join(file_ori))
            # calculate lpips
            l = loss_fn_alex(original, edited)
            # print(f'LPIPS score: {l.item()}')
            lpips_scores.append(l.item())
        if row.concept not in meta_concept:
            meta_concept[row.concept] = []
        meta_concept[row.concept].append(np.mean(lpips_scores))
        df_prompts.loc[index, 'lpips_loss'] = np.mean(lpips_scores)
    for concept in meta_concept:
        meta_concept[concept] = np.mean(meta_concept[concept])
    meta_path = os.path.join(save_path, 'lpipsloss.json')
    with open(meta_path, "w") as f:
        json.dump(meta_concept, f)
    if save_path is not None:
        df_prompts.to_csv(os.path.join(save_path, 'lpipsloss.csv'))


def compute_generated_LPIPS(args):
    data_path_format = args.data_path
    image_concept = args.image_concept
    # model_name = args.edited_path.split('/')[-1]
    edited_path_format = args.edited_path_format
    prompts_path = args.prompts_path
    num_samples = args.num_samples
    if args.edited_concept_path is not None:
        df = pd.read_csv(args.edited_concept_path)
    else:
        df = None
    model_method = args.model_method
    save_path_format = args.save_path_format
    image_concept_path = args.image_concept_path
    invert_concept_path = args.invert_concept_path
    invert_concept_list = []
    specific_concept_path = args.specific_concept_path
    specific_concept_set = set()
    if specific_concept_path is not None:
        with open(specific_concept_path, "r") as f:
            for line in f:
                specific_concept_set.add(line.strip())

    if invert_concept_path is None:
        invert_concept_list = [None]
    else:
        with open(invert_concept_path, "r") as f:
            for line in f:
                invert_concept_list.append(line.strip())
    for invert_concept in invert_concept_list:
        with open(image_concept_path, "r") as f:
            for line in f:
                model_concept = line.strip()
                if df is not None:
                    for _, row in df.iterrows():
                        data_path = data_path_format.format(row.edit_prompt)
                        if specific_concept_path is None:
                            specific_concept_set = set()
                            specific_concept_set.add(model_concept)
                        if model_method == "esd":
                            edited_path_tem = edited_path_format.format(model_concept.replace(" ", ""), model_concept,
                                                                        row.prompt)
                        elif model_method == "sd":
                            edited_path_tem = edited_path_format.format(model_concept,
                                                                        row.prompt, invert_concept)
                        elif model_method == "generate":
                            edited_path_tem = edited_path_format.format(model_concept, model_concept,
                                                                        row.edit_prompt, invert_concept)
                        elif model_method == "generate_esd":
                            edited_path_tem = edited_path_format.format(
                                model_concept.replace("Mickey Mouse", "mickey").replace(" ", ""),
                                row.edit_prompt, invert_concept)

                        elif model_method == "generate_sd":
                            edited_path_tem = edited_path_format.format(model_concept,
                                                                        row.edit_prompt, invert_concept)
                        elif model_method == "generate_tensor":
                            edited_path_tem = edited_path_format.format(model_concept,
                                                                        row.edit_prompt, model_concept)
                        elif model_method == "generate_sd_v1-4":
                            edited_path_tem = edited_path_format.format(
                                row.edit_prompt, invert_concept)
                        else:
                            edited_path_tem = edited_path_format.format(model_concept, "{}", row.prompt)
                        if "{}" in edited_path_tem:
                            for image_concept in specific_concept_set:
                                edited_path_tem_tem = edited_path_tem.format(image_concept)
                                specific_concept_set_tem = set()
                                specific_concept_set_tem.add(image_concept)
                                print(edited_path_tem_tem)
                                lpips_evaluation(edited_path=edited_path_tem_tem,
                                                 save_path=edited_path_tem_tem,
                                                 prompts_path=prompts_path,
                                                 num_samples=num_samples,
                                                 data_path=data_path,
                                                 specific_concept_set=specific_concept_set_tem)
                        else:
                            print(edited_path_tem)
                            lpips_evaluation(edited_path=edited_path_tem,
                                             save_path=edited_path_tem,
                                             prompts_path=prompts_path,
                                             num_samples=num_samples,
                                             data_path=data_path,
                                             specific_concept_set=specific_concept_set)
                else:
                    edited_path_tem = edited_path_format.format(model_concept)
                    lpips_evaluation(edited_path=edited_path_tem,
                                     save_path=edited_path_tem,
                                     prompts_path=prompts_path,
                                     num_samples=num_samples,
                                     data_path=data_path_format,
                                     specific_concept_set=specific_concept_set)


def compute_LPIPS_between_concept(args):
    prompts_path = args.prompts_path
    df = pd.read_csv(prompts_path)
    specific_concept_dict = {}

    for _, row in df.iterrows():
        if row.concept not in specific_concept_dict:
            specific_concept_dict[row.concept] = []
            specific_concept_dict[row.concept].append(row.case_number)
        else:
            specific_concept_dict[row.concept].append(row.case_number)
    num_samples = args.num_samples
    save_path = f"evaluation-outputs/{args.csv_name}/SD3"
    for image_concept in specific_concept_dict:

        df_prompts = pd.read_csv(prompts_path)  # read the prompts csv to get correspoding case_number and prompts
        file_names = []
        df_prompts['lpips_loss'] = df_prompts['case_number'] * 0  # initialise lpips column in df
        meta_concept = {}
        for index, row in df_prompts.iterrows():
            case_number = row.case_number
            data_case_number = specific_concept_dict[image_concept][case_number % args.concept_num]
            save_path_tem = os.path.join(save_path, image_concept, f"{data_case_number}" + "_{}.png")
            # print(f"case_number:{case_number},data_case_number:{data_case_number}")
            data_path_tem = os.path.join(
                save_path,
                row.concept,
                f"{case_number}" + "_{}.png",
            )
            lpips_scores = []
            for i in range(0, num_samples):
                file = save_path_tem.format(i)
                file_ori = data_path_tem.format(i)
                # print(file)
                # read both the files (original image to compare with and the edited image)
                edited = image_loader(os.path.join(file))
                original = image_loader(os.path.join(file_ori))
                # calculate lpips
                l = loss_fn_alex(original, edited)
                # print(f'LPIPS score: {l.item()}')
                lpips_scores.append(l.item())
            if row.concept not in meta_concept:
                meta_concept[row.concept] = []
            meta_concept[row.concept].append(np.mean(lpips_scores))
            df_prompts.loc[index, 'lpips_loss'] = np.mean(lpips_scores)
        for concept in meta_concept:
            meta_concept[concept] = np.mean(meta_concept[concept])
        meta_path = os.path.join(save_path, f'lpipsloss_{image_concept}.json')
        with open(meta_path, "w") as f:
            json.dump(meta_concept, f)
        if save_path is not None:
            df_prompts.to_csv(os.path.join(save_path, f'lpipsloss_{image_concept}.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='LPIPS',
        description='Takes the path to two images and gives LPIPS')
    parser.add_argument("--edited_path", help='path to edited image', type=str, required=False)
    parser.add_argument("--edited_path_format", help='path to edited image', type=str, required=False)
    parser.add_argument("--edited_concept_path", type=str, required=False)
    parser.add_argument("--prompts_path", help='path to csv prompts', type=str, required=False)
    parser.add_argument("--save_path_format", help='path to save results', type=str, required=False, default=None)
    parser.add_argument("--num_samples", help='path to save results', type=int, required=True, default=None)
    parser.add_argument("--data_path", type=str, required=True, default=None)
    parser.add_argument("--image_concept", type=str, required=False, default=None)
    parser.add_argument("--image_concept_path", type=str, required=False, default=None)
    parser.add_argument("--model_method", type=str, required=False, default=None)
    parser.add_argument("--invert_concept_path", type=str, required=False, default=None)
    parser.add_argument("--specific_concept_path", type=str, required=False, default=None)
    parser.add_argument("--method", type=str, required=False, default=None)
    parser.add_argument("--csv_name", type=str, required=False, default=None)
    parser.add_argument("--concept_num", type=int, required=False, default=None)
    loss_fn_alex = lpips.LPIPS(net='alex')
    args = parser.parse_args()
    if args.method == "test_concept":
        compute_LPIPS_between_concept(args)
    else:
        compute_generated_LPIPS(args)
