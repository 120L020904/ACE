import json
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd


def compute_and_save_result(score, save_path, name):
    score = np.array(score)
    score_mean = np.mean(score)
    result = {"mean_score": score_mean}
    os.makedirs(save_path, exist_ok=True)
    result_json_path = os.path.join(save_path, name.split("/")[-1])
    with open(result_json_path, "w") as fp:
        json.dump(result, fp)


def compute_result_image_concept_SD3(args):
    image_concept_path = "data/concept_text/IP_character_concept_10.txt"
    output_path = "evaluation-outputs/cartoon_eval_test/SD3/"
    save_path = "evaluation-outputs/cartoon_eval_test_result/SD3/"
    json_path = os.path.join(output_path, "evaluation_results(concept only).json")
    clip_score = []
    with open(json_path, "r") as f:
        json_data = json.load(f)
    with open(image_concept_path, "r") as f:
        for line in f:
            concept = line.strip()
            clip_score.append(json_data[0][concept])
    compute_and_save_result(clip_score, save_path, "evaluation_results_image_concept_clip_10_concept.json")


def compute_result_edit_concept_SD3(args):
    df = pd.read_csv(args.edit_prompt_path)
    clip_score = []
    concept_text_path = "data/concept_text/concept_input_3.txt"
    save_path = "evaluation-outputs/cartoon_eval_test_result/SD3/"
    path = "evaluation-outputs/cartoon_eval_test/SD3/evaluation_results_{}_{}.json"
    concept_list = []
    if concept_text_path is not None:
        with open(concept_text_path, "r") as f_concept:
            for line_concept in f_concept:
                concept_list.append(line_concept.strip())
    else:
        concept_list.append(None)
    for image_concept in concept_list:
        for _, row in df.iterrows():
            with open(path.format(row.concept, image_concept), "r") as f:
                json_data = json.load(f)
                clip_score.append(json_data[0][row.concept])
    compute_and_save_result(clip_score, save_path, "evaluation_results_edit_clip_3_concept.json")
    return


def compute_influence_result_erase(args):
    folder_name = args.csv_name + args.add_name + "_result"
    is_one_concept = args.is_one_concept
    # spm_model_name_format = [
    #     "SD-v1-4_{}_{}_num_inversion_50_method_LEDITS_edit_guidance_10.0_skip_0.1_edit_threshold_0.9_is_DDIMinversion_True_inversion_prompt_True_use_mask"]
    spm_model_name_format = args.model_name_format
    spm_image_concept_path = args.image_concept_path
    generate_concept_path = args.specific_concept_path
    is_edit = args.is_edit
    result_key = "mean_score"
    model_name = args.model_name

    image_concept_list = []
    generate_concept_list = []
    with open(spm_image_concept_path, "r") as f:
        for line in f:
            image_concept_list.append(line.strip())
    with open(generate_concept_path, "r") as f:
        for line in f:
            generate_concept_list.append(line.strip())
    erase_clip = []
    relate_clip = []
    erase_lpips = []
    relate_lpips = []
    for concept in image_concept_list:
        if not is_edit:
            output_path = os.path.join("evaluation-outputs",
                                       folder_name,
                                       spm_model_name_format.format(concept,
                                                                    concept, "{}", "{}"))
        else:
            output_path = os.path.join("evaluation-outputs",
                                       folder_name,
                                       spm_model_name_format.format(concept,
                                                                    "{}", "{}", "{}"))
        if is_one_concept:
            erase_clip_one = []
            relate_clip_one = []
            erase_lpips_one = []
            relate_lpips_one = []
        for generate_concept in generate_concept_list:
            if not is_edit:
                json_path_clip = os.path.join(output_path, f"evaluation_results_{generate_concept}_character_concept.json")
                json_path_lpips = os.path.join(output_path, f"evaluation_results_lpips_{generate_concept}.json")
            else:
                json_path_clip = os.path.join(output_path.format(generate_concept,
                                                                    "{}", "{}"),
                                              f"evaluation_results_{generate_concept}_character_concept.json")
                json_path_lpips = os.path.join(output_path.format(generate_concept,
                                                                    "{}", "{}"), f"evaluation_results_lpips_{generate_concept}.json")
            with open(json_path_clip, "r") as f:
                clip = json.load(f)
                if concept == generate_concept:
                    erase_clip.append(clip[result_key])
                    if is_one_concept:
                        erase_clip_one.append(clip[result_key]/2.5)
                else:
                    relate_clip.append(clip[result_key])
                    if is_one_concept:
                        relate_clip_one.append(clip[result_key]/2.5)
            with open(json_path_lpips, "r") as f:
                lpips = json.load(f)
                if concept == generate_concept:
                    erase_lpips.append(lpips["mean_score"])
                    if is_one_concept:
                        erase_lpips_one.append(lpips["mean_score"])
                else:
                    relate_lpips.append(lpips["mean_score"])
                    if is_one_concept:
                        relate_lpips_one.append(lpips["mean_score"])
        if is_one_concept:
            save_path_one = os.path.join("evaluation-outputs",
                                     folder_name + f"_average_one",concept,
                                     model_name)
            compute_and_save_result(erase_clip_one, save_path=save_path_one, name=f"erase_clip.json")
            compute_and_save_result(relate_clip_one, save_path=save_path_one, name=f"relate_clip.json")
            compute_and_save_result(relate_lpips_one, save_path=save_path_one, name=f"relate_lpips.json")
            compute_and_save_result(erase_lpips_one, save_path=save_path_one, name=f"erase_lpips.json")
    save_path = os.path.join("evaluation-outputs",
                                 folder_name + "_average",
                                 model_name)
    compute_and_save_result(erase_clip, save_path=save_path, name=f"erase_clip.json")
    compute_and_save_result(relate_clip, save_path=save_path, name=f"relate_clip.json")
    compute_and_save_result(relate_lpips, save_path=save_path, name=f"relate_lpips.json")
    compute_and_save_result(erase_lpips, save_path=save_path, name=f"erase_lpips.json")


def compute_influence_result_edit_reverse(args):
    df = pd.read_csv(args.edit_prompt_path)
    model_method = args.model_method
    is_cs = args.is_cs
    is_specific = args.is_specific
    invert_concept_path = args.invert_concept_path
    invert_concept_list = []
    image_concept_list = []
    model_name = args.model_name
    with open(args.image_concept_path, "r") as f:
        for line in f:
            image_concept_list.append(line.strip())

    erase_clip = []
    relate_clip = []
    folder_name = args.csv_name + args.add_name + "_result"
    for image_concept in image_concept_list:
        for _, row in df.iterrows():
            output_path = os.path.join("evaluation-outputs",
                                       args.csv_name + args.add_name,
                                       args.model_name_format.format(row.concept, image_concept
                                                                     ))
            json_path = os.path.join(output_path, f"evaluation_results_{image_concept}.json")
            with open(json_path, "r") as f:
                json_data = json.load(f)
            if image_concept == row.concept:
                erase_clip.append(json_data[0][f"{image_concept}"])
            else:
                relate_clip.append(json_data[0][f"{image_concept}"])
    save_path = os.path.join("evaluation-outputs",
                             folder_name + "_average",
                             model_name)
    compute_and_save_result(erase_clip, save_path=save_path, name=f"{model_name}_erase_clip.json")
    compute_and_save_result(relate_clip, save_path=save_path, name=f"{model_name}_relate_clip.json")


def compute_result_all_concept(args):
    if args.edit_prompt_path is not None:
        df = pd.read_csv(args.edit_prompt_path)
    else:
        df = None
    model_method = args.model_method
    is_lpips = args.is_lpips
    is_cs = args.is_cs
    is_full = args.is_full
    is_specific = args.is_specific
    invert_concept_path = args.invert_concept_path
    invert_concept_list = []
    if invert_concept_path is None:
        invert_concept_list = [None]
    else:
        with open(invert_concept_path, "r") as f:
            for line in f:
                invert_concept_list.append(line.strip())
    image_concept_list = []
    if not is_specific:
        with open(args.specific_concept_path, "r") as f:
            for line in f:
                image_concept_list.append(line.strip())

    with open(args.image_concept_path, "r") as f:
        for line in f:
            model_concept = line.strip()
            if is_specific:
                image_concept_list = []
                image_concept_list.append(model_concept)
            for image_concept in image_concept_list:
                for invert_concept in invert_concept_list:
                    clip_score = []
                    clip_score_edit = []
                    lpips_score = []
                    if df is not None:
                        for _, row in df.iterrows():
                            if model_method == "esd":
                                output_path = os.path.join("evaluation-outputs",
                                                           args.csv_name + args.add_name,
                                                           args.model_name_format.format(model_concept.replace(" ", ""),
                                                                                         model_concept, row.prompt))
                            elif model_method == "sd":
                                output_path = os.path.join("evaluation-outputs",
                                                           args.csv_name + args.add_name,
                                                           args.model_name_format.format(model_concept,
                                                                                         row.prompt))

                            elif model_method == "generate":
                                output_path = os.path.join("evaluation-outputs",
                                                           args.csv_name + args.add_name,
                                                           args.model_name_format.format(model_concept,
                                                                                         model_concept,
                                                                                         row.edit_prompt,
                                                                                         invert_concept))
                            elif model_method == "generate_tensor":
                                output_path = os.path.join("evaluation-outputs",
                                                           args.csv_name + args.add_name,
                                                           args.model_name_format.format(model_concept,
                                                                                         row.edit_prompt,
                                                                                         model_concept))
                            elif model_method == "generate_sd":
                                output_path = os.path.join("evaluation-outputs",
                                                           args.csv_name + args.add_name,
                                                           args.model_name_format.format(model_concept, row.edit_prompt,
                                                                                         invert_concept))
                            elif model_method == "generate_esd":
                                output_path = os.path.join("evaluation-outputs",
                                                           args.csv_name + args.add_name,
                                                           args.model_name_format.format(
                                                               model_concept.replace("Mickey Mouse", "mickey"),
                                                               row.edit_prompt,
                                                               invert_concept))
                            else:
                                output_path = os.path.join("evaluation-outputs",
                                                           args.csv_name + args.add_name,
                                                           args.model_name_format.format(model_concept,
                                                                                         image_concept, row.prompt))
                            print(output_path)

                            if is_cs:
                                if not args.clip_path_change:
                                    json_path = os.path.join(output_path, f"evaluation_results_{image_concept}.json")
                                else:
                                    json_path = os.path.join(output_path,
                                                             f"evaluation_results_clip_None_image_{image_concept}.json")
                            elif is_lpips:
                                json_path = os.path.join(output_path, "lpipsloss.json")
                            elif is_full:
                                if not args.clip_path_change:
                                    json_clip_path = os.path.join(output_path, f"evaluation_results(concept only).json")
                                    json_edit_clip_path = os.path.join(output_path,
                                                                       "evaluation_results_{}_{}.json".format(
                                                                           row.concept,
                                                                           model_concept))
                                else:
                                    json_clip_path = os.path.join(output_path,
                                                                  f"evaluation_results_clip_None_image_{image_concept}.json")
                                    json_edit_clip_path = os.path.join(output_path,
                                                                       f"evaluation_results_clip_{row.concept}_image_{image_concept}.json")

                                json_lpips_path = os.path.join(output_path, "lpipsloss.json")
                            else:
                                json_path = os.path.join(output_path,
                                                         "evaluation_results_{}_{}.json".format(row.concept,
                                                                                                model_concept))
                            if is_full:
                                with open(json_clip_path, "r") as f:
                                    json_clip_data = json.load(f)
                                with open(json_edit_clip_path, "r") as f:
                                    json_edit_clip_data = json.load(f)
                                with open(json_lpips_path, "r") as f:
                                    json_lpips_data = json.load(f)
                                clip_score.append(json_clip_data[0][f"{image_concept}"])
                                clip_score_edit.append(json_edit_clip_data[0][row.concept])
                                lpips_score.append(json_lpips_data[image_concept])
                            else:
                                with open(json_path, "r") as f:
                                    json_data = json.load(f)
                                    if is_cs:
                                        clip_score.append(json_data[0][f"{image_concept}"])
                                    elif is_lpips:
                                        clip_score.append(json_data[f"{image_concept}"])
                                    else:
                                        clip_score.append(json_data[0][f"{row.concept}"])
                    else:
                        output_path = os.path.join("evaluation-outputs",
                                                   args.csv_name + args.add_name,
                                                   args.model_name_format.format(model_concept))
                        if is_cs:
                            if not args.clip_path_change:
                                json_path = os.path.join(output_path, f"evaluation_results_{image_concept}.json")
                            else:
                                json_path = os.path.join(output_path,
                                                         f"evaluation_results_clip_None_image_{image_concept}.json")
                        elif is_lpips:
                            json_path = os.path.join(output_path, "lpipsloss.json")
                        elif is_full:
                            if not args.clip_path_change:
                                json_clip_path = os.path.join(output_path, f"evaluation_results_{image_concept}.json")
                            else:
                                json_clip_path = os.path.join(output_path,
                                                              f"evaluation_results_clip_None_image_{image_concept}.json")

                            json_edit_clip_path = os.path.join(output_path,
                                                               "evaluation_results_{}_{}.json".format(row.concept,
                                                                                                      model_concept))
                            json_lpips_path = os.path.join(output_path, "lpipsloss.json")
                        else:
                            json_path = os.path.join(output_path,
                                                     "evaluation_results_{}_{}.json".format(row.concept, model_concept))
                        if is_full:
                            with open(json_clip_path, "r") as f:
                                json_clip_data = json.load(f)
                            with open(json_edit_clip_path, "r") as f:
                                json_edit_clip_data = json.load(f)
                            with open(json_lpips_path, "r") as f:
                                json_lpips_data = json.load(f)
                            clip_score.append(json_clip_data[0][f"{model_concept}"])
                            clip_score_edit.append(json_edit_clip_data[0][row.concept])
                            lpips_score.append(json_lpips_data[model_concept])
                        else:
                            with open(json_path, "r") as f:
                                json_data = json.load(f)
                                if is_cs:
                                    clip_score.append(json_data[0][f"{image_concept}"])
                                elif is_lpips:
                                    clip_score.append(json_data[f"{image_concept}"])
                                else:
                                    clip_score.append(json_data[0][f"{row.concept}"])

                if is_cs:
                    format_path = os.path.join("evaluation-outputs",
                                               args.csv_name + args.add_name + "_result",
                                               args.model_name_format.format(model_concept, model_concept, "{}", "{}"))
                    compute_and_save_result(clip_score, save_path=format_path,
                                            name="evaluation_results_{}_character_concept.json".format(image_concept))
                elif is_lpips:
                    format_path = os.path.join("evaluation-outputs",
                                               args.csv_name + args.add_name + "_result",
                                               args.model_name_format.format(model_concept, model_concept, "{}", "{}"))
                    compute_and_save_result(clip_score, save_path=format_path,
                                            name="evaluation_results_lpips_{}.json".format(image_concept))
                elif is_full:
                    format_path = os.path.join("evaluation-outputs",
                                               args.csv_name + args.add_name + "_result",
                                               args.model_name_format.format(model_concept, image_concept, "{}", "{}"))
                    compute_and_save_result(lpips_score, save_path=format_path,
                                            name="evaluation_results_lpips_{}.json".format(image_concept))
                    compute_and_save_result(clip_score, save_path=format_path,
                                            name="evaluation_results_{}_character_concept.json".format(image_concept))
                    compute_and_save_result(clip_score_edit, save_path=format_path,
                                            name="evaluation_results_{}_edit.json".format(image_concept))
                else:
                    compute_and_save_result(clip_score, save_path=format_path,
                                            name="evaluation_results_{}.json".format(model_concept))


def compute_final_result(args):
    folder_name = args.csv_name + args.add_name + "_result"
    # spm_model_name_format = [
    #     "SD-v1-4_{}_{}_num_inversion_50_method_LEDITS_edit_guidance_10.0_skip_0.1_edit_threshold_0.9_is_DDIMinversion_True_inversion_prompt_True_use_mask"]
    model_name_format = args.model_name_format
    image_concept_path = args.image_concept_path
    is_specific = args.is_specific
    edit_concept_clip_erase = []
    edit_concept_clip_relate = []
    image_concept_clip_erase = []
    image_concept_clip_relate = []
    image_lpips_erase = []
    image_lpips_relate = []
    result_key = "mean_score"
    model_name = args.model_name
    edit = not args.is_cs
    j = 0
    concept_set = set()
    with open(image_concept_path, "r") as f:
        for line in f:
            concept = line.strip()

            concept_set.add(concept)
    for model_concept in concept_set:
        for image_concept in concept_set:
            if not is_specific:
                output_path = os.path.join("evaluation-outputs",
                                           folder_name,
                                           model_name_format.format(model_concept,
                                                                    image_concept, "{}", "{}"))
            else:
                output_path = os.path.join("evaluation-outputs",
                                           folder_name,
                                           model_name_format.format(model_concept,
                                                                    model_concept, "{}", "{}"))
                image_concept = model_concept
            print(output_path)
            j += 1
            print(j)
            json_path_edit_clip = os.path.join(output_path, f"evaluation_results_{image_concept}_edit.json")
            json_path_clip = os.path.join(output_path, f"evaluation_results_{image_concept}_character_concept.json")
            json_path_lpips = os.path.join(output_path, f"evaluation_results_lpips_{image_concept}.json")
            if model_concept == image_concept:
                if edit:
                    with open(json_path_edit_clip, "r") as f:
                        edit_clip = json.load(f)
                        edit_concept_clip_erase.append(edit_clip[result_key])

                    with open(json_path_lpips, "r") as f:
                        lpips = json.load(f)
                        image_lpips_erase.append(lpips[result_key])
                with open(json_path_clip, "r") as f:
                    clip = json.load(f)
                    image_concept_clip_erase.append(clip[result_key])
            else:
                if edit:
                    with open(json_path_edit_clip, "r") as f:
                        edit_clip = json.load(f)
                        edit_concept_clip_relate.append(edit_clip[result_key])

                    with open(json_path_lpips, "r") as f:
                        lpips = json.load(f)
                        image_lpips_relate.append(lpips[result_key])
                with open(json_path_clip, "r") as f:
                    clip = json.load(f)
                    image_concept_clip_relate.append(clip[result_key])
    save_path = os.path.join("evaluation-outputs",
                             folder_name + "_average",
                             model_name)
    if edit:
        compute_and_save_result(edit_concept_clip_erase, save_path=save_path, name=f"{model_name}_edit_erase_clip.json")
        compute_and_save_result(edit_concept_clip_relate, save_path=save_path,
                                name=f"{model_name}_edit_relate_clip.json")
        compute_and_save_result(image_lpips_erase, save_path=save_path, name=f"{model_name}_lpips_erase.json")
        compute_and_save_result(image_lpips_relate, save_path=save_path, name=f"{model_name}_lpips_relate.json")
    compute_and_save_result(image_concept_clip_erase, save_path=save_path,
                            name=f"{model_name}_image_concept_clip_erase.json")
    compute_and_save_result(image_concept_clip_relate, save_path=save_path,
                            name=f"{model_name}_image_concept_clip_relate.json")
    print(save_path)
    return


def compute_nudity(args):
    save_folder = args.save_folder
    save_folder_name_list = os.listdir(save_folder)
    output_path = args.output_path
    nudity_num = {}

    for save_folder_name in save_folder_name_list:

        json_path = os.path.join(args.save_folder, save_folder_name, "evaluation_results.json")
        with open(json_path, "r") as f:
            nudity_num_tem = json.load(f)
        for key in nudity_num_tem:
            if key in nudity_num:
                nudity_num[key] += nudity_num_tem[key]
            else:
                nudity_num[key] = nudity_num_tem[key]
            print(os.path.join(args.save_folder, save_folder_name))
    os.makedirs(output_path, exist_ok=True)
    json_output_path = os.path.join(output_path, "result.json")
    for key in nudity_num_tem:
        nudity_num[key] = nudity_num[key] / len(save_folder_name_list)
    with open(json_output_path, "w") as f:
        json.dump(nudity_num, f)

    return


def compute_fid(args):
    save_folder = args.save_folder
    specific_concept_path = args.image_concept_path
    specific_concept_list = []
    with open(specific_concept_path, "r") as f:
        for line in f:
            specific_concept_list.append(line.strip())
    output_path = args.output_path + "_average"
    is_cs = args.is_cs
    fid_list = []
    clip_list = []
    for specific_concept in specific_concept_list:
        json_path = os.path.join(save_folder.format(specific_concept), f"fid_{specific_concept}.json")
        with open(json_path, "r") as f:
            fid = json.load(f)
        if is_cs:
            clip_json_path = os.path.join(save_folder.format(specific_concept), f"evaluation_results.json")
            with open(clip_json_path,"r") as f:
                clip_score = json.load(f)
            clip_list.append(clip_score["mean_score"])
        fid_list.append(fid[specific_concept])
    compute_and_save_result(fid_list, output_path, "fid_result.json")
    if is_cs:
        compute_and_save_result(clip_list, output_path, "clip_result.json")


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
        "--is_lpips",
        action='store_true',
        help='',
        required=False,
        default=False
    )
    parser.add_argument(
        "--is_cs",
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
        "--model_name",
        type=str,
        required=False,
        default="",
        help="Specify the concept"
    )
    parser.add_argument(
        "--model_method",
        type=str,
        required=False,
        default="",
        help=""
    )
    parser.add_argument(
        "--invert_concept_path",
        type=str,
        required=False,
        default=None,
        help=""
    )
    parser.add_argument(
        "--clip_path_change",
        action='store_true',
        help='',
        required=False,
        default=False
    )
    parser.add_argument(
        "--is_final",
        action='store_true',
        help='',
        required=False,
        default=False
    )
    parser.add_argument(
        "--is_specific",
        action='store_true',
        help='',
        required=False,
        default=False
    )
    parser.add_argument(
        "--is_one_concept",
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
    parser.add_argument(
        "--method",
        type=str,
        required=False,
        default=None,
        help=""
    )
    parser.add_argument(
        "--specific_concept_path",
        type=str,
        required=False,
        default=None,
        help=""
    )
    args = parser.parse_args()
    # compute_result_all_concept(args)
    if args.method == "final":
        compute_final_result(args)
    elif args.method == "edit_reverse":
        compute_influence_result_edit_reverse(args)
    elif args.method == "erase_influence":
        compute_influence_result_erase(args)
    elif args.method == "compute_nudity":
        compute_nudity(args)
    elif args.method == "compute_fid":
        compute_fid(args)
    else:
        compute_result_all_concept(args)
    # compute_result_image_concept_SD3(args)
    # compute_result_edit_concept_SD3(args)
