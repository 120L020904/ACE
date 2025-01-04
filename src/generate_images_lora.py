import gc
import sys
from typing import List, Union, Literal
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import LMSDiscreteScheduler
import torch
from PIL import Image
import pandas as pd
import argparse
import os
sys.path.append("/home/yxwei/wangzihao/ACE")
from utils.figure_grid import merge_images
from models.merge_ace import load_state_dict
from models.ace import ACELayer, ACENetwork
from src.eval.evaluation.eval_util import clip_score, create_meta_json
from src.eval.evaluation.clip_evaluator import ClipEvaluator
import train_util

RES = [8, 16, 32, 64]
MATCHING_METRICS = Literal[
    "clipcos",
    "clipcos_tokenuni",
    "tokenuni",
]


def calculate_matching_score(
        prompt_tokens,
        prompt_embeds,
        erased_prompt_tokens,
        erased_prompt_embeds,
        matching_metric: MATCHING_METRICS,
        special_token_ids: set,
        weight_dtype: torch.dtype = torch.float32,
):
    scores = []
    if "clipcos" in matching_metric:
        clipcos = torch.cosine_similarity(
            prompt_embeds.flatten(1, 2),
            erased_prompt_embeds.flatten(1, 2),
            dim=-1).cpu()
        scores.append(clipcos)
    if "tokenuni" in matching_metric:
        prompt_set = set(prompt_tokens[0].tolist()) - special_token_ids
        tokenuni = []
        for ep in erased_prompt_tokens:
            ep_set = set(ep.tolist()) - special_token_ids
            tokenuni.append(len(prompt_set.intersection(ep_set)) / len(ep_set))
        scores.append(torch.tensor(tokenuni).to("cpu", dtype=weight_dtype))
    return torch.max(torch.stack(scores), dim=0)[0]


def clip_score_text(images: List[Union[torch.Tensor, np.ndarray, Image.Image, str]],
                    ablated_texts: List[str],
                    w: float = 2.5,
                    clip_model: str = "ViT-B/32",
                    n_px: int = 224, ):
    score_result = []
    for text in ablated_texts:
        ablated_clip_score = clip_score(images, text, w, clip_model, n_px)
        score = np.mean(ablated_clip_score).item()
        score_result.append(score)
    return score_result


def get_images(latents, vae):
    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


def save_images(pil_images,
                folder_path,
                case_number,
                concept,
                prompts_path,
                is_coco):
    attn_score_erased = None
    # attn_score_eval = None
    attn_score_list = []
    for num, im in enumerate(pil_images):
        if not is_coco:
            os.makedirs(f"{folder_path}/{concept}", exist_ok=True)
            im.save(f"{folder_path}/{concept}/{case_number}_{num}.png")
        else:
            os.makedirs(f"{folder_path}", exist_ok=True)
            im.save(f"{folder_path}/{case_number}_{num}.png")
    return attn_score_list, attn_score_erased


def flush():
    torch.cuda.empty_cache()
    gc.collect()




@torch.no_grad()
def generate_images(model_name,
                    prompts_path,
                    save_path,
                    device='cuda:0',
                    guidance_scale=7.5,
                    image_size=512,
                    ddim_steps=100,
                    num_samples=10,
                    from_case=0,
                    is_lora=True,
                    aligned_multipliers=None,
                    need_mid_image=False,
                    check_rate=0.5,
                    erased_concept=None,
                    lora_rank=4,
                    matching_metric: MATCHING_METRICS = "clipcos_tokenuni",
                    edit_concept_path=None,
                    specific_concept=None,
                    specific_concept_set=None,
                    is_Mace=False,
                    lora_path=None,
                    cab_path=None,
                    is_specific=True,
                    lora_name=None,
                    tensor_name=None,
                    is_coco=False,
                    test_unet=False,
                    model_path="CompVis/stable-diffusion-v1-4",
                    is_textencoder=False,
                    model_weight_path=None):
    '''
    Function to generate images from diffusers code

    The program requires the prompts to be in a csv format with headers
        1. 'case_number' (used for file naming of image)
        2. 'prompt' (the prompt used to generate image)
        3. 'seed' (the inital seed to generate gaussion noise for diffusion input)

    Parameters
    ----------
    model_name : list
        name of the model to load.
    prompts_path : str
        path for the csv file with prompts and corresponding seeds.
    save_path : str
        save directory for images.
    device : str, optional
        device to be used to load the model. The default is 'cuda:0'.
    guidance_scale : float, optional
        guidance value for inference. The default is 7.5.
    image_size : int, optional
        image size. The default is 512.
    ddim_steps : int, optional
        number of denoising steps. The default is 100.
    num_samples : int, optional
        number of samples generated per prompt. The default is 10.
    from_case : int, optional
        The starting offset in csv to generate images. The default is 0.

    Returns
    -------
    None.

    '''
    if not is_Mace:
        dir_ = model_path
        # dir_ = "CompVis/stable-diffusion-v1-4"  # all the erasure models built on SDv1-4
    else:
        dir_ = ""
    weight_dtype = torch.float32

    # 1. Load the autoencoder model which will be used to decode the latents into image space.
    vae = AutoencoderKL.from_pretrained(dir_, subfolder="vae")
    # 2. Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained(dir_, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(dir_, subfolder="text_encoder")
    # 3. The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained(dir_, subfolder="unet")

    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                     num_train_timesteps=1000)
    special_token_ids = set(tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map.values()))
    vae.to(device)
    text_encoder.to(device)
    unet.to(device)
    torch_device = device
    if is_specific:
        specific_concept_set = set().add(specific_concept)
    print(specific_concept_set)
    if is_lora:
        if lora_name is not None:

            if tensor_name is None:
                spm_paths = [f"models/{lora_name}/{lora_name}_last/{lora_name}_last.safetensors"]
            else:
                spm_paths = [f"models/{lora_name}/{tensor_name}/{lora_name}_last/{lora_name}_last.safetensors"]
        else:
            spm_paths = [lora_path]
        used_multipliers = []
        network = ACENetwork(
            unet,
            rank=lora_rank,
            alpha=1.0,
            module=ACELayer,
        ).to(device, dtype=weight_dtype)
        spms, metadatas = zip(*[
            load_state_dict(spm_model_path, weight_dtype) for spm_model_path in spm_paths
        ])

        erased_prompts = [md["prompts"].split(",") for md in metadatas]
        erased_prompts_count = [len(ep) for ep in erased_prompts]
        erased_prompts_flatten = [item for sublist in erased_prompts for item in sublist]
        erased_prompt_embeds, erased_prompt_tokens = train_util.encode_prompts(
            tokenizer, text_encoder, erased_prompts_flatten, return_tokens=True
        )
    else:
        erased_prompts = erased_concept
        weighted_spm = None
        used_multipliers = []
        network = None
        erased_prompts_count = None
        if 'SD-v1-4' not in model_name[0] and not is_Mace and cab_path is None and "MACE" not in model_name[0]:
            try:
                if not is_textencoder:
                    model_path_tem = model_weight_path
                    if not test_unet:
                        unet.load_state_dict(torch.load(model_path_tem))
                    else:
                        print("test_unet")
                        unet = torch.load(model_path_tem)
                        unet.to(device)
                else:
                    text_encoder.load_state_dict(torch.load(model_weight_path, map_location=device), strict=False)
            except Exception as e:
                print(f'Model path is not valid, please check the file name and structure: {e}')
                exit()
    df = pd.read_csv(prompts_path)
    name_path = ''
    concept_set = set()
    for name in model_name:
        if name_path == '':
            name_path = name_path + name
        else:
            name_path = name_path + '-' + name

    print(f"erased_prompts is {erased_prompts}")
    attn_score = {}
    edit_concept_set = set()
    if edit_concept_path != None:
        with open(edit_concept_path, "r") as f:
            for line in f:
                edit_concept_set.add(line.strip())
    else:
        edit_concept_set.add(None)
    for edit_concept in edit_concept_set:
        folder_path = f'{save_path}/{name_path}_edit_{edit_concept}/{tensor_name}'
        os.makedirs(folder_path, exist_ok=True)
        print(folder_path)
        for index, row in df.iterrows():
            if index < from_case:
                continue
            if edit_concept != None:
                prompt_str = row.prompt.format(edit_concept)
            else:
                prompt_str = row.prompt
            if not is_coco:
                if row.concept not in specific_concept_set:
                    continue

            prompt = [str(prompt_str)] * num_samples

            seed = row.evaluation_seed
            case_number = row.case_number
            if is_lora:
                weighted_spm = dict.fromkeys(spms[0].keys())
                prompt_embeds, prompt_tokens = train_util.encode_prompts(
                    tokenizer, text_encoder, [row.prompt], return_tokens=True
                )
                if aligned_multipliers is not None:
                    multipliers = torch.tensor(aligned_multipliers).to("cpu", dtype=weight_dtype)
                    if multipliers == [0, 0, 0]:
                        matching_metric = "aazeros"
                    elif multipliers == [1, 1, 1]:
                        matching_metric = "zzone"
                else:
                    multipliers = calculate_matching_score(
                        prompt_tokens,
                        prompt_embeds,
                        erased_prompt_tokens,
                        erased_prompt_embeds,
                        matching_metric=matching_metric,
                        special_token_ids=special_token_ids,
                        weight_dtype=weight_dtype
                    )
                    multipliers = torch.split(multipliers, erased_prompts_count)
                spm_multipliers = torch.tensor(multipliers).to("cpu", dtype=weight_dtype)
                for spm, multiplier in zip(spms, spm_multipliers):
                    max_multiplier = torch.max(multiplier)
                    for key, value in spm.items():
                        if weighted_spm[key] is None:
                            weighted_spm[key] = value * max_multiplier
                        else:
                            weighted_spm[key] += value * max_multiplier
                    used_multipliers.append(max_multiplier.item())
                network.load_state_dict(weighted_spm)

            if not is_coco:
                concept = row.concept
                concept_set.add(concept)
            else:
                concept = None


            height = image_size  # default height of Stable Diffusion
            width = image_size  # default width of Stable Diffusion

            num_inference_steps = ddim_steps  # Number of denoising steps
            if hasattr(row,"evaluation_guidance"):
                guidance_scale = row.evaluation_guidance
            else:
                guidance_scale = guidance_scale  # Scale for classifier-free guidance

            generator = torch.manual_seed(seed)  # Seed generator to create the inital latent noise

            batch_size = len(prompt)

            text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length,
                                   truncation=True,
                                   return_tensors="pt")

            text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

            max_length = text_input.input_ids.shape[-1]
            uncond_input = tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
            # print(f"text_embeddings size is {text_embeddings.shape}")
            latents = torch.randn(
                (batch_size, unet.in_channels, height // 8, width // 8),
                generator=generator,
            )
            latents = latents.to(torch_device)

            scheduler.set_timesteps(num_inference_steps)

            latents = latents * scheduler.init_noise_sigma

            from tqdm.auto import tqdm

            scheduler.set_timesteps(num_inference_steps)
            step = 0
            pbar = tqdm(scheduler.timesteps)
            for t in pbar:
                pbar.set_postfix({"guidance": guidance_scale})
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

                # predict the noise residual
                with torch.no_grad():
                    if is_lora:
                        with network:
                            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                    else:
                        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = scheduler.step(noise_pred, t, latents).prev_sample
                # print(f"latents size is {latents.shape}")
                step += 1
                if step == int(check_rate * num_inference_steps) and need_mid_image:
                    pil_mid_images = get_images(latents, vae)
                    folder_mid_path = os.path.join(folder_path, f"mid_{check_rate}")
                    os.makedirs(folder_mid_path, exist_ok=True)
                    save_images(pil_images=pil_mid_images,
                                folder_path=folder_mid_path,
                                case_number=case_number,
                                concept=concept,
                                prompts_path=prompts_path,
                                is_coco=is_coco)

                    del pil_mid_images
            pil_images = get_images(latents, vae)
            save_images(pil_images=pil_images,
                        folder_path=folder_path,
                        case_number=case_number,
                        concept=concept,
                        prompts_path=prompts_path,
                        is_coco=is_coco)

            del latents
        if not is_coco:
            for concept in concept_set:
                if concept not in specific_concept_set:
                    continue
                image_dir_path = f"{folder_path}/{concept}/"  # 图片集地址
                image_save_path = f"{folder_path}/{concept}" + "_{}.png"
                merge_images(image_dir_path, image_save_path, 256, 5, 5)
            for image_concept in specific_concept_set:
                create_meta_json(csv_df=df, save_folder=folder_path, num_samples=num_samples,
                                 image_concept=image_concept)
                evaluator = ClipEvaluator(
                    save_folder=folder_path, output_path=folder_path, image_concept=image_concept
                )
                evaluator.evaluation()
    flush()

def main(args):
    model_name = args.model_name
    prompts_path = args.prompts_path
    save_path = args.save_path
    device = args.device
    guidance_scale = args.guidance_scale
    image_size = args.image_size
    ddim_steps = args.ddim_steps
    num_samples = args.num_samples
    from_case = args.from_case
    is_lora = args.is_lora

    test_unet = args.test_unet
    need_mid_image = args.need_mid_image
    check_rate = args.check_rate
    erased_concept = args.erased_concept
    lora_rank = args.lora_rank
    edit_concept_path = args.edit_concept_path
    specific_concept = args.specific_concept
    specific_concept_path = args.specific_concept_path
    is_Mace = args.is_Mace
    is_specific = args.is_specific
    conf_path = args.fuse_lora_config_path
    lora_path = args.lora_path
    model_path = args.model_path
    model_concept_path = args.model_concept_path
    lora_name = args.lora_name
    tensor_name = args.tensor_name
    is_coco = args.is_coco
    is_SD = args.is_SD
    is_text_encoder = args.is_text_encoder
    model_weight_path = args.model_weight_path
    specific_concept_set = set()
    generate_concept_set = set()
    generate_concept_path = args.generate_concept_path
    if specific_concept_path is not None:
        with open(specific_concept_path, "r") as concepts:
            for concept in concepts:
                specific_concept_set.add(concept.strip())
    else:
        specific_concept_set.add(None)
    if generate_concept_path is None:
        generate_concept_set = specific_concept_set
    else:
        with open(generate_concept_path, "r") as concepts:
            for concept in concepts:
                generate_concept_set.add(concept.strip())
    if model_concept_path is not None:
        df_model = pd.read_csv(model_concept_path)
        model_name_dict = {}
        for _, row in df_model.iterrows():
            if row.concept in specific_concept_set:
                model_name_dict[row.concept] = row.prompt
    else:
        model_name_dict = None
    for concept in specific_concept_set:
        lora_path_tem = None
        lora_name_tem = None
        tensor_name_tem = None
        if is_lora:
            model_name_tem = model_name[0].format(concept)
            if model_concept_path is None:
                if lora_path is None:
                    lora_name_tem = lora_name.format(concept)
                    if tensor_name is not None:
                        tensor_name_tem = tensor_name.format(concept)
                else:
                    lora_path_tem = lora_path.format(concept, concept, concept)
                print(lora_path_tem, lora_name_tem)
            else:
                model_prompt = model_name_dict[concept]
                if lora_path is None:
                    lora_name_tem = lora_name.format(concept)
                    if tensor_name is not None:
                        tensor_name_tem = tensor_name.format(concept)
                else:
                    lora_path_tem = lora_path.format(model_prompt, model_prompt, model_prompt)
        elif model_concept_path is not None:
            model_name_tem = model_name[0].format(model_name_dict[concept])
        elif is_SD:
            model_name_tem = model_name[0]
        else:
            model_name_tem = model_name[0].format(concept)
        generate_images([model_name_tem],
                        prompts_path,
                        save_path,
                        device=device,
                        guidance_scale=guidance_scale,
                        image_size=image_size,
                        ddim_steps=ddim_steps,
                        num_samples=num_samples,
                        from_case=from_case,
                        is_lora=is_lora,
                        aligned_multipliers=args.multipliers,
                        need_mid_image=need_mid_image,
                        check_rate=check_rate,
                        erased_concept=erased_concept,
                        lora_rank=lora_rank,
                        edit_concept_path=edit_concept_path,
                        specific_concept=specific_concept,
                        lora_path=lora_path_tem,
                        specific_concept_set=generate_concept_set,
                        is_Mace=is_Mace,
                        is_specific=is_specific,
                        lora_name=lora_name_tem,
                        tensor_name=tensor_name_tem,
                        is_coco=is_coco,
                        test_unet=test_unet,
                        model_path=model_path,
                        is_textencoder=is_text_encoder,
                        model_weight_path=model_weight_path)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='generateImages',
        description='Generate Images using Diffusers Code')
    parser.add_argument('--model_name', help='name of model', nargs='+', type=str, required=False)
    parser.add_argument('--prompts_path', help='path to csv file with prompts', type=str, required=False)
    parser.add_argument('--save_path', help='folder where to save images', type=str, required=False)
    parser.add_argument('--device', help='cuda device to run on', type=str, required=False, default='cuda:0')
    parser.add_argument('--guidance_scale', help='guidance to run eval', type=float, required=False, default=7.5)
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
    parser.add_argument('--from_case', help='continue generating from case_number', type=int, required=False, default=0)
    parser.add_argument('--num_samples', help='number of samples per prompt', type=int, required=False, default=1)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False,
                        default=100)
    parser.add_argument('--from_ckpt', action='store_true', help='whether get pretrained model from ckpt 500',
                        required=False, default=False)
    parser.add_argument('--need_attention_map', action='store_true', help='whether obtain maps of generated images',
                        required=False, default=False)
    parser.add_argument('--need_mid_image', action='store_true', help='whether obtain mid of generated images',
                        required=False, default=False)
    parser.add_argument('--is_lora', action='store_true', help='whether use lora',
                        required=False, default=False)
    parser.add_argument('--is_Mace', action='store_true', help='whether use lora',
                        required=False, default=False)
    parser.add_argument('--is_specific', action='store_true',
                        required=False, default=False)
    parser.add_argument('--is_SD', action='store_true',
                        required=False, default=False)
    parser.add_argument('--is_text_encoder', action='store_true',
                        required=False, default=False)
    parser.add_argument('--is_coco', action='store_true',
                        required=False, default=False)
    parser.add_argument('--test_unet', action='store_true',
                        required=False, default=False)
    parser.add_argument('--check_rate', help='check rate of generate images', type=float, required=False, default=0.5)
    parser.add_argument('--multipliers', help='coefficient of spm', nargs='*', type=float, required=False)
    parser.add_argument('--erased_concept', nargs='*', type=str, required=False)
    parser.add_argument('--lora_rank', help='lora rank of model used to train', type=int, required=False, default=4)
    parser.add_argument('--edit_concept_path', type=str, required=False, default=None)
    parser.add_argument('--specific_concept', type=str, required=False, default=None)
    parser.add_argument('--specific_concept_path', type=str, required=False, default=None)
    parser.add_argument('--fuse_lora_config_path', type=str, required=False, default=None)
    parser.add_argument('--lora_path', type=str, required=False, default=None)
    parser.add_argument('--lora_name', type=str, required=False, default=None)
    parser.add_argument('--model_path', type=str, required=False, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument('--model_concept_path', type=str, required=False, default=None)
    parser.add_argument('--cab_path', type=str, required=False)
    parser.add_argument('--model_weight_path', type=str, required=False)
    parser.add_argument('--tensor_name', type=str, required=False)
    parser.add_argument('--generate_concept_path', type=str, required=False)
    args = parser.parse_args()
    main(args)
