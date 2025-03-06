import sys
from typing import List, Union
sys.path.append("/home/yxwei/wangzihao/ACE")
import numpy as np
from diffusers import DiffusionPipeline, StableDiffusion3Pipeline
import torch
from PIL import Image
import pandas as pd
import argparse
import os
from utils.figure_grid import merge_images
from src.eval.evaluation.eval_util import clip_score

RES = [8, 16, 32, 64]


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
                prompts_path):
    for num, im in enumerate(pil_images):
        if 'nudity' not in prompts_path:
            os.makedirs(f"{folder_path}/{concept}", exist_ok=True)
            im.save(f"{folder_path}/{concept}/{case_number}_{num}.png")
        else:
            os.makedirs(f"{folder_path}", exist_ok=True)
            im.save(f"{folder_path}/{case_number}_{num}.png")
    return


def generate_images(model_name,
                    prompts_path,
                    save_path,
                    device='cuda:0',
                    guidance_scale=7.5,
                    image_size=1024,
                    ddim_steps=100,
                    num_samples=10,
                    from_case=0,
                    is_SD3=True,
                    specific_concept_set=None):
    '''
    Function to generate images from diffusers code

    The program requires the prompts to be in a csv format with headers
        1. 'case_number' (used for file naming of image)
        2. 'prompt' (the prompt used to generate image)
        3. 'seed' (the inital seed to generate gaussion noise for diffusion input)

    Parameters
    ----------
    model_name : str
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

    weight_dtype = torch.float16
    if not is_SD3:
        dir_ = "CompVis/stable-diffusion-v1-4"  # all the erasure models built on SDv1-4
        pipe = DiffusionPipeline.from_pretrained(dir_, torch_dtype=torch.float16,
                                                 use_safetensors=True)
    else:
        dir_ = "stabilityai/stable-diffusion-3-medium-diffusers"
        pipe = StableDiffusion3Pipeline.from_pretrained(dir_, torch_dtype=torch.float16,use_safetensors=True)
    pipe.to(device)

    torch_device = device
    df = pd.read_csv(prompts_path)
    name_path = ''
    concept_set = set()
    for name in model_name:
        if name_path == '':
            name_path = name_path + name
        else:
            name_path = name_path + '-' + name
    folder_path = f'{save_path}/{name_path}'
    os.makedirs(folder_path, exist_ok=True)
    print(folder_path)
    for _, row in df.iterrows():
        prompt = str(row.prompt)
        seed = row.evaluation_seed
        case_number = row.case_number
        if row.concept not in specific_concept_set:
            continue
        if 'nudity' not in prompts_path:
            concept = row.concept
            concept_set.add(concept)
        else:
            concept = None
        if case_number < from_case:
            continue

        num_inference_steps = ddim_steps  # Number of denoising steps

        guidance_scale = guidance_scale  # Scale for classifier-free guidance

        generator = torch.manual_seed(seed)  # Seed generator to create the inital latent noise
        pil_images = pipe(prompt=prompt,
                          num_inference_steps=num_inference_steps,
                          num_images_per_prompt=num_samples,
                          guidance_scale=guidance_scale,
                          generator=generator,
                          height=image_size,
                          width=image_size).images



        save_images(pil_images=pil_images,
                    folder_path=folder_path,
                    case_number=case_number,
                    concept=concept,
                    prompts_path=prompts_path)
        del pil_images

    for concept in concept_set:
        image_dir_path = f"{folder_path}/{concept}/"  # 图片集地址
        image_save_path = f"{folder_path}/{concept}.png"
        merge_images(image_dir_path, image_save_path, 256, 5, 5)

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
    specific_concept_path = args.specific_concept_path
    is_SD3 = args.is_SD3
    specific_concept_set = set()
    if specific_concept_path is not None:
        with open(specific_concept_path, "r") as concepts:
            for concept in concepts:
                specific_concept_set.add(concept.strip())
    else:
        specific_concept_set.add(None)
    generate_images(model_name,
                    prompts_path,
                    save_path,
                    device=device,
                    guidance_scale=guidance_scale,
                    image_size=image_size,
                    ddim_steps=ddim_steps,
                    num_samples=num_samples,
                    from_case=from_case,
                    is_SD3=is_SD3,
                    specific_concept_set=specific_concept_set)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='generateImages',
        description='Generate Images using Diffusers Code')
    parser.add_argument('--model_name', help='name of model', nargs='+', type=str, required=True)
    parser.add_argument('--prompts_path', help='path to csv file with prompts', type=str, required=True)
    parser.add_argument('--save_path', help='folder where to save images', type=str, required=True)
    parser.add_argument('--device', help='cuda device to run on', type=str, required=False, default='cuda:0')
    parser.add_argument('--guidance_scale', help='guidance to run eval', type=float, required=False, default=7.5)
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
    parser.add_argument('--from_case', help='continue generating from case_number', type=int, required=False, default=0)
    parser.add_argument('--num_samples', help='number of samples per prompt', type=int, required=False, default=1)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False,
                        default=100)
    parser.add_argument('--from_ckpt', action='store_true', help='whether get pretrained model from ckpt 500',
                        required=False, default=False)
    parser.add_argument('--is_SD3', action='store_true', help='whether use lora',
                        required=False, default=False)
    parser.add_argument('--specific_concept_path', type=str, required=False, default=None)
    args = parser.parse_args()
    main(args)
