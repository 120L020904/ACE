import argparse
import gc
import json
import os
import pathlib
import sys
from typing import Union, List, BinaryIO, Optional, Dict, Callable, Any
import torch.nn.functional as F
from torchvision.io import read_image

import numpy as np
import pandas as pd
import torch
from diffusers.pipelines.ledits_pp.pipeline_output import LEditsPPDiffusionPipelineOutput
from diffusers.pipelines.ledits_pp.pipeline_leditspp_stable_diffusion import LeditsGaussianSmoothing, \
    LeditsAttentionStore, rescale_noise_cfg

sys.path.append("/home/yxwei/wangzihao/ACE")
from src.models.ace import ACENetwork, ACELayer
from utils.figure_grid import merge_images
from PIL import Image
from torchvision.utils import make_grid, _log_api_usage_once

from src.eval.evaluation.eval_util import create_meta_json
from masactrl.masactrl import MutualSelfAttentionControl
from diffusers import LEditsPPPipelineStableDiffusion, DDIMScheduler, LEditsPPPipelineStableDiffusionXL, \
    StableDiffusionPipeline, StableDiffusionInpaintPipeline
from diffusers.utils import load_image
from masactrl.diffuser_utils import MasaCtrlPipeline
from masactrl.masactrl_utils import regiter_attention_editor_diffusers
from models.merge_ace import load_state_dict
from src.eval.evaluation.clip_evaluator import ClipEvaluator

device = 'cuda:0'


def flush():
    torch.cuda.empty_cache()
    gc.collect()


def extract_text_encoder_ckpt(ckpt_path):
    full_ckpt = torch.load(ckpt_path)
    new_ckpt = {}
    for key in full_ckpt.keys():
        if 'text_encoder.text_model' in key:
            new_ckpt[key.replace("text_encoder.", "")] = full_ckpt[key]
    return new_ckpt


class Inversion:
    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample

    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(
            timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample

    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else self.guidance_scale
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(self.num_ddim_steps):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents

    def __init__(self, model, guidance_scale, num_ddim_steps):
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(num_ddim_steps)
        self.num_ddim_steps = num_ddim_steps
        self.guidance_scale = guidance_scale
        self.prompt = None
        self.context = None


def load_image_mc(image_path, device):
    image = read_image(image_path)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    image = F.interpolate(image, (512, 512))
    image = image.to(device)
    return image


def save_image(
        tensor: Union[torch.Tensor, List[torch.Tensor]],
        fp: Union[str, pathlib.Path, BinaryIO],
        format: Optional[str] = None,
        **kwargs,
) -> None:
    """
    Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """

    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(save_image)
    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)


@torch.no_grad()
def generate_images(
        pipe,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        editing_prompt: Optional[Union[str, List[str]]] = None,
        editing_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        reverse_editing_direction: Optional[Union[bool, List[bool]]] = False,
        edit_guidance_scale: Optional[Union[float, List[float]]] = 5,
        edit_warmup_steps: Optional[Union[int, List[int]]] = 0,
        edit_cooldown_steps: Optional[Union[int, List[int]]] = None,
        edit_threshold: Optional[Union[float, List[float]]] = 0.9,
        user_mask: Optional[torch.Tensor] = None,
        sem_guidance: Optional[List[torch.Tensor]] = None,
        use_cross_attn_mask: bool = False,
        use_intersect_mask: bool = True,
        sem_embeds_list: bool = False,
        attn_store_steps: Optional[List[int]] = [],
        store_averaged_over_steps: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
):
    if pipe.inversion_steps is None:
        raise ValueError(
            "You need to invert an input image first before calling the pipeline. The `invert` method has to be called beforehand. Edits will always be performed for the last inverted image(s)."
        )

    eta = pipe.eta
    num_images_per_prompt = 1
    latents = pipe.init_latents

    zs = pipe.zs
    pipe.scheduler.set_timesteps(len(pipe.scheduler.timesteps))

    if use_intersect_mask:
        use_cross_attn_mask = True

    if use_cross_attn_mask:
        pipe.smoothing = LeditsGaussianSmoothing(pipe.device)

    if user_mask is not None:
        user_mask = user_mask.to(pipe.device)

    org_prompt = ""

    # 1. Check inputs. Raise error if not correct
    pipe.check_inputs(
        negative_prompt,
        editing_prompt_embeds,
        negative_prompt_embeds,
        callback_on_step_end_tensor_inputs,
    )

    pipe._guidance_rescale = guidance_rescale
    pipe._clip_skip = clip_skip
    pipe._cross_attention_kwargs = cross_attention_kwargs

    # 2. Define call parameters
    batch_size = pipe.batch_size

    if editing_prompt:
        enable_edit_guidance = True
        if isinstance(editing_prompt, str):
            editing_prompt = [editing_prompt]
        pipe.enabled_editing_prompts = len(editing_prompt)
    elif editing_prompt_embeds is not None:
        enable_edit_guidance = True
        pipe.enabled_editing_prompts = editing_prompt_embeds.shape[0]
    else:
        pipe.enabled_editing_prompts = 0
        enable_edit_guidance = False

    # 3. Encode input prompt
    lora_scale = (
        pipe.cross_attention_kwargs.get("scale", None) if pipe.cross_attention_kwargs is not None else None
    )
    if not sem_embeds_list:
        edit_concepts, uncond_embeddings, num_edit_tokens = pipe.encode_prompt(
            editing_prompt=editing_prompt,
            device=pipe.device,
            num_images_per_prompt=num_images_per_prompt,
            enable_edit_guidance=enable_edit_guidance,
            negative_prompt=negative_prompt,
            editing_prompt_embeds=editing_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=pipe.clip_skip,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if enable_edit_guidance:
            text_embeddings = torch.cat([uncond_embeddings, edit_concepts])
            pipe.text_cross_attention_maps = [editing_prompt] if isinstance(editing_prompt, str) else editing_prompt
        else:
            text_embeddings = torch.cat([uncond_embeddings])

    # 4. Prepare timesteps
    # pipe.scheduler.set_timesteps(num_inference_steps, device=pipe.device)
    timesteps = pipe.inversion_steps
    t_to_idx = {int(v): k for k, v in enumerate(timesteps[-zs.shape[0]:])}

    if use_cross_attn_mask:
        pipe.attention_store = LeditsAttentionStore(
            average=store_averaged_over_steps,
            batch_size=batch_size,
            max_size=(latents.shape[-2] / 4.0) * (latents.shape[-1] / 4.0),
            max_resolution=None,
        )
        pipe.prepare_unet(pipe.attention_store, PnP=False)
        resolution = latents.shape[-2:]
        att_res = (int(resolution[0] / 4), int(resolution[1] / 4))

    # 5. Prepare latent variables
    num_channels_latents = pipe.unet.config.in_channels
    latents = pipe.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        None,
        None,
        torch.float32,
        pipe.device,
        latents,
    )

    # 6. Prepare extra step kwargs.
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(eta)

    pipe.sem_guidance = None
    pipe.activation_mask = None

    # 7. Denoising loop
    num_warmup_steps = 0
    with pipe.progress_bar(total=len(timesteps)) as progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            if sem_embeds_list:
                edit_concepts, uncond_embeddings, num_edit_tokens = pipe.encode_prompt(
                    editing_prompt=editing_prompt,
                    device=pipe.device,
                    num_images_per_prompt=num_images_per_prompt,
                    enable_edit_guidance=enable_edit_guidance,
                    negative_prompt=negative_prompt,
                    editing_prompt_embeds=editing_prompt_embeds[i],
                    negative_prompt_embeds=negative_prompt_embeds[i],
                    lora_scale=lora_scale,
                    clip_skip=pipe.clip_skip,
                )
                if enable_edit_guidance:
                    text_embeddings = torch.cat([uncond_embeddings, edit_concepts])
                    pipe.text_cross_attention_maps = [editing_prompt] if isinstance(editing_prompt,
                                                                                    str) else editing_prompt
                else:
                    text_embeddings = torch.cat([uncond_embeddings])
            if enable_edit_guidance:
                latent_model_input = torch.cat([latents] * (1 + pipe.enabled_editing_prompts))
            else:
                latent_model_input = latents

            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

            text_embed_input = text_embeddings

            # predict the noise residual
            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embed_input).sample

            noise_pred_out = noise_pred.chunk(1 + pipe.enabled_editing_prompts)  # [b,4, 64, 64]
            noise_pred_uncond = noise_pred_out[0]
            noise_pred_edit_concepts = noise_pred_out[1:]

            noise_guidance_edit = torch.zeros(
                noise_pred_uncond.shape,
                device=pipe.device,
                dtype=noise_pred_uncond.dtype,
            )

            if sem_guidance is not None and len(sem_guidance) > i:
                noise_guidance_edit += sem_guidance[i].to(pipe.device)

            elif enable_edit_guidance:
                if pipe.activation_mask is None:
                    pipe.activation_mask = torch.zeros(
                        (len(timesteps), len(noise_pred_edit_concepts), *noise_pred_edit_concepts[0].shape)
                    )

                if pipe.sem_guidance is None:
                    pipe.sem_guidance = torch.zeros((len(timesteps), *noise_pred_uncond.shape))

                for c, noise_pred_edit_concept in enumerate(noise_pred_edit_concepts):
                    if isinstance(edit_warmup_steps, list):
                        edit_warmup_steps_c = edit_warmup_steps[c]
                    else:
                        edit_warmup_steps_c = edit_warmup_steps
                    if i < edit_warmup_steps_c:
                        continue

                    if isinstance(edit_guidance_scale, list):
                        edit_guidance_scale_c = edit_guidance_scale[c]
                    else:
                        edit_guidance_scale_c = edit_guidance_scale

                    if isinstance(edit_threshold, list):
                        edit_threshold_c = edit_threshold[c]
                    else:
                        edit_threshold_c = edit_threshold
                    if isinstance(reverse_editing_direction, list):
                        reverse_editing_direction_c = reverse_editing_direction[c]
                    else:
                        reverse_editing_direction_c = reverse_editing_direction

                    if isinstance(edit_cooldown_steps, list):
                        edit_cooldown_steps_c = edit_cooldown_steps[c]
                    elif edit_cooldown_steps is None:
                        edit_cooldown_steps_c = i + 1
                    else:
                        edit_cooldown_steps_c = edit_cooldown_steps

                    if i >= edit_cooldown_steps_c:
                        continue

                    noise_guidance_edit_tmp = noise_pred_edit_concept - noise_pred_uncond

                    if reverse_editing_direction_c:
                        noise_guidance_edit_tmp = noise_guidance_edit_tmp * -1

                    noise_guidance_edit_tmp = noise_guidance_edit_tmp * edit_guidance_scale_c

                    if user_mask is not None:
                        noise_guidance_edit_tmp = noise_guidance_edit_tmp * user_mask

                    if use_cross_attn_mask:
                        out = pipe.attention_store.aggregate_attention(
                            attention_maps=pipe.attention_store.step_store,
                            prompts=pipe.text_cross_attention_maps,
                            res=att_res,
                            from_where=["up", "down"],
                            is_cross=True,
                            select=pipe.text_cross_attention_maps.index(editing_prompt[c]),
                        )
                        attn_map = out[:, :, :, 1: 1 + num_edit_tokens[c]]  # 0 -> startoftext

                        # average over all tokens
                        if attn_map.shape[3] != num_edit_tokens[c]:
                            raise ValueError(
                                f"Incorrect shape of attention_map. Expected size {num_edit_tokens[c]}, but found {attn_map.shape[3]}!"
                            )

                        attn_map = torch.sum(attn_map, dim=3)

                        # gaussian_smoothing
                        attn_map = F.pad(attn_map.unsqueeze(1), (1, 1, 1, 1), mode="reflect")
                        attn_map = pipe.smoothing(attn_map).squeeze(1)

                        # torch.quantile function expects float32
                        if attn_map.dtype == torch.float32:
                            tmp = torch.quantile(attn_map.flatten(start_dim=1), edit_threshold_c, dim=1)
                        else:
                            tmp = torch.quantile(
                                attn_map.flatten(start_dim=1).to(torch.float32), edit_threshold_c, dim=1
                            ).to(attn_map.dtype)
                        attn_mask = torch.where(
                            attn_map >= tmp.unsqueeze(1).unsqueeze(1).repeat(1, *att_res), 1.0, 0.0
                        )

                        # resolution must match latent space dimension
                        attn_mask = F.interpolate(
                            attn_mask.unsqueeze(1),
                            noise_guidance_edit_tmp.shape[-2:],  # 64,64
                        ).repeat(1, 4, 1, 1)
                        pipe.activation_mask[i, c] = attn_mask.detach().cpu()
                        if not use_intersect_mask:
                            noise_guidance_edit_tmp = noise_guidance_edit_tmp * attn_mask

                    if use_intersect_mask:
                        if t <= 800:
                            noise_guidance_edit_tmp_quantile = torch.abs(noise_guidance_edit_tmp)
                            noise_guidance_edit_tmp_quantile = torch.sum(
                                noise_guidance_edit_tmp_quantile, dim=1, keepdim=True
                            )
                            noise_guidance_edit_tmp_quantile = noise_guidance_edit_tmp_quantile.repeat(
                                1, pipe.unet.config.in_channels, 1, 1
                            )

                            # torch.quantile function expects float32
                            if noise_guidance_edit_tmp_quantile.dtype == torch.float32:
                                tmp = torch.quantile(
                                    noise_guidance_edit_tmp_quantile.flatten(start_dim=2),
                                    edit_threshold_c,
                                    dim=2,
                                    keepdim=False,
                                )
                            else:
                                tmp = torch.quantile(
                                    noise_guidance_edit_tmp_quantile.flatten(start_dim=2).to(torch.float32),
                                    edit_threshold_c,
                                    dim=2,
                                    keepdim=False,
                                ).to(noise_guidance_edit_tmp_quantile.dtype)

                            intersect_mask = (
                                    torch.where(
                                        noise_guidance_edit_tmp_quantile >= tmp[:, :, None, None],
                                        torch.ones_like(noise_guidance_edit_tmp),
                                        torch.zeros_like(noise_guidance_edit_tmp),
                                    )
                                    * attn_mask
                            )

                            pipe.activation_mask[i, c] = intersect_mask.detach().cpu()

                            noise_guidance_edit_tmp = noise_guidance_edit_tmp * intersect_mask

                        else:
                            # print(f"only attention mask for step {i}")
                            noise_guidance_edit_tmp = noise_guidance_edit_tmp * attn_mask

                    elif not use_cross_attn_mask:
                        # calculate quantile
                        noise_guidance_edit_tmp_quantile = torch.abs(noise_guidance_edit_tmp)
                        noise_guidance_edit_tmp_quantile = torch.sum(
                            noise_guidance_edit_tmp_quantile, dim=1, keepdim=True
                        )
                        noise_guidance_edit_tmp_quantile = noise_guidance_edit_tmp_quantile.repeat(1, 4, 1, 1)

                        # torch.quantile function expects float32
                        if noise_guidance_edit_tmp_quantile.dtype == torch.float32:
                            tmp = torch.quantile(
                                noise_guidance_edit_tmp_quantile.flatten(start_dim=2),
                                edit_threshold_c,
                                dim=2,
                                keepdim=False,
                            )
                        else:
                            tmp = torch.quantile(
                                noise_guidance_edit_tmp_quantile.flatten(start_dim=2).to(torch.float32),
                                edit_threshold_c,
                                dim=2,
                                keepdim=False,
                            ).to(noise_guidance_edit_tmp_quantile.dtype)

                        pipe.activation_mask[i, c] = (
                            torch.where(
                                noise_guidance_edit_tmp_quantile >= tmp[:, :, None, None],
                                torch.ones_like(noise_guidance_edit_tmp),
                                torch.zeros_like(noise_guidance_edit_tmp),
                            )
                            .detach()
                            .cpu()
                        )

                        noise_guidance_edit_tmp = torch.where(
                            noise_guidance_edit_tmp_quantile >= tmp[:, :, None, None],
                            noise_guidance_edit_tmp,
                            torch.zeros_like(noise_guidance_edit_tmp),
                        )

                    noise_guidance_edit += noise_guidance_edit_tmp

                pipe.sem_guidance[i] = noise_guidance_edit.detach().cpu()

            noise_pred = noise_pred_uncond + noise_guidance_edit

            if enable_edit_guidance and pipe.guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(
                    noise_pred,
                    noise_pred_edit_concepts.mean(dim=0, keepdim=False),
                    guidance_rescale=pipe.guidance_rescale,
                )

            idx = t_to_idx[int(t)]
            latents = pipe.scheduler.step(
                noise_pred, t, latents, variance_noise=zs[idx], **extra_step_kwargs
            ).prev_sample

            # step callback
            if use_cross_attn_mask:
                store_step = i in attn_store_steps
                pipe.attention_store.between_steps(store_step)

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(pipe, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                # prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                progress_bar.update()

    # 8. Post-processing
    if not output_type == "latent":
        image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False, generator=generator)[
            0
        ]
        image, has_nsfw_concept = pipe.run_safety_checker(image, pipe.device, text_embeddings.dtype)
    else:
        image = latents
        has_nsfw_concept = None

    if has_nsfw_concept is None:
        do_denormalize = [True] * image.shape[0]
    else:
        do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

    image = pipe.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

    # Offload all models
    pipe.maybe_free_model_hooks()

    if not return_dict:
        return (image, has_nsfw_concept)

    return LEditsPPDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


@torch.no_grad()
def eval_edit(model_name,
              prompts_path,
              save_path,
              is_lora,
              reverse_editing_direction,
              num_samples,
              num_inversion_steps,
              is_LEDITS,
              is_Masactrl,
              is_StyleID,
              is_SD_v1_4,
              edit_guidance_scale,
              skip,
              image_size,
              data_path,
              edit_prompt_path=None,
              inversion_prompt=None,
              multipliers=1.0,
              lora_rank=1,
              generate_concept_set=None,
              is_Mace=False,
              lora_path=None,
              cab_path=None,
              is_DDIMinversion=False,
              inversion_guidance=1.0,
              inversion_equal_inference=False,
              use_mask=True,
              tensor_path=None,
              sem_embeds_list=False,
              negative_prompt=None,
              lora_name=None,
              tensor_name=None,
              is_inpainting=False,
              mask_prompt="",
              model_weight_path=None,
              is_text_encoder=False,
              is_specific=False,
              model_path_input=None
              ):
    device = 'cuda:0'
    weight_type = torch.float32
    if is_SD_v1_4 and not is_Mace and not is_inpainting:
        model_path = model_path_input
    elif is_inpainting:
        model_path = "sd-legacy/stable-diffusion-inpainting"
    else:
        model_path = "sd-legacy/stable-diffusion-inpainting"
    print(model_path)
    if is_LEDITS:
        if is_SD_v1_4:
            if is_DDIMinversion:
                scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                          clip_sample=False,
                                          set_alpha_to_one=False, steps_offset=1)
                pipe = LEditsPPPipelineStableDiffusion.from_pretrained(model_path, local_files_only=True,
                                                                       safety_checker=None, scheduler=scheduler)
            else:
                pipe = LEditsPPPipelineStableDiffusion.from_pretrained(model_path, local_files_only=True,
                                                                       safety_checker=None)

        else:
            pipe = LEditsPPPipelineStableDiffusionXL.from_pretrained(
                model_path, local_files_only=True, torch_dtype=weight_type,
            )

        method = "LEDITS" + f"_eg_{edit_guidance_scale}_DDIM_{is_DDIMinversion}_inv_prompt_{inversion_prompt}_mask_{use_mask}_inv_rd_{reverse_editing_direction}"

    elif is_Masactrl:
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                  set_alpha_to_one=False, steps_offset=1)
        pipe = MasaCtrlPipeline.from_pretrained(model_path, scheduler=scheduler, cross_attention_kwargs={"scale": 0.5},
                                                torch_dtype=weight_type)
        method = "Masactrl"
    elif is_inpainting:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_path, torch_dtype=weight_type, safety_checker=None,
            local_files_only=True,
        )
        method = "inpainting"
    else:
        scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler", torch_dtype=weight_type)
        pipe = StableDiffusionPipeline.from_pretrained(model_path,
                                                       scheduler=scheduler,
                                                       torch_dtype=weight_type,
                                                       safety_checker=None,
                                                       local_files_only=True, )

        method = "None" + f"_eg_{edit_guidance_scale}_is_DDIM_{is_DDIMinversion}_inv_prompt_{inversion_prompt}_equ_infer_{inversion_equal_inference}_neg_{negative_prompt}"
    # 3. The UNet model for generating the latents.

    if is_lora:
        if lora_path is None:
            if tensor_name is None:
                spm_paths = [f"models/{lora_name}/{lora_name}_last/{lora_name}_last.safetensors"]
            else:
                spm_paths = [f"models/{lora_name}/{tensor_name}/{lora_name}_last/{lora_name}_last.safetensors"]
        else:
            spm_paths = [lora_path]
        model_name_ = ''
        for i, name in enumerate(model_name):
            if i == 0:
                model_name_ = name
            else:
                model_name_ = model_name_ + '-' + name
        used_multipliers = []
        network = ACENetwork(
            pipe.unet,
            rank=lora_rank,
            alpha=1.0,
            module=ACELayer,
        ).to(device, dtype=weight_type)
        spms, metadatas = zip(*[
            load_state_dict(spm_model_path, weight_type) for spm_model_path in spm_paths
        ])
        erased_prompts = [md["prompts"].split(",") for md in metadatas]
        erased_prompts_count = [len(ep) for ep in erased_prompts]
        weighted_spm = dict.fromkeys(spms[0].keys())
        spm_multipliers = torch.tensor(multipliers).to("cpu", dtype=weight_type)
        for spm, multiplier in zip(spms, spm_multipliers):
            max_multiplier = torch.max(multiplier)
            for key, value in spm.items():
                if weighted_spm[key] is None:
                    # print(key)
                    weighted_spm[key] = value * max_multiplier
                else:
                    weighted_spm[key] += value * max_multiplier
            used_multipliers.append(max_multiplier.item())
        network.load_state_dict(weighted_spm)
    else:
        model_name_ = model_name[0]
        spms = None
        erased_prompts = None
        erased_prompts = None
        weighted_spm = None
        used_multipliers = []
        network = None
        if 'SD' not in model_name_ and not is_Mace and cab_path is None and "MACE" not in model_name_:
            try:
                print(model_weight_path)
                if is_text_encoder:
                    pipe.text_encoder.load_state_dict(extract_text_encoder_ckpt(model_weight_path), strict=False)
                else:
                    pipe.unet.load_state_dict(torch.load(model_weight_path))
            except Exception as e:
                print(f'Model path is not valid, please check the file name and structure: {e}')
                exit()
    pipe = pipe.to(device)
    df = pd.read_csv(prompts_path)
    folder_path = f'{save_path}/{model_name_}'
    if tensor_path is not None:
        tensor_dict = torch.load(tensor_path)
        row_edit_list = [tensor_dict]
    elif edit_prompt_path is not None:
        df_edit = pd.read_csv(edit_prompt_path)
        row_edit_list = [row_edit for _, row_edit in df_edit.iterrows()]
    else:
        row_edit_list = [None]
    for specific_concept in generate_concept_set:
        concepts = set()

        for row_edit in row_edit_list:
            if is_inpainting:
                edit_flag = False
            if tensor_path is not None:
                edit_prompt_embeds = row_edit["embedding"]
                negative_prompt_embeds = row_edit["surrogate_embedding"]
                edit_prompt = None
            else:
                edit_prompt_embeds = None
                negative_prompt_embeds = None
                if row_edit is not None:
                    if is_DDIMinversion:
                        edit_prompt = row_edit.edit_prompt
                    else:
                        edit_prompt = row_edit.prompt
                    edit_concept = row_edit.concept
                else:
                    edit_prompt = None
                    edit_concept = None
            for _, row in df.iterrows():
                prompt = [str(row.prompt)]
                seed = row.evaluation_seed
                generator = torch.manual_seed(seed)
                concept = row.concept
                if specific_concept is not None and concept != specific_concept:
                    continue
                if is_inpainting and concept != row_edit.concept:
                    continue
                else:
                    edit_flag = True
                concepts.add(concept)
                case_number = row.case_number
                if len(edit_prompt) > 50:
                    edit_prompt_print = edit_prompt[:50]
                else:
                    edit_prompt_print = edit_prompt
                dir_path = f"{folder_path}/{specific_concept}_{edit_prompt_print}_inv_num_{num_inversion_steps}_inv_g_{inversion_guidance}_specific_{is_specific}_mask_{mask_prompt}_eg_{edit_guidance_scale}_method_{method}"
                print(dir_path)
                for i in range(num_samples):
                    image = load_image(f"{data_path}/{concept}/{case_number}_{i}.png")
                    image = image.resize((image_size, image_size))
                    if is_LEDITS:

                        if is_DDIMinversion:
                            if "{}" in row.prompt:
                                edit_prompt_tem = row.prompt.format(edit_prompt)
                            else:
                                edit_prompt_tem = edit_prompt
                            print(f"edit prompt:{edit_prompt_tem}, inversion prompt {inversion_prompt}")
                            image = np.array(image)
                            nullInversion = Inversion(pipe, guidance_scale=inversion_guidance,
                                                      num_ddim_steps=num_inversion_steps)
                            if inversion_prompt is None:
                                nullInversion.init_prompt(row.prompt.format(""))
                                print(row.prompt.format(""))
                            else:
                                nullInversion.init_prompt(inversion_prompt)
                            if is_lora:
                                with network:
                                    _, latents_list = nullInversion.ddim_inversion(image)
                            else:
                                _, latents_list = nullInversion.ddim_inversion(image)
                            init_latent = latents_list.pop()
                            pipe.eta = 0
                            pipe.scheduler.config.timestep_spacing = "leading"
                            pipe.scheduler.set_timesteps(int(num_inversion_steps * (1 + skip)))
                            pipe.init_latents = init_latent
                            pipe.inversion_steps = pipe.scheduler.timesteps[-num_inversion_steps:]
                            zs = torch.zeros(size=(num_inversion_steps, *init_latent.shape), device=device,
                                             dtype=weight_type)
                            pipe.zs = zs.flip(0)
                            pipe.batch_size = 1
                        else:
                            edit_prompt_tem = edit_prompt
                            if is_lora:
                                with network:
                                    _ = pipe.invert(image=image,
                                                    num_inversion_steps=num_inversion_steps,
                                                    skip=0.1,
                                                    source_prompt=row.prompt,
                                                    generator=generator,
                                                    source_guidance_scale=inversion_guidance)
                            else:
                                _ = pipe.invert(image=image,
                                                num_inversion_steps=num_inversion_steps,
                                                skip=0.1,
                                                source_prompt=row.prompt,
                                                generator=generator,
                                                source_guidance_scale=inversion_guidance)
                        if is_lora:
                            with network:
                                edited_image = generate_images(
                                    pipe=pipe,
                                    editing_prompt=[edit_prompt_tem],
                                    edit_guidance_scale=edit_guidance_scale,
                                    editing_prompt_embeds=edit_prompt_embeds,
                                    negative_prompt_embeds=negative_prompt_embeds,
                                    edit_threshold=0.9,
                                    generator=generator,
                                    reverse_editing_direction=reverse_editing_direction,
                                    use_cross_attn_mask=use_mask,
                                    height=image_size,
                                    width=image_size,
                                    use_intersect_mask=use_mask,
                                    sem_embeds_list=sem_embeds_list,
                                    negative_prompt=negative_prompt
                                ).images[0]
                        else:
                            edited_image = generate_images(
                                pipe=pipe,
                                editing_prompt=[edit_prompt_tem],
                                edit_guidance_scale=edit_guidance_scale,
                                editing_prompt_embeds=edit_prompt_embeds,
                                negative_prompt_embeds=negative_prompt_embeds,
                                edit_threshold=0.9,
                                generator=generator,
                                reverse_editing_direction=reverse_editing_direction,
                                use_cross_attn_mask=use_mask,
                                height=image_size,
                                width=image_size,
                                use_intersect_mask=use_mask,
                                sem_embeds_list=sem_embeds_list,
                                negative_prompt=negative_prompt
                            ).images[0]

                    elif is_Masactrl:
                        image_tem = load_image_mc(f"{data_path}/{concept}/{case_number}_{i}.png", device=device)
                        prompts = [row.prompt, f"a photo of {concept} {edit_prompt}"]
                        print(prompts)
                        # inference the synthesized image with MasaCtrl
                        STEP = 4
                        LAYPER = 10
                        if is_lora:
                            with network:
                                start_code, latents_list = pipe.invert(image_tem,
                                                                       row.prompt,
                                                                       guidance_scale=inversion_guidance,
                                                                       num_inference_steps=num_inversion_steps,
                                                                       return_intermediates=True)
                        else:
                            start_code, latents_list = pipe.invert(image_tem,
                                                                   row.prompt,
                                                                   guidance_scale=inversion_guidance,
                                                                   num_inference_steps=num_inversion_steps,
                                                                   return_intermediates=True)
                        start_code = start_code.expand(len(prompts), -1, -1, -1)
                        start_code = start_code.to(device)
                        # hijack the attention module
                        editor = MutualSelfAttentionControl(STEP, LAYPER, total_steps=30)
                        regiter_attention_editor_diffusers(pipe, editor)
                        # inference the synthesized image
                        if is_lora:
                            with network:
                                edited_image_tensor = pipe(prompts, num_inference_steps=num_inversion_steps,
                                                           latents=start_code, guidance_scale=edit_guidance_scale,
                                                           generator=generator)[-1:]
                        else:
                            edited_image_tensor = pipe(prompts, num_inference_steps=num_inversion_steps,
                                                       latents=start_code, guidance_scale=edit_guidance_scale,
                                                       generator=generator)[-1:]
                        grid = make_grid(edited_image_tensor)
                        # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
                        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                        edited_image = Image.fromarray(ndarr)
                    elif is_inpainting:
                        mask_image = load_image(
                            f"{data_path}_mask_box_0.4_text_{mask_prompt}/{concept}/{case_number}_{i}.png")
                        mask_image = mask_image.resize((image_size, image_size))
                        print(edit_prompt)
                        if is_lora:
                            with network:
                                edited_image = pipe(prompt=edit_prompt, image=image, mask_image=mask_image,
                                                    guidance_scale=edit_guidance_scale, generator=generator).images[0]
                        else:
                            edited_image = pipe(prompt=edit_prompt, image=image, mask_image=mask_image,
                                                guidance_scale=edit_guidance_scale, generator=generator).images[0]
                        edited_image_np = np.array(edited_image)
                        mask_image_np = np.array(mask_image)
                        image_np = np.array(image)
                        edited_image_np_final = mask_image_np / 255 * edited_image_np + (
                                    1 - mask_image_np / 255) * image_np
                        edited_image = Image.fromarray(np.uint8(edited_image_np_final))
                    else:
                        if is_DDIMinversion:
                            image = np.array(image)
                            nullInversion = Inversion(pipe, guidance_scale=1.0, num_ddim_steps=num_inversion_steps)
                            if "{}" in row.prompt:
                                edit_prompt_tem = row.prompt.format(edit_prompt)
                            else:
                                edit_prompt_tem = edit_prompt
                            if inversion_prompt is None and not inversion_equal_inference:
                                nullInversion.init_prompt(row.prompt.format(""))
                                print(f"inversion prompt: {row.prompt.format('')}")
                            elif inversion_equal_inference:
                                nullInversion.init_prompt(edit_prompt)
                                print(f"inversion prompt: {edit_prompt}")
                            else:
                                nullInversion.init_prompt(inversion_prompt)
                            print(f"edit prompt:{edit_prompt_tem}, inversion prompt {inversion_prompt}")
                            _, latents_list = nullInversion.ddim_inversion(image)
                            init_latent = latents_list.pop()
                        else:
                            edit_prompt_tem = edit_prompt
                            init_latent = None
                        edited_image = pipe(prompt=edit_prompt_tem,
                                            height=image_size,
                                            width=image_size,
                                            num_inference_steps=num_inversion_steps,
                                            guidance_scale=edit_guidance_scale,
                                            latents=init_latent).images[0]
                    if not is_StyleID:
                        os.makedirs(f"{dir_path}/{concept}", exist_ok=True)
                        edited_image.save(
                            f"{dir_path}/{concept}/{case_number}_{i}.png")
            if is_inpainting:
                continue
            for concept in concepts:
                image_dir_path = f"{dir_path}/{concept}/"  # 图片集地址
                image_save_path = os.path.join(dir_path, concept + "_{}.png")
                image_size_grid = 256  # 每张小图片的大小
                image_colnum = 5  # 合并成一张图后，一行有几个小图
                merge_images(os.path.join(image_dir_path), image_save_path, image_size_grid, image_colnum, num_samples)
            create_meta_json(csv_df=df,
                             save_folder=f"{dir_path}",
                             num_samples=num_samples,
                             image_concept=specific_concept)
            evaluator = ClipEvaluator(
                save_folder=os.path.join(dir_path),
                output_path=dir_path,
                image_concept=specific_concept
            )
            evaluator.evaluation()
            if tensor_path is None:
                create_meta_json(csv_df=df,
                                 save_folder=f"{dir_path}",
                                 num_samples=num_samples,
                                 concept=edit_concept,
                                 image_concept=specific_concept)
                evaluator = ClipEvaluator(
                    save_folder=os.path.join(dir_path),
                    output_path=dir_path,
                    given_concept=edit_concept,
                    image_concept=specific_concept
                )
                evaluator.evaluation()

    del pipe
    flush()
    return


def main(args):
    model_name = args.model_name
    prompts_path = args.prompts_path
    save_path = args.save_path
    is_lora = args.is_lora
    reverse_editing_direction = args.reverse_editing_direction
    num_samples = args.num_samples
    num_inversion_steps = args.num_inversion_steps
    is_LEDITS = args.is_LEDITS
    is_Masactrl = args.is_Masactrl
    is_SD_v1_4 = args.is_SD_v1_4
    edit_guidance_scale = args.edit_guidance_scale
    is_Mace = args.is_Mace
    skip = args.skip
    image_size = args.image_size
    is_StyleID = args.is_StyleID
    data_path = args.data_path
    multipliers = args.multipliers
    lora_rank = args.lora_rank
    specific_concept_path = args.specific_concept_path
    lora_path = args.lora_path
    tensor_path = args.tensor_path
    spm_model_concept_path = args.spm_model_concept_path
    is_DDIMinversion = args.is_DDIMinversion
    edit_prompt_path = args.edit_prompt_path
    inversion_prompt_path = args.inversion_prompt_path
    is_specific = args.is_specific
    inversion_equal_inference = args.inversion_equal_inference
    inversion_guidance = args.inversion_guidance_scale
    lora_name = args.lora_name
    mask_prompt = args.mask_prompt
    tensor_name = args.tensor_name
    is_text_encoder = args.is_text_encoder
    use_mask = args.use_mask
    inversion_prompt_list = []
    generate_concept_set = set()
    cab_path_tem = None
    tensor_path_tem = None
    is_list = args.is_list
    negative_prompt = args.negative_prompt
    generate_concept_path = args.generate_concept_path
    model_weight_path = args.model_weight_path
    model_path_input = args.model_path
    is_inpainting = args.is_inpainting
    model_weight_path_tem = None
    if generate_concept_path is not None:
        with open(generate_concept_path, "r") as f:
            for line in f:
                generate_concept_set.add(line.strip())
    if inversion_prompt_path is None:
        inversion_prompt_list = [None]
    else:
        with open(inversion_prompt_path, "r") as f:
            for line in f:
                inversion_prompt = line.strip()
                inversion_prompt_list.append(inversion_prompt)
    specific_concept_list = []
    if specific_concept_path is not None:
        with open(specific_concept_path, "r") as f:
            for line in f:
                specific_concept_list.append(line.strip())
    else:
        specific_concept_list.append(None)
    for inversion_prompt in inversion_prompt_list:
        for specific_concept in specific_concept_list:
            lora_name_tem = None
            tensor_name_tem = None
            lora_path_tem = None
            if "SD-v1-4" in model_name[0] or ("MACE" in model_name[0] and not is_Mace):
                model_name_tem = model_name[0].format(specific_concept)
                if tensor_path is not None:
                    tensor_path_tem = tensor_path.format(specific_concept, specific_concept)
                print("SD-v1-4")
            elif is_lora:
                if spm_model_concept_path is not None:
                    model_concept_dict = {}
                    with open(spm_model_concept_path, "r") as f_spm:
                        json_data = json.load(f_spm)
                    model_prompt = json_data[specific_concept]
                    if lora_path is None:
                        lora_name_tem = lora_name.format(specific_concept)
                        if tensor_name is not None:
                            tensor_name_tem = tensor_name.format(specific_concept)
                    else:
                        lora_path_tem = lora_path.format(model_prompt, model_prompt, model_prompt)
                    model_name_tem = model_name[0].format(specific_concept)
                    print(lora_path_tem)
                else:
                    if lora_path is None:
                        lora_name_tem = lora_name.format(specific_concept)
                        if tensor_name is not None:
                            tensor_name_tem = tensor_name.format(specific_concept)
                    else:
                        lora_path_tem = lora_path.format(specific_concept, specific_concept, specific_concept)

                    model_name_tem = model_name[0].format(specific_concept)
                    print(lora_path_tem)
            else:
                model_name_tem = model_name[0].format(specific_concept)
                model_weight_path_tem = model_weight_path.format(specific_concept, specific_concept)
            if is_specific:
                generate_concept_set = set()
                generate_concept_set.add(specific_concept)
            eval_edit(model_name=[model_name_tem],
                      save_path=save_path,
                      is_lora=is_lora,
                      reverse_editing_direction=reverse_editing_direction,
                      num_samples=num_samples,
                      prompts_path=prompts_path,
                      num_inversion_steps=num_inversion_steps,
                      is_LEDITS=is_LEDITS,
                      image_size=image_size,
                      is_Masactrl=is_Masactrl,
                      inversion_guidance=inversion_guidance,
                      is_SD_v1_4=is_SD_v1_4,
                      edit_guidance_scale=edit_guidance_scale,
                      lora_rank=lora_rank,
                      data_path=data_path,
                      multipliers=multipliers,
                      skip=skip,
                      lora_path=lora_path_tem,
                      generate_concept_set=generate_concept_set,
                      is_DDIMinversion=is_DDIMinversion,
                      edit_prompt_path=edit_prompt_path,
                      inversion_prompt=inversion_prompt,
                      inversion_equal_inference=inversion_equal_inference,
                      cab_path=cab_path_tem,
                      use_mask=use_mask,
                      tensor_path=tensor_path_tem,
                      sem_embeds_list=is_list,
                      negative_prompt=negative_prompt,
                      tensor_name=tensor_name_tem,
                      lora_name=lora_name_tem,
                      is_StyleID=is_StyleID,
                      is_Mace=is_Mace,
                      model_weight_path=model_weight_path_tem,
                      is_specific=is_specific,
                      model_path_input=model_path_input,
                      is_text_encoder=is_text_encoder,
                      is_inpainting=is_inpainting,
                      mask_prompt=mask_prompt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='generateImages',
        description='Generate Images using Diffusers Code')
    parser.add_argument('--model_name', help='name of model', type=str, nargs='+', required=True)
    parser.add_argument('--prompts_path', help='path to csv file with prompts', type=str, required=True)
    parser.add_argument('--save_path', help='folder where to save images', type=str, required=True)
    parser.add_argument('--is_lora', action='store_true', help='whether model is lora', required=False, default=False)
    parser.add_argument('--num_samples', help='number of samples per prompt', type=int, required=False, default=1)
    parser.add_argument('--edit_prompt_path', help='edit prompt', type=str, required=False,
                        default=None)
    parser.add_argument('--num_inversion_steps', help='number of samples per prompt', type=int, required=False,
                        default=50)
    parser.add_argument('--is_LEDITS', action='store_true', help='whether use LEDITS++', required=False, default=False)
    parser.add_argument('--is_Masactrl', action='store_true', help='whether use LEDITS++', required=False,
                        default=False)
    parser.add_argument('--is_Mace', action='store_true', required=False, default=False)
    parser.add_argument('--is_StyleID', action='store_true', required=False, default=False)
    parser.add_argument('--is_DDIMinversion', action='store_true', required=False, default=False)
    parser.add_argument('--reverse_editing_direction', action='store_true', help='whether use LEDITS++', required=False,
                        default=False)
    parser.add_argument('--edit_guidance_scale', type=float, required=False, default=10)
    parser.add_argument('--inversion_guidance_scale', type=float, required=False, default=1)
    parser.add_argument('--skip', type=float, required=False, default=0.1)
    parser.add_argument('--is_SD_v1_4', action='store_true', required=False, default=False)
    parser.add_argument('--inversion_equal_inference', action='store_true', required=False, default=False)
    parser.add_argument('--image_size', type=int, required=False, default=512)
    parser.add_argument('--data_path', type=str, required=True, default="")
    parser.add_argument('--multipliers', help='coefficient of spm', nargs='*', type=float, required=True)
    parser.add_argument('--lora_rank', type=int, required=False, default=1)
    parser.add_argument('--specific_concept', type=str, required=False, default=None)
    parser.add_argument('--specific_concept_path', type=str, required=False, default=None)
    parser.add_argument('--fuse_lora_config_path', type=str, required=False, default=None)
    parser.add_argument('--lora_path', type=str, required=False, default=None)
    parser.add_argument('--lora_name', type=str, required=False, default=None)
    parser.add_argument('--tensor_name', type=str, required=False, default=None)
    parser.add_argument('--esd_model_concept_path', type=str, required=False, default=None)
    parser.add_argument('--spm_model_concept_path', type=str, required=False, default=None)
    parser.add_argument('--inversion_prompt_path', type=str, required=False, default=None)
    parser.add_argument('--is_specific', action='store_true', required=False, default=False)
    parser.add_argument('--use_mask', action='store_true', required=False, default=False)
    parser.add_argument('--tensor_path', type=str, required=False, default=None)
    parser.add_argument('--is_list', action='store_true', required=False, default=False)
    parser.add_argument('--negative_prompt', type=str, required=False, default=None)
    parser.add_argument('--content_path', type=str, required=False, default=None)
    parser.add_argument('--generate_concept_path', type=str, required=False, default=None)
    parser.add_argument('--mask_prompt', type=str, required=False, default=None)
    parser.add_argument('--is_inpainting', action='store_true', required=False, default=False)
    parser.add_argument('--model_weight_path', type=str, required=False, default=None)
    parser.add_argument('--model_path', type=str, required=False, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument('--is_text_encoder', action='store_true', required=False, default=False)
    args = parser.parse_args()
    main(args)
