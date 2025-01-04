import argparse
import copy
import gc
import json
import os
import random

import numpy as np
from torch.utils.data import Dataset, DataLoader

from diffusers import StableDiffusionPipeline, UNet2DConditionModel, SchedulerMixin
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import train_util
from models.ace import ACENetwork, ACELayer
from typing import List
import torch.nn.functional as F
from generate_images_lora import get_images
from src.eval.evaluation.eval_util import clip_score


def flush():
    torch.cuda.empty_cache()
    gc.collect()


class AnchorsDataset(Dataset):
    def __init__(self, prompt_path, concept):
        self.anchor_list = []
        with open(prompt_path, "r", encoding='utf-8') as f:
            for anchor in f.readlines():
                anchor_concept = anchor.strip('\n')
                if anchor_concept != concept:
                    self.anchor_list.append(anchor_concept)

    def __len__(self):
        return len(self.anchor_list)

    def __getitem__(self, index):
        anchor = self.anchor_list[index % len(self.anchor_list)]
        return anchor


class InfiniteDataLoader(DataLoader):
    def __iter__(self):
        return self.iter_function()

    def iter_function(self):
        while True:
            for batch in super().__iter__():
                yield batch


def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs


def embeds_prompt(prompt, text_encoder, tokenizer, attention_mask, text_encoder_use_attention_mask=None):
    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_input.input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
        return_dict=False,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds


def get_anchors(unconditional, anchor_prompts, tokenizer, text_encoder):
    return


def sample(target, unconditional, sampling_batch_size, tokenizer=None, text_encoder=None):
    samples = []
    while len(samples) < sampling_batch_size:
        while True:
            # sample from gaussian distribution
            noise = torch.randn_like(target)
            # normalize the noise
            noise = noise / noise.view(-1).norm(dim=-1)
            # compute the similarity
            sim = torch.cosine_similarity(target.view(-1), noise.view(-1), dim=-1)
            # the possibility of accepting the sample = 1 - sim
            if random.random() < 1 - sim:
                break
        scale = random.random() * 0.4 + 0.8
        sample = scale * noise * target.view(-1).norm(dim=-1)
        samples.append(sample)

    samples = [torch.cat([unconditional, s]) for s in samples]
    samples = torch.cat(samples, dim=0)
    return samples


def diffusion_to_get_x_t(
        unet: UNet2DConditionModel,
        scheduler: SchedulerMixin,
        latents: torch.FloatTensor,  # ただのノイズだけのlatents
        text_embeddings: torch.FloatTensor,
        vae,
        concept,
        total_timesteps: int = 1000,
        start_timesteps=0,
        get_t: int = 1000,
        need_total=False,
        need_score=True,
        **kwargs,
):
    # latents_steps = []
    step = 0
    x_t = None
    for timestep in scheduler.timesteps[start_timesteps:total_timesteps]:
        if step is get_t:
            x_t = latents
            if not need_total:
                break
        noise_pred = train_util.predict_noise(
            unet, scheduler, timestep, latents, text_embeddings, **kwargs
        )

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, timestep, latents).prev_sample
        step = step + 1
    if x_t is None:
        x_t = latents
    if need_score:
        images = get_images(latents, vae)
        score = clip_score(images, concept)
        score = (score - 0.5) / (0.55 - 0.5)
        score = np.clip(score, a_min=0, a_max=1)
        score_result = score.item()
        del images
    else:
        score_result = None
    del latents
    # return latents_steps
    return x_t, score_result




def train_esd(prompt,
              start_guidance,
              negative_guidance_scale,
              iterations,
              lr,
              model_path,
              devices,
              seperator=None,
              image_size=512,
              ddim_steps=50,
              weight_dtype=torch.float32,
              batch_size=1,
              change_step_rate=1.0,
              surrogate_concept="",
              lora_rank=4,
              textencoder_lr=5e-05,
              with_prior_preservation=True,
              anchor_prompt_path=None,
              anchor_batch_size=2,
              is_train_null=False,
              no_certain_sur=False,
              tensor_path=None,
              with_prior_latent=False,
              no_cond=False,
              only_sur=False,
              null_weight=0.8,
              pl_weight=0.5,
              surrogate_guidance_scale=3.0,
              surrogate_concept_clip_path=None):
    '''
    Function to train diffusion models to erase concepts from model weights

    Parameters
    ----------
    prompt : str
        The concept to erase from diffusion model (Eg: "Van Gogh").
    train_method : str
        The parameters to train for erasure (ESD-x, ESD-u, full, selfattn).
    start_guidance : float
        Guidance to generate images for training.
    negative_guidance : float
        Guidance to erase the concepts from diffusion model.
    iterations : int
        Number of iterations to train.
    lr : float
        learning rate for fine tuning.
    config_path : str
        config path for compvis diffusion format.
    model_path : str
        path for pre-trained diffusers diffusion model.
    diffusers_config_path : str
        Config path for diffusers unet in json format.
    devices : str
        2 devices used to load the models (Eg: '0,1' will load in cuda:0 and cuda:1).
    seperator : str, optional
        If the prompt has commas can use this to seperate the prompt for individual simulataneous erasures. The default is None.
    image_size : int, optional
        Image size for generated images. The default is 512.
    ddim_steps : int, optional
        Number of diffusion time steps. The default is 50.

    Returns
    -------
    None

    '''
    torch.autograd.set_detect_anomaly(True)
    # PROMPT CLEANING
    word_print = prompt
    surrogate_concept_print = surrogate_concept
    if prompt == 'allartist':
        prompt = "Kelly Mckernan, Thomas Kinkade, Ajin Demi Human, Alena Aenami, Tyler Edlin, Kilian Eng"
    if prompt == 'i2p':
        prompt = "hate, harassment, violence, suffering, humiliation, harm, suicide, sexual, nudity, bodily fluids, blood"
    if prompt == "artifact":
        prompt = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy"

    if seperator is not None:
        words = prompt.split(seperator)
        words = [word.strip() for word in words]
    else:
        words = [prompt]
    print(words)
    print(surrogate_concept)


    ddim_eta = 0
    model = StableDiffusionPipeline.from_pretrained(model_path,
                                                    torch_dtype=torch.float32,
                                                    safety_checker=None,
                                                    local_files_only=True, ).to(devices[1])

    # MODEL TRAINING SETUP

    model.unet.enable_xformers_memory_efficient_attention()
    model.unet.to(devices[0], dtype=weight_dtype)
    model.vae.to(devices[0], dtype=weight_dtype)
    scheduler_ori = copy.deepcopy(model.scheduler)
    scheduler_ori.set_timesteps(1000)

    # set model to eval
    model.unet.requires_grad_(False)
    model.unet.eval()
    lora_alpha = 1.0
    network = ACENetwork(
        model.unet,
        rank=lora_rank,
        multiplier=1.0,
        alpha=lora_alpha,
        module=ACELayer,
    ).to(device=devices[0], dtype=weight_dtype)
    model_metadata = {
        "prompts": ",".join([prompt]),
        "rank": str(lora_rank),
        "alpha": str(lora_alpha),
    }
    model.text_encoder.eval()
    # get lora params
    unet_lora_params = network.prepare_optimizer_params(
        textencoder_lr, lr, lr
    )

    # print(f"unet_train parameters is {(unet_lora_params.parameters())}")
    losses = []
    opt = torch.optim.Adam(unet_lora_params, lr=lr)
    criteria = torch.nn.MSELoss()
    history = []
    if surrogate_concept_clip_path is None:
        is_sc_clip = False
        sc_clip = None
    else:
        with open(surrogate_concept_clip_path, "r") as f:
            sc_clip = json.load(f)
        is_sc_clip = True
    if tensor_path is not None:
        tensor_dict = torch.load(tensor_path)
        tensor_path_split = tensor_path.split("/")
        tensor_name = tensor_path_split[1]
        is_tensor_path = True
        if not only_sur:
            erase_prompt_embeds = tensor_dict["embedding"].to(devices[0])
        sur_prompt_embeds = tensor_dict["surrogate_embedding"].to(devices[0])
        if sur_prompt_embeds.shape[0] == 1:
            if not only_sur:
                erase_prompt_embeds = erase_prompt_embeds.repeat(ddim_steps, 1, 1, 1)
            sur_prompt_embeds = sur_prompt_embeds.repeat(ddim_steps, 1, 1, 1)
    else:
        is_tensor_path = False
        tensor_dict = None
    if with_prior_preservation:
        anchor_dataset = AnchorsDataset(prompt_path=anchor_prompt_path, concept=prompt)
        print(anchor_prompt_path)
        name = f'ACE_lora_{word_print}-sc_{surrogate_concept_print}-ng_{negative_guidance_scale}-iter_{iterations}-lr_{lr}-lora-prior_{anchor_batch_size}_tr_null_{is_train_null}_nc_{no_cond}_no_cer_sur_{no_certain_sur}_tensor_{is_tensor_path}_nw_{null_weight}_pl_{pl_weight}_sg_new_{surrogate_guidance_scale}_is_sc_clip_{is_sc_clip}'
    else:
        anchor_dataset = None
        name = f'ACE_lora_{word_print}-sc_{surrogate_concept_print}-ng_{negative_guidance_scale}-iter_{iterations}-lr_{lr}-lora_tr_null_{is_train_null}_nc_{no_cond}_no_cer_sur_{no_certain_sur}_tensor_{is_tensor_path}_nw_{null_weight}_pl_{pl_weight}_sg_new_{surrogate_guidance_scale}_is_sc_clip_{is_sc_clip}'
    print(name)

    # TRAINING CODE
    pbar = tqdm(range(iterations))
    if with_prior_preservation:
        anchor_dataloader = InfiniteDataLoader(anchor_dataset, anchor_batch_size, shuffle=True)
    else:
        anchor_dataloader = range(iterations)
    for i, data in zip(pbar, anchor_dataloader):
        # model.unet.train()
        word = random.sample(words, 1)[0]
        # get text embeddings for unconditional and conditional prompts
        emb_0 = embeds_prompt([''],
                              text_encoder=model.text_encoder,
                              tokenizer=model.tokenizer,
                              attention_mask=None,
                              text_encoder_use_attention_mask=False
                              )
        emb_s = embeds_prompt([surrogate_concept],
                              text_encoder=model.text_encoder,
                              tokenizer=model.tokenizer,
                              attention_mask=None,
                              text_encoder_use_attention_mask=False
                              )
        emb_p = embeds_prompt(word,
                              text_encoder=model.text_encoder,
                              tokenizer=model.tokenizer,
                              attention_mask=None,
                              text_encoder_use_attention_mask=False
                              )
        # [1, 77, 768]
        emb_n = embeds_prompt(word,
                              text_encoder=model.text_encoder,
                              tokenizer=model.tokenizer,
                              attention_mask=None,
                              text_encoder_use_attention_mask=False
                              )
        # print(emb_n.shape)
        if with_prior_preservation:
            # print(data)
            emb_anchor = embeds_prompt(data,
                                       text_encoder=model.text_encoder,
                                       tokenizer=model.tokenizer,
                                       attention_mask=None,
                                       text_encoder_use_attention_mask=False
                                       )
            emb_anchor = torch.cat([emb_0.repeat(len(data), 1, 1), emb_anchor], dim=0)
            # print(emb_anchor.shape)
        else:
            emb_anchor = None
        opt.zero_grad()

        t_end = torch.randint(int((1 - change_step_rate) * ddim_steps), ddim_steps, (1,), device=devices[0])
        # print(f"change ddim step is {t_end}")
        # time step from 1000 to 0 (0 being good)
        # get init latent
        init_latent = torch.randn(
            (1, 4, 64, 64)
        ).to(devices[0])
        init_latent = init_latent * model.scheduler.init_noise_sigma
        with torch.no_grad():
            # override the _call_ method of CrossAttn block in unet to record attention maps
            # unet_tem = copy.deepcopy(model.unet)
            # unet_tem.eval()
            # unet_tem.requires_grad_(False)

            model.scheduler.set_timesteps(ddim_steps)
            # generate an image with the concept from ESD model
            # emb_p seems to work better instead of emb_0
            # get conditional and unconditional scores from frozen model at time step t and image z
            with network:
                # [1,4,64,64]
                latent_t, _ = diffusion_to_get_x_t(
                    model.unet,
                    model.scheduler,
                    init_latent,
                    train_util.concat_embeddings(
                        emb_0,
                        emb_p,
                        batch_size,
                    ),
                    vae=model.vae,
                    concept=prompt,
                    start_timesteps=0,
                    total_timesteps=ddim_steps,
                    guidance_scale=start_guidance,
                    get_t=t_end,
                    need_total=False,
                    need_score=False
                )
            if is_train_null and with_prior_latent:
                latent_t_prior, _ = diffusion_to_get_x_t(
                    model.unet,
                    model.scheduler,
                    init_latent.repeat(len(data), 1, 1, 1),
                    emb_anchor,
                    vae=model.vae,
                    concept=prompt,
                    start_timesteps=0,
                    total_timesteps=ddim_steps,
                    guidance_scale=start_guidance,
                    get_t=t_end,
                    need_total=False,
                    need_score=False
                )
            # set training timestep
            model.scheduler.set_timesteps(1000)
            current_timestep = model.scheduler.timesteps[
                int(t_end * 1000 / ddim_steps)
            ]
            if with_prior_preservation:
                e_prior_ori = train_util.predict_noise(
                    model.unet,
                    model.scheduler,
                    current_timestep,
                    latent_t.repeat(len(data), 1, 1, 1),
                    emb_anchor,
                    guidance_scale=1,
                ).to(devices[0], dtype=torch.float32)
                if is_train_null and with_prior_latent:
                    e_0_prior_ori = train_util.predict_noise(
                        model.unet,
                        model.scheduler,
                        current_timestep,
                        latent_t_prior,
                        train_util.concat_embeddings(
                            emb_0,
                            emb_0,
                            len(data),
                        ),
                        guidance_scale=1,
                    ).to(devices[0], dtype=torch.float32)
                    e_0_prior_ori.requires_grad = False
                e_prior_ori.requires_grad = False
            else:
                e_prior_ori = None
            if is_train_null and no_certain_sur:
                e_s = None
            else:
                if tensor_path is not None:
                    e_s = train_util.predict_noise(
                        model.unet,
                        scheduler_ori,
                        current_timestep,
                        latent_t,
                        train_util.concat_embeddings(
                            emb_0,
                            sur_prompt_embeds[int(t_end)],
                            batch_size,
                        ),
                        guidance_scale=1,
                    ).to(devices[0], dtype=torch.float32)
                else:
                    e_s = train_util.predict_noise(
                        model.unet,
                        scheduler_ori,
                        current_timestep,
                        latent_t,
                        train_util.concat_embeddings(
                            emb_0,
                            emb_s,
                            batch_size,
                        ),
                        guidance_scale=1,
                    ).to(devices[0], dtype=torch.float32)
            e_0 = train_util.predict_noise(
                model.unet,
                scheduler_ori,
                current_timestep,
                latent_t,
                train_util.concat_embeddings(
                    emb_0,
                    emb_0,
                    batch_size,
                ),
                guidance_scale=1,
            ).to(devices[0], dtype=torch.float32)
            if tensor_path is not None and not only_sur:
                e_p = train_util.predict_noise(
                    model.unet,
                    scheduler_ori,
                    current_timestep,
                    latent_t,
                    train_util.concat_embeddings(
                        emb_0,
                        erase_prompt_embeds[int(t_end)],
                        batch_size,
                    ),
                    guidance_scale=1,
                ).to(devices[0], dtype=torch.float32)
            else:
                e_p = train_util.predict_noise(
                    model.unet,
                    scheduler_ori,
                    current_timestep,
                    latent_t,
                    train_util.concat_embeddings(
                        emb_0,
                        emb_p,
                        batch_size,
                    ),
                    guidance_scale=1,
                ).to(devices[0], dtype=torch.float32)
        # breakpoint()
        # get conditional score from ESD model
        if is_train_null:
            with network:
                e_n_0 = train_util.predict_noise(
                    model.unet,
                    model.scheduler,
                    current_timestep,
                    latent_t,
                    train_util.concat_embeddings(
                        emb_0,
                        emb_0,
                        batch_size,
                    ),
                    guidance_scale=1,
                ).to(devices[0], dtype=torch.float32)
        if not no_cond:
            with network:
                e_n = train_util.predict_noise(
                    model.unet,
                    model.scheduler,
                    current_timestep,
                    latent_t,
                    train_util.concat_embeddings(
                        emb_0,
                        emb_n,
                        batch_size,
                    ),
                    guidance_scale=1,
                ).to(devices[0], dtype=torch.float32)
        if with_prior_preservation:
            with network:
                e_prior = train_util.predict_noise(
                    model.unet,
                    model.scheduler,
                    current_timestep,
                    latent_t.repeat(len(data), 1, 1, 1),
                    emb_anchor,
                    guidance_scale=1,
                ).to(devices[0], dtype=torch.float32)
                if is_train_null and with_prior_latent:
                    e_0_prior = train_util.predict_noise(
                        model.unet,
                        model.scheduler,
                        current_timestep,
                        latent_t_prior,
                        train_util.concat_embeddings(
                            emb_0,
                            emb_0,
                            len(data),
                        ),
                        guidance_scale=1,
                    ).to(devices[0], dtype=torch.float32)

        if e_s is not None:
            e_s.requires_grad = False
        e_p.requires_grad = False

        if is_train_null:
            if no_certain_sur:
                surrogate_guidance = torch.zeros_like(latent_t)
                if is_sc_clip:
                    own_clip = sc_clip[0][prompt]
                for j in range(len(data)):
                    if is_sc_clip:
                        sc_clip_tem = sc_clip[0][data[j]]
                        clip_scale = sc_clip_tem / own_clip
                        surrogate_guidance += clip_scale * (e_prior_ori[j] - e_0)
                    else:
                        surrogate_guidance += e_prior_ori[j] - e_0
            else:
                surrogate_guidance = e_s - e_0
            loss_erase_null = criteria(e_n_0,
                                       e_0 + negative_guidance_scale * (
                                                   e_p - e_0) - surrogate_guidance_scale * surrogate_guidance)

        if not no_cond:
            loss_erase_cond = criteria(e_n,
                                       e_0 - (negative_guidance_scale * (e_p - e_0)))
        if is_train_null:
            if no_cond:
                loss_erase = loss_erase_null
            else:
                loss_erase = null_weight * loss_erase_null + (1 - null_weight) * loss_erase_cond
        else:
            loss_erase = loss_erase_cond
        # reconstruction loss for ESD objective from frozen model and conditional score of ESD model
        # loss = criteria(e_n, e_0) works the best try 5000 epochs

        if with_prior_preservation:

            loss_prior = criteria(e_prior, e_prior_ori)
            if is_train_null and with_prior_latent:
                loss_prior += criteria(e_0_prior, e_0_prior_ori)
            loss = (1 - pl_weight) * loss_erase + pl_weight * loss_prior
        else:
            loss = loss_erase
        # update weights to erase the concept
        loss.backward()
        losses.append(loss.item())
        if is_train_null and with_prior_preservation:
            pbar.set_postfix({"loss": loss.item(),
                              "cond_loss": loss_erase_cond.item(),
                              "null_loss": loss_erase_null.item(),
                              "loss_prior": loss_prior.item()})
        else:
            pbar.set_postfix({"loss": loss.item()})
        history.append(loss.item())
        opt.step()
        if with_prior_preservation:
            del latent_t, e_prior, e_prior_ori, e_p, emb_anchor
        else:
            del latent_t, e_p
        flush()
        # save checkpoint and loss curve
        if (i + 1) % 1000 == 0 and i + 1 != iterations and i + 1 >= 1000:
            folder_path = f'models/{name}/{name}_epoch_{i + 1}'
            os.makedirs(folder_path, exist_ok=True)
            network.save_weights(
                os.path.join(folder_path, f"{name}_{i}steps.safetensors"),
                dtype=weight_dtype,
                metadata=model_metadata,
            )

        if i % 100 == 0:
            save_history(losses, name, word_print)
    # model.unet.eval()
    if tensor_path is not None:
        folder_path = f'models/{name}/{tensor_name}/{name}_last'
    else:
        folder_path = f'models/{name}/{name}_last'
    os.makedirs(folder_path, exist_ok=True)
    network.save_weights(
        os.path.join(folder_path, f"{name}_last.safetensors"),
        dtype=weight_dtype,
        metadata=model_metadata,
    )
    del model
    save_history(losses, name, word_print)


def save_history(losses, name, word_print):
    folder_path = f'models/{name}'
    os.makedirs(folder_path, exist_ok=True)
    with open(f'{folder_path}/loss.txt', 'w') as f:
        f.writelines([str(i) for i in losses])
    plot_loss(losses, f'{folder_path}/loss.png', word_print, n=3)


def plot_loss(losses, path, word, n=100):
    v = moving_average(losses, n)
    plt.plot(v, label=f'{word}_loss')
    plt.legend(loc="upper left")
    plt.title('Average loss in trainings', fontsize=20)
    plt.xlabel('Data point', fontsize=16)
    plt.ylabel('Loss value', fontsize=16)
    plt.savefig(path)


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def main(args):
    prompt = args.prompt
    start_guidance = args.start_guidance
    surrogate_guidance_scale = args.surrogate_guidance
    negative_guidance = args.negative_guidance
    iterations = args.iterations
    lr = args.lr
    model_path = args.model_path
    devices = [f'cuda:{int(d.strip())}' for d in args.devices.split(',')]
    seperator = args.seperator
    image_size = args.image_size
    ddim_steps = args.ddim_steps
    change_step_rate = args.change_step_rate
    surrogate_concept = args.surrogate
    lora_rank = args.lora_rank
    textencoder_lr = args.textencoder_lr
    with_prior_preservation = args.with_prior_preservation
    anchor_prompt_path = args.anchor_prompt_path
    anchor_batch_size = args.anchor_batch_size
    is_train_null = args.is_train_null
    no_certain_sur = args.no_certain_sur
    tensor_path = args.tensor_path
    with_prior_latent = args.with_prior_latent
    no_cond = args.no_cond
    only_sur = args.only_sur
    null_weight = args.null_weight
    pl_weight = args.pl_weight
    sc_clip_path = args.sc_clip_path
    train_esd(prompt=prompt,
              start_guidance=start_guidance,
              negative_guidance_scale=negative_guidance,
              iterations=iterations,
              lr=lr,
              model_path=model_path,
              devices=devices,
              seperator=seperator,
              image_size=image_size,
              ddim_steps=ddim_steps,
              change_step_rate=change_step_rate,
              surrogate_concept=surrogate_concept,
              lora_rank=lora_rank,
              textencoder_lr=textencoder_lr,
              with_prior_preservation=with_prior_preservation,
              anchor_prompt_path=anchor_prompt_path,
              anchor_batch_size=anchor_batch_size,
              is_train_null=is_train_null,
              no_certain_sur=no_certain_sur,
              tensor_path=tensor_path,
              with_prior_latent=with_prior_latent,
              no_cond=no_cond,
              only_sur=only_sur,
              null_weight=null_weight,
              pl_weight=pl_weight,
              surrogate_guidance_scale=surrogate_guidance_scale,
              surrogate_concept_clip_path=sc_clip_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='TrainESD',
        description='Finetuning stable diffusion model to erase concepts using ESD method')
    parser.add_argument('--prompt', help='prompt corresponding to concept to erase', type=str, required=True)
    parser.add_argument('--train_method', help='method of training', type=str, required=True)
    parser.add_argument('--start_guidance', help='guidance of start image used to train', type=float, required=False,
                        default=3)
    parser.add_argument('--negative_guidance', help='guidance of negative training used to train', type=float,
                        required=False, default=1)
    parser.add_argument('--surrogate_guidance', help='guidance of negative training used to train', type=float,
                        required=False, default=3)
    parser.add_argument('--iterations', help='iterations used to train', type=int, required=False, default=1000)
    parser.add_argument('--lr', help='learning rate used to train', type=float, required=False, default=1e-5)
    parser.add_argument('--model_path', help='ckpt path for stable diffusion v1-4', type=str, required=False,
                        default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--diffusers_config_path', help='diffusers unet config json path', type=str, required=False,
                        default='unet-config/diffusers_unet_config.json')
    parser.add_argument('--devices', help='cuda devices to train on', type=str, required=False, default='0,0')
    parser.add_argument('--seperator', help='separator if you want to train bunch of words separately', type=str,
                        required=False, default=None)
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False,
                        default=50)
    parser.add_argument('--change_step_rate', help='inference steps used to train', type=float, required=False,
                        default=1.0)
    parser.add_argument('--attention_strength', help='strength of attention map to loss', type=float, required=False,
                        default=1.0)
    parser.add_argument('--surrogate', help='surrogate_concept for original concept', type=str, required=True,
                        default="")
    parser.add_argument("--lora_rank", type=int, default=4, help="Rank of LoRA approximation.")
    parser.add_argument("--textencoder_lr", type=float, default=5e-05, help="learning rate of text encoder")
    parser.add_argument('--with_prior_preservation', action='store_true', help='whether obtain mid of generated images',
                        required=False, default=False)
    parser.add_argument('--is_anchor', action='store_true',
                        required=False, default=False)
    parser.add_argument('--is_train_null', action='store_true',
                        required=False, default=False)
    parser.add_argument('--with_prior_latent', action='store_true',
                        required=False, default=False)
    parser.add_argument('--no_certain_sur', action='store_true',
                        required=False, default=False)
    parser.add_argument('--no_cond', action='store_true',
                        required=False, default=False)
    parser.add_argument('--anchor_concept', type=str, required=False, default=None)
    parser.add_argument('--anchor_prompt_path', type=str, required=False)
    parser.add_argument('--anchor_batch_size', type=int, default=2, required=False)
    parser.add_argument('--tensor_path', type=str, required=False)
    parser.add_argument('--only_sur', action='store_true', required=False, default=False)
    parser.add_argument('--pr_weight', type=float, required=False, default=0.5)
    parser.add_argument('--null_weight', type=float, required=False, default=0.8)
    parser.add_argument('--pl_weight', type=float, required=False, default=0.5)
    parser.add_argument('--is_cos', action='store_true', required=False, default=False)
    parser.add_argument('--is_norm', action='store_true', required=False, default=False)
    parser.add_argument('--sc_clip_path', type=str, required=False)
    args = parser.parse_args()
    main(args)
