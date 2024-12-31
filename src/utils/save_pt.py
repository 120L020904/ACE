import sys

import torch

sys.path.append("train-scripts/src")
from diffusers import StableDiffusionPipeline
from models.merge_spm import load_state_dict
from models.spm import SPMNetwork, SPMLayer


def save_pt(lora_name, weight_type=torch.float32):
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4",
                                                   torch_dtype=weight_type,
                                                   safety_checker=None,
                                                   local_files_only=True, )
    # spm_paths = [f"models/{lora_name}/{lora_name}_last/{lora_name}_last.safetensors"]
    spm_paths = ["output/Van Gogh/Van Gogh_last.safetensors"]
    device = 'cuda:0'
    network = SPMNetwork(
        pipe.unet,
        rank=1,
        alpha=1.0,
        module=SPMLayer,
    ).to(device, dtype=weight_type)
    spms, metadatas = zip(*[
        load_state_dict(spm_model_path, weight_type) for spm_model_path in spm_paths
    ])
    used_multipliers = []
    weighted_spm = dict.fromkeys(spms[0].keys())
    multipliers = torch.tensor([1.0]).to("cpu", dtype=weight_type)
    spm_multipliers = torch.tensor(multipliers).to("cpu", dtype=weight_type)
    for spm, multiplier in zip(spms, spm_multipliers):
        max_multiplier = torch.max(multiplier)
        for key, value in spm.items():
            if weighted_spm[key] is None:
                weighted_spm[key] = value * max_multiplier
            else:
                weighted_spm[key] += value * max_multiplier
        used_multipliers.append(max_multiplier.item())
    network.load_state_dict(weighted_spm)
    with network:
        # torch.save(pipe.unet, f"models/{lora_name}/{lora_name}.pt")
        torch.save(pipe.unet, "output/Van Gogh/Van Gogh.pt")
    return


if __name__ == "__main__":
    save_pt(
        "ESD_lora_Van Gogh-sc_-ng_3.0-iter_750-lr_0.001-lora-prior_2_tr_null_True_nc_False_no_cer_sur_True_tensor_False_nw_0.5_pl_0.9_sg_new_1.5_is_sc_clip_True",
    )
