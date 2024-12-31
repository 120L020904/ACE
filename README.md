# ACE: Anti-Editing Concept Erasure in Text-to-Image Models

This respository contains the code for paper ACE: Anti-Editing Concept Erasure in Text-to-Image Models.

## Setup

### Dependency Installation
You can create virtual environment use following 
```bash
git clone https://github.com/120L020904/ACE.git
cd ACE
conda env create -f environment.yaml
```


### Preparation
First prepare images about test concept.

```bash
#! /bin/bash
export CSV="cartoon_eval_test"
export ADD_NAME=""
export OUTPUT_DIR="evaluation-outputs/$CSV$ADD_NAME"
export MODEL_NAME="SD3"
accelerate launch --config_file config.yaml train-scripts/src/generate_images_sd3.py \
  --model_name "${MODEL_NAME}" \
  --prompts_path "data/concept_csv/$CSV.csv" \
  --specific_path "data/concept_text/IP_character_concept_10.txt"\
  --save_path="$OUTPUT_DIR" \
  --image_size 1024 \
  --ddim_steps 30 \
  --num_samples 5 \
  --is_SD3
```
Then get concept relation json.
```bash
#! /bin/bash
export CSV="cartoon_eval_test"
export PRE=""
export ADD_NAME="_512"
export CSV_FOLDER=$CSV$ADD_NAME
export MODEL="SD3"
CUDA_VISIBLE_DEVICES=0 python train-scripts/src/eval/evaluation/clip_evaluator.py \
  --csv_path="data/concept_csv/$CSV.csv" \
  --save_folder="evaluation-outputs/$CSV_FOLDER/$MODEL" \
  --output_path="evaluation-outputs/$CSV_FOLDER/$MODEL" \
  --num_samples=5 \
  --csv_name="$CSV" \
  --add_name=$ADD_NAME \
  --method "concept_relation" 
```
### Train
Train concept erasing lora.
```bash
#! /bin/bash
export CONCEPT="Elsa (Frozen)"
CUDA_VISIBLE_DEVICES=5 python train-scripts/src/lora_train_esd_test.py \
  --prompt "$CONCEPT" \
  --surrogate '' \
  --train_method 'full' \
  --devices '0,0' \
  --iterations 1500 \
  --change_step_rate 1 \
  --lr 0.001 \
  --negative_guidance 3 \
  --surrogate_guidance 3 \
  --ddim_steps 30 \
  --anchor_prompt_path "data/concept_text/cartoon_character_concept.txt" \
  --anchor_batch_size 2 \
  --pl_weight 0.8 \
  --null_weight 0.99 \
  --sc_clip_path "evaluation-outputs/cartoon_eval_test/SD3/evaluation_results_clip_${CONCEPT}_image_None.json" \
  --is_train_null \
  --with_prior_preservation \
  --no_certain_sur 
```

### Test
```bash
#! /bin/bash
export CSV="cartoon_eval_format"
export ADD_NAME="_512"
export OUTPUT_DIR="evaluation-outputs/$CSV$ADD_NAME"
export MODEL_NAME="ACE_lora_Elsa (Frozen)-sc_-ng_3.0-iter_1500-lr_0.001-lora-prior_2_tr_null_True_nc_False_no_cer_sur_True_tensor_False_nw_0.99_pl_0.8_sg_new_3.0_is_sc_clip_True"
accelerate launch --config_file config4.yaml train-scripts/src/generate_images_lora.py \
  --model_name "${MODEL_NAME}" \
  --prompts_path "data/concept_csv/$CSV.csv" \
  --save_path="$OUTPUT_DIR" \
  --image_size 512 \
  --ddim_steps 30 \
  --num_samples 1 \
  --multipliers 1 \
  --lora_rank 4 \
  --is_lora \
  --lora_name "ACE_lora_Elsa (Frozen)-sc_-ng_3.0-iter_1500-lr_0.001-lora-prior_2_tr_null_True_nc_False_no_cer_sur_True_tensor_False_nw_0.99_pl_0.8_sg_new_3.0_is_sc_clip_True"

```
## Checkpoints

Checkpoints are coming soon


## Acknowledgments

In this code we refer to the following codebase: [Diffusers](https://github.com/huggingface/diffusers) and [SPM](https://lyumengyao.github.io/projects/spm). Great thanks to them!

