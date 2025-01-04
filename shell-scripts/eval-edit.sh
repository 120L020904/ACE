#! /bin/bash
export CSV="IP_character"
export ADD_NAME="_512"
export OUTPUT_DIR="evaluation-outputs/$CSV$ADD_NAME"
export MODEL_NAME="ACE_lora_Elsa (Frozen)-sc_-ng_3.0-iter_1500-lr_0.001-lora-prior_2_tr_null_True_nc_False_no_cer_sur_True_tensor_False_nw_0.99_pl_0.8_sg_new_3.0_is_sc_clip_True"
accelerate launch --config_file config.yaml src/eval_edit.py \
  --model_name="${MODEL_NAME}" \
  --prompts_path "data/concept_csv/$CSV.csv" \
  --save_path=$OUTPUT_DIR \
  --num_inversion_steps 30 \
  --num_samples 1 \
  --skip 0.1 \
  --edit_guidance_scale 10 \
  --image_size 512 \
  --data_path "evaluation-outputs/${CSV}/SD3" \
  --multipliers 1.0 \
  --inversion_guidance_scale 1.5 \
  --lora_rank 4 \
  --is_SD_v1_4 \
  --use_mask \
  --edit_prompt_path "data/edit_concept/edit_concept_input.csv" \
  --generate_concept_path "data/concept_text/IP_character_concept_10.txt" \
  --is_specific \
  --is_lora \
  --lora_name "${MODEL_NAME}" \
  --is_LEDITS