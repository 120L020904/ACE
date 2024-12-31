#! /bin/bash
export CSV="woman_full_body"
export ADD_NAME="_512"
export OUTPUT_DIR="evaluation-outputs/$CSV$ADD_NAME"
export MODEL_NAME="AdvUnlearn_nudity"
accelerate launch --config_file config7.yaml train-scripts/src/eval_edit.py \
  --model_name="${MODEL_NAME}" \
  --prompts_path "data/concept_csv/$CSV.csv" \
  --save_path=$OUTPUT_DIR \
  --num_inversion_steps 30 \
  --num_samples 1 \
  --skip 0.1 \
  --edit_threshold 0.9 \
  --edit_guidance_scale 10 \
  --image_size 512 \
  --data_path "evaluation-outputs/${CSV}/flux" \
  --multipliers 1.0 \
  --inversion_guidance_scale 1.5 \
  --lora_rank 4 \
  --is_SD_v1_4 \
  --use_mask \
  --edit_prompt_path "data/edit_concept/edit_nudity_woman_input.csv" \
  --specific_concept_path "data/concept_text/woman_concept.txt" \
  --is_specific \
  --is_lora \
  --lora_name "${MODEL_NAME}" \
  --is_LEDITS