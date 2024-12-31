#! /bin/bash
export CSV=""
export ADD_NAME=""
export OUTPUT_DIR="evaluation-outputs/$CSV$ADD_NAME"
export MODEL_NAME="SD3"
accelerate launch --config_file config.yaml train-scripts/src/generate_images_sd3.py \
  --model_name "${MODEL_NAME}" \
  --prompts_path "data/concept_csv/$CSV.csv" \
  --specific_concept_path "data/concept_text/IP_character_concept_10.txt"\
  --save_path="$OUTPUT_DIR" \
  --image_size 1024 \
  --ddim_steps 30 \
  --num_samples 5 \
  --is_SD3