#! /bin/bash
export CSV="10_others"
export ADD_NAME="_512"
export PRE=""
export MODEL="SPM_{}_edit_None/None"
export CSV_FOLDER="$CSV$ADD_NAME"
CUDA_VISIBLE_DEVICES=4 python eval-scripts/lpips_eval.py \
  --prompts_path="data/concept_csv/$CSV.csv" \
  --csv_name="${CSV}" \
  --edited_path_format="evaluation-outputs/$CSV_FOLDER/$MODEL" \
  --save_path_format="evaluation-outputs/$CSV_FOLDER/$MODEL" \
  --image_concept_path="data/concept_text/IP_character_concept_10.txt" \
  --num_samples=5 \
  --data_path="evaluation-outputs/${CSV}${ADD_NAME}/SD-v1-4_edit_None/None" \
  --model_method="generate_sd" \
  --method "" \
  --concept_num 5 \
  --specific_concept_path "data/concept_text/generate_IP_10_others.txt" \
#  --invert_concept_path="data/concept_text/inversion_prompt.txt"
# --edited_concept_path="data/edit_concept/edit_concept_input.csv" \