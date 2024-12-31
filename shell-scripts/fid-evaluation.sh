#! /bin/bash
export CSV="coco1000"
export ORI_MODEL="SPM_nudity_edit_None"
export ADD_NAME="_512"
export PRE=""
export MODEL="SD-v1-4_edit_None"
python eval-scripts/calculate_metrics.py \
  --generated="evaluation-outputs/$CSV$ADD_NAME/$MODEL" \
  --save_dir="evaluation-outputs/$CSV$ADD_NAME/$MODEL" \
  --csv_path="data/concept_csv/$CSV.csv" \
#  --image_concept_path="data/concept_text/IP_character_concept_10.txt"
