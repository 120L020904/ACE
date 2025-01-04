#! /bin/bash
export CSV="IP_character"
export PRE=""
export ADD_NAME=""
export CSV_FOLDER=$CSV$ADD_NAME
export MODEL="SD3"
CUDA_VISIBLE_DEVICES=0 python src/eval/evaluation/clip_evaluator.py \
  --csv_path="data/concept_csv/$CSV.csv" \
  --save_folder="evaluation-outputs/$CSV_FOLDER/$MODEL" \
  --output_path="evaluation-outputs/$CSV_FOLDER/$MODEL" \
  --num_samples=3 \
  --csv_name="$CSV" \
  --add_name=$ADD_NAME \
  --method "concept_relation" \
  --image_concept_path "data/concept_text/IP_character_concept_10.txt"