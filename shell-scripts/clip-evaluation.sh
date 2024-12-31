#! /bin/bash
export CSV="cartoon_eval_test"
export PRE=""
export ADD_NAME="_512"
export CSV_FOLDER=$CSV$ADD_NAME
export MODEL="SD3"
CUDA_VISIBLE_DEVICES=6 python train-scripts/src/eval/evaluation/clip_evaluator.py \
  --csv_path="data/concept_csv/$CSV.csv" \
  --save_folder="evaluation-outputs/$CSV_FOLDER/$MODEL" \
  --output_path="evaluation-outputs/$CSV_FOLDER/$MODEL" \
  --num_samples=5 \
  --csv_name="$CSV" \
  --add_name=$ADD_NAME \
  --method "concept_relation"