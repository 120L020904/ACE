#! /bin/bash
export CONCEPT="Hello Kitty"
CUDA_VISIBLE_DEVICES=0 python src/train_ace.py \
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
  --anchor_prompt_path "data/concept_text/IP_character_50.txt" \
  --anchor_batch_size 2 \
  --pl_weight 0.8 \
  --null_weight 0.99 \
  --sc_clip_path "evaluation-outputs/IP_character_50/SD3/evaluation_results_clip_${CONCEPT}_image_None.json" \
  --is_train_null \
  --with_prior_preservation \
  --no_certain_sur \
