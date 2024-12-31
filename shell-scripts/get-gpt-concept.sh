#! /bin/bash
python train-scripts/diffusers/gpt_utils.py \
  --ablated_concept "mickey" \
  --category "cartoon character" \
  --anchor_num 20 \
  --synonym_num 1
