#!/bin/bash

mask_sizes=("thick" "thin" "medium")
# mask_sizes=("thick")

size=512
script_path="data/modules/lama/gen_mask_dataset_with_segmentations.py" # cal from project root

for mask_size in "${mask_sizes[@]}";
do
  # python -V
  # echo $VIRTUAL_ENV
  # echo $mask_size
  python $script_path "--config" "data/modules/lama/data_gen_configs/random_${mask_size}_${size}.yaml" "--indir" "data/open_source_samples/interiornet_mask_generated/views_with_seg/" "--seg_postfix" "_nyu_mask" "--outdir" "data/open_source_samples/interiornet_mask_generated/generated_with_seg/masked_views_${mask_size}_${size}/" "--ext" "png" "--n-jobs" "4" 
done