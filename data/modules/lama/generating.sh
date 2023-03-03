#!/bin/bash

# mask_sizes=("thin" "thick" "medium")
mask_sizes=("thick")

size=512
script_path="data/modules/lama/gen_mask_dataset.py" # cal from project root

for mask_size in "${mask_sizes[@]}";
do
  # python -V
  # echo $VIRTUAL_ENV
  # echo $mask_size
  python $script_path "--config" "data/modules/lama/data_gen_configs/random_${mask_size}_${size}.yaml" "--indir" "data/open_source_samples/interiornet_mask_generated/views/" "--outdir" "data/open_source_samples/interiornet_mask_generated/masked_views_${mask_size}_${size}/" "--ext" "png" 
done