# IN INPAITING FOLDER
wget -O models/ldm/inpainting_big/model.ckpt https://ommer-lab.com/files/latent-diffusion/inpainting_big.zip --no-check-certificate


## PLACES NOT EMA VS EMA

python scripts/inpaint_runaway_correct.py --indir "data/INPAINTING/inpainting_examples/" --outdir "data/INPAINTING/output_images_debug/" --ckpt "logs/2023-02-08_custom_place_training_different_samplerSAMESEEDNOTEMA3/checkpoints/epoch=000685.ckpt" --yaml_profile "configs/latent-diffusion/inpainting_runaway_inference.yaml" --device cuda:0 --prefix "PLACES_OVERFIT"

python scripts/inpaint_runaway_correct.py --indir "data/INPAINTING/inpainting_examples/" --outdir "data/INPAINTING/output_images_debug/" --ckpt "logs/2023-02-08_custom_place_training_different_samplerSAMESEEDNOTEMA3/checkpoints/epoch=000685.ckpt" --yaml_profile "configs/latent-diffusion/inpainting_runaway_inference.yaml" --device cuda:0 --ema --prefix "PLACES_OVERFIT"

## CHRISTINA

python scripts/inpaint_runaway_correct.py --indir "data/samples/interiornet_test/" --outdir "data/samples/custom_inpainting/output/" --ckpt "models/ldm/inpainting_big/model_compvis.ckpt" --yaml_profile "configs/latent-diffusion/inpainting_runaway_inference.yaml" --device cuda:0 --prefix "christina"


python scripts/inpaint_runaway_correct.py --indir "data/samples/interiornet_test/" --outdir "data/samples/custom_inpainting/output/" --ckpt "logs/2023-02-08_custom_christina_FULL_TRAINING/checkpoints/last.ckpt" --yaml_profile "configs/latent-diffusion/inpainting_runaway_inference.yaml" --device cuda:0 --prefix "christina"

python scripts/inpaint_runaway_correct.py --indir "data/samples/interiornet_test/" --outdir "data/samples/custom_inpainting/output/" --ckpt "logs/2023-02-08_custom_christina_FULL_TRAINING/checkpoints/epoch=000004.ckpt" --yaml_profile "configs/latent-diffusion/inpainting_runaway_inference.yaml" --device cuda:0 --prefix "christina"



## KEYBOARD NOT EMA VS EMA

python scripts/inpaint_runaway_correct.py --indir "data/samples/custom_inpainting/mouse_keyboard/" --outdir "data/samples/custom_inpainting/output/" --ckpt "logs/2023-02-08_custom_keyboard_training_FULL_TRAINING/checkpoints/last.ckpt" --yaml_profile "configs/latent-diffusion/inpainting_runaway_inference.yaml" --device cuda:0 --prefix "trained_key2"

python scripts/inpaint_runaway_correct.py --indir "data/samples/custom_inpainting/mouse_keyboard/" --outdir "data/samples/custom_inpainting/output/" --ckpt "logs/2023-02-08_custom_keyboard_training_FULL_TRAINING2/checkpoints/last.ckpt" --yaml_profile "configs/latent-diffusion/inpainting_runaway_inference.yaml" --device cuda:0 --prefix "trained_key2"

python scripts/inpaint_runaway_correct.py --indir "data/samples/custom_inpainting/mouse_keyboard/" --outdir "data/samples/custom_inpainting/output/" --ckpt "logs/2023-02-08_custom_keyboard_training_different_samplerSAMESEEDNOTEMA/checkpoints/epoch=000269.ckpt" --yaml_profile "configs/latent-diffusion/inpainting_runaway_inference.yaml" --device cuda:0 --prefix "trained_key"

python scripts/inpaint_runaway_correct.py --indir "data/samples/custom_inpainting/mouse_keyboard/" --outdir "data/samples/custom_inpainting/output/" --ckpt "logs/2023-02-08_custom_keyboard_training_FULL_SAMESEEDNOTEMA/checkpoints/last.ckpt" --yaml_profile "configs/latent-diffusion/inpainting_runaway_inference.yaml" --device cuda:0 --prefix "trained_key_FULL"



## DEBUG

python scripts/INPAINTDEBUG.py --indir "/data01/lorenzo.stacchio/TU GRAZ/Stable_Diffusion_Inpaiting/stable-diffusion_custom_inpaint/data/INPAINTING/custom_inpainting/" --outdir "/data01/lorenzo.stacchio/TU GRAZ/Stable_Diffusion_Inpaiting/stable-diffusion_custom_inpaint/data/INPAINTING/output_images_debug/"

python scripts/inpaint_runaway_correctBIG.py --indir "/data01/lorenzo.stacchio/TU GRAZ/Stable_Diffusion_Inpaiting/stable-diffusion_custom_inpaint/data/INPAINTING/custom_inpainting/" --outdir "/data01/lorenzo.stacchio/TU GRAZ/Stable_Diffusion_Inpaiting/stable-diffusion_custom_inpaint/data/INPAINTING/output_images_debug/"




