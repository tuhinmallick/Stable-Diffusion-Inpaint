############ PLACES ############

python3 main_inpainting.py --train --name  custom_place_training_different_samplerSAMESEEDNOTEMA3 --base  configs/latent-diffusion/inpainting_runaway_customPLACE.yaml  --gpus 1,   --seed  42


###################################


############ KEYBOARD ############

python3 main_inpainting.py --train --name  custom_keyboard_training_FULL_TRAINING --base  configs/latent-diffusion/inpainting_runaway_customKEYBOARDFULL.yaml  --gpus 0,   --seed  42

python3 main_inpainting.py --train --name  custom_keyboard_training_FULL_TRAINING2 --base  configs/latent-diffusion/inpainting_runaway_customKEYBOARDFULL.yaml  --gpus 0,   --seed  42

python3 main_inpainting.py --train --name  custom_keyboard_training_FULL_FT_scratch --base  configs/latent-diffusion/inpainting_runaway_customKEYBOARDFULL_scratch.yaml  --gpus 0,   --seed  42


python3 main_inpainting.py --train --name  custom_keyboard_training_FINETUNING_UNET --base  configs/latent-diffusion/inpainting_runaway_customFINETUNING_UNET.yaml  --gpus 1,   --seed  42


python3 main_inpainting.py --train --base  configs/latent-diffusion/inpainting_runaway_customFINETUNING_UNET_ATT.yaml  --gpus 1,   --seed  42 --resume logs/2023-02-08_custom_keyboard_training_FINETUNE_ATT/checkpoints/epoch=000029.ckpt


############ interionet ############

python3 main_inpainting.py --train --name  custom_interionet_FULL_TRAINING --base  configs/latent-diffusion/inpainting_runaway_interiornet_FULL.yaml  --gpus 0,   --seed  42


python3 main_inpainting.py --train --name  custom_interionet_FULL_TRAINING_fixed_LR --base  configs/latent-diffusion/inpainting_runaway_interiornet_FULL_modified.yaml  --gpus 1,   --seed  42


python3 main_inpainting.py --train --name  custom_interionet_FULL_TRAINING_test_scheduler --base  configs/latent-diffusion/inpainting_runaway_interiornet_FULL_LR_LOWER.yaml  --gpus 0,   --seed  42


############ interionet LAMA ############

python3 main_inpainting.py --train --name  custom_interionet_FULL_TRAINING_LAMA --base  configs/latent-diffusion/inpainting_runaway_interiornet_LAMA_LR_LOWER.yaml  --gpus 1,   --seed  42


############## CONTROLNET INTERIORNET #######################

python3 main_inpainting.py  --train  --name  custom_keyboard_training_CONTROLNET --base  configs/latent-diffusion/inpainting_runaway_interiornet_ControlNet_training.yaml   --gpus  0,   --seed  42


python3 main_inpainting.py  --train  --name  custom_keyboard_training_CONTROLNET_LAMA --base  configs/latent-diffusion/inpainting_runaway_interiornet_ControlNet_training_LAMA.yaml   --gpus  0,   --seed  42