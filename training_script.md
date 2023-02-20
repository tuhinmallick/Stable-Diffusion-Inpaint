############ PLACES ############

python3 main_inpainting.py --train --name  custom_place_training_different_samplerSAMESEEDNOTEMA3 --base  configs/latent-diffusion/inpainting_runaway_customPLACE.yaml  --gpus 1,   --seed  42


###################################


############ KEYBOARD ############

python3 main_inpainting.py --train --name  custom_keyboard_training_different_samplerSAMESEEDNOTEMA --base  configs/latent-diffusion/inpainting_runaway_customKEYBOARD.yaml  --gpus 1,   --seed  42


python3 main_inpainting.py --train --name  custom_keyboard_training_FULL_SAMESEEDNOTEMA --base  configs/latent-diffusion/inpainting_runaway_customKEYBOARDFULL.yaml  --gpus 1,   --seed  42

