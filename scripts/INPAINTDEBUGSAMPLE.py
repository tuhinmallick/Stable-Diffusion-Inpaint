import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import sys
sys.path.insert(
   1, "/data01/lorenzo.stacchio/TU GRAZ/Stable_Diffusion_Inpaiting/stable-diffusion_custom_inpaint")
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from inpaint_utils import seed_everything, validate_state_dicts
import json , cv2
from torchvision.transforms import ToPILImage
from inpaint_utils import make_batch, sample_model_original
seed_everything(42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir",
        type=str,
        nargs="?",
        help="dir containing image-mask pairs (`example.png` and `example_mask.png`)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )

    opt = parser.parse_args()

    masks = sorted(glob.glob(os.path.join(opt.indir, "*_mask.*")))
    images = [x.replace("_mask.png", ".png") for x in masks]
    print(f"Found {len(masks)} inputs.")

    # config = OmegaConf.load("models/ldm/inpainting_big/config.yaml")
    # config = OmegaConf.load("configs/latent-diffusion/inpainting_runaway4_NOCLIP.yaml")
    config = OmegaConf.load("configs/latent-diffusion/inpainting_runaway.yaml")
    # print(config)
    model = instantiate_from_config(config.model)

    print("Loading modeling")
    # model.load_state_dict(torch.load("models/ldm/inpainting_big/last.ckpt")["state_dict"],
    #                        strict=False)

    # dict_old_weights = torch.load("logs/2023-02-08NODECAY_inpainting_runaway/checkpoints/epoch=000002.ckpt")["state_dict"]

    # # print(list(dict_old_weights.keys()))
    # print(len(dict_old_weights))
    
    # dict_new_weights = torch.load("logs/2023-02-08NODECAY_inpainting_runaway/checkpoints/last.ckpt")["state_dict"]
    # print(len(dict_new_weights))
    # validate_state_dicts(dict_old_weights, dict_new_weights)
    # exit()
    # validate_state_dicts()

    state_dict_to_load = torch.load("models/ldm/inpainting_big/model_compvis.ckpt")["state_dict"]

    # state_dict_to_load = torch.load("logs/2023-02-08NODECAY_inpainting_runaway/checkpoints/epoch=000002-v1.ckpt")["state_dict"]
    # state_dict_to_load = torch.load("logs/2023-02-08NODECAY_inpainting_runaway/checkpoints/last.ckpt")["state_dict"]

    # print(len(state_dict_to_load))
    # exit()
    model.load_state_dict(state_dict_to_load,
                           strict=False)

    # dict_old_weights = torch.load("models/ldm/inpainting_big/model_compvis.ckpt")["state_dict"]
    # # print(list(dict_old_weights.keys()))
    # print(len(dict_old_weights))
    
    # dict_new_weights = torch.load("logs/2023-02-07T13-57-20_inpainting_runaway/checkpoints/epoch=000002.ckpt")["state_dict"]
    # # validate_state_dicts(dict_old_weights, dict_new_weights)
    # print(len(dict_new_weights))
    # # validate_state_dicts()

    # model.load_state_dict(dict_old_weights,
    #                         strict=True)
    
    # # NUOVI PESI TRAINATI PER UNA EPOCA CHE ORA HANNO ANCHE I PESI DDIM 
    # model.load_state_dict(dict_new_weights,
    #                         strict=True)
    

    # CON STRICT TRUE DA FALSE SE SI PROVA A CARICARE I PESI VECCHI CON QUALSIASI CONFIGURAZIONE DEL MODELLO PERCHE' MANCANO I PESI DEL DDIM, CON FALSE NO PERCHE' SEMPLICEMENTE NON LI CARICA
    # model.load_state_dict(dict_old_weights,
    #                         strict=True)

    # model.load_state_dict(torch.load("models/ldm/inpainting_big/sd-v1-5-inpainting.ckpt")["state_dict"],
    #                        strict=False)

    print("Model loaded")
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    with torch.no_grad():
        with model.ema_scope():
            for image, mask in tqdm(zip(images, masks)):
                outpath = os.path.join(opt.outdir, os.path.split(image)[1])

                batch = make_batch(image, mask, device=device)
                sample_model_original(model, batch)