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
from inpaint_utils import make_batch
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

    # masks = sorted(glob.glob(os.path.join(opt.indir, "*_mask.*")))
    masks = sorted(glob.glob(os.path.join(opt.indir, "16693_12_mask.*")))

    images = [x.replace("_mask.png", ".png") for x in masks]
    print(f"Found {len(masks)} inputs.")

    config = OmegaConf.load("configs/latent-diffusion/inpainting_runaway.yaml")
    model = instantiate_from_config(config.model)

    # weight_path_old = "logs/2023-02-08_custom_place_training_different_sampler/checkpoints/last.ckpt"

    weight_path_old = "models/ldm/inpainting_big/last.ckpt"

    # weight_path_new = "logs/2023-02-08_custom_place_training_different_sampler/checkpoints/epoch=000023.ckpt"

    # weight_path_old = "models/ldm/inpainting_big/last.ckpt"

    # weight_path_new = "models/ldm/inpainting_big/model_compvis.ckpt"

    print("Loading modeling from %s" % weight_path_old)

    state_dict_to_load = torch.load(weight_path_old)["state_dict"]

    # state_dict_to_load_old = torch.load(weight_path_old)["state_dict"]
    
    
    # validate_state_dicts(state_dict_to_load_old,state_dict_to_load)
    
    # exit()
    model.load_state_dict(state_dict_to_load,
                           strict=False)

    
    device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    
    with torch.no_grad():
        with model.ema_scope():
            for image, mask in tqdm(zip(images, masks)):
                outpath = os.path.join(opt.outdir, os.path.split(image)[1])

                batch = make_batch(image, mask, device=device, resize_to=512)
                
                # encode masked image and concat downsampled mask
                c = model.cond_stage_model.encode(batch["masked_image"])
               
                # print(batch["mask"])
                
                cc = torch.nn.functional.interpolate(batch["mask"],
                                                     size=c.shape[-2:])
                # print(cc)
                c = torch.cat((c, cc), dim=1)

                # print("SHAPE FED TO INPUT", c.shape)
                # shape = (c.shape[1]-2,)+c.shape[2:]
                shape = (3,) + c.shape[2:]

                print("SHAPE FED TO INPUT", shape)
                # print("\nSAMPLING\n")
                
                samples_ddim, _ = sampler.sample(S=opt.steps,
                                                 conditioning=c,
                                                 batch_size=c.shape[0],
                                                 shape=shape,
                                                 verbose=False)
                # print(samples_ddim.shape)
                # exit()

                x_samples_ddim = model.decode_first_stage(samples_ddim)

                image = torch.clamp((batch["image"]+1.0)/2.0,
                                    min=0.0, max=1.0)
                mask = torch.clamp((batch["mask"]+1.0)/2.0,
                                   min=0.0, max=1.0)
                predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0,
                                              min=0.0, max=1.0)

                inpainted = (1-mask)*image+mask*predicted_image
                inpainted = inpainted.cpu().numpy().transpose(0,2,3,1)[0]*255
                print("Save in %s" % outpath)
                Image.fromarray(inpainted.astype(np.uint8)).save(outpath)