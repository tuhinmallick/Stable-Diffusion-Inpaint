import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
import sys
sys.path.insert(
   1, "/data01/lorenzo.stacchio/TU GRAZ/Stable_Diffusion_Inpaiting/stable-diffusion_custom_inpaint")
from main_inpainting import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from inpaint_utils import seed_everything, make_batch, make_image
from contextlib import suppress
from torchmetrics.image.lpip_similarity import LPIPS
import cv2
seed_everything(42)


# RUN SCRIPT


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Stable Diffusion Inpainting")
    
    parser.add_argument(
        "--csv_file",
        type=str,
        nargs="?",
        help="csv file containing image_path mask_path and partition column for evaluation",
        required=True
    )
    
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write output inpainted images",
        required=True
    )
    
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="path of weights to load",
        required=True        
    )
    
    parser.add_argument(
        "--ckpt",
        type=str,
        help="path of weights to load",
        required=True        
    )
    
    parser.add_argument(
        "--yaml_profile",
        type=str,
        help="yaml file describing the model to initialize",
        required=True        
    )

      
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="specify the device for inference (cpu, cuda, cuda:x)",
    )
      
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    
    parser.add_argument(
        "--lpips",
        type=str,
        default="alex",
        help="net_type for LPIPS metrics: alex | vgg",
    )

    parser.add_argument(
        "--ema",
        action='store_true',
        help="use ema weights",
    )
    
    opt = parser.parse_args()

    # READ FROM CSV
    df = pd.read_csv(opt.csv_file)
    
    interpolation_images = 2
    # FIRST 7 AND TAKE THE FOURTH AS THE ONE TO BE GENERATED
    df = df.head(interpolation_images)
 
    
    masks_for_mean = df["mask_path"]
    images_for_mean = df["image_path"]
    
    print(f"Found {len(masks_for_mean)} inputs.")

    config = OmegaConf.load(opt.yaml_profile)
    model = instantiate_from_config(config.model)

    model.load_state_dict(torch.load(opt.ckpt)["state_dict"],
                            strict=False)

    print("Loading modeling from %s" % opt.ckpt)
    
    
    device = torch.device(opt.device) if torch.cuda.is_available() and opt.device is not "cpu" else torch.device("cpu")
    
    model = model.to(device)

    sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    
    scope = model.ema_scope if opt.ema else suppress
    ema_prefix = "EMA" if opt.ema else "NOT_EMA"
    
    acc_representations = []
    out_paths = []
    
    with torch.no_grad():
        with scope("Sampling"):
            for image_path, mask_path in tqdm(zip(images_for_mean, masks_for_mean)):
                
                batch = make_batch(image_path, mask_path, device=device, resize_to=512)
                
                c = model.cond_stage_model.encode(batch["masked_image"])
                # Average of different representations and encode 
                
                cc = torch.nn.functional.interpolate(batch["mask"],
                                                        size=c.shape[-2:])

                c = torch.cat((c, cc), dim=1)


                shape = (3,) + c.shape[2:]

                
                samples_ddim, _ = sampler.sample(S=opt.steps,
                                                    conditioning=c,
                                                    batch_size=c.shape[0],
                                                    shape=shape,
                                                    verbose=False)

                x_samples_ddim = model.decode_first_stage(samples_ddim)

               
                

                image = torch.clamp((batch["image"]+1.0)/2.0,
                                    min=0.0, max=1.0)
                mask = torch.clamp((batch["mask"]+1.0)/2.0,
                                    min=0.0, max=1.0)
                
                masked_image = torch.clamp((batch["masked_image"]+1.0)/2.0,
                                    min=0.0, max=1.0)
                
                predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0,
                                                min=0.0, max=1.0)
                               
                inpainted = (1-mask)*image+mask*predicted_image
                
                inpainted = inpainted.cpu().numpy().transpose(0,2,3,1)[0]*255
                                
                outpath = os.path.join(opt.outdir, "%s_%s_%s_%s.png" % (os.path.split(os.path.basename(image_path))[1].split(".")[0], opt.prefix, ema_prefix, os.path.basename(opt.ckpt).split(".")[0]))
                Image.fromarray(inpainted.astype(np.uint8)).save(outpath)
                out_paths.append(outpath)
                
            # RE-ENCODE FIRST STAGE ALL GENERATED IMAGES AND MEAN THEM WITH NOVEL TO INPAINT
            stacked_representation = torch.stack([model.cond_stage_model.encode(make_image(x,device=device, resize_to=512)) for x in out_paths], dim = 0).squeeze()
            print(stacked_representation.shape)
            assert stacked_representation.shape[0]==2
            for i in range(0,10):
                i = i/10
                interpolated = torch.lerp(stacked_representation[0], stacked_representation[1], i).unsqueeze(dim=0)
                
                x_samples_ddim = model.decode_first_stage(interpolated)
                
                # predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0,
                #                                 min=0.0, max=1.0)
                                            
                predicted_image = x_samples_ddim.cpu().numpy().transpose(0,2,3,1)[0]*255
                                
                outpath = os.path.join(opt.outdir, "%s_%s_%s_INTERPOLATED_%s.png" % (os.path.split(os.path.basename(image_path))[1].split(".")[0], opt.prefix, ema_prefix, i))
                Image.fromarray(predicted_image.astype(np.uint8)).save(outpath)
