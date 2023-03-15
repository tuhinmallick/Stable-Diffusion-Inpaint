import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import sys
sys.path.insert(
   1, "/data01/lorenzo.stacchio/TU GRAZ/Stable_Diffusion_Inpaiting/stable-diffusion_custom_inpaint")
from main_inpainting import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from inpaint_utils import seed_everything
from inpaint_utils import make_batch_seg,plot_row_original_mask_output
from contextlib import suppress
seed_everything(42)


# RUN SCRIPT
# python scripts/inpaint_runaway_correct.py --indir "/data01/lorenzo.stacchio/TU GRAZ/Stable_Diffusion_Inpaiting/stable-diffusion_custom_inpaint/data/INPAINTING/inpainting_examples/" --outdir "/data01/lorenzo.stacchio/TU GRAZ/Stable_Diffusion_Inpaiting/stable-diffusion_custom_inpaint/data/INPAINTING/output_images_debug/"


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Stable Diffusion Inpainting")
    
    parser.add_argument(
        "--indir",
        type=str,
        nargs="?",
        help="dir containing image-mask pairs (`example.png` and `example_mask.png`)",
        required=True
    )
    
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
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
        "--mask_inverted",
        action='store_true',
        help="mask is black over white and not white over black",
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
        "--ema",
        action='store_true',
        help="use ema weights",
    )
    
    opt = parser.parse_args()

    masks = sorted(glob.glob(os.path.join(opt.indir, "*_mask.*")))

    images = [x.replace("_mask.png", ".png") for x in masks]
    
    segmentation = [x.replace("_mask.png", "_seg.png") for x in masks]
    print(f"Found {len(masks)} inputs.")

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
    
    with torch.no_grad():
        with scope("Sampling"):
            for image, mask, segm_mask in tqdm(zip(images, masks, segmentation)):
                outpath = os.path.join(opt.outdir, "%s_%s_%s_%s.png" % (os.path.split(image)[1].split(".")[0], opt.prefix, ema_prefix, os.path.basename(opt.ckpt)))

                batch = make_batch_seg(image, mask, segm_mask, device=device, resize_to=512, mask_inverted = opt.mask_inverted)
                
                c = model.cond_stage_model.encode(batch["masked_image"])
                                
                cc = torch.nn.functional.interpolate(batch["mask"],
                                                        size=c.shape[-2:])

                c_concat = torch.cat((c, cc), dim=1)


                dict_cond = dict(control=batch["seg_mask"], c_concat=c_concat)


                shape = (3,) + c.shape[2:]

                
                samples_ddim, _ = sampler.sample(S=opt.steps,
                                                    conditioning=dict_cond,
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

                seg_mask = torch.clamp((batch["seg_mask"]+1.0)/2.0,
                                    min=0.0, max=1.0)
                 
                predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0,
                                                min=0.0, max=1.0)

                if opt.mask_inverted:
                    inpainted = mask*image+mask*predicted_image
                else:
                    inpainted = (1-mask)*image+mask*predicted_image
                
                inpainted = inpainted.cpu().numpy().transpose(0,2,3,1)[0]*255
                
                predicted_image = predicted_image.cpu().numpy().transpose(0,2,3,1)[0]*255
                print("Save in %s" % outpath)
                
                mask = mask.cpu().numpy().transpose(0,2,3,1)[0]*255
                image = image.cpu().numpy().transpose(0,2,3,1)[0]*255
                masked_image = masked_image.cpu().numpy().transpose(0,2,3,1)[0]*255
                seg_mask = seg_mask.cpu().numpy().transpose(0,2,3,1)[0]*255

                image_to_print = plot_row_original_mask_output([{"seg_mask":seg_mask, "masked_image":masked_image,  "image":image, "predicted_image":predicted_image}], image_size = 512)
                Image.fromarray(image_to_print.astype(np.uint8)).save(outpath)