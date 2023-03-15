import glob,re,os, pandas as pd 
import cv2 
import tqdm

def invert_binary_mask(image_path_mask, save_path):
    img = cv2.imread(image_path_mask)
    inverted = cv2.bitwise_not(img)
    cv2.imwrite(save_path, inverted)
    

# mother_dir = "/data01/lorenzo.stacchio/TU GRAZ/Stable_Diffusion_Inpaiting/stable-diffusion_custom_inpaint/data/christina_sample/interiornet_sample/"
# mask_folder = mother_dir + "mask0/table_10/"
# mask_out_folder = mother_dir + "mask0/table_10_inverted/"

mother_dir = "/data01/lorenzo.stacchio/TU GRAZ/Stable_Diffusion_Inpaiting/stable-diffusion_custom_inpaint/data/open_source_samples/dynafill/subset/"
mask_folder = mother_dir + "masks/dynamic/"
mask_out_folder = mother_dir + "masks/dynamic_inverted/"

if not os.path.exists(mask_out_folder): os.mkdir(mask_out_folder)

## ALL VIEWS
res_masks = [f for f in glob.glob(mask_folder +  "*") if re.search(r'.(jpg|png)', f)]

for path in tqdm.tqdm(res_masks, desc="inverting binary masks"):
    save_path = mask_out_folder + os.path.basename(path)
    invert_binary_mask(path,save_path)