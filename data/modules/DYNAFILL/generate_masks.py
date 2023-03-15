import glob,os,cv2,tqdm
import pandas as pd 
import numpy as np 

def generate_binary_mask(path, df_color_to_mask, out_path):
    image = cv2.imread(path)
    masks_acc = np.zeros(shape=image.shape)
  
    for _, row in df_color_to_mask.iterrows():
        bgr_vector = [int(row["B"]),int(row["G"]),int(row["R"])]
        masked = np.zeros(shape=(list(image.shape[:2])), dtype="uint8")
        indices = np.where(np.all(image == bgr_vector, axis=-1))
        masked[indices] = 255
        masks_acc[indices] =[255,255,255]

    cv2.imwrite(out_path,masks_acc)



mother_dir = "/data01/lorenzo.stacchio/TU GRAZ/Stable_Diffusion_Inpaiting/Datasets/DynaFill/"

# postfixs = ["training", "validation"]
partition_postfixs = ["training"]


image_folder_input_local_path = "semseg/dynamic"
image_folder_outputput_local_path = "semseg/dynamic_masks"

# Masks should be create on dynamic images only with dynamic objects
df_dynamic_masks = pd.read_csv("data/modules/DYNAFILL/mask_colors.csv")
df_dynamic_colors = df_dynamic_masks[df_dynamic_masks["Dynamic"]==True] 

for partition in partition_postfixs:
    dynamic_out_folder = "%s/%s/%s/" % (mother_dir, partition, image_folder_outputput_local_path)
    if not os.path.exists(dynamic_out_folder):
        os.mkdir(dynamic_out_folder)
    # print(dynamic_out_folder)

    dynamic_image_path = "%s/%s/%s/" % (mother_dir, partition, image_folder_input_local_path)
    # print(dynamic_image_path)
    all_paths = list(glob.glob(dynamic_image_path + "*"))
    for path in tqdm.tqdm(all_paths, desc="Generating mask for %s partition" % partition, total=len(all_paths)):
        out_path = "%s%s" % (dynamic_out_folder, os.path.basename(path))
        # print(out_path)
        generate_binary_mask(path, df_dynamic_colors, out_path)
        # break
    # exit()

