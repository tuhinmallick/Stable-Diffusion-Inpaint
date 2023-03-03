import glob,re,os, pandas as pd 

def all_basename(list_paths):
    return [os.path.basename(x) for x in list_paths]


def replace_all(list_paths, to_remove):
    return [ x.replace(to_remove) for x in list_paths]

mother_dir = "interiornet_sample/"
cam_folder = mother_dir + "cam0/data/"
depth_folder = mother_dir + "depth0/"
label_folder = mother_dir + "label0/"
mask_folder = mother_dir + "mask0/table_10_inverted/"

## ALL VIEWS
res = [f for f in glob.glob(cam_folder +  "*") if re.search(r'.(jpg|png)', f)]
print(len(res))

## ALL MASKS
res_masks = [f for f in glob.glob(mask_folder +  "*") if re.search(r'.(jpg|png)', f)]
print(len(res_masks))


path_intersection = sorted(list(set(all_basename(res)).intersection(set(all_basename(res_masks)))))

## Create dataframe
df = pd.DataFrame(columns=["image_path","mask_path","partition"])

# sampling validation rate
validation_percentage = 0.2
val_samples = int(round(len(path_intersection)*validation_percentage))

sampl_validation_rate = len(path_intersection)//val_samples
print(sampl_validation_rate)
how_many_val = 1
prefix_to_remove = "/data01/lorenzo.stacchio/TU GRAZ/Stable_Diffusion_Inpaiting/stable-diffusion_custom_inpaint/data/open_source_samples/"

prefix_image = cam_folder.replace(prefix_to_remove,"")
prefix_mask = mask_folder.replace(prefix_to_remove,"")

list_samples_image, list_samples_mask, list_partition = [],[],[]

for i in range(0, len(path_intersection), sampl_validation_rate):
    # print(i, i+sampl_validation_rate)
    samples = path_intersection[i:i+sampl_validation_rate]
    list_samples_image.extend([prefix_image + x for x in samples])
    list_samples_mask.extend([prefix_mask + x for x in samples])
    list_partition.extend(["train"]*(len(samples)-how_many_val) + ["validation"]*(how_many_val))
    
for col, lists in zip(df.columns, [list_samples_image,list_samples_mask,list_partition]):
    df[col] = lists

df.to_csv("dataframe_interiornet.csv",index=False)