import glob,re,os, pandas as pd 

def all_basename(list_paths, to_replace = ""):
    return [os.path.basename(x).replace(to_replace, "") for x in list_paths]

def reconstruct_final_postfix(path, postfix):
    path, ext = ".".join(path.split(".")[:-1]),path.split(".")[-1]
    path += postfix
    return path + "." + ext

def replace_all(list_paths, to_remove):
    return [ x.replace(to_remove) for x in list_paths]

mother_dir = "interiornet_sample/"
cam_folder = mother_dir + "cam0/data/"
depth_folder = mother_dir + "depth0/"
label_folder = mother_dir + "label0/data/"
mask_folder = mother_dir + "mask0/table_10_inverted/"

## ALL VIEWS
res = [f for f in glob.glob(cam_folder +  "*") if re.search(r'.(jpg|png)', f)]
print(len(res))

## ALL MASKS
res_masks = [f for f in glob.glob(mask_folder +  "*") if re.search(r'.(jpg|png)', f)]
print(len(res_masks))


## ALL SEG MASKS
res_segmasks = [f for f in glob.glob(label_folder +  "*") if re.search(r'.(jpg|png)', f)]
print(len(res_segmasks))

init_set = set(all_basename(res))

to_replace = ["", "_nyu_mask"]
for s_, tr in zip((res_masks,res_segmasks),to_replace):
    # print(set(all_basename(s_,tr)))
    init_set = init_set.intersection(set(all_basename(s_,tr)))
    
path_intersection = sorted(list(init_set))

# print(len(path_intersection))

# exit()
## Create dataframe
df = pd.DataFrame(columns=["image_path","mask_path","segmask_path","partition"])

# sampling validation rate
validation_percentage = 0.2
val_samples = int(round(len(path_intersection)*validation_percentage))

sampl_validation_rate = len(path_intersection)//val_samples
print(sampl_validation_rate)
how_many_val = 1
prefix_to_remove = "/data01/lorenzo.stacchio/TU GRAZ/Stable_Diffusion_Inpaiting/stable-diffusion_custom_inpaint/data/open_source_samples/"

prefix_image = cam_folder.replace(prefix_to_remove,"")
prefix_mask = mask_folder.replace(prefix_to_remove,"")
prefix_segmask = label_folder.replace(prefix_to_remove,"")

list_samples_image, list_samples_mask,list_samples_segmask, list_partition = [],[],[],[]


for i in range(0, len(path_intersection), sampl_validation_rate):
    # print(i, i+sampl_validation_rate)
    samples = path_intersection[i:i+sampl_validation_rate]
    list_samples_image.extend([prefix_image + x for x in samples])
    list_samples_mask.extend([prefix_mask + x for x in samples])
    list_samples_segmask.extend([prefix_segmask + reconstruct_final_postfix(x, "_nyu_mask") for x in samples])
    list_partition.extend(["train"]*(len(samples)-how_many_val) + ["validation"]*(how_many_val))
    
for col, lists in zip(df.columns, [list_samples_image,list_samples_mask,list_samples_segmask,list_partition]):
    df[col] = lists

df.to_csv("dataframe_interiornet_segmentation.csv",index=False)