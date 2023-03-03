import glob,re,os, pandas as pd 

def all_basename(list_paths):
    return [os.path.basename(x) for x in list_paths]


def replace_all(list_paths, to_remove):
    return [ x.replace(to_remove) for x in list_paths]




def from_mask_to_normal(path):
    return "_mask".join(path.split("_mask")[:-1]) + "." + os.path.basename(path).split(".")[-1]

df = pd.DataFrame(columns=["image_path","mask_path","partition"])

modality_dir = "interiornet_mask_generated/"
validation_percentage = 0.2

for folder in glob.glob(modality_dir + "masked_views_*"):
    mother_dir = "interiornet_mask_generated/%s/" % (os.path.basename(folder))

    ## ALL VIEWS
    masked_images = [f for f in glob.glob(mother_dir +  "*_mask*") if re.search(r'.(jpg|png)', f)]

    # images = [x.replace("_mask", "") for x in masked_images]

    ## Create dataframe

    # sampling validation rate
    val_samples = int(round(len(masked_images)*validation_percentage))
    print("train samples %s Val samples %s" %(len(masked_images)-val_samples,val_samples))
    sampl_validation_rate = len(masked_images)//val_samples
    how_many_val = 1
    prefix_to_remove = "/data01/lorenzo.stacchio/TU GRAZ/Stable_Diffusion_Inpaiting/stable-diffusion_custom_inpaint/data/open_source_samples/"

    list_samples_image, list_samples_mask, list_partition = [],[],[]

    for i in range(0, len(masked_images), sampl_validation_rate):
        # print(i, i+sampl_validation_rate)
        samples = masked_images[i:i+sampl_validation_rate]
        list_samples_mask.extend([x.replace(prefix_to_remove,"") for x in samples])
        list_samples_image.extend([from_mask_to_normal(x).replace(prefix_to_remove, "") for x in samples])
        list_partition.extend(["train"]*(len(samples)-how_many_val) + ["validation"]*(how_many_val))
        
    dict_to_append = {}
    
    for col, lists in zip(df.columns, [list_samples_image,list_samples_mask,list_partition]):
        dict_to_append[col] = lists
    
    df = df.append(pd.DataFrame(dict_to_append))

df.to_csv("dataframe_interiornet_lama.csv",index=False)