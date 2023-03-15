import pandas as pd, glob, os


columns = ["image_path","image_path_target","mask_path","segmask_path","partition"]

df = pd.DataFrame(columns = columns)

mother_dir = "/data01/lorenzo.stacchio/TU GRAZ/Stable_Diffusion_Inpaiting/Datasets/DynaFill/"

out_dir_df = "data/modules/DYNAFILL/"

partitions = ["training","validation"]

# POSTFIX
mask_dir = "semseg/dynamic_masks/"
seg_dir = "semseg/static/"
image_dir_real =  "rgb/dynamic/"
image_dir_target =  "rgb/static/"

list_partition = []
total_paths = []

for partition in partitions:
    partition_dir = mother_dir + partition + "/"
    paths = list(glob.glob(partition_dir + "*.png")) 
    total_paths.extend(paths)      
    list_partition.extend(["train"]*len(paths))


df["partition"] = list_partition

for col, dir_root in zip(columns, [image_dir_real,image_dir_target,mask_dir,seg_dir]):
    curr_paths = [dir_root + os.path.basename(x) for x in paths]
    df[col] = curr_paths

print(df)
exit()
df.to_csv(out_dir_df + "full_to_target.csv",index=False)   
    





