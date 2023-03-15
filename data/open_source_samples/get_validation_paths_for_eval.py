import pandas as pd 
import os, random
random.seed(42)

mother_dir = "/data01/lorenzo.stacchio/TU GRAZ/Stable_Diffusion_Inpaiting/stable-diffusion_custom_inpaint/data/open_source_samples/"
csv_file = mother_dir + "dataframe_interiornet_lama.csv"
out_path = mother_dir + ".".join(os.path.basename(csv_file).split(".")[:-1]) + "_validation.csv"

filter_samples_per_category = 100

df_filtered = pd.read_csv(csv_file)

df_filtered =  df_filtered[df_filtered["partition"]=="validation"]

df_temp = pd.DataFrame(columns = df_filtered.columns)

if filter_samples_per_category:
    df_filtered["mask_type"] = df_filtered["image_path"].apply(lambda x: x.split("/")[1]) 
    
    for kind_mask in df_filtered["mask_type"].unique():  
        sample_res = df_filtered[df_filtered['mask_type'].isin([kind_mask])]
        sample_res = sample_res.sample(n=filter_samples_per_category, random_state=42)
        df_temp = df_temp.append(sample_res,ignore_index = True)

df_temp = df_temp.drop(["mask_type"], axis=1)
df_filtered = df_temp


for col in list(df_filtered.columns)[:2]:
    df_filtered[col] = df_filtered[col].apply(lambda x: mother_dir + x) 


df_filtered.to_csv(out_path,index=False)