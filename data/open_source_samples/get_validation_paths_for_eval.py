import pandas as pd 

mother_dir = "/data01/lorenzo.stacchio/TU GRAZ/Stable_Diffusion_Inpaiting/stable-diffusion_custom_inpaint/data/open_source_samples/"
csv_file = mother_dir + "dataframe_interiornet.csv"

df_filtered = pd.read_csv(csv_file)

df_filtered =  df_filtered[df_filtered["partition"]=="validation"]

for col in list(df_filtered.columns)[:2]:
    df_filtered[col] = df_filtered[col].apply(lambda x: mother_dir + x) 

df_filtered.to_csv(mother_dir + "dataframe_filtered_test.csv",index=False)