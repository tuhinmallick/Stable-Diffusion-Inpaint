import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from torch.utils.data import DataLoader
import torch
from einops import rearrange

class InpaintingBase(Dataset):
    def __init__(self,
                 csv_file,
                 data_root,
                 partition,
                 size,
                 interpolation="bicubic",
                 ):

        self.csv_df = pd.read_csv(csv_file)
        self.csv_df = self.csv_df[self.csv_df["partition"]==partition] # filter partition
        self._length = len(self.csv_df)
        self.data_root = data_root
        self.size = size
        self.transform = None
        self.transform_mask = None

        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]

        self._length = len(self.csv_df)
        self.image_paths = self.csv_df["image_path"]
        self.mask_image = self.csv_df["mask_path"]
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
            "relative_file_path_mask_": [l for l in self.mask_image],
            "file_path_mask_": [os.path.join(self.data_root, l)
                           for l in self.mask_image],
        }

    def __len__(self):
        return self._length

    def _transform_and_normalize(self, image_path, mask_path):
        image = Image.open(image_path)
        image = image.convert("RGB")

        mask = Image.open(mask_path)

        # mask = mask.convert("RGB")
        pil_mask = mask.convert('L')
        black_image = Image.new('RGB', image.size)
        masked_image = Image.composite(image, black_image, pil_mask)
        # masked_image.save("masked_test.jpg")

        masked_image = masked_image.convert("RGB")

        # transpose because of ldm/models/diffusion/ddpm.py get_input()

        # image.save("before_transform.jpg")
        # Transformations
        image = self.transform(image)
        image = rearrange(image, 'c h w -> h w c')

        pil_mask = self.transform_mask(pil_mask)
        pil_mask = rearrange(pil_mask, 'c h w -> h w c')

        masked_image = self.transform(masked_image)
        masked_image = rearrange(masked_image, 'c h w -> h w c')

        return image, masked_image, pil_mask

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image,masked_image, mask = self._transform_and_normalize(example["file_path_"],example["file_path_mask_"])
        example["image"] = image
        example["masked_image"] = masked_image
        example["mask"] = mask
        return example


class InpaintingTrain(InpaintingBase):
    def __init__(self, **kwargs):
        super().__init__(csv_file="data/inpainting_dataset_surrogate/dataset.csv", partition="train",data_root="data/inpainting_dataset_surrogate/images/",**kwargs)
        self.transform = transforms.Compose([
                transforms.Resize((self.size,self.size)),
                transforms.ToTensor(),
        ])

        self.transform_mask = transforms.Compose([
                transforms.Resize((self.size,self.size)),
                transforms.ToTensor(),
        ])




class InpaintingValidation(InpaintingBase):
    def __init__(self, **kwargs):
        super().__init__(csv_file="data/inpainting_dataset_surrogate/dataset.csv", partition="validation", data_root="data/inpainting_dataset_surrogate/images/", **kwargs)
        self.transform = transforms.Compose([
                        transforms.Resize((self.size,self.size)),
                        transforms.ToTensor(),
        ])
        self.transform_mask = transforms.Compose([
                transforms.Resize((self.size,self.size)),
                transforms.ToTensor(),
        ])


if __name__=="__main__":
    size = 256
    transform = transforms.Compose([
        transforms.Resize((size,size)),
        transforms.ToTensor(),
    ])
    de_transform =  transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/255, 1/255 ,1/255 ]),
                    ])
    
    de_transform_mask =  transforms.Compose([ transforms.Normalize(mean = [ 0. ],
                                                     std = [ 1/255]),
                    ])

    ip_train = InpaintingTrain(size = 256)
    ip_train_loader = DataLoader(ip_train, batch_size=1, num_workers=4,
                          pin_memory=True, shuffle=True)

    for idx, batch in enumerate(ip_train_loader):
        im_keys = ['image', 'masked_image', 'mask']
        for k in im_keys:
            # print(batch[k].shape)               
            image_de = rearrange(batch[k], 'b h w c -> b c h w'  )
            if k=="mask":
                image_de = de_transform_mask(image_de)
            else:
                image_de = de_transform(image_de)
            rgb_img = (image_de).type(torch.uint8).squeeze(0)
            # rgb_img = rearrange(rgb_img, 'h w c -> c h w'  )
            # print(rgb_img.shape)
            img = transforms.ToPILImage()(rgb_img)  
            # print(img.size)
            img.save("ldm/data/test_loader_inpaint/%s_test.jpg" % k)
