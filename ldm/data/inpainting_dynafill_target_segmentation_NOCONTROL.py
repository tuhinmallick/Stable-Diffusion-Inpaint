import os
import numpy as np
import PIL
from PIL import Image,ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from torch.utils.data import DataLoader
import torch
from einops import rearrange
import cv2 

class InpaintingDynaFillNOCONTROLTargetSegmentation(Dataset):
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

        self.image_paths = self.csv_df["image_path"]
        self.image_paths_target = self.csv_df["image_path_target"]
        self.mask_image = self.csv_df["mask_path"]
        self.segmask_image = self.csv_df["segmask_path"]
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
            "relative_file_path_target_": [l for l in self.image_paths_target],
            "file_path_target_": [os.path.join(self.data_root, l)
                           for l in self.image_paths_target],
            "relative_file_path_mask_": [l for l in self.mask_image],
            "file_path_mask_": [os.path.join(self.data_root, l)
                           for l in self.mask_image],
            "relative_file_path_segmask_": [l for l in self.segmask_image],
            "file_path_segmask_": [os.path.join(self.data_root, l)
                           for l in self.segmask_image],
        }

    def __len__(self):
        return self._length

    def _transform_and_normalize_inference(self, image_path, image_target_path, mask_path, seg_mask_path, resize_to):
        image = np.array(Image.open(image_path).convert("RGB"))
        
        if image.shape[0]!=resize_to or image.shape[1]!=resize_to:
            image = cv2.resize(src=image, dsize=(resize_to,resize_to), interpolation = cv2.INTER_AREA)
        image = image.astype(np.float32)/255.0

        image = image[None].transpose(0,3,1,2)
        image = torch.from_numpy(image)

        mask = np.array(Image.open(mask_path).convert("L"))
        
        if mask.shape[0]!=resize_to or mask.shape[1]!=resize_to:
            mask = cv2.resize(src=mask, dsize=(resize_to,resize_to), interpolation = cv2.INTER_AREA)
        mask = mask.astype(np.float32)/255.0

        mask = mask[None,None]
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)

        masked_image = (1-mask)*image
        
        ##IMAGE TARGET
        image_target = np.array(Image.open(image_target_path).convert("RGB"))
        
        if image_target.shape[0]!=resize_to or image_target.shape[1]!=resize_to:
            image_target = cv2.resize(src=image_target, dsize=(resize_to,resize_to), interpolation = cv2.INTER_AREA)
        image_target = image_target.astype(np.float32)/255.0

        image_target = image_target[None].transpose(0,3,1,2)
        image_target = torch.from_numpy(image_target)
        
        ## SEGMENTATION MASK
        seg_mask = np.array(Image.open(seg_mask_path).convert("RGB"))

        if seg_mask.shape[0]!=resize_to or seg_mask.shape[1]!=resize_to:
            seg_mask = cv2.resize(seg_mask, (resize_to, resize_to), interpolation=cv2.INTER_NEAREST)
        seg_mask = seg_mask.astype(np.float32)/255.0
        seg_mask = seg_mask[None].transpose(0,3,1,2)
        seg_mask = torch.from_numpy(seg_mask)

        # The input image is the target, conditioned by masked version of the origin        
        batch = {"image": image_target, "mask": mask, "masked_image": masked_image, "seg_mask": seg_mask}


        for k in batch:
            batch[k] = batch[k]*2.0-1.0
            if k=="mask":
                batch[k] = torch.squeeze(batch[k], dim=1) # we are in get item here, so one at a time
            else:
                batch[k] = torch.squeeze(batch[k], dim=0)
        return batch

    def __getitem__(self, i):
                
        example2 = dict((k, self.labels[k][i]) for k in self.labels)

        add_dict = self._transform_and_normalize_inference(example2["file_path_"],example2["file_path_target_"],example2["file_path_mask_"],example2["file_path_segmask_"], resize_to=self.size)
        
        example2.update(add_dict)

        return example2


class InpaintingDynaFillNOCONTROLTargetSegmentationTrain(InpaintingDynaFillNOCONTROLTargetSegmentation):
    def __init__(self, csv_file, data_root, **kwargs):
        super().__init__(csv_file=csv_file, partition="train",data_root=data_root,**kwargs)
        self.transform = transforms.Compose([
                transforms.Resize((self.size,self.size)),
                transforms.ToTensor(),
        ])

        self.transform_mask = transforms.Compose([
                transforms.Resize((self.size,self.size)),
                transforms.ToTensor(),
        ])


class InpaintingDynaFillNOCONTROLTargetSegmentationValidation(InpaintingDynaFillNOCONTROLTargetSegmentation):
    def __init__(self, csv_file,data_root, **kwargs):
        super().__init__(csv_file=csv_file, partition="validation", data_root=data_root, **kwargs)
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

    data_root = "/data01/lorenzo.stacchio/TU GRAZ/Stable_Diffusion_Inpaiting/stable-diffusion_custom_inpaint/data/open_source_samples/dynafill/subset/"
    csv_file = data_root + "full_to_target.csv"
    
    ip_train = InpaintingDynaFillNOCONTROLTargetSegmentationTrain(size = 256,csv_file=csv_file, data_root=data_root)
    ip_train_loader = DataLoader(ip_train, batch_size=1, num_workers=4,
                          pin_memory=True, shuffle=True)

    for idx, batch in enumerate(ip_train_loader):
        im_keys = ['image', 'image_target', 'masked_image', 'mask', "hint"]
        for k in im_keys:
            # print(batch[k].shape)               
            image_de = batch[k]
            image_de = (image_de + 1)/2

            if k=="mask":
                image_de = de_transform_mask(image_de)
            else:
                image_de = de_transform(image_de)
                
            rgb_img = (image_de).type(torch.uint8).squeeze(0)
            # print(rgb_img.shape)
            img = transforms.ToPILImage()(rgb_img)  
            # print(img.size)
            img.save("ldm/data/test_loader_dynafill/%s_test.jpg" % k)

        break