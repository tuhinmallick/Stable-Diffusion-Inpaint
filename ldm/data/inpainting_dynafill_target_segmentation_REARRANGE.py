import os
import numpy as np
import PIL
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from torch.utils.data import DataLoader
import torch
from einops import rearrange
import cv2


class InpaintingDynaFillARRANGETargetSegmentation(Dataset):
    def __init__(self,
                 csv_file,
                 data_root,
                 partition,
                 size,
                 interpolation="bicubic",
                 controlNet = False
                 ):

        self.csv_df = pd.read_csv(csv_file)
        # filter partition
        self.csv_df = self.csv_df[self.csv_df["partition"] == partition]
        self._length = len(self.csv_df)
        self.controlNet = controlNet
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
        # image = np.array(Image.open(image_path).convert("RGB"))
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image = image.astype(np.float32)/255.0
        # MASK AND MASKED IMAGE
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = mask.astype(np.float32)/255.0
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = mask[...,None]
        masked_image = (1-mask)*image  # nullify white part of the images

        # IMAGE TARGET
        image_target = cv2.imread(image_target_path)
        image_target = cv2.cvtColor(image_target, cv2.COLOR_BGR2RGB)
        image_target = image_target.astype(np.float32)/255.0

        # SEGMENTATION MASK
        seg_mask = cv2.imread(seg_mask_path)
        seg_mask = cv2.cvtColor(seg_mask, cv2.COLOR_BGR2RGB)
        seg_mask = seg_mask.astype(np.float32)/255.0

        mask_with_seg = mask*seg_mask
        seg_mask_maked = (1-mask)*seg_mask

        # The input image is the target, conditioned by masked version of the origin
        if self.controlNet:
            batch = {"image": image_target, "mask": mask,
                 "masked_image": masked_image, "hint": seg_mask, "masked_hint": seg_mask_maked, "hint_with_mask": mask_with_seg}
        else:
            batch = {"image": image_target, "mask": mask,
                    "masked_image": masked_image, "seg_mask": seg_mask, "masked_seg_mask": seg_mask_maked, "mask_with_seg":mask_with_seg}

        return batch

    def __getitem__(self, i):
        
        example_i = dict((k, self.labels[k][i]) for k in self.labels)

        transformed_example = self._transform_and_normalize_inference(
            example_i["file_path_"], example_i["file_path_target_"], example_i["file_path_mask_"], example_i["file_path_segmask_"], resize_to=self.size)

        transformed_example = {k:self.transform(v) for k,v in transformed_example.items()}

        return transformed_example


class InpaintingDynaFillARRANGETargetSegmentationTrain(InpaintingDynaFillARRANGETargetSegmentation):
    def __init__(self, csv_file, data_root, **kwargs):
        super().__init__(csv_file=csv_file, partition="train", data_root=data_root, **kwargs)

        self.transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize((self.size,self.size)),
                        transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])


class InpaintingDynaFillARRANGETargetSegmentationValidation(InpaintingDynaFillARRANGETargetSegmentation):
    def __init__(self, csv_file, data_root, **kwargs):
        super().__init__(csv_file=csv_file, partition="validation",
                         data_root=data_root, **kwargs)

        self.transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize((self.size,self.size)),
                        # transforms.Lambda(lambda x: x * 2. - 1.)])
                        transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])



if __name__ == "__main__":
    size = 256

    de_transform = transforms.Compose([
        #transforms.Lambda(lambda x: (x +1.) / 2.),
        transforms.Lambda(lambda x: rearrange((x +1.) / 2., 'b h w c ->b c h w')),
        transforms.Normalize(mean=[0., 0., 0.],
                             std=[1/255, 1/255, 1/255]),
        # transforms.Lambda(lambda x: rearrange((x +1.) / 2., 'b h w c ->b c h w'))
        # transforms.Normalize(mean=0., # it projects to all the channels
        #                      std=1/255),
    ])

    de_transform_mask = transforms.Compose([
        #transforms.Lambda(lambda x: (x +1.) / 2.),
        transforms.Lambda(lambda x: rearrange((x +1.) / 2., 'b h w c ->b c h w')),
        transforms.Normalize(mean=[0.],
                             std=[1/255]),
    ])
    data_root = "/data01/lorenzo.stacchio/TU GRAZ/Stable_Diffusion_Inpaiting/Datasets/DynaFill/"
    csv_file = "data/modules/DYNAFILL/full_to_target.csv"

    ip_train = InpaintingDynaFillARRANGETargetSegmentationTrain(
        size=256, csv_file=csv_file, data_root=data_root)

    ip_train_loader = DataLoader(ip_train, batch_size=1, num_workers=4,
                                 pin_memory=True, shuffle=False)

    for idx, batch in enumerate(ip_train_loader):
        im_keys = ['image', 'masked_image', 'mask', "seg_mask", "masked_seg_mask", "mask_with_seg"]
        for k in im_keys:
            # print(batch[k].shape)
            image_de = batch[k]
            # image_de = (image_de + 1.)/2.
            if k=="mask":
                image_de = de_transform_mask(image_de)
            else: 
                image_de = de_transform(image_de)

            rgb_img = (image_de).type(torch.uint8).squeeze(0)
            rgb_img = rgb_img.permute(1, 2, 0).numpy() # again because we did a transformation

            cv2.imwrite(img=cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB), filename="ldm/data/test_loader_dynafill_rearrange/%s_test.jpg" % k)
        break