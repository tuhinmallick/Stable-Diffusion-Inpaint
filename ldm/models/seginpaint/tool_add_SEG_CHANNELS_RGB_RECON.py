import numpy as np
from omegaconf import OmegaConf
import sys
import os
import torch
# run from root directory only! python ldm/models/ControlNet/tool_add_control.py
sys.path.append(os.getcwd())
from ldm.util import instantiate_from_config


def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model


root_dir = "/data01/lorenzo.stacchio/TU GRAZ/Stable_Diffusion_Inpaiting/stable-diffusion_custom_inpaint/"

input_path = root_dir + "models/ldm/inpainting_big/model_compvis.ckpt"
output_path = root_dir + "models/ldm/SEG_INPAINT/model_SEG_RGB_RECON.ckpt"
config_path = root_dir + 'models/ldm/SEG_INPAINT/inpainting_DYNAFILL_SEG_RECON_256.yaml'


assert os.path.exists(input_path), 'Input model does not exist.'
assert not os.path.exists(output_path), 'Output filename already exists.'
assert os.path.exists(os.path.dirname(output_path)
                      ), 'Output path is not valid.'


def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]


model = create_model(config_path=config_path)

pretrained_weights = torch.load(input_path)
if 'state_dict' in pretrained_weights:
    pretrained_weights = pretrained_weights['state_dict']

scratch_dict = model.state_dict()

target_dict = {}

# REPLICATE WEIGHTS FROM MASK
# INPUT
index_start = 0
index_end = 6 # retain feature learned from noise and masked image

# OUTPUT
out_index_start = 0
out_index_end = 3


for k, v in scratch_dict.items():
    print(k, v.shape)
    if "model.diffusion_model.input_blocks.0.0.weight" == k:  # here place old scratch_dict
        # print(k)
        # print(v.shape)
        print(f'These weights are newly added: {k}')
        old_weights = pretrained_weights[k].clone()
        # considering the order of the channels, the last three input channels should be repeated from
        # the mask one (the actual last) to start managing the novel segmentation condition
        target_dict[k] = scratch_dict[k].clone()
        # new dimension with 6 channels as output
        target_dict[k][:, index_start:index_end, :, :] = old_weights[:,
                                                                  index_start:index_end, :, :]  # retained pre-trained one
    elif "model.diffusion_model.out.2.weight" == k:  # here place old scratch_dict
        old_weights = pretrained_weights[k].clone()
        # considering the order of the channels, the last three input channels should be repeated from
        # the mask one (the actual last) to start managing the novel segmentation condition
        target_dict[k] = scratch_dict[k].clone()
        # new dimension with 6 channels as output
        target_dict[k][out_index_start:out_index_end, :, :, :] = old_weights[out_index_start:out_index_end, :, :, :]  # retained pre-trained one
    elif "model.diffusion_model.out.2.bias" == k:
        old_weights = pretrained_weights[k].clone()
        # considering the order of the channels, the last three input channels should be repeated from
        # the mask one (the actual last) to start managing the novel segmentation condition
        target_dict[k] = scratch_dict[k].clone()
        # new dimension with 6 channels as output
        target_dict[k][out_index_start:out_index_end] = old_weights[out_index_start:out_index_end] 

    else:
        target_dict[k] = pretrained_weights[k].clone()
        # print(f'Clone old weights from: {k}')


model.load_state_dict(target_dict, strict=True)
torch.save(model.state_dict(), output_path)
print('Done.')