import sys, os
import torch
sys.path.append(os.getcwd()) # run from root directory only! python ldm/models/ControlNet/tool_add_control.py
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import einops

def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model

root_dir = "/data01/lorenzo.stacchio/TU GRAZ/Stable_Diffusion_Inpaiting/stable-diffusion_custom_inpaint/"

input_path = root_dir + "models/ldm/inpainting_big/model_compvis.ckpt"
output_path = root_dir + "models/ldm/SEG_INPAINT/model_SEG_RGB.ckpt"

config_path =  root_dir + 'models/ldm/SEG_INPAINT/SEG_INPAINT.yaml'


assert os.path.exists(input_path), 'Input model does not exist.'
assert not os.path.exists(output_path), 'Output filename already exists.'
assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'


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
index_start = 0 
index_end=3
to_clone_channels = 3




for k,v in scratch_dict.items():
    if "model.diffusion_model.input_blocks.0.0.weight" == k: # here place old scratch_dict 
        # print(k)
        # print(v.shape)
        print(f'These weights are newly added: {k}')
        old_weights = pretrained_weights[k].clone()
        # considering the order of the channels, the last three input channels should be repeated from
        # the mask one (the actual last) to start managing the novel segmentation condition
        rgb_weights = old_weights[:,index_start:index_end,:,:]

        target_dict[k] = scratch_dict[k].clone()
        # print(target_dict[k].shape)
        # print(seg_weiths.shape)
        channels_to_clone = (target_dict[k].shape[1]-to_clone_channels)

        target_dict[k][:,:channels_to_clone,:,:] = old_weights # fill old
        target_dict[k][:,channels_to_clone:,:,:] = rgb_weights # novel weights
        # new_weights = scratch_dict
    else:
        target_dict[k] = pretrained_weights[k].clone()
        # print(f'Clone old weights from: {k}')

 
model.load_state_dict(target_dict, strict=True)
torch.save(model.state_dict(), output_path)
print('Done.')