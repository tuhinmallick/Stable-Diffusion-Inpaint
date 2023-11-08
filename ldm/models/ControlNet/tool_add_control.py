import sys, os
import torch
sys.path.append(os.getcwd()) # run from root directory only! python ldm/models/ControlNet/tool_add_control.py
from ldm.models.ControlNet.model import create_model

root_dir = "/data01/lorenzo.stacchio/TU GRAZ/Stable_Diffusion_Inpaiting/stable-diffusion_custom_inpaint/"

input_path = f"{root_dir}models/ldm/inpainting_big/model_compvis.ckpt"
output_path = f"{root_dir}models/ldm/ControlNet/model_cn.ckpt"

config_path = f'{root_dir}configs/latent-diffusion/inpainting_runaway_interiornet_FULL_ControlNet.yaml'


assert os.path.exists(input_path), 'Input model does not exist.'
assert not os.path.exists(output_path), 'Output filename already exists.'
assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'


def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    return (False, '') if p != parent_name else (True, name[len(parent_name):])


model = create_model(config_path=config_path)

pretrained_weights = torch.load(input_path)
if 'state_dict' in pretrained_weights:
    pretrained_weights = pretrained_weights['state_dict']

scratch_dict = model.state_dict()

target_dict = {}
for k in scratch_dict.keys():
    is_control, name = get_node_name(k, 'control_')
    copy_k = f'model.diffusion_{name}' if is_control else k
    if copy_k in pretrained_weights:
        target_dict[k] = pretrained_weights[copy_k].clone()
    else:
        target_dict[k] = scratch_dict[k].clone()
        print(f'These weights are newly added: {k}')

model.load_state_dict(target_dict, strict=True)
torch.save(model.state_dict(), output_path)
print('Done.')
