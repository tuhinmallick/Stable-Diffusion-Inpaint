## BUG FIXEX
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

```Import error taming```: just install from pip with pip install taming-transformers

```ImportError: cannot import name 'VectorQuantizer2'```: replace quantize.py in taming/modules/vqvae/quantize.py with [this](https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py)

## Input image format

b h w c ---> a causa del ldm/models/diffusion/ddpm.py get_inpt()

## Inpainting weights
Taken from: https://github.com/CompVis/stable-diffusion/issues/17#issuecomment-1224165074


wget -O models/ldm/inpainting_big/last.ckpt https://heibox.uni-heidelberg.de/f/4d9ac7ea40c64582b7c9/?dl=1

To use that of compvis official (the zip file is instead a cpkt)

wget -O models/ldm/inpainting_big/model.ckpt https://ommer-lab.com/files/latent-diffusion/inpainting_big.zip --no-check-certificate

## UNdertanding UNET

https://nn.labml.ai/diffusion/ddpm/unet.html


## CUSTOM TRAINING
E' tutto nel file main.py

Per le config (base_configs) bisogna dargliere in ordine scegliendo i vari config file per i vari pezzi e fornendo quindi anche i cpkt per tutti i pezzi della rete.
Nel nostro caso, vogliamo finetunare solo il latnt, come hanno fatto su tutti i train definiti in configs/latent-diffusion che hanno come modello target definito nello yaml file:

target: ldm.models.diffusion.ddpm.LatentDiffusion


Comunque bisogna fare un grosso merge di tutte quelle configurazioni e poi fare un training col main.py.


## PROBLEMA
sono riuscito a caricare i pesi del modello ma per partire da un resume mi richiedeva quelli del sampler ddpim, 
che non sono stati rilasciati.


## PROBLEMA 2

Unexpected key(s) in state_dict: "ddim_sigmas", "ddim_alphas", "ddim_alphas_prev", "ddim_sqrt_one_minus_alphas".

In pratica nei pesi forniti da Stable Diffusion, mancano i pesi del sampler ddim, quindi del modello implicito più efficiente.
Questo vuol dire che verranno trainati da zero anche facendo fine-tuning. 

## Visualize Network

https://github.com/szagoruyko/pytorchviz


# SAVE ONLY LAST EPOCH

default_modelckpt_cfg["params"]["save_top_k"] = 0
