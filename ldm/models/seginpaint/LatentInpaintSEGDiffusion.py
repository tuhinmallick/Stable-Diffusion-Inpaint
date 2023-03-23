import torch
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
import contextlib
from torchvision.utils import make_grid
from ldm.util import exists,  instantiate_from_config
from ldm.models.diffusion.ddpm import LatentDiffusion


class LatentInpaintDiffusionSEG_CHANNELS(LatentDiffusion):
    """
    can just run as pure inpainting model (only concat mode).
    To disable finetuning mode, set finetune_keys to None
     """
    def __init__(self,
                # DEFAULT FINETUNE KEYS --> use to add novel channels to the concatenation
                 finetune_keys_to_retain=("model.diffusion_model.input_blocks.0.0.weight",
                                "model_ema.diffusion_modelinput_blocks00weight"
                                ),
                 concat_keys=("masked_image","mask","seg_mask"), # AS IN INFERENCE
                 masked_image_key="masked_image",
                 encode_seg_with_first_stage = False,
                #  keep_finetune_dims=4,  # if model was trained without concat mode before and we would like to keep these channels
                 keep_finetune_dims=7,  # if model was trained with concat mode before and we would like to keep these channels and add more channels to condition
                 c_concat_log_start=0, # to log reconstruction of c_concat codes, first three channels in our case
                 c_concat_log_end=3,
                 c_seg_cat_log_start = 4,
                 c_seg_cat_log_end =7,
                 *args, **kwargs
                 ):
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", list())
        super().__init__(*args, **kwargs)
        self.masked_image_key = masked_image_key
        self.encode_seg_with_first_stage = encode_seg_with_first_stage
        assert self.masked_image_key in concat_keys
        self.finetune_keys_to_retain = finetune_keys_to_retain
        
        self.concat_keys = concat_keys
        self.keep_dims = keep_finetune_dims
        self.c_concat_log_start = c_concat_log_start
        self.c_concat_log_end = c_concat_log_end
        
        self.c_seg_cat_log_start = c_seg_cat_log_start
        self.c_seg_cat_log_end = c_seg_cat_log_end

        if exists(self.finetune_keys_to_retain): assert exists(ckpt_path), 'can only finetune from a given checkpoint'
        if exists(ckpt_path): self.init_from_ckpt(ckpt_path, ignore_keys)
        
    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]

            # make it explicit, finetune by including extra input channels
            # HERE THE FINETUNED IS INTENDED TO BE IN THE CASE YOU WANT TO ADD MORE THAN
            # CLASSICAL CONDITIONING IN THE INPUT CHANNELS AND YOU WANT TO RETAIN THE KNOWLEDGE FROM BEFORE
            if exists(self.finetune_keys_to_retain) and k in self.finetune_keys_to_retain:
                new_entry = None
                for name, param in self.named_parameters():
                    if name in self.finetune_keys_to_retain:
                        print(f"modifying key '{name}' and keeping its original {self.keep_dims} (channels) dimensions only")
                        new_entry = torch.zeros_like(param)  # zero init
                assert exists(new_entry), 'did not find matching parameter to modify'
                new_entry[:, :self.keep_dims, ...] = sd[k][:, :self.keep_dims, ...]
                sd[k] = new_entry
                
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
    
    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())

        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        
        # Optimization made by me
        opt = torch.optim.AdamW(params, lr=lr)

        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt
                       
    @torch.no_grad()
    def get_input(self, batch, k, cond_key=None, bs=None, return_first_stage_outputs=False):
        cdict = {n:p for n,p in self.named_parameters()}
        a = cdict["model.diffusion_model.input_blocks.0.0.weight"]
        ttt = a[:,self.keep_dims:,:]
        # note: restricted to non-trainable encoders currently
        assert not self.cond_stage_trainable, 'trainable cond stages not yet supported for inpainting'
        z, c, x, xrec, xc = super().get_input(batch, self.first_stage_key, return_first_stage_outputs=True,
                                              force_c_encode=True, return_original_cond=True, bs=bs)

        assert exists(self.concat_keys)
        c_cat = list()
        
        # THE ORDER of self.concat_keys MUST BE AS IN INFERENCE
        for ck in self.concat_keys:
            # TODO: ATTENZIONE TOLTO REARRANGE (ORA RIMESSO)

            cc = rearrange(batch[ck], 'b h w c -> b c h w').to(memory_format=torch.contiguous_format).float()
            # cc = batch[ck].to(memory_format=torch.contiguous_format).float()
            if bs is not None:
                cc = cc[:bs]
                cc = cc.to(self.device)
            bchw = z.shape
            if self.encode_seg_with_first_stage and ck=="seg_mask":
                # print("Encoding %s with first stage" % ck )
                cc = self.get_first_stage_encoding(self.encode_first_stage(cc))
            else:
                if ck != self.masked_image_key:
                    # interpolating also the seg mask
                    cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
                else:
                    cc = self.get_first_stage_encoding(self.encode_first_stage(cc))
            c_cat.append(cc)

        c_cat = torch.cat(c_cat, dim=1)
        
        # NO CROSS ATTENTION
        all_conds = [c_cat]
        
        if return_first_stage_outputs:
            return z, all_conds, x, xrec, xc
        return z, all_conds

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=50, ddim_eta=0,
                  plot_denoise_rows=False, plot_diffusion_rows=True, use_ema_scope=True,
                   **kwargs):
        ema_scope =  self.ema_scope if use_ema_scope else contextlib.suppress
        use_ddim = ddim_steps is not None

        log = dict()
        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key, bs=N, return_first_stage_outputs=True)
        # c_cat, c = c["c_concat"][0], c["c_crossattn"][0]
        c = c[0]
        c_cat = c#no cross attention

        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["reconstruction"] = xrec

        if not (self.c_concat_log_start is None and self.c_concat_log_end is None):
            log["c_concat_decoded"] = self.decode_first_stage(c_cat[:,self.c_concat_log_start:self.c_concat_log_end])

        if self.encode_seg_with_first_stage:
            log["seg_decoded"] = self.decode_first_stage(c_cat[:,self.c_seg_cat_log_start:self.c_seg_cat_log_end])

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            with ema_scope("Sampling"):
                samples, z_denoise_row = self.sample_log(cond=c_cat,
                                                            batch_size=N, ddim=use_ddim,
                                                            ddim_steps=ddim_steps, eta=ddim_eta)
           
                                    
            x_samples = self.decode_first_stage(samples)

            log["samples"] = x_samples
            
           
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        log["seg_mask"] = rearrange(batch["seg_mask"], 'b h w c -> b c h w').to(memory_format=torch.contiguous_format).float()
        log["mask"] = rearrange(batch["mask"], 'b h w c -> b c h w').to(memory_format=torch.contiguous_format).float()
        log["masked_image"] = rearrange(batch["masked_image"], 'b h w c -> b c h w').to(memory_format=torch.contiguous_format).float()

        return log



########## TODO ##########
class LatentInpaintDiffusionSEG_CROSSATTENTION_CHANNELS(LatentDiffusion):
    
    """
    can just run as pure inpainting model (only concat mode).
    To disable finetuning mode, set finetune_keys to None
     """
    def __init__(self,
                # DEFAULT FINETUNE KEYS --> use to add novel channels to the concatenation
                 finetune_training_keys=("model.diffusion_model.input_blocks.0.0.weight",
                                "model_ema.diffusion_modelinput_blocks00weight"
                                ),
                 finetune_keys_to_retain=("model.diffusion_model.input_blocks.0.0.weight",
                                "model_ema.diffusion_modelinput_blocks00weight"
                                ),
                 concat_keys=("masked_image","mask","seg_mask"), # AS IN INFERENCE
                 masked_image_key="masked_image",
                #  keep_finetune_dims=4,  # if model was trained without concat mode before and we would like to keep these channels
                 keep_finetune_dims=7,  # if model was trained with concat mode before and we would like to keep these channels and add more channels to condition
                 c_concat_log_start=0, # to log reconstruction of c_concat codes, first three channels in our case
                 c_concat_log_end=3,
                 *args, **kwargs
                 ):
        pass