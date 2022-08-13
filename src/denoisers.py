import gc
import io
import math
import sys

sys.path.append('./k-diffusion')
sys.path.append('./guided-diffusion')
sys.path.append('./v-diffusion-pytorch')

import k_diffusion as K
import torch
from torch import nn
from loguru import logger





class GuidedDenoiserWithGrad(nn.Module):
    def __init__(self, model, cond_fn):
        super().__init__()
        self.inner_model = model
        self.cond_fn = cond_fn
        self.orig_denoised = None

    def forward(self, x, sigma, **kwargs):
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            denoised = self.inner_model(x, sigma, **kwargs)
            self.orig_denoised = denoised.detach()
            cond_grad = self.cond_fn(x, sigma, denoised=denoised, **kwargs)
        cond_denoised = denoised + cond_grad * K.utils.append_dims(sigma ** 2, x.ndim)
        return cond_denoised



class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):                    
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)

        cond_in = torch.cat([uncond, cond])            
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)   
        val = uncond + (cond - uncond) * cond_scale            
        #print("DEBUGGING ONLY! SLOW!:  CFGDenoiser fwd return: " + str(val))
        return val  
