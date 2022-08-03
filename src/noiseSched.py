import gc
import io
import math
import sys

sys.path.append('./k-diffusion')
sys.path.append('./guided-diffusion')
sys.path.append('./v-diffusion-pytorch')

from loguru import logger
import accelerate
import clip
import k_diffusion as K
import lpips
import torch
from torchvision import transforms



from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
import diffusion.models.cc12m_1 as Vdiff


import paramsGen
import denoisers
import cond_fns

def GetSigmas(genParams: paramsGen.ParamsGen, model_wrap, device):

    # default to model sigmas
    sigmas = model_wrap.get_sigmas(genParams.n_steps)
    if len(sigmas) > 1:
        smax = max(sigmas).to('cpu')
        smin = min(sigmas).to('cpu')
    
    if genParams.sigma_max >= 0:
        smax = min(genParams.sigma_max, smax.to('cpu'))

    if genParams.sigma_min >= 0:
        smin = max(genParams.sigma_min, smin.to('cpu'))

    if genParams.noiseSchedule.lower() == 'model':
        sigmas = model_wrap.get_sigmas(genParams.n_steps)
    
    if genParams.noiseSchedule.lower() == 'karras':
        sigmas = K.sampling.get_sigmas_karras(genParams.n_steps, smin, smax, rho=7., device=device)

    if genParams.noiseSchedule.lower() == 'exp':
        sigmas = K.sampling.get_sigmas_exponential(genParams.n_steps, smin, smax, device=device)
    
    return sigmas
