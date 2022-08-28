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



import paramsGen
import denoisers

def GetSigmas(genParams: paramsGen.ParamsGen, model_wrap, device):

    # default to model sigmas
    if hasattr(model_wrap, 'get_sigmas'):
        sigmas = model_wrap.get_sigmas(genParams.n_steps)
    else:
        genParams.noiseSchedule = 'karras'
        sigmas = []
        smax = 10
        smin = 0.03

    if len(sigmas) > 1:
        smax = max(sigmas).to('cpu')
        smin = min(sigmas).to('cpu')
    
    if genParams.sigma_max >= 0:
        smax = min(genParams.sigma_max, smax.to('cpu'))

    if genParams.sigma_min >= 0:
        smin = max(genParams.sigma_min, smin.to('cpu'))

    if genParams.noiseSchedule.lower() == 'model':
        print("Noise Schedule: MODEL")
        sigmas = model_wrap.get_sigmas(genParams.n_steps)
    
    if genParams.noiseSchedule.lower() == 'karras':
        print("Noise Schedule:  KARRAS")
        sigmas = K.sampling.get_sigmas_karras(genParams.n_steps, smin, smax, rho=7., device=device)

    if genParams.noiseSchedule.lower() == 'exp':
        print("Noise Schedule: EXP")
        sigmas = K.sampling.get_sigmas_exponential(genParams.n_steps, smin, smax, device=device)
    
    return sigmas
