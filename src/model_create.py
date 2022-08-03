from email.policy import strict
import gc
import io
import math
import sys

sys.path.append('./k-diffusion')
sys.path.append('./guided-diffusion')
sys.path.append('./v-diffusion-pytorch')
sys.path.append('./latent-diffusion')

from loguru import logger
import accelerate
import clip
import k_diffusion as K
import lpips
import torch
from torchvision import transforms

from omegaconf import OmegaConf

from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
import diffusion.models.cc12m_1 as Vdiff
from ldm.util import instantiate_from_config


import paramsGen
import denoisers
import cond_fns
import noiseSched


# returns: model config, initial model, model wraper
def CreateOpenAIModel(model_path: str, image_size: int, device, strictMode=True):
    model_config = model_and_diffusion_defaults()
    model_config.update({
        'attention_resolutions': '32, 16, 8',
        'class_cond': False,
        'diffusion_steps': 1000,
        'rescale_timesteps': True,
        'timestep_respacing': '1000',
        'learn_sigma': True,
        'noise_schedule': 'linear',
        'num_channels': 256,
        'num_head_channels': 64,
        'num_res_blocks': 2,
        'resblock_updown': True,
        'use_checkpoint': False,
        'use_fp16': True,
        'use_scale_shift_norm': True,
    })
    model_config['image_size'] = image_size
    model_path = model_path

    model, diffusion = create_model_and_diffusion(**model_config)
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=strictMode)
    model.requires_grad_(False).eval().to(device)
    if model_config['use_fp16']:
        model.convert_to_fp16()

    model_wrap = K.external.OpenAIDenoiser(model, diffusion, device=device)

    return model_config, model, model_wrap




# returns: model config, initial model, model wraper
def CreateCompVisModel(model_path: str, config_path:str, image_size: int, device, requireGrad):
        model_config = OmegaConf.load(config_path)

        model_config['image_size'] = image_size 

        model = instantiate_from_config(model_config.model)
        model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)


        #false is better, but apparently something in there requires grad... ( its clip)
        model.requires_grad_(requireGrad).eval().to(device)

        model_wrap = K.external.CompVisDenoiser(model, False, device=device)

        return model_config, model, model_wrap


