
import gc
import io
import math
import sys

sys.path.append('./../k-diffusion')
sys.path.append('./../guided-diffusion')
sys.path.append('./../v-diffusion-pytorch')

from functools import partial

import clip
import k_diffusion as K
import lpips
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms, utils
from torchvision.transforms import functional as TF
from tqdm.notebook import tqdm
from loguru import logger
from torchvision.utils import make_grid

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

import denoisers
import cutouts
import paramsGen
import cond_fns
import lossFunctions
import utilFuncs
import noiseSched
import model_create
import clipWrap
import cond_fns

from ldm.modules.encoders.modules import FrozenCLIPTextEmbedder

class ModelInstance:
    def __init__(self):
        self.extra_args = None
        self.sigmas = None
        self.condFns = None
        self.modelWrap = None
        self.kdiffModelWrap = None


class ModelWrap:
    def __init__(self):
        self.model_path = None
        self.config_path = None        
        self.modelNum = -1
        self.strictModelLoad = True

        self.using_openAi = True
        self.using_compvisLDM = False     

        self.image_size = None
        
        #loaded info
        self.model = None
        self.model_config = None
        self.kdiffExternalModelWrap = None #kdiff model wrapper
        self.imageTensorSize = None


    def ModelLoadSettings(self, modelNum):
        self.modelNum = modelNum

        self.using_openAi = True
        self.using_compvisLDM = False

        self.model_path = "D:/ml/models-diffusion/the-eye/512x512_diffusion_uncond_finetune_008100.pt"
        self.image_size = 512

        if self.modelNum == 2:
            self.model_path = "D:/ml/models-diffusion/the-eye/256x256_diffusion_uncond.pt"
            self.image_size = 256

        if self.modelNum == 3:
            self.model_path = "D:/ml/models-diffusion/openAI-diff/512x512_FeiArt-Handpainted-CG-Diffusion.pt"
            self.image_size = 512
            self.strictModelLoad = False

        if self.modelNum == 4 or self.modelNum == 6:
            self.using_openAi = False
            self.using_compvisLDM = True
            self.config_path = "D:/AIrtist/wes-diffusion-wrap/latent-diffusion/configs/latent-diffusion/txt2img-1p4B-eval.yaml"
            self.model_path = "D:/ml/models-diffusion/ldm-compvis/txt2img-f8-large-jack000-finetuned-fp16.ckpt" 
            self.image_size = 512
            if self.modelNum == 6:
                self.image_size = 256

        #model_path = "D:/ml/models-diffusion/vdiff/cc12m_1_cfg.pth"
        #image_size = 256

        if self.modelNum == 5:
            self.using_openAi = False
            self.using_compvisLDM = True
            self.config_path = "D:/ml/models-diffusion/laion-5B-kl-f8/laion-5B-kl-f8.yaml"
            self.model_path = "D:/ml/models-diffusion/laion-5B-kl-f8/laion-5B-kl-f8.ckpt" 
            self.image_size = 512


        if self.modelNum == 7 or self.modelNum == 8:
            self.using_openAi = False
            self.using_compvisLDM = True
            self.config_path = "D:/AIrtist/wes-diffusion-wrap/latent-diffusion/configs/retrieval-augmented-diffusion/768x768.yaml"
            self.model_path = "D:/ml/models-diffusion/ldm-compvis/RDM768x768_LD.ckpt" 
            self.image_size = 768
            if self.modelNum == 8:
                self.image_size = 1024





    def LoadModel(self, device):
        #this is very hacky
        if self.using_openAi == True:
            self.model_config, self.model, self.kdiffExternalModelWrap = model_create.CreateOpenAIModel(self.model_path, self.image_size, device, self.strictModelLoad)    
            self.imageTensorSize = self.image_size


        elif self.modelNum == 7 or self.modelNum == 8:
            self.model_config = OmegaConf.load(self.config_path)

            self.model = instantiate_from_config(self.model_config.model)
            self.model.load_state_dict(torch.load(self.model_path, map_location='cpu')["state_dict"], strict=False)
            self.model.requires_grad_(True).eval().to(device)

            self.kdiffExternalModelWrap = K.external.CompVisDenoiser(self.model, False, device=device)
            self.imageTensorSize = self.image_size//16  

        
        elif self.using_compvisLDM == True:
            #TODO: fix grad true/false based on needing clip...
            #defaulting to using grad now because its easier
            self.model_config, self.model, self.kdiffExternalModelWrap = model_create.CreateCompVisModel(self.model_path, self.config_path, self.image_size, device, True)

            self.imageTensorSize = self.image_size//8




    def CreateModelInstance(self, device, clipWrapper:clipWrap.ClipWrap, genParams:paramsGen.ParamsGen, clip_guided) -> ModelInstance:
        inst = ModelInstance()
        inst.modelWrap = self

        if self.using_openAi == True:
            inst.condFns = cond_fns.CondFnClipGuidedObj()
            inst.sigmas = noiseSched.GetSigmas(genParams, self.kdiffExternalModelWrap, device)
            inst.kdiffModelWrap = denoisers.GuidedDenoiserWithGrad(self.kdiffExternalModelWrap, inst.condFns.cond_fn, 3)        

        if self.modelNum == 7 or self.modelNum == 8:

            clipTextEmbed = clipWrap.FrozenCLIPTextEmbedder(clipWrapper.model)
            c = clipTextEmbed.encode(genParams.prompts).float()
            uc = torch.zeros_like(c)
            inst.extra_args = {'cond': c, 'uncond': uc, 'cond_scale': genParams.conditioning_scale}

            inst.sigmas = noiseSched.GetSigmas(genParams, self.kdiffExternalModelWrap, device)

            inst.kdiffModelWrap = denoisers.CFGDenoiser(self.kdiffExternalModelWrap, 16)

            if clip_guided == True:
                inst.condFns = cond_fns.CondFnClipGuidedCompvisObj(self.model)            
                inst.kdiffModelWrap = denoisers.GuidedDenoiserWithGrad(inst.kdiffModelWrap, inst.condFns.cond_fn, 16) 



        elif self.using_compvisLDM == True:
            n_samples = 1        

            if genParams.CFGprompts and len(genParams.CFGprompts[0]) > 2:
                c = self.model.get_learned_conditioning(n_samples * genParams.CFGprompts)
            else:
                c = self.model.get_learned_conditioning(n_samples * genParams.prompts)

            uc = self.model.get_learned_conditioning(n_samples * [""])
            inst.extra_args = {'cond': c, 'uncond': uc, 'cond_scale': genParams.conditioning_scale}

            inst.sigmas = noiseSched.GetSigmas(genParams, self.kdiffExternalModelWrap, device)
            inst.kdiffModelWrap = denoisers.CFGDenoiser(self.kdiffExternalModelWrap, 4)

            if clip_guided == True:
                inst.condFns = cond_fns.CondFnClipGuidedCompvisObj(self.model)            
                inst.kdiffModelWrap = denoisers.GuidedDenoiserWithGrad(inst.kdiffModelWrap, inst.condFns.cond_fn, 4) 

        return inst
