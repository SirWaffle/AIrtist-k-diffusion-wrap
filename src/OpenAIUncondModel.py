
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
import modelWrap

from ldm.modules.encoders.modules import FrozenCLIPTextEmbedder



class OpenAIUncondModel(modelWrap.ModelWrap):
    def __init__(self):
        self.model_path = None
        #self.config_path = None        

        self.default_image_size_x = None
        self.default_image_size_y = None

        #loaded info
        self.model = None
        self.model_config = None
        self.kdiffExternalModelWrap = None #kdiff model wrapper
        self.default_imageTensorSize = None

        self.default_guiding = 'CLIP'


    def ModelLoadSettings(self):
        self.model_path = "D:/ml/models-diffusion/the-eye/512x512_diffusion_uncond_finetune_008100.pt" 
        #self.model_path = "D:/ml/models-diffusion/the-eye/256x256_diffusion_uncond.pt"
        self.default_image_size_x = 512
        self.default_image_size_y = 512
        self.channels = 3


    def LoadModel(self, device):
        self.model_config, self.model, self.kdiffExternalModelWrap = model_create.CreateOpenAIModel(self.model_path, self.default_image_size_x, device, True)    
        self.imageTensorSize = self.default_image_size_x

        #self.model_config = OmegaConf.load(self.config_path)


        #self.kdiffExternalModelWrap = K.external.CompVisDenoiser(self.model, False, device=device)


    def RequestImageSize(self, inst:modelWrap.ModelContext, x, y):

        if x == -1:
            x = self.default_image_size_x
        if y == -1:
            y = self.default_image_size_y

        image_size_x = x - (x % 16)
        image_size_y = image_size_x
        inst.image_tensor_size = image_size_x
        inst.image_size_x = image_size_x
        inst.image_size_y = image_size_y

        return inst


    def CreateCFGDenoiser(self, inst:modelWrap.ModelContext, clipEncoder:clipWrap.ClipWrap, cfgPrompts, condScale, genParams:paramsGen.ParamsGen):
        print("No CFG denoiser for uncond openAI models")
        return inst


    def CreateClipGuidedDenoiser(self, inst:modelWrap.ModelContext,  cw:clipWrap.ClipWrap, clipPrompts, genParams:paramsGen.ParamsGen, device):
        target_embeds, weights = cw.GetTextPromptEmbeds(clipPrompts, device)
        inst.target_clip_embeds.extend(target_embeds)
        inst.clip_weights.extend(weights)

        inst.condFns = cond_fns.CondFnClipGuidedObj(self.model, cw, inst, genParams)       

        inner_model = inst.kdiffModelWrap
        if inner_model == None:
            inner_model = self.kdiffExternalModelWrap   

        #TODO: figure out how to handle multiples of these needing extra args or something
        #clip denoiser doesnt need any conditional stuff at all, and doesnt need unconditional
        #if inst.extra_args == None:
        #    #TODO: get rid of this, used to get size of embeds the model is expecting
        #    clipTextEmbed = clipWrap.FrozenCLIPTextEmbedder(cw.model)
        #    embeds = clipTextEmbed.encode(clipPrompts).float()

        #    c = torch.zeros_like(embeds)
        #    inst.extra_args = {'cond': c }

        inst.kdiffModelWrap = denoisers.GuidedDenoiserWithGrad(inner_model, inst.condFns.cond_fn)  
        return inst