
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



class ModelContext:
    def __init__(self):
        self.extra_args = None
        self.sigmas = None
        self.condFns = None
        self.modelWrap:ModelWrap = None
        self.kdiffModelWrap = None

        self.target_cfg_embeds = None
        self.cfg_weights = None

        self.target_clip_embeds = []
        self.clip_weights = []

        self.init = None
        self.make_cutouts = None

        self.image_size_x = None
        self.image_size_y = None
        self.image_tensor_size = None


class ModelWrap:
    def __init__(self):
        self.model_path = None
        self.config_path = None        

        self.image_size_x = None
        self.image_size_y = None

        self.channels = None
        #loaded info
        self.model = None
        self.model_config = None
        self.kdiffExternalModelWrap = None #kdiff model wrapper

        self.default_guiding = 'CFG'

    def RequestImageSize(self, inst:ModelContext, x, y):
        self.image_size_x = x
        self.image_size_y = y

    def ModelLoadSettings(self):
        raise NotImplementedError

    def LoadModel(self, device):
        raise NotImplementedError

    def CreateModelInstance(self, device, clipWrapper:clipWrap.ClipWrap, genParams:paramsGen.ParamsGen, clip_guided) -> ModelContext:
        inst = ModelContext()
        inst.modelWrap = self
        return inst

    def CreateCFGDenoiser(self, inst:ModelContext, clipEncoder:clipWrap.ClipWrap, cfgPrompts, condScale, genParams:paramsGen.ParamsGen):
        raise NotImplementedError

    def CreateClipGuidedDenoiser(self, inst:ModelContext,  cw:clipWrap.ClipWrap, clipPrompts, genParams:paramsGen.ParamsGen, device):
        raise NotImplementedError     

    def GetSigmas(self, inst:ModelContext, genParams: paramsGen.ParamsGen, device):
        inst.sigmas = noiseSched.GetSigmas(genParams, self.kdiffExternalModelWrap, device)
        return inst

    def EncodeInitImage(self, initTensor):
        return initTensor

    def DecodeImage(self, imageTensor):
        return imageTensor
