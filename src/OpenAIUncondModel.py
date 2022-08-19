import sys

sys.path.append('./../k-diffusion')
sys.path.append('./../guided-diffusion')
sys.path.append('./../v-diffusion-pytorch')

import torch

from loguru import logger

import denoisers
import paramsGen
import lossFunctions
import clipWrap
import modelWrap

import k_diffusion as K
import torch
from guided_diffusion.script_util import (create_model_and_diffusion,
                                          model_and_diffusion_defaults)



class CondFnClipGuidedObj:
    def __init__(self, model, clipWrapper:clipWrap.ClipWrap, modelCtx:modelWrap.ModelContext, genParams):
        super().__init__()
        self.normalize = clipWrapper.normalize
        self.clipWrapper:clipWrap.ClipWrap = clipWrapper
        self.lpips_model = clipWrapper.lpips_model
        self.genParams:paramsGen.ParamsGen = genParams

        self.model = model
        self.modelCtx:modelWrap.ModelContext = modelCtx
    
    def uncond_fn(self, x, sigma, denoised, **kwargs):
        logger.debug("uncond_fn")

    def cond_fn(self, x, sigma, denoised, **kwargs):
        batches = denoised.shape[0]

        #print("batches:", batches)
        #print("denoise: ", denoised.shape)
        #print("x:", x.shape)
        #print("sigma: ", sigma.shape)
        
        n = x.shape[0]
        # Anti-grain hack for the 256x256 ImageNet model
        #fac = sigma / (sigma ** 2 + 1) ** 0.5
        #denoised_in = x.lerp(denoised, fac)
        #denoised_in = denoised

        clip_in = self.normalize(self.modelCtx.make_cutouts(denoised.add(1).div(2)))
        image_embeds = self.clipWrapper.model.encode_image(clip_in).float()

        loss = None

        if self.genParams.clip_guidance_scale != 0:
            dists = lossFunctions.spherical_dist_loss(image_embeds[:, None], self.modelCtx.target_clip_embeds[None])
            dists = dists.view([self.genParams.cutn, n, -1])
            losses = dists.mul(self.modelCtx.clip_weights).sum(2).mean(0)
            tv_losses = lossFunctions.tv_loss(denoised)
            range_losses = lossFunctions.range_loss(denoised)#todo: denopised or denoised in
            loss = losses.sum() * self.genParams.overall_clip_scale * ( self.genParams.clip_guidance_scale + tv_losses.sum() * self.genParams.tv_scale + range_losses.sum() * self.genParams.range_scale)

        if self.genParams.aesthetics_scale != 0:
            ascore = self.clipWrapper.GetAestheticRatingFromEmbed(image_embeds)
            loss = loss.sum() + self.genParams.aesthetics_scale * ascore

        if self.modelCtx.init is not None and self.genParams.init_scale:
            init_losses = self.lpips_model(denoised, self.modelCtx.init)
            loss = loss + init_losses.sum() * self.genParams.init_scale

        return -torch.autograd.grad(loss, x)[0]




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
        self.model_config = model_and_diffusion_defaults()
        self.model_config.update({
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
        self.model_config['image_size'] = self.default_image_size_x
        model_path = model_path

        self.model, diffusion = create_model_and_diffusion(**self.model_config)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
        self.model.requires_grad_(False).eval().to(device)
        if self.model_config['use_fp16']:
            self.model.convert_to_fp16()

        self.kdiffExternalModelWrap = K.external.OpenAIDenoiser(self.model, diffusion, device=device)



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

        inst.condFns = CondFnClipGuidedObj(self.model, cw, inst, genParams)       

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