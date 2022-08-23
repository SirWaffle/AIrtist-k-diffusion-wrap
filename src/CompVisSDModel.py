
import sys

sys.path.append('./../k-diffusion')
sys.path.append('./../stable-diffusion')


import k_diffusion as K
import torch
from loguru import logger
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import onnx

import denoisers
import paramsGen
import lossFunctions
import clipWrap
import modelWrap

#from ldm.modules.encoders.modules import FrozenCLIPTextEmbedder



class CondFnClipGuidedCompvisObj:
    def __init__(self, model, clipWrapper:clipWrap.ClipWrap, modelCtx:modelWrap.ModelContext, genParams):
        super().__init__()
        self.normalize = clipWrapper.normalize
        self.clipWrapper:clipWrap.ClipWrap = clipWrapper
        self.lpips_model = clipWrapper.lpips_model
        self.genParams = genParams

        self.model = model
        self.modelCtx = modelCtx

    
    def uncond_fn(self, x, sigma, denoised, **kwargs):
        logger.debug("uncond_fn")

    def cond_fn(self, x, sigma, denoised, **kwargs):
        n = x.shape[0]
        
        denoised_in = self.model.first_stage_model.decode(denoised / self.model.scale_factor)

        clip_in = self.normalize(self.modelCtx.make_cutouts(denoised_in.add(1).div(2)))
        image_embeds = self.clipWrapper.model.encode_image(clip_in).float()

        loss = None

        #clip_in = self.normalize(make_cutouts(denoised_in.add(1).div(2)))
        if self.genParams.clip_guidance_scale != 0:
            #dists = lossFunctions.spherical_dist_loss(image_embeds[:, None], self.target_embeds[None])
            #dists = dists.view([self.genParams.cutn, n, -1])
            #losses = dists.mul(self.weights).sum(2).mean(0)
            #loss = losses.sum() * self.genParams.clip_guidance_scale
            dists = lossFunctions.spherical_dist_loss(image_embeds[:, None], self.modelCtx.target_clip_embeds[None])
            dists = dists.view([self.genParams.cutn, n, -1])
            losses = dists.mul(self.modelCtx.clip_weights).sum(2).mean(0)
            tv_losses = lossFunctions.tv_loss(denoised_in)
            range_losses = lossFunctions.range_loss(denoised_in)
            loss = losses.sum() * self.genParams.overall_clip_scale * (self.genParams.clip_guidance_scale + tv_losses.sum() * self.genParams.tv_scale + range_losses.sum() * self.genParams.range_scale)

        if self.genParams.aesthetics_scale != 0:
            ascore = self.clipWrapper.GetAestheticRatingFromEmbed(image_embeds)
            #norm_embed = image_embeds.div(image_embeds.norm(dim=-1, keepdim=True))
            #ascore = self.clipWrapper.aestheticModel.amodel(norm_embed)
            #print(ascore)
    
            if loss == None:
                loss = self.genParams.aesthetics_scale * ascore.sum()
            else:
                loss = -1 * ( loss.sum() + self.genParams.aesthetics_scale * ascore.sum() )

        if self.modelCtx.init is not None and self.genParams.init_scale:
            init_losses = self.lpips_model(denoised_in, self.modelCtx.init)
            loss = loss + init_losses.sum() * self.genParams.init_scale
                    
        return -torch.autograd.grad(loss, x)[0]   



class CompVisSDModel(modelWrap.ModelWrap):
    def __init__(self):
        self.model_path = None
        self.config_path = None        

        self.default_image_size_x = None
        self.default_image_size_y = None

        #loaded info
        self.model = None
        self.model_config = None
        self.kdiffExternalModelWrap = None #kdiff model wrapper
        self.default_imageTensorSize = None

        self.default_guiding = 'CFG'

        self.ONNX = False


    def ModelLoadSettings(self):
        self.config_path = "D:/AIrtist/k-diffusion-wrap/stable-diffusion/configs/stable-diffusion/v1-inference.yaml"
        #self.model_path = "E:/MLModels/stableDiffusion/sd-v1-3-full-ema.ckpt" 
        self.default_image_size_x = 512
        self.default_image_size_y = 512
        self.channels = 4


    def LoadModel(self, device):
        if self.ONNX == True:
            #TODO: this just loads for checking,m need to use onnxruntime, 
            # create a session, and change how inference is done...
            #maybe making an ONNXwrapper would help? ugh.
            self.model = onnx.load("E:/onnxOut/sd-v1-3-full-ema.onnx")
            self.default_imageTensorSize = self.default_image_size_x//16  
            self.kdiffExternalModelWrap = self.model
            self.ONNX = True

        else:
            self.model_config = OmegaConf.load(self.config_path)
            self.model = instantiate_from_config(self.model_config.model)

            self.model.load_state_dict(torch.load(self.model_path, map_location='cpu')["state_dict"], strict=False)
            self.model.eval().half().to(device)
            #self.model.eval().to(device)

            self.kdiffExternalModelWrap = K.external.CompVisDenoiser(self.model, False, device=device)
            self.default_imageTensorSize = self.default_image_size_x//16  


    def RequestImageSize(self, inst:modelWrap.ModelContext, x, y):

        if x == -1:
            x = self.default_image_size_x
        if y == -1:
            y = self.default_image_size_y

        image_size_x = x - (x % 8) #rdm was 16
        image_size_y = y - (y % 8)
        inst.image_tensor_size_x = image_size_x//8 #rdm was 16 
        inst.image_tensor_size_y = image_size_y//8
        inst.image_size_x = image_size_x
        inst.image_size_y = image_size_y

        return inst
        
    def CreateModelInstance(self, device, clipWrapper:clipWrap.ClipWrap, genParams:paramsGen.ParamsGen, clip_guided) -> modelWrap.ModelContext:
        inst = modelWrap.ModelContext()
        inst.modelWrap = self
        inst.precision = 'autocast' #faster, but is it worse?
        return inst

    def CreateCFGDenoiser(self, inst:modelWrap.ModelContext, clipEncoder:clipWrap.ClipWrap, cfgPrompts, condScale, genParams:paramsGen.ParamsGen):
        if self.ONNX == True:
            clipTextEmbed = clipWrap.FrozenCLIPTextEmbedder(clipEncoder.model)
            inst.target_cfg_embeds = clipTextEmbed.encode(cfgPrompts).float()
            c = inst.target_cfg_embeds
            uc = torch.zeros_like(c)
        else:
            c = inst.modelWrap.model.get_learned_conditioning(cfgPrompts)
            uc = inst.modelWrap.model.get_learned_conditioning(genParams.num_images_to_sample * [""])

        inst.extra_args = {'cond': c, 'uncond': uc, 'cond_scale': condScale}

        if self.ONNX == False:
            inst.kdiffModelWrap = denoisers.CFGDenoiser(self.kdiffExternalModelWrap)
        else:
            inst.kdiffModelWrap = self.model
        
        return inst


    def CreateClipGuidedDenoiser(self, inst:modelWrap.ModelContext,  cw:clipWrap.ClipWrap, clipPrompts, genParams:paramsGen.ParamsGen, device):
        
        target_embeds, weights = cw.GetTextPromptEmbeds(clipPrompts, device)
        inst.target_clip_embeds.extend(target_embeds)
        inst.clip_weights.extend(weights)

        inst.condFns = CondFnClipGuidedCompvisObj(self.model, cw, inst, genParams)       

        inner_model = inst.kdiffModelWrap
        if inner_model == None:
            inner_model = self.kdiffExternalModelWrap   

        #TODO: figure out how to handle multiples of these needing extra args or something
        #clip denoiser doesnt need any conditional stuff at all, and doesnt need unconditional
        if inst.extra_args == None:
            #TODO: get rid of this, used to get size of embeds the model is expecting
            clipTextEmbed = clipWrap.FrozenCLIPTextEmbedder(cw.model)
            embeds = clipTextEmbed.encode(clipPrompts).float()

            c = torch.zeros_like(embeds)
            inst.extra_args = {'cond': c }

        inst.kdiffModelWrap = denoisers.GuidedDenoiserWithGrad(inner_model, inst.condFns.cond_fn) 

        return inst



    def EncodeInitImage(self, initTensor):
        init = self.model.encode_first_stage(initTensor)
        init = self.model.get_first_stage_encoding(init)
        return init


    def DecodeImage(self, imageTensor):
        return self.model.decode_first_stage(imageTensor)