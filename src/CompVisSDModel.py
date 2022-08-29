
import sys

sys.path.append('./../k-diffusion')
sys.path.append('./../stable-diffusion')


import k_diffusion as K
import torch
from loguru import logger
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import onnx
import onnxruntime
from einops import rearrange, repeat
from transformers import CLIPTokenizer, CLIPTextModel
import torch.nn as nn

import denoisers
import paramsGen
import lossFunctions
import clipWrap
import modelWrap

#from ldm.modules.encoders.modules import FrozenCLIPTextEmbedder


# clip guidance
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



# clip text encoder. dont rely on the one ins table-diffusion, it always loads to cuda
class FrozenCLIPEmbedder(nn.Module):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, clipPath="openai/clip-vit-large-patch14", device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(clipPath)
        self.transformer = CLIPTextModel.from_pretrained(clipPath).to(device)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)





class CompVisSDModel(modelWrap.ModelWrap):
    def __init__(self, torchdtype = torch.float16):
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

        self.frozenClip:FrozenCLIPEmbedder = None

        #defaults to fp16 on cuda
        self.tensordtype = torchdtype


    def ModelLoadSettings(self):
        self.config_path = "D:/AIrtist/k-diffusion-wrap/stable-diffusion/configs/stable-diffusion/v1-inference.yaml"
        #self.model_path = "E:/MLModels/stableDiffusion/sd-v1-3-full-ema.ckpt" 
        self.default_image_size_x = 512
        self.default_image_size_y = 512
        self.channels = 4


    def LoadModel(self, device):
        #load the clip text embedder we need for converting tex tto embeds.
        #we could probably dump this if we have the clipwrapper loaded for clip guidance to save memory...
        #TODO!
        #but for now, this will work

        #hrm, huggingface repo of: clip-vit-large-patch14
        # is NOT the same as the model pulled down... it follows a different code path. figure this out...
        #self.frozenClip = FrozenCLIPEmbedder("E:/MLModels/clip/clip-vit-large-patch14",device)
        self.frozenClip = FrozenCLIPEmbedder(device = device)

        self.model_config = OmegaConf.load(self.config_path)
        self.model = instantiate_from_config(self.model_config.model)

        self.model.load_state_dict(torch.load(self.model_path, map_location='cpu')["state_dict"], strict=False)
        
        if str(device) == 'cpu':
            self.model.eval().to(torch.float32).to(device)
            self.tensordtype = torch.float32
        else:
            if self.tensordtype == torch.float32:
                self.model.eval().to(device)
            else:
                self.model.eval().half().to(device)

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
        if str(device) == 'cpu':
            inst.precision = None
        else:
            inst.precision = 'autocast' #faster, but is it worse?
        return inst

    def CreateCFGDenoiser(self, inst:modelWrap.ModelContext, clipEncoder:clipWrap.ClipWrap, cfgPrompts, condScale, genParams:paramsGen.ParamsGen):
        #c = inst.modelWrap.model.get_learned_conditioning(cfgPrompts)
        #uc = inst.modelWrap.model.get_learned_conditioning(genParams.num_images_to_sample * [""])
        c = self.frozenClip.encode(cfgPrompts)
        uc = self.frozenClip.encode(genParams.num_images_to_sample * [""])

        inst.extra_args = {'cond': c, 'uncond': uc, 'cond_scale': condScale}

        inst.kdiffModelWrap = denoisers.CFGDenoiser(self.kdiffExternalModelWrap)
        
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
        init = self.model.encode_first_stage(initTensor.to(self.tensordtype))
        init = self.model.get_first_stage_encoding(init)
        return init


    def DecodeImage(self, imageTensor):
        return self.model.decode_first_stage(imageTensor)