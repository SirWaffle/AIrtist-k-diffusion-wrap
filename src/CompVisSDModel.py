
import sys

sys.path.append('./../k-diffusion')
sys.path.append('./../stable-diffusion')

import numpy as np
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
from safetensors.torch import load_file

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
        #self.config_path = "D:/AIrtist/k-diffusion-wrap/stable-diffusion/configs/stable-diffusion/v1-inference.yaml"
        self.config_path = "D:/AIrtist/k-diffusion-wrap/stablediffusion/configs/stable-diffusion/v1-inference.yaml"
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

        if self.model_path.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
            except ImportError as e:
                raise ImportError(f"The model is in safetensors format and it is not installed, use `pip install safetensors`: {e}")
            print('========= Attempting to load safetensors ==========')
            pl_sd = load_file(self.model_path, device='cpu')
            self.model.load_state_dict(pl_sd, strict=False)
        else:
            pl_sd = torch.load(self.model_path, map_location='cpu')
            self.model.load_state_dict(pl_sd["state_dict"], strict=False)
        
        if str(device) == 'cpu':
            self.model.eval().to(torch.float32).to(device)
            self.tensordtype = torch.float32
        else:
            self.model.eval().to(device).to(self.tensordtype)

        #self.kdiffExternalModelWrap = K.external.CompVisDenoiser(self.model, False, device=device)
        self.kdiffExternalModelWrap = CompVisDenoiser(self.model, False, device=device)

        self.default_imageTensorSize = self.default_image_size_x//16  



    def transform_checkpoint_dict_key(self, k):
        chckpoint_dict_replacements = {
            'cond_stage_model.transformer.embeddings.': 'cond_stage_model.transformer.text_model.embeddings.',
            'cond_stage_model.transformer.encoder.': 'cond_stage_model.transformer.text_model.encoder.',
            'cond_stage_model.transformer.final_layer_norm.': 'cond_stage_model.transformer.text_model.final_layer_norm.',
        }

        for text, replacement in chckpoint_dict_replacements.items():
            if k.startswith(text):
                k = replacement + k[len(text):]

        return k

    def get_state_dict_from_checkpoint(self, pl_sd):
        pl_sd = pl_sd.pop("state_dict", pl_sd)
        pl_sd.pop("state_dict", None)

        sd = {}
        for k, v in pl_sd.items():
            new_key = self.transform_checkpoint_dict_key(k)

            if new_key is not None:
                sd[new_key] = v

        pl_sd.clear()
        pl_sd.update(sd)

        return pl_sd



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

        print("image size: (" + str(image_size_x) + ", " + str(image_size_y) + ")")
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

        c = self.frozenClip.encode(cfgPrompts)

        if genParams.CFGNegPrompts == None:
            uc = self.frozenClip.encode(genParams.num_images_to_sample * [""])
        else:
            uc = self.frozenClip.encode(genParams.CFGNegPrompts)

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














######################
### trying to make wrapper classes that can be parsed by jitscript thing
class DiscreteSchedule(nn.Module):
    """A mapping between continuous noise levels (sigmas) and a list of discrete noise
    levels."""
    #@torch.jit.script
    def __init__(self, sigmas, quantize):
        super().__init__()
        self.register_buffer('sigmas', sigmas)
        self.quantize:bool = quantize

   #@torch.jit.script
    def get_sigmas(self, n=None):
        if n is None:
            return K.sampling.append_zero(self.sigmas.flip(0))
        t_max = len(self.sigmas) - 1
        t = torch.linspace(t_max, 0, n, device=self.sigmas.device)
        return K.sampling.append_zero(self.t_to_sigma(t))

    #@torch.jit.script
    def sigma_to_t(self, sigma, quantize=None):
        quantize = self.quantize if quantize is None else quantize
        dists = torch.abs(sigma - self.sigmas[:, None])
        if quantize:
            return torch.argmin(dists, dim=0).view(sigma.shape)
        #low_idx, high_idx = torch.sort(torch.topk(dists, dim=0, k=2, largest=False).indices, dim=0)[0]
        topks = torch.topk(dists, dim=0, k=2, largest=False).indices
        high_idx = torch.max(topks, dim=0)[0]
        low_idx = torch.min(topks, dim=0)[0]
        low, high = self.sigmas[low_idx], self.sigmas[high_idx]
        w = (low - sigma) / (low - high)
        w = w.clamp(0, 1)
        t = (1 - w) * low_idx + w * high_idx
        return t.view(sigma.shape)

    #@torch.jit.script
    def t_to_sigma(self, t):
        t = t.float()
        low_idx, high_idx, w = t.floor().long(), t.ceil().long(), t.frac()
        return (1 - w) * self.sigmas[low_idx] + w * self.sigmas[high_idx]


class DiscreteEpsDDPMDenoiser(DiscreteSchedule):
    """A wrapper for discrete schedule DDPM models that output eps (the predicted
    noise)."""

    #@torch.jit.script
    def __init__(self, model, alphas_cumprod, quantize):
        super().__init__(((1 - alphas_cumprod) / alphas_cumprod) ** 0.5, quantize)
        self.inner_model = model
        self.sigma_data = 1.

    #@torch.jit.script
    def get_scalings(self, sigma):
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_out, c_in

    #@torch.jit.script
    def get_eps(self, sigma, cond):
        return self.inner_model(sigma, cond)

    #@torch.jit.script
    def loss(self, input, noise, sigma, cond):
        c_out, c_in = [K.utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        noised_input = input + noise * K.utils.append_dims(sigma, input.ndim)
        eps = self.get_eps(noised_input * c_in, self.sigma_to_t(sigma), cond)
        return (eps - noise).pow(2).flatten(1).mean(1)

    #@torch.jit.script
    def forward(self, input, sigma, cond):
        c_out, c_in = [K.utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        eps = self.get_eps(input * c_in, self.sigma_to_t(sigma), cond)
        return input + eps * c_out


class CompVisDenoiser(DiscreteEpsDDPMDenoiser):
    """A wrapper for CompVis diffusion models."""

   # @torch.jit.script
    def __init__(self, model, quantize=False, device='cpu'):
        super().__init__(model, model.alphas_cumprod, quantize=quantize)

    #@torch.jit.script
    def get_eps(self, input, sigma, cond):
        return self.inner_model.apply_model(input, sigma, cond)