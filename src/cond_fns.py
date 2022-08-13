import lossFunctions
import cutouts
import torch
from loguru import logger
import clipWrap
from modelWrap import ModelContext, ModelWrap
import paramsGen
import modelWrap


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








