import lossFunctions
import cutouts
import torch
from loguru import logger
import clipWrap
import paramsGen

class CondFnClipGuidedObj:
    def __init__(self):
        super().__init__()
        self.normalize = None
        self.clipWrapper:clipWrap.ClipWrap = None
        self.lpips_model = None
        self.genParams:paramsGen.ParamsGen = None

        self.target_embeds = None
        self.weights = None
        self.init = None
        self.make_cutouts = None
    
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
        denoised_in = denoised

        clip_in = self.normalize(self.make_cutouts(denoised_in.add(1).div(2)))
        image_embeds = self.clipWrapper.model.encode_image(clip_in).float()

        loss = None

        if self.genParams.clip_guidance_scale != 0:
            dists = lossFunctions.spherical_dist_loss(image_embeds[:, None], self.target_embeds[None])
            dists = dists.view([self.genParams.cutn, n, -1])
            losses = dists.mul(self.weights).sum(2).mean(0)
            tv_losses = lossFunctions.tv_loss(denoised_in)
            range_losses = lossFunctions.range_loss(denoised)
            loss = losses.sum() * self.genParams.clip_guidance_scale + tv_losses.sum() * self.genParams.tv_scale + range_losses.sum() * self.genParams.range_scale

        if self.genParams.aesthetics_scale != 0:
            ascore = self.clipWrapper.GetAestheticRatingFromEmbed(image_embeds)
            loss = loss.sum() + self.genParams.aesthetics_scale * ascore

        if self.init is not None and self.genParams.init_scale:
            init_losses = self.lpips_model(denoised_in, self.init)
            loss = loss + init_losses.sum() * self.genParams.init_scale

        return -torch.autograd.grad(loss, x)[0]








class CondFnClipGuidedCompvisObj:
    def __init__(self, model):
        super().__init__()
        self.normalize = None
        self.clipWrapper:clipWrap.ClipWrap = None
        self.lpips_model = None
        self.genParams = None

        self.model = model

        self.target_embeds = None
        self.weights = None
        self.init = None
        self.make_cutouts = None
        self.isCompvisLDM = False
    
    def uncond_fn(self, x, sigma, denoised, **kwargs):
        logger.debug("uncond_fn")

    def cond_fn(self, x, sigma, denoised, **kwargs):
        n = x.shape[0]
        
        # Anti-grain hack for the 256x256 ImageNet model
        if self.isCompvisLDM == False:
            fac = sigma / (sigma ** 2 + 1) ** 0.5
            denoised_in = x.lerp(denoised, fac)
        else:
            denoised_in = self.model.first_stage_model.decode(denoised / self.model.scale_factor)

        clip_in = self.normalize(self.make_cutouts(denoised_in.add(1).div(2)))
        image_embeds = self.clipWrapper.model.encode_image(clip_in).float()

        loss = None

        #clip_in = self.normalize(make_cutouts(denoised_in.add(1).div(2)))
        if self.genParams.clip_guidance_scale != 0:
            #dists = lossFunctions.spherical_dist_loss(image_embeds[:, None], self.target_embeds[None])
            #dists = dists.view([self.genParams.cutn, n, -1])
            #losses = dists.mul(self.weights).sum(2).mean(0)
            #loss = losses.sum() * self.genParams.clip_guidance_scale
            dists = lossFunctions.spherical_dist_loss(image_embeds[:, None], self.target_embeds[None])
            dists = dists.view([self.genParams.cutn, n, -1])
            losses = dists.mul(self.weights).sum(2).mean(0)
            tv_losses = lossFunctions.tv_loss(denoised_in)
            range_losses = lossFunctions.range_loss(denoised)
            loss = losses.sum() * self.genParams.clip_guidance_scale + tv_losses.sum() * self.genParams.tv_scale + range_losses.sum() * self.genParams.range_scale

        if self.genParams.aesthetics_scale != 0:
            ascore = self.clipWrapper.GetAestheticRatingFromEmbed(image_embeds)
            #norm_embed = image_embeds.div(image_embeds.norm(dim=-1, keepdim=True))
            #ascore = self.clipWrapper.aestheticModel.amodel(norm_embed)
            #print(ascore)
    
            if loss == None:
                loss = self.genParams.aesthetics_scale * ascore.sum()
            else:
                loss = -1 * ( loss.sum() + self.genParams.aesthetics_scale * ascore.sum() )

        if self.init is not None and self.genParams.init_scale:
            init_losses = self.lpips_model(denoised_in, self.init)
            loss = loss + init_losses.sum() * self.genParams.init_scale
                    
        return -torch.autograd.grad(loss, x)[0]   