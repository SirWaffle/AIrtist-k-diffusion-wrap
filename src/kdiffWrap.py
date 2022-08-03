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
import modelWrap

from ldm.modules.encoders.modules import FrozenCLIPTextEmbedder


class KDiffWrap:
    def __init__(self):
        #torch
        self.torchDevice = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Using device:', self.torchDevice, flush=True)

        self.CurClipWrap = None
        self.CurModelWrap = None




    def CreateModels(self, modelNum, clipModelNum) -> tuple[clipWrap.ClipWrap, modelWrap.ModelWrap]: 
        # CLIP model settings
        clipwrapper = clipWrap.ClipWrap()
        clipwrapper.ModelLoadSettings(clipModelNum)
        clipwrapper.LoadModel(self.torchDevice)

        # MODEL wrapper settings...
        modelwrapper = modelWrap.ModelWrap()
        modelwrapper.ModelLoadSettings(modelNum)

        modelwrapper.LoadModel(self.torchDevice)

        return clipwrapper, modelwrapper



    def internal_run(self, genParams:paramsGen.ParamsGen, cw:clipWrap.ClipWrap, mw:modelWrap.ModelWrap): 

        
        clipguided = genParams.clip_guidance_scale != 0 or genParams.aesthetics_scale != 0

        if genParams.aesthetics_scale != 0:
            if cw.aestheticModel.amodel == None:
                cw.LoadAestheticsModel(self.torchDevice)
        else:
            cw.aestheticModel.amodel = None
            gc.collect()

        if genParams.image_prompts != None and len(genParams.image_prompts) > 0 and len(genParams.image_prompts) < genParams.num_images_to_sample:
            while len(genParams.image_prompts) < genParams.num_images_to_sample:
                genParams.image_prompts.append( genParams.image_prompts[0])

        #need to sloppily fix the prompts for sampling more stuff, it requires more prompts...
        if len(genParams.prompts) < genParams.num_images_to_sample:
            while len(genParams.prompts) < genParams.num_images_to_sample:
                genParams.prompts.append( genParams.prompts[0])

        elif len(genParams.prompts) > genParams.num_images_to_sample + 1:
            genParams.prompts = genParams.prompts[0:genParams.num_images_to_sample]

        elif len(genParams.prompts) == genParams.num_images_to_sample + 1:
            #append the first entry to the other entries, make a new list
            new_list = []
            first_entry = genParams.prompts[0]
            count = 1
            for x in genParams.prompts:
                if count > 1:
                    new_list.append(first_entry + ' ' + x)
                count += 1
            
            genParams.prompts = new_list
            


        #cheat for now
        device = self.torchDevice

        #for caching..
        self.CurClipWrap = cw
        self.CurModelWrap = mw

        #get the prompts to use for clip
        clip_prompts = genParams.prompts

        #if there are cfg prompts, whats in prompts is going to be for clip
        if genParams.CFGprompts and len(genParams.CFGprompts[0]) > 2:
            clip_prompts = genParams.prompts
        
        # if clip prompts are defined, use that
        if genParams.CLIPprompts and len(genParams.CLIPprompts[0]) > 2:
            clip_prompts = genParams.CLIPprompts


        print('========= PARAMS ==========')
        print("ModelNum: " + str(mw.modelNum))
        print("Clip Num:" + str(cw.modelNum))
        attrs = vars(genParams)
        # now dump this in some way or another
        print(', '.join("%s: %s" % item for item in attrs.items()))
        print('========= /PARAMS ==========')


        

        modelInst = mw.CreateModelInstance(self.torchDevice, cw, genParams, clipguided)

        print("sigmas: " + str(modelInst.sigmas))


        ############
        # start gen loop
        ################
        clip_size = cw.model.visual.input_resolution
        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711])


        make_cutouts = None

        if genParams.cutoutMethod.lower() == 'grid':
            make_cutouts = cutouts.MakeGridCutouts(clip_size, genParams.cutn, genParams.cut_pow)
        else:
            make_cutouts = cutouts.MakeCutoutsRandom(clip_size, genParams.cutn, genParams.cut_pow)


        side_x = side_y = mw.image_size

        target_embeds, weights = [], []

        for prompt in clip_prompts:
            txt, weight = utilFuncs.parse_prompt(prompt)
            target_embeds.append(cw.model.encode_text(clip.tokenize(txt).to(device)).float())
            weights.append(weight)

        if genParams.image_prompts != None:
            for prompt in genParams.image_prompts:
                path, weight = utilFuncs.parse_prompt(prompt)
                img = Image.open(utilFuncs.fetch(path)).convert('RGB')
                img = TF.resize(img, min(side_x, side_y, *img.size), transforms.InterpolationMode.LANCZOS)

                gridCuts = cutouts.MakeGridCutouts(clip_size, 1, genParams.cut_pow)
                #make_cutouts(TF.to_tensor(img)[None].to(device))
                batch = gridCuts(TF.to_tensor(img)[None].to(device))
                embed = cw.model.encode_image(normalize(batch)).float()
                target_embeds.append(embed)
                weights.extend([weight / genParams.cutn] * genParams.cutn)

            #we need to set the conditional phrase here...
            if mw.using_compvisLDM == True:
                #expects like [1,1,768], output from encode image above is [1,768]
                embed = embed[None,:].to(device)
                c = embed
                uc = torch.zeros_like(c)
                modelInst.extra_args = {'cond': c, 'uncond': uc, 'cond_scale': genParams.conditioning_scale}






        target_embeds = torch.cat(target_embeds)
        weights = torch.tensor(weights, device=device)
        if weights.sum().abs() < 1e-3:
            raise RuntimeError('The weights must not sum to 0.')
        weights /= weights.sum().abs()

        init = None
        if genParams.init_image is not None:
            init = Image.open(utilFuncs.fetch(genParams.init_image)).convert('RGB')
            init = init.resize((side_x, side_y), Image.Resampling.LANCZOS)
            init = TF.to_tensor(init).to(device)[None] * 2 - 1

            if mw.using_compvisLDM == True:
                init = mw.model.encode_first_stage(init)
                init = mw.model.get_first_stage_encoding(init)
                init.to(device)

        if init is not None:
            modelInst.sigmas = modelInst.sigmas[modelInst.sigmas <= genParams.sigma_start]

        def callback(info):
            i = info['i'] 
            if info['i'] % 25 == 0:
                tqdm.write(f'Step {info["i"]} of {len(modelInst.sigmas) - 1}, sigma {info["sigma"]:g}:')

            if info['i'] != 0 and info['i'] % genParams.saveEvery == 0:
                denoised = modelInst.kdiffModelWrap.orig_denoised
                nrow = math.ceil(denoised.shape[0] ** 0.5)
                grid = utils.make_grid(denoised, nrow, padding=0)
                filename = f'step_{i}.png'
                K.utils.to_pil_image(grid).save(filename)


        if genParams.seed is not None:
            torch.manual_seed(genParams.seed)


        #hacky way to get these vars into the cond_fns
        if modelInst.condFns != None:
            modelInst.condFns.normalize = normalize
            modelInst.condFns.clipWrapper = cw
            modelInst.condFns.lpips_model = cw.lpips_model
            modelInst.condFns.genParams = genParams
            modelInst.condFns.target_embeds = target_embeds
            modelInst.condFns.weights = weights
            modelInst.condFns.init = init
            modelInst.condFns.make_cutouts = make_cutouts
            modelInst.condFns.isCompvisLDM = mw.using_compvisLDM


        utilFuncs.log_torch_mem("Starting sample loops")
        
        def doSamples(sm: str):
            x = torch.randn([genParams.num_images_to_sample, modelInst.kdiffModelWrap.channels, 
                            mw.imageTensorSize, mw.imageTensorSize], device=device) * modelInst.sigmas[0]
            if init is not None:
                x += init

            if  sm == "heun":
                x_0 = K.sampling.sample_heun(modelInst.kdiffModelWrap, x, modelInst.sigmas, second_order=False, s_churn=20, callback=callback, extra_args=modelInst.extra_args)
            elif sm == "lms":
                x_0 = K.sampling.sample_lms(modelInst.kdiffModelWrap, x, modelInst.sigmas, callback=callback,extra_args=modelInst.extra_args)
            elif sm == "euler":
                x_0 = K.sampling.sample_euler(modelInst.kdiffModelWrap, x, modelInst.sigmas, s_churn=20, callback=callback,extra_args=modelInst.extra_args)
            elif sm == "euler_a":
                x_0 = K.sampling.sample_euler_ancestral(modelInst.kdiffModelWrap, x, modelInst.sigmas, callback=callback,extra_args=modelInst.extra_args)
            elif sm == "dpm_2":
                x_0 = K.sampling.sample_dpm_2(modelInst.kdiffModelWrap, x, modelInst.sigmas, s_churn=20, callback=callback,extra_args=modelInst.extra_args)
            elif sm == "dpm_2_a":
                x_0 = K.sampling.sample_dpm_2_ancestral(modelInst.kdiffModelWrap, x, modelInst.sigmas, callback=callback,extra_args=modelInst.extra_args)
            else:
                print("ERROR: invalid sampling method, defaulting to LMS")
                x_0 = K.sampling.sample_lms(modelInst.kdiffModelWrap, x, modelInst.sigmas, callback=callback,extra_args=modelInst.extra_args)
                
            return x_0


        #for i in range(genParams.n_batches):
        with torch.no_grad():
            if mw.using_compvisLDM == True:
                with mw.model.ema_scope():
                    samples = doSamples(genParams.sampleMethod.lower())
            else:
                samples = doSamples(genParams.sampleMethod.lower())

        ############
        # end gen loop
        ################

        if mw.using_compvisLDM == True:
            samples = mw.model.decode_first_stage(samples)
            #samples = torch.clamp((samples + 1.0) / 2.0, min=0.0, max=1.0)

        #TODO: combine all the samples into a grid somehow
        imgsList = []
        for i, out in enumerate(samples):
            imgsList.append( K.utils.to_pil_image(out) )
            #grid = K.utils.to_pil_image(out)

        if clipguided == True:
            denoised = modelInst.kdiffModelWrap.orig_denoised
            if mw.using_compvisLDM == True:
                denoised = mw.model.decode_first_stage(denoised)
            nrow = math.ceil(denoised.shape[0] ** 0.5)
            grid = make_grid(denoised, nrow, padding=0)
            grid = K.utils.to_pil_image(grid)
        else:
            #nrow = math.ceil(denoised.shape[0] ** 0.5)
            rows = 8
            if genParams.num_images_to_sample <= 4:
                rows = 2
            elif genParams.num_images_to_sample <= 9:
                rows = 3
            elif genParams.num_images_to_sample <= 16:
                rows = 4
            grid = make_grid(samples, rows, padding=0)
            grid = K.utils.to_pil_image(grid)

        return grid, imgsList
