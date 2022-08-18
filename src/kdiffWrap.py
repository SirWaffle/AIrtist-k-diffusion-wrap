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
#from ldm.util import instantiate_from_config

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
import CompVisRDMModel
import OpenAIUncondModel

from ldm.modules.encoders.modules import FrozenCLIPTextEmbedder


class KDiffWrap:
    def __init__(self):
        #torch
        self.torchDevice = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Using device:', self.torchDevice, flush=True)

        self.CurClipWrap = None
        self.CurModelWrap = None




    def CreateModels(self, clipModelNum, modelNum = 0) -> tuple[clipWrap.ClipWrap, modelWrap.ModelWrap]: 
        # CLIP model settings
        clipwrapper = clipWrap.ClipWrap()
        clipwrapper.ModelLoadSettings(clipModelNum)
        clipwrapper.LoadModel(self.torchDevice)

        # MODEL wrapper settings...
        if modelNum == 1:
            modelwrapper = OpenAIUncondModel.OpenAIUncondModel()
        else:
            modelwrapper = CompVisRDMModel.CompVisRDMModel()

        modelwrapper.ModelLoadSettings()
        modelwrapper.LoadModel(self.torchDevice)

        return clipwrapper, modelwrapper



    def pad_or_truncate(some_list, target_len):
        return some_list[:target_len] + [0]*(target_len - len(some_list))



    def internal_run(self, genParams:paramsGen.ParamsGen, cw:clipWrap.ClipWrap, mw:modelWrap.ModelWrap): 

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
        if genParams.prompts != None:
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
                


        if genParams.seed is not None:
            torch.manual_seed(genParams.seed)

        #cheat for now
        device = self.torchDevice

        #for caching..
        self.CurClipWrap = cw
        self.CurModelWrap = mw

        #get the prompts to use for clip
        clip_prompts = None
        cfg_prompts = None

        if mw.default_guiding == 'CFG':
            cfg_prompts = genParams.prompts
        if mw.default_guiding == 'CLIP' or genParams.clip_guidance_scale != 0:
            clip_prompts = genParams.prompts

        #if there are cfg prompts, whats in prompts is going to be for clip
        if genParams.CFGprompts and len(genParams.CFGprompts[0]) > 2:
            cfg_prompts = genParams.CFGprompts
        
        # if clip prompts are defined, use that
        if genParams.CLIPprompts and len(genParams.CLIPprompts[0]) > 2:
            clip_prompts = genParams.CLIPprompts

        clipguided = False
        if genParams.clip_guidance_scale != 0 or genParams.aesthetics_scale != 0:
            if clip_prompts != None:
                clipguided = True
        if clipguided == False:
            clip_prompts = None

        print('========= PARAMS ==========')
        print("Model: " + str(mw.model_path))
        print("Clip Num:" + str(cw.modelNum))
        attrs = vars(genParams)
        # now dump this in some way or another
        print(', '.join("%s: %s" % item for item in attrs.items()))
        print('========= /PARAMS ==========')


        torch.cuda.empty_cache()

        modelCtx = mw.CreateModelInstance(self.torchDevice, cw, genParams, clipguided)

        modelCtx = mw.RequestImageSize(modelCtx, genParams.image_size_x, genParams.image_size_y)

        if cfg_prompts != None:
            print("using CFG denoiser, prompts: " + str(cfg_prompts))
            modelCtx = mw.CreateCFGDenoiser(modelCtx, cw, cfg_prompts, genParams.conditioning_scale, genParams)

        if clip_prompts != None:
            print("using CLIP denoiser, prompts: " + str(clip_prompts))
            modelCtx = mw.CreateClipGuidedDenoiser(modelCtx, cw, clip_prompts, genParams, device)

        modelCtx = mw.GetSigmas(modelCtx, genParams, device)

        print("got sigmas: " + str(modelCtx.sigmas))


        ############
        # start gen loop
        ################



        if genParams.cutoutMethod.lower() == 'grid':
            modelCtx.make_cutouts = cutouts.MakeGridCutouts(cw.clip_size, genParams.cutn, genParams.cut_pow)
        else:
            modelCtx.make_cutouts = cutouts.MakeCutoutsRandom(cw.clip_size, genParams.cutn, genParams.cut_pow)


        side_x = modelCtx.image_size_x
        side_y = modelCtx.image_size_y

        if genParams.image_prompts != None:
            for prompt in genParams.image_prompts:
                path, weight = utilFuncs.parse_prompt(prompt)
                img = Image.open(utilFuncs.fetch(path)).convert('RGB')
                img = TF.resize(img, min(side_x, side_y, *img.size), transforms.InterpolationMode.LANCZOS)

                gridCuts = cutouts.MakeGridCutouts(cw.clip_size, 1, genParams.cut_pow)
                #make_cutouts(TF.to_tensor(img)[None].to(device))
                batch = gridCuts(TF.to_tensor(img)[None].to(device))
                embed = cw.model.encode_image(cw.normalize(batch)).float()
                modelCtx.target_clip_embeds.append(embed)
                modelCtx.clip_weights.extend([weight / genParams.cutn] * genParams.cutn)

            #we need to set the conditional phrase here...
            if mw.using_compvisLDM == True:
                #expects like [1,1,768], output from encode image above is [1,768]
                embed = embed[None,:].to(device)
                c = embed
                uc = torch.zeros_like(c)
                modelCtx.extra_args = {'cond': c, 'uncond': uc, 'cond_scale': genParams.conditioning_scale}





        if modelCtx.target_clip_embeds != None and len(modelCtx.target_clip_embeds) > 0:
            modelCtx.target_clip_embeds = torch.cat(modelCtx.target_clip_embeds)
            modelCtx.clip_weights = torch.tensor(modelCtx.clip_weights, device=device)
            if modelCtx.clip_weights.sum().abs() < 1e-3:
                raise RuntimeError('The weights must not sum to 0.')
            modelCtx.clip_weights /= modelCtx.clip_weights.sum().abs()

        init = None
        if genParams.init_image is not None:
            print("using init image: " + genParams.init_image)
            init = Image.open(utilFuncs.fetch(genParams.init_image)).convert('RGB')
            init = init.resize((side_x, side_y), Image.Resampling.LANCZOS)
            init = TF.to_tensor(init).to(device)[None] * 2 - 1

            init = mw.EncodeInitImage(init).to(device)


        if init is not None:
            modelCtx.sigmas = modelCtx.sigmas[modelCtx.sigmas <= genParams.sigma_start]



        #modelCtx.target_clip_embeds = target_embeds
        #modelCtx.clip_weights = weights
        modelCtx.init = init       



        def callback(info):
            i = info['i'] 
            if info['i'] % 25 == 0:
                tqdm.write(f'Step {info["i"]} of {len(modelCtx.sigmas) - 1}, sigma {info["sigma"]:g}:')

            if info['i'] != 0 and info['i'] % genParams.saveEvery == 0:
                denoised = modelCtx.kdiffModelWrap.orig_denoised
                nrow = math.ceil(denoised.shape[0] ** 0.5)
                grid = utils.make_grid(denoised, nrow, padding=0)
                filename = f'step_{i}.png'
                K.utils.to_pil_image(grid).save(filename)


        utilFuncs.log_torch_mem("Starting sample loops")
        
        def doSamples(sm: str):
            x = torch.randn([genParams.num_images_to_sample, modelCtx.modelWrap.channels, 
                            modelCtx.image_tensor_size, modelCtx.image_tensor_size], device=device) * modelCtx.sigmas[0]
            if init is not None:
                x += init

            if  sm == "heun":
                print("sampling: HUEN")
                x_0 = K.sampling.sample_heun(modelCtx.kdiffModelWrap, x, modelCtx.sigmas, s_churn=20, callback=callback, extra_args=modelCtx.extra_args)
            elif sm == "lms":
                print("sampling: LMS")
                x_0 = K.sampling.sample_lms(modelCtx.kdiffModelWrap, x, modelCtx.sigmas, callback=callback,extra_args=modelCtx.extra_args)
            elif sm == "euler":
                print("sampling: EULER")
                x_0 = K.sampling.sample_euler(modelCtx.kdiffModelWrap, x, modelCtx.sigmas, s_churn=20, callback=callback,extra_args=modelCtx.extra_args)
            elif sm == "euler_a":
                print("sampling: EULER_A")
                x_0 = K.sampling.sample_euler_ancestral(modelCtx.kdiffModelWrap, x, modelCtx.sigmas, callback=callback,extra_args=modelCtx.extra_args)
            elif sm == "dpm_2":
                print("sampling: DPM_2")
                x_0 = K.sampling.sample_dpm_2(modelCtx.kdiffModelWrap, x, modelCtx.sigmas, s_churn=20, callback=callback,extra_args=modelCtx.extra_args)
            elif sm == "dpm_2_a":
                print("sampling: DPM_2_A")
                x_0 = K.sampling.sample_dpm_2_ancestral(modelCtx.kdiffModelWrap, x, modelCtx.sigmas, callback=callback,extra_args=modelCtx.extra_args)
            else:
                print("ERROR: invalid sampling method, defaulting to LMS")
                x_0 = K.sampling.sample_lms(modelCtx.kdiffModelWrap, x, modelCtx.sigmas, callback=callback,extra_args=modelCtx.extra_args)
                
            return x_0


        #for i in range(genParams.n_batches):
        with torch.no_grad():
            if hasattr(mw.model, 'ema_scope'):
                with mw.model.ema_scope():
                    samples = doSamples(genParams.sampleMethod.lower())
            else:
                samples = doSamples(genParams.sampleMethod.lower())

        ############
        # end gen loop
        ################

        samples = mw.DecodeImage(samples)

        #TODO: combine all the samples into a grid somehow
        imgsList = []
        for i, out in enumerate(samples):
            imgsList.append( K.utils.to_pil_image(out) )
            #grid = K.utils.to_pil_image(out)

        if clipguided == True:
            denoised = modelCtx.kdiffModelWrap.orig_denoised

            denoised = mw.DecodeImage(denoised)

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
