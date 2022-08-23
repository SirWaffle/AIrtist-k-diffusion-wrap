import gc
import io
import sys

sys.path.append('./src')
sys.path.append('./k-diffusion')
sys.path.append('./guided-diffusion')
sys.path.append('./v-diffusion-pytorch')

from loguru import logger

import paramsGen
import kdiffWrap

from System import Byte, Array



def serve_pil_image(pil_img):
    img_io = io.BytesIO()
    pil_img.save(img_io, 'PNG')
    img_io.seek(0)
    return img_io


def CreateKDiffer() -> kdiffWrap.KDiffWrap:
    kdiffer = kdiffWrap.KDiffWrap()
    return kdiffer


def DoGenerate(kdiffReq, kdiffer:kdiffWrap.KDiffWrap, clipwrap, modelwrap):

    ## Generation Settings
    print(str(kdiffReq))

    genParams = paramsGen.ParamsGen()

    genParams.prompts = kdiffReq.prompt.split('|')
    genParams.CFGprompts = kdiffReq.cfgprompt.split('|')
    genParams.CLIPprompts = kdiffReq.clipprompt.split('|')
    
    genParams.image_prompts = kdiffReq.image_prompts
    if genParams.image_prompts != None and len(genParams.image_prompts) < 3:
        genParams.image_prompts = []

    genParams.n_steps = kdiffReq.steps                # 80 LMS or 1000 HEUN - The number of timesteps to use
    genParams.clip_guidance_scale = kdiffReq.clip_scale  # 1000 - Controls how much the image should look like the prompt.
    genParams.tv_scale = kdiffReq.tv_scale             # 100 - Controls the smoothness of the final output.
    genParams.range_scale = kdiffReq.range_scale            # 50 - Controls how far out of range RGB values are allowed to be.
    genParams.cutn = kdiffReq.cuts                  # 16 - The number of random crops per step.
                                # Good values are 16 for 256x256 and 64-128 for 512x512.
    genParams.cut_pow = kdiffReq.cut_pow               # 0.5 - 
    genParams.seed = kdiffReq.seed
    if genParams.seed == -1:
        genParams.seed = None

    genParams.num_images_to_sample = kdiffReq.grid     # 64 - not sure? -- seems to act as 'number of batches'
    # This can be an URL or Colab local path and must be in quotes.

    genParams.init_image = kdiffReq.init_image #init image URL
    if len(genParams.init_image) < 3:
        genParams.init_image = None

    genParams.sigma_start = 10   # The starting noise level when using an init image.
                    # Higher values make the output look more like the init.
    genParams.init_scale = 1000  # This enhances the effect of the init image, a good value is 1000.

    genParams.conditioning_scale = kdiffReq.cond_scale

    genParams.sampleMethod = kdiffReq.sample #"LMS" #LMS or HEUN

    genParams.sigma_max = kdiffReq.smax
    genParams.sigma_min = kdiffReq.smin

    genParams.cutoutMethod = kdiffReq.cutoutMethod
    genParams.aesthetics_scale = kdiffReq.aesthetics_scale

    genParams.noiseSchedule = kdiffReq.sched
    

    genParams.image_size_x = kdiffReq.image_size_x
    genParams.image_size_y = kdiffReq.image_size_y

    #clipwrap, modelwrap = kdiffer.CreateModels(kdiffReq.model, kdiffReq.clipModel)
    #clipwrap, modelwrap = CreateModelsWithCaching(kdiffer, kdiffReq.model, kdiffReq.clipModel)
    gridPilImage, individualPilImages = kdiffer.internal_run(genParams, clipwrap, modelwrap)

    #gridPilImage, individualPilImages = internal_run.internal_run(genParams, kdiffReq.model, kdiffReq.clipModel)

    #pilImg = K.utils.to_pil_image(image)
    conv = serve_pil_image(gridPilImage)
    op = conv.getvalue()

    c_sharp_bytes = Array[Byte](op)
    print(type(c_sharp_bytes))

    return c_sharp_bytes


########
## AIArtist callable from c#
#########

def Generate(kdiffReq, kdiffer, clipwrap, modelwrap): 
    gc.collect()
    ret = DoGenerate(kdiffReq, kdiffer, clipwrap, modelwrap)
    gc.collect()
    return ret
