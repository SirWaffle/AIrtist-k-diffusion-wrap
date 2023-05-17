import gc
import io
import sys
import torch

sys.path.append('./src')
sys.path.append('./k-diffusion')
sys.path.append('./guided-diffusion')
sys.path.append('./v-diffusion-pytorch')
sys.path.append('./kandinsky2')

from kandinsky2 import get_kandinsky2

from loguru import logger

import paramsGen
import kdiffWrap

from System import Byte, Array

from PIL import Image

def serve_pil_image(pil_img):
    img_io = io.BytesIO()
    pil_img.save(img_io, 'PNG')
    img_io.seek(0)
    return img_io


def CreateKDiffer() -> kdiffWrap.KDiffWrap:
    kdiffer = kdiffWrap.KDiffWrap()
    return kdiffer


def DoGenerate(kdiffReq, resp, kdiffer:kdiffWrap.KDiffWrap, clipwrap, modelwrap):

    ## Generation Settings
    print(str(kdiffReq))

    genParams = paramsGen.ParamsGen()

    genParams.prompts = kdiffReq.prompt.split('|')
    genParams.CFGprompts = kdiffReq.cfgprompt.split('|')
    genParams.CLIPprompts = kdiffReq.clipprompt.split('|')

    genParams.CFGNegPrompts = kdiffReq.cfgNegPrompt.split('|')
    
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

    genParams.num_images_to_sample = kdiffReq.grid     # 64 - not sure? -- seems to act as 'number of batches'
    # This can be an URL or Colab local path and must be in quotes.

    genParams.init_image = kdiffReq.init_image #init image URL
    if len(genParams.init_image) < 3:
        genParams.init_image = None

    genParams.sigma_start = kdiffReq.sigma_start   # The starting noise level when using an init image.
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
    gridPilImage, individualPilImages, seeds = kdiffer.internal_run(genParams, clipwrap, modelwrap)

    #gridPilImage, individualPilImages = internal_run.internal_run(genParams, kdiffReq.model, kdiffReq.clipModel)

    #pilImg = K.utils.to_pil_image(image)
    conv = serve_pil_image(gridPilImage)
    op = conv.getvalue()

    c_sharp_bytes = Array[Byte](op)
    print(type(c_sharp_bytes))

    resp.imageBytes = c_sharp_bytes

    for i, out in enumerate(individualPilImages):
        resp.seed[i] = seeds[i]

        conv = serve_pil_image(out)
        op = conv.getvalue()

        bytes = Array[Byte](op)

        resp.imageInfos[i] = bytes

    del gridPilImage
    del individualPilImages
    del conv
    del op

    gc.collect()
    torch.cuda.empty_cache()

    #return c_sharp_bytes
    return resp




def DeleteKandinskyModel(model):
        if model != None:
            model = model.cpu()
        del model
        gc.collect()
        return None
    
def CreateKandinskyModel():
    model = get_kandinsky2('cuda', task_type='text2img', model_version='2.1', use_flash_attention=False)
    return model
            
def DoGenerateKandinsky(kdiffReq, resp, model):
    ## Generation Settings
    print("Running kandinsky or whatever model")
    print(str(kdiffReq))

    genParams = paramsGen.ParamsGen()

    genParams.prompts = kdiffReq.prompt.split('|')
    genParams.CFGprompts = kdiffReq.cfgprompt.split('|')
    genParams.CLIPprompts = kdiffReq.clipprompt.split('|')

    genParams.CFGNegPrompts = kdiffReq.cfgNegPrompt.split('|')
    
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

    genParams.num_images_to_sample = kdiffReq.grid     # 64 - not sure? -- seems to act as 'number of batches'
    # This can be an URL or Colab local path and must be in quotes.

    genParams.init_image = kdiffReq.init_image #init image URL
    if len(genParams.init_image) < 3:
        genParams.init_image = None

    genParams.sigma_start = kdiffReq.sigma_start   # The starting noise level when using an init image.
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


    if genParams.image_size_y < 768:
        genParams.image_size_y = 768
    if genParams.image_size_x < 768:
        genParams.image_size_x = 768

    images = model.generate_text2img(
        genParams.prompts[0],
        negative_decoder_prompt=genParams.CFGNegPrompts[0],
        num_steps=kdiffReq.steps, #30
        batch_size=genParams.num_images_to_sample, #4
        guidance_scale=genParams.conditioning_scale, #4
        h=genParams.image_size_y,#genParams.image_size_x, #768
        w=genParams.image_size_x,#genParams.image_size_y, #768
        sampler=kdiffReq.kan_sampler, #"p_sampler",
        prior_cf_scale=kdiffReq.kan_prior_cfg, #4,
        prior_steps=str(kdiffReq.kan_prior_steps), #"5",
        #denoised_type='dynamic_threshold', 
        # dynamic_threshold_v = 99.5,
    )

    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = sum(heights)
    total_width = int((total_width / 2) + 1)
    max_height = int((max_height / 2) + 1)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    y_offset = 0
    count = 0
    for im in images:
        count = count + 1
        new_im.paste(im, (x_offset,y_offset))
        x_offset += im.size[0]
        if count == 2:
            x_offset = 0
            y_offset += im.size[0]

    gridPilImage = new_im


    conv = serve_pil_image(gridPilImage)
    op = conv.getvalue()

    c_sharp_bytes = Array[Byte](op)
    print(type(c_sharp_bytes))

    resp.imageBytes = c_sharp_bytes

    for i, out in enumerate(images):
        resp.seed[i] = 0 #seeds[i]

        conv = serve_pil_image(out)
        op = conv.getvalue()

        bytes = Array[Byte](op)

        resp.imageInfos[i] = bytes

    del new_im
    del gridPilImage
    del images
    del conv
    del op

    gc.collect()
    torch.cuda.empty_cache()

    #return c_sharp_bytes
    return resp

########
## AIArtist callable from c#
#########

def Generate(kdiffReq, resp, kdiffer, clipwrap, modelwrap): 
    torch.cuda.empty_cache()
    gc.collect()
    ret = DoGenerate(kdiffReq, resp, kdiffer, clipwrap, modelwrap)
    gc.collect()
    torch.cuda.empty_cache()
    return ret
