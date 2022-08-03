import gc
import sys

sys.path.append('./src')
sys.path.append('./k-diffusion')
sys.path.append('./guided-diffusion')
sys.path.append('./v-diffusion-pytorch')

import paramsGen
import kdiffWrap

import torch

def DoGenerate(kdiffReq): 
    # kdiff req is a struct KDiffRequest from c#

    ## Generation Settings
    print(str(kdiffReq))

    genParams = paramsGen.ParamsGen()

    #genParams.prompts =  ['a virus monster is playing guitar, oil on canvas','a virus monster is playing guitar, oil on canvas','a virus monster is playing guitar, oil on canvas'] #['A mysterious orb by Ernst Fuchs']
    #genParams.prompts =  ["""Cyberpunk Knight showing his face, Biopunk, Trending on Artstation, Intelligent, symmetrical, realistic, oil painting, Biopunk, Biopunk, Biopunk, Biopunk, Brushstrokes, Symmetrical, Proportional"""] #['A mysterious orb by Ernst Fuchs']
    #genParams.prompts = ['a cat']
    
    genParams.init_image = None #[]
    genParams.image_prompts = None
    genParams.image_prompts = ["https://images.saymedia-content.com/.image/t_share/MTc2Mjg0ODMwNTQ2NDA0NTI1/yin-yang-symbol-meaning-chinese-philosophy.jpg"]
    
    genParams.CFGprompts = None #[]
    genParams.CLIPprompts = None #[]
    genParams.n_steps = 80                # 1000 - The number of timesteps to use    
    genParams.tv_scale = 100              # 100 - Controls the smoothness of the final output.
    genParams.range_scale = 50            # 50 - Controls how far out of range RGB values are allowed to be.
    genParams.cutn = 8                  # 16 - The number of random crops per step.
                                # Good values are 16 for 256x256 and 64-128 for 512x512.
    genParams.cut_pow = 0.5               # 0.5 - 
    genParams.seed = None
    genParams.num_images_to_sample = 1    # 64 - not sure? -- seems to act as 'number of batches'
    # This can be an URL or Colab local path and must be in quotes.
    #genParams.init_image = None
    genParams.sigma_start = 10   # The starting noise level when using an init image.
                    # Higher values make the output look more like the init.
    genParams.init_scale = 1000  # This enhances the effect of the init image, a good value is 1000.

    genParams.conditioning_scale = 3.0 #??? 
    genParams.sampleMethod = "LMS"#"HEUN" #"LMS" #LMS or HEUN

    genParams.noiseSchedule = "MODEL"#"KARRAS" #"MODEL"
    genParams.sigma_min = -1.0 #1.4
    genParams.sigma_max = -1.0 #20.0

    genParams.saveEvery = 99999999#5
    
    genParams.clip_guidance_scale = 0#1000  # 1000 - Controls how much the image should look like the prompt.
    genParams.aesthetics_scale = 0

    genParams.cutoutMethod = "RANDOM" #EVEN

    modelNum = 7 #7
    clipModelNum = 3  #0

    kdiffer = kdiffWrap.KDiffWrap()

    clipguided = genParams.clip_guidance_scale != 0

    clipwrap, modelwrap = kdiffer.CreateModels(modelNum, clipModelNum)

    torch.autograd.set_detect_anomaly(True)

    gridPilImage, individualPilImages = kdiffer.internal_run(genParams, clipwrap, modelwrap)

    for i, out in enumerate(individualPilImages):
        filename = f'out_{i}.png'
        out.save(filename)

    gridPilImage.save('out_grid.png')      







gc.collect()
DoGenerate(None)
gc.collect()