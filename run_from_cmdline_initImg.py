import gc
import sys

sys.path.append('./stable-diffusion')
sys.path.append('./src')
sys.path.append('./k-diffusion')
sys.path.append('./guided-diffusion')
sys.path.append('./v-diffusion-pytorch')

import paramsGen
import kdiffWrap
import ProfilerHelper
import torch
from torch.profiler import profile, record_function, ProfilerActivity

#@ProfilerHelper.profile(sort_by='cumulative', lines_to_print=80, strip_dirs=False)
def DoGenerate(kdiffReq): 
    # kdiff req is a struct KDiffRequest from c#

    ## Generation Settings
    print(str(kdiffReq))

    genParams = paramsGen.ParamsGen()

    #genParams.prompts =  ['a virus monster is playing guitar, oil on canvas','a virus monster is playing guitar, oil on canvas','a virus monster is playing guitar, oil on canvas'] #['A mysterious orb by Ernst Fuchs']
    #genParams.prompts =  ["""Cyberpunk Knight showing his face, Biopunk, Trending on Artstation, Intelligent, symmetrical, realistic, oil painting, Biopunk, Biopunk, Biopunk, Biopunk, Brushstrokes, Symmetrical, Proportional"""] #['A mysterious orb by Ernst Fuchs']
    #genParams.prompts = ['a cat']
    #genParams.prompts = ['a cat by anton fadeev'] #] #['a cat'] #None #'a cat'
    genParams.CFGprompts = None #['a cat']
    genParams.CLIPprompts = None #['a digital painting by anton fadeev']#['A mysterious orb by Ernst Fuchs'] #NONE
    
    genParams.prompts = ["a wizard holding an ice staf and torch, with firey eyes, casting a spell of defiance, digital illustration, epic and detailed"]
    genParams.init_image = "./mspaint.png"
    genParams.image_prompts = None #["https://images.saymedia-content.com/.image/t_share/MTc2Mjg0ODMwNTQ2NDA0NTI1/yin-yang-symbol-meaning-chinese-philosophy.jpg"]
    
    #DPM_2 / DPM_2_A .. works for RDM CFG
    #LMS seems to work for clip guidance ( HUEUN as well, i think )

    genParams.n_steps = 100              # 1000 - The number of timesteps to use    


    genParams.seed = 646941043242600 #None #2226809351
    genParams.num_images_to_sample = 1    # 64 - not sure? -- seems to act as 'number of batches'
    # This can be an URL or Colab local path and must be in quotes.
    #genParams.init_image = None
    genParams.sigma_start = 3   # The starting noise level when using an init image.
                    # Higher values make the output look more like the init. ( worng, lower values do )


    genParams.conditioning_scale = 7.0 #7.0 seems pretty good in general for RDM, 7.0 to 10.0 pretty good for SD
    genParams.sampleMethod = "LMS" #LMS"#"HEUN" #"LMS" #LMS or HEUN

    genParams.noiseSchedule = "MODEL"#"KARRAS" #"MODEL"
    genParams.sigma_min = 0.0001 #1.4 for rdm, -1 = default, doesnt change MODEL sched
    genParams.sigma_max = 10 #20.0 for rdm, -1 = default, doesnt change MODEL sched

    genParams.saveEvery = 999999#5    

    genParams.image_size_x = 512
    genParams.image_size_y = 512


    ##########CLIP RELATED######################
    genParams.cutoutMethod = "RANDOM" 
    genParams.tv_scale = 100              # 100 - Controls the smoothness of the final output.
    genParams.range_scale = 50            # 50 - Controls how far out of range RGB values are allowed to be.
    genParams.cutn = 32                  # 16 - The number of random crops per step.
                                # Good values are 16 for 256x256 and 64-128 for 512x512.
    genParams.cut_pow = 0.5               # 0.5 - 
    genParams.init_scale = 1000  # This enhances the effect of the init image, a good value is 1000.
    genParams.clip_guidance_scale = 10000#1000  # 1000 - Controls how much the image should look like the prompt.
    genParams.overall_clip_scale = 1.0 #tune down the total value of clip losses...
    genParams.aesthetics_scale = 0





    kdiffer = kdiffWrap.KDiffWrap() #default device is cuda:0, but you can pass 'cpu' to run on CPU

    #dont need to load an extra clip model with sd-v1-4, comment it out for SD
    clipwrap = None#kdiffer.CreateClipModel("vit-l-14") 
    modelwrap = kdiffer.CreateModel("sd-v1-4")

    #torch.autograd.set_detect_anomaly(True)

    #with profile(activities=[
    #        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True,) as prof:
    #    with record_function("model_inference"):
    #        gridPilImage, individualPilImages = kdiffer.internal_run(genParams, clipwrap, modelwrap)

    #with open("prof.txt", "w") as external_file:
    #    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10), file=external_file)
    #    external_file.close()
            

    gridPilImage, individualPilImages, seeds = kdiffer.internal_run(genParams, clipwrap, modelwrap)

    for i, out in enumerate(individualPilImages):
        filename = f'out_{i}.png'
        out.save(filename)

    gridPilImage.save('out_grid.png')      







gc.collect()
DoGenerate(None)
gc.collect()