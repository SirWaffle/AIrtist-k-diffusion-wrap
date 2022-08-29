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

def DoGenerate(kdiffReq): 
    genParams = paramsGen.ParamsGen()

    genParams.CFGprompts = None 
    genParams.CLIPprompts = None 
    
    genParams.prompts = ["cat"]#["photorealistic painting portrait of a beautiful gorgeous majestic young goddess princess figurative liminal complex flat geometric minimalism by oskar schlemmer rembrandt sorolla oil on canvas cosmic levels shimmer pastel color "]
    genParams.init_image = None #"https://images.saymedia-content.com/.image/t_share/MTc2Mjg0ODMwNTQ2NDA0NTI1/yin-yang-symbol-meaning-chinese-philosophy.jpg"
    genParams.image_prompts = None #["https://images.saymedia-content.com/.image/t_share/MTc2Mjg0ODMwNTQ2NDA0NTI1/yin-yang-symbol-meaning-chinese-philosophy.jpg"]
    

    genParams.n_steps = 50               # 1000 - The number of timesteps to use    
   
    genParams.seed = 646941043242600 
    genParams.num_images_to_sample = 4    

    genParams.conditioning_scale = 7.0 #7.0 seems pretty good in general for RDM, 7.0 to 10.0 pretty good for SD

    genParams.sampleMethod = "LMS" #only have an LMS for ONNX right now

    genParams.noiseSchedule = "MODEL"

    genParams.saveEvery = 99999 # high value prevents saving, save an image every x steps
    

    kdiffer = kdiffWrap.KDiffWrap() #default device is cuda:0, but you can pass 'cpu' to run on CPU


    genParams.image_size_x = 512
    genParams.image_size_y = 512

    #dont need to load an extra clip model with sd-v1-4, comment it out for SD
    clipwrap = None# kdiffer.CreateClipModel("vit-l-14") 
    modelwrap = kdiffer.CreateModel("onnx-test")#"sd-v1-4")#


    gridPilImage, individualPilImages, seeds = kdiffer.internal_run(genParams, clipwrap, modelwrap)


    for i, out in enumerate(individualPilImages):
        filename = f'out_{i}.png'
        out.save(filename)

    gridPilImage.save('out_grid.png')      







gc.collect()
DoGenerate(None)
gc.collect()