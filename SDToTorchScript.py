import sys
import gc
import os

sys.path.append('./stable-diffusion')
sys.path.append('./k-diffusion')
sys.path.append('./src')

import k_diffusion as K
import torch
import torch.onnx 
from torch import nn
from torch import autocast
from onnxruntime.quantization import quantize_dynamic, QuantType, quantize, QuantizationMode
from contextlib import nullcontext
import onnx
import onnxruntime
import modelWrap
import CompVisSDModel
import denoisers

#
# Edit paths below in 'main' to point to models / output paths
#


## Note:
#  .\stable-diffusion\ldm\modules\diffusionmodules\util.py
# will give an error when converting to ONNX due to being unable to handle 'CheckpointFunction'
# this call is not needed for inference, so just hard code the if statement to always be false:
#
#def checkpoint(func, inputs, params, flag):
#    ...
#    flag = False  <---- add this
#    if flag:
#        args = tuple(inputs) + tuple(params)
#        return CheckpointFunction.apply(func, len(inputs), *args)
#    else:
#        return func(*inputs)
#


# main executing function
def Main():

    #fastest on GPU appears to be autocast = true, dtype = float32
    #decoder for FP16 no autocast fails to load as a session for some reason...

    torchDevice = torch.device('cuda:0') #'cpu'

    modelPrefix = "sd-v1-4"
    model_path = "E:/MLModels/stableDiffusion/sd-v1-4.ckpt" 

    #CreateModels(model_path, torch.float32, True, modelPrefix + "-fp32-cuda0-auto-jit-trace", torchDevice)
    CreateModels(model_path, torch.float32, False, modelPrefix + "-fp32-cuda0-TS-trace", torchDevice)
    #CreateONNXModels(model_path, torch.float16, True, modelPrefix + "-fp16-cuda-auto", torchDevice)
    #cant process this one
    #CreateONNXModels(model_path, torch.float16, False, modelPrefix + "-fp16-cuda", torchDevice)


def MakeDir(path:str):
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)


def CreateModels(model_path, dtype, autocast, modelName, device):

    modelwrapper:modelWrap.ModelContext = CompVisSDModel.CompVisSDModel(dtype)
    modelwrapper.model_path = model_path
    modelwrapper.modelName = modelName
    modelwrapper.ModelLoadSettings()
    modelwrapper.LoadModel(device)

    outFile = "E:/onnxOut/" + modelwrapper.modelName + "/model.pt"
    MakeDir("E:/onnxOut/" + modelwrapper.modelName )
    ConvertToCuda(modelwrapper, outFile, dtype, autocast, device)
    
    outFile = "E:/onnxOut/" + modelwrapper.modelName + "-decode/model.pt"
    MakeDir("E:/onnxOut/" + modelwrapper.modelName + "-decode" )
    ConvertDecodeToCuda(modelwrapper, outFile, dtype, autocast, device)

    modelwrapper.model = None    
    modelwrapper = None
    gc.collect()
    torch.cuda.empty_cache()


#hacky stuff to treat encode / decode as models
class FwdWrap(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x_noisy, t, cond):
        return self.model.apply_model(x_noisy, t, cond)



def ConvertToCuda(modelWrapper:modelWrap.ModelWrap, outFilePath:str, dtype, autocast_enable:bool, device):

    sigmaTensor = torch.FloatTensor([0.1]).to(device).to(dtype)
    condTensor = modelWrapper.frozenClip.encode(['']).to(device).to(dtype)
    uncondTensor = modelWrapper.frozenClip.encode(['sample stupid prompt hellooo']).to(device).to(dtype)
    condscaleTensor = torch.FloatTensor([0.1]).to(device).to(dtype)
    image_input = torch.rand((1, 4, 64, 64)).to(device).to(dtype) #1 512x512 image

    #put this through the KDiff wrapper, it seems to work well
    modelCtx:modelWrap.ModelContext = modelWrap.ModelContext()
    modelCtx.modelWrap = modelWrapper

    #modelCtx.extra_args = {'cond': c, 'uncond': uc, 'cond_scale': condScale}
    modelCtx.kdiffModelWrap = denoisers.CFGDenoiser(modelWrapper.kdiffExternalModelWrap)
    
    if True:
        sm = torch.jit.script(modelCtx.kdiffModelWrap)
    else:
        #lets try without the wrapper, will have to redo the external wrapper and CFG denoiser...
        if True: #without wrapper
            fwdwrap = FwdWrap(modelCtx.modelWrap.model)
            image_input = torch.cat([image_input] * 2)
            sigmaTensor = torch.cat([sigmaTensor] * 2)
            cond_in = torch.cat([uncondTensor, condTensor]) 
            sm = torch.jit.trace(fwdwrap, 
                (image_input, sigmaTensor, cond_in), 
                strict=True,
                check_trace=False) #this is off by more than 1e-5...
        else: #with wrapper
            sm = torch.jit.trace(modelCtx.kdiffModelWrap, 
                (image_input, sigmaTensor, condTensor, uncondTensor, condscaleTensor), 
                strict=True,
                check_trace=False) #this is off by more than 1e-5...
    sm.save(outFilePath)

    print(" ") 
    print('Model has been converted to TorchScript') 





#hacky stuff to treat encode / decode as models
class decodeWrap(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputImageTensor):
        return self.model.decode_first_stage(inputImageTensor)



def ConvertDecodeToCuda(modelWrapper:modelWrap.ModelWrap, outFilePath:str, dtype, autocast_enable:bool, device):
    #input_tensor = torch.rand((1, 4, 64, 64)).to(device).to(dtype) #1 512x512 image

    input_tensor = torch.randn([1, 4, 64, 64], device=device).to(dtype) + 0.01 * 10

    dw = decodeWrap(modelWrapper.model)

    if True:
        sm = torch.jit.script(dw)
    else:
        sm = torch.jit.trace(dw, input_tensor, 
                strict=True,
                check_trace=False) #this is off by more than 1e-5...)

    sm.save(outFilePath)
    
    print(" ") 
    print('Decode has been converted to TorchScript') 


    return









Main()