import sys
import gc

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

    CreateONNXModels(model_path, torch.float32, True, modelPrefix + "-fp32-cuda-auto", torchDevice)
    CreateONNXModels(model_path, torch.float32, False, modelPrefix + "-fp32-cuda", torchDevice)
    CreateONNXModels(model_path, torch.float16, True, modelPrefix + "-fp16-cuda-auto", torchDevice)
    #cant process this one
    #CreateONNXModels(model_path, torch.float16, False, modelPrefix + "-fp16-cuda", torchDevice)



def CreateONNXModels(model_path, dtype, autocast, modelName, device):

    modelwrapper:modelWrap.ModelContext = CompVisSDModel.CompVisSDModel(dtype)
    modelwrapper.model_path = model_path
    modelwrapper.modelName = modelName
    modelwrapper.ModelLoadSettings()
    modelwrapper.LoadModel(device)

    outFile = "E:/onnxOut/" + modelwrapper.modelName + "/model.onnx"
    ConvertToONNXCuda(modelwrapper, outFile, dtype, autocast, device)
    
    outFile = "E:/onnxOut/" + modelwrapper.modelName + "-decode/model.onnx"
    ConvertDecodeToONNXCuda(modelwrapper, outFile, dtype, autocast, device)

    modelwrapper.model = None    
    modelwrapper = None
    gc.collect()
    torch.cuda.empty_cache()



def CheckONNX(modelPath:str):
    onnx_model = onnx.load(modelPath)

    print('The model is:\n{}'.format(onnx_model))

    # Check the model
    try:
        onnx.checker.check_model(onnx_model)
    except onnx.checker.ValidationError as e:
        print('The model is invalid: %s' % e)
    else:
        print('The model is valid!')


def ConvertToONNXCuda(modelWrapper:modelWrap.ModelWrap, outFilePath:str, dtype, autocast_enable:bool, device):

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
        
    if str(device) == 'cpu':    
        prec_dev = 'cpu'
    else:
        prec_dev = 'cuda'

    precision_scope = autocast if autocast_enable else nullcontext       
    with torch.no_grad():
        with precision_scope(prec_dev):
            torch.onnx.export(modelCtx.kdiffModelWrap,         # model being run 
                (image_input, sigmaTensor, condTensor, uncondTensor, condscaleTensor),       # model input (or a tuple for multiple inputs)         
                outFilePath,       # where to save the model  
                export_params=True,  # store the trained parameter weights inside the model file 
                opset_version=16,    # the ONNX version to export the model to 
                do_constant_folding=True,  # whether to execute constant folding for optimization 
                input_names = ['modelInput', 'sigma', 'uncond', 'cond', 'cond_scale'],   # the model's input names 
                output_names = ['modelOutput'], # the model's output names 
                dynamic_axes={'modelInput' : {0 : 'batch_size'},
                                'uncond' : {0 : 'batch_size'},
                                'cond' : {0 : 'batch_size'},
                                'sigma' : {0 : 'batch_size'},
                            'modelOutput' : {0 : 'batch_size'}})

    print(" ") 
    print('Model has been converted to ONNX') 

    #CheckONNX(outFilePath)





# this doesnt work yet
def OptimizeONNX(onnxModelInPath:str, onnxModelOutPath:str):
    
    
    #self.model = onnx.load(self.model_path)
    model_in = onnxModelInPath
    model_out = onnxModelOutPath
    quantize.quantize_dynamic(model_in, model_out)
    #onnx.save(qm, model_out)







#hacky stuff to treat encode / decode as models
class decodeWrap(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputImageTensor):
        return self.model.decode_first_stage(inputImageTensor)



def ConvertDecodeToONNXCuda(modelWrapper:modelWrap.ModelWrap, outFilePath:str, dtype, autocast_enable:bool, device):
    #input_tensor = torch.rand((1, 4, 64, 64)).to(device).to(dtype) #1 512x512 image

    input_tensor = torch.randn([1, 4, 64, 64], device=device).to(dtype) + 0.01 * 10

    dw = decodeWrap(modelWrapper.model)

    if str(device) == 'cpu':    
        prec_dev = 'cpu'
    else:
        prec_dev = 'cuda'
    precision_scope = autocast if autocast_enable else nullcontext         
    with torch.no_grad():
        with precision_scope(prec_dev):
            torch.onnx.export(dw,         # model being run 
                (input_tensor),       # model input (or a tuple for multiple inputs)         
                outFilePath,       # where to save the model  
                export_params=True,  # store the trained parameter weights inside the model file 
                opset_version=16,    # the ONNX version to export the model to 
                do_constant_folding=True,  # whether to execute constant folding for optimization 
                input_names = ['modelInput'],   # the model's input names 
                output_names = ['modelOutput'], # the model's output names 
                dynamic_axes={'modelInput' : {0 : 'batch_size'},
                            'modelOutput' : {0 : 'batch_size'}})

    print(" ") 
    print('Decode has been converted to ONNX') 

    #CheckONNX(outFilePath)

    return









Main()