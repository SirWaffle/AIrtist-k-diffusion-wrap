
import sys

sys.path.append('./../k-diffusion')
sys.path.append('./../stable-diffusion')


import k_diffusion as K
import torch
from loguru import logger
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import onnx
import onnxruntime
from einops import rearrange, repeat
from transformers import CLIPTokenizer, CLIPTextModel
import torch.nn as nn
from onnxruntime.quantization import quantize_dynamic, QuantType, quantize, QuantizationMode

import denoisers
import paramsGen
import lossFunctions
import clipWrap
import modelWrap
import CompVisSDModel



class CompVisSDOnnxModel(CompVisSDModel.CompVisSDModel):
    def __init__(self):
        super().__init__()
        self.ONNX = True #hacky flag im using for ONNX testing
        self.ONNXSession:onnxruntime.InferenceSession = None # hacky way to get a session here for testing



    def LoadModel(self, device):

        #hack for optimization stuff
        if False:
            #TODO: this just loads for checking,m need to use onnxruntime, 
            # create a session, and change how inference is done...
            #maybe making an ONNXwrapper would help? ugh.
            #self.model = onnx.load(self.model_path)

            #hack here to quantize modelll.
            
            model_fp32 = self.model_path
            model_quant = 'e:/onnxOut/model.quant.onnx'
            #quantized_model = quantize_dynamic(model_fp32, model_quant)
            #quantize.quantize_dynamic(model_fp32, model_quant)
            #onnx.save(qm, model_quant)


        self.kdiffExternalModelWrap = self.model
        self.ONNX = True

        
        sess_options = onnxruntime.SessionOptions()

        # Set graph optimization level
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        if str(device) == 'cpu':
            providers=['CPUExecutionProvider']
        else:                
            #see if this gives us a speed bump at the cost of VRAM
            #also increases the initial run by 1 min ...
            #but becomes 30% faster after that
            providers=['CUDAExecutionProvider']
            #providers = [("CUDAExecutionProvider", {"cudnn_conv_use_max_workspace": '1'})]

        self.ONNXSession = onnxruntime.InferenceSession(self.model_path, sess_options, providers=providers)#,'CUDAExecutionProvider'])
        #HACK again, we need to laod the normal model to decode the image..
        #self.model_path = 'E:/MLModels/stableDiffusion/sd-v1-3-full-ema.ckpt'sd-v1-4.onnx
        self.model_path = 'E:/MLModels/stableDiffusion/sd-v1-4.ckpt'


        if True:
            #TODO
            #still loading the old model for the encode/decode methods...need to figure out how to get 
            #that seperated. put it in the cpu for now
            self.model_config = OmegaConf.load(self.config_path)
            self.model = instantiate_from_config(self.model_config.model)

            self.model.load_state_dict(torch.load(self.model_path, map_location='cpu')["state_dict"], strict=False)
            #if str(device) == 'cpu':
            self.model.eval().to(torch.float32)#.to(device)
            self.tensordtype = torch.float32
            #else:
            #    self.model.eval().half()#.to(device)
            #    self.tensordtype = torch.float16

            self.kdiffExternalModelWrap = K.external.CompVisDenoiser(self.model, False, device=device)

        self.default_imageTensorSize = self.default_image_size_x//16  

        #load the clip text embedder we need for converting tex tto embeds.
        #we could probably dump this if we have the clipwrapper loaded for clip guidance to save memory...
        #TODO!
        #but for now, this will work

        #hrm, huggingface repo of: clip-vit-large-patch14
        # is NOT the same as the model pulled down... it follows a different code path. figure this out...
        #self.frozenClip = FrozenCLIPEmbedder("E:/MLModels/clip/clip-vit-large-patch14",device)
        self.frozenClip = CompVisSDModel.FrozenCLIPEmbedder(device = device)

        return

    def CreateCFGDenoiser(self, inst:modelWrap.ModelContext, clipEncoder:clipWrap.ClipWrap, cfgPrompts, condScale, genParams:paramsGen.ParamsGen):
        if self.ONNX == True:
            inst.target_cfg_embeds = self.frozenClip.encode(cfgPrompts).float()
            c = inst.target_cfg_embeds
            uc = torch.zeros_like(c)
        else:            
            #c = inst.modelWrap.model.get_learned_conditioning(cfgPrompts)
            #uc = inst.modelWrap.model.get_learned_conditioning(genParams.num_images_to_sample * [""])
            c = self.frozenClip.encode(cfgPrompts)
            uc = self.frozenClip.encode(genParams.num_images_to_sample * [""])

        inst.extra_args = {'cond': c, 'uncond': uc, 'cond_scale': condScale}

        if self.ONNX == False:
            inst.kdiffModelWrap = denoisers.CFGDenoiser(self.kdiffExternalModelWrap)
        else:
            inst.kdiffModelWrap = self.model
        
        return inst


    def EncodeInitImage(self, initTensor):
        initTensor = initTensor.cpu()
        init = self.model.encode_first_stage(initTensor.to(self.tensordtype))
        init = self.model.get_first_stage_encoding(init)
        return init


    def DecodeImage(self, imageTensor):
        #we currently keep the original model in RAM for decode/encode
        #convert from device back to float32 cpu 
        imageTensor = imageTensor.cpu().to(torch.float32)
        return self.model.decode_first_stage(imageTensor)