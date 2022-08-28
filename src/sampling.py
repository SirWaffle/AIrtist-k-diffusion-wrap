import math
import numpy as np
from scipy import integrate
import torch
from torchdiffeq import odeint
from tqdm.auto import trange, tqdm
import onnxruntime


import k_diffusion as K

@torch.no_grad()
def sample_lms_ONNX(model:onnxruntime.InferenceSession, x, sigmas, extra_args=None, callback=None, disable=None, order=4):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    ds = []

    for i in trange(len(sigmas) - 1, disable=disable):
        #q = sigmas.cpu().detach()[i]
        q = np.array((sigmas[i] * s_in).cpu().detach())
        denoised = model.run(None,         
                   {'modelInput': x.cpu().detach().numpy(), 
                   'sigma': np.array(q), 
                   'uncond': extra_args["uncond"].cpu().detach().numpy(),
                   'cond': extra_args["cond"].cpu().detach().numpy(), 
                   'cond_scale': np.array([extra_args["cond_scale"]],dtype='float32')})
                   
        denoised = torch.from_numpy(np.array(denoised)).cuda().squeeze(0)#.unsqueeze(0)
        #denoised = torch.from_numpy(np.array(denoised)).cpu().squeeze(0)#.unsqueeze(0)
        d = K.sampling.to_d(x, sigmas[i], denoised)
        ds.append(d)
        if len(ds) > order:
            ds.pop(0)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised}, denoised)
        cur_order = min(i + 1, order)
        coeffs = [K.sampling.linear_multistep_coeff(cur_order, sigmas.cpu(), i, j) for j in range(cur_order)]
        x = x + sum(coeff * d for coeff, d in zip(coeffs, reversed(ds)))
    return x


@torch.no_grad()
def sample_lms_ONNX_with_binding(model:onnxruntime.InferenceSession, x, sigmas, extra_args=None, callback=None, disable=None, order=4):
    
    #need to smash thing sdown to FP16 to test fp16 mode
    #fp16 is slower than fp32 model?!
    bindingType = np.float32 #16

    #x = x.half()
    #sigmas = sigmas.half()
    #extra_args["cond"] = extra_args["cond"].half()
    #extra_args["uncond"] = extra_args["uncond"].half()

    condscaleTens = torch.FloatTensor([extra_args["cond_scale"]]).cuda()
    #condscaleTens = condscaleTens.half()

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    ds = []


    #this is just shape of x : logit_output = torch.empty((2, 128, 50257), dtype=torch.float32, device='cuda')
    denoised = torch.zeros_like(x).cuda()
    

    #do i need to rebind every time?
    binding:onnxruntime.IOBinding = model.io_binding()
    binding.bind_output(name='modelOutput', device_type='cuda', device_id=0, element_type=bindingType, shape=tuple(denoised.shape), buffer_ptr=denoised.data_ptr())    
    binding.bind_input(name='uncond', device_type='cuda',device_id=0, element_type=bindingType, shape=tuple(extra_args["uncond"].shape), buffer_ptr=extra_args["uncond"].data_ptr())
    binding.bind_input(name='cond', device_type='cuda',device_id=0, element_type=bindingType, shape=tuple(extra_args["cond"].shape), buffer_ptr=extra_args["cond"].data_ptr())
    binding.bind_input(name='cond_scale', device_type='cuda',device_id=0, element_type=bindingType, shape=tuple(condscaleTens.shape),buffer_ptr=condscaleTens.data_ptr())

    for i in trange(len(sigmas) - 1, disable=disable):
        q = (sigmas[i] * s_in).cuda() 
        
        #lets try just binding what we feed in that changes...
        binding.bind_input(name='modelInput', device_type='cuda',device_id=0, element_type=bindingType, shape=tuple(x.shape), buffer_ptr=x.data_ptr())    
        binding.bind_input(name='sigma', device_type='cuda',device_id=0, element_type=bindingType, shape=tuple(q.shape), buffer_ptr=q.data_ptr())
        
        model.run_with_iobinding(binding)

        d = K.sampling.to_d(x, sigmas[i], denoised)
        ds.append(d)
        if len(ds) > order:
            ds.pop(0)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised}, denoised)
        cur_order = min(i + 1, order)
        coeffs = [K.sampling.linear_multistep_coeff(cur_order, sigmas.cpu(), i, j) for j in range(cur_order)]
        x = x + sum(coeff * d for coeff, d in zip(coeffs, reversed(ds)))
      
    return x
