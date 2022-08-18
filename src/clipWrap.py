import gc
import io
import math
import sys

sys.path.append('./../aesthetic-predictor')

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms, utils
from torchvision.transforms import functional as TF

import clip
import torch
from torch import nn
from einops import rearrange, repeat
import lpips
import utilFuncs
import cutouts



class ClipWrap:
    def __init__(self):
        self.modelPath = None
        self.model = None
        self.modelNum = -1
        self.lpips_model = None
        self.clip_size = None
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711])

        self.aestheticModel = AestheticModel()


    def ModelLoadSettings(self, modelNum):
        self.modelNum = modelNum

        if self.modelNum != -1:
            self.modelPath = "E:/MLModels/clip/ViT-B-16.pt"

        if self.modelNum == 2:
            self.modelPath = "E:/MLModels/clip/ViT-L-14-336px.pt"

        if self.modelNum == 3:
            self.modelPath = "E:/MLmodels/clip/ViT-L-14.pt"

    def GetAestheticRatingFromImage(self, image):
        image_features = self.model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        prediction = self.aestheticModel.amodel(image_features)
        print(prediction)
        return prediction

    def GetAestheticRatingFromEmbed(self, embed):
        embed = embed.div(embed.norm(dim=-1, keepdim=True))
        prediction = self.aestheticModel.amodel(embed)
        #print(prediction)
        return prediction

    def LoadModel(self, torchDevice):
        self.model = clip.load(self.modelPath, jit=False)[0].eval().requires_grad_(False).to(torchDevice)
        self.lpips_model = lpips.LPIPS(net='vgg').to(torchDevice)
        self.clip_size = self.model.visual.input_resolution
        
    def LoadAestheticsModel(self, torchDevice):
        self.aestheticModel.CreateModel(torchDevice)


    def GetTextPromptEmbeds(self, clip_prompts, device):
        target_embeds = []
        weights = []
        if clip_prompts != None:
            for prompt in clip_prompts:
                txt, weight = utilFuncs.parse_prompt(prompt)
                target_embeds.append(self.model.encode_text(clip.tokenize(txt).to(device)).float())
                weights.append(weight)
        return target_embeds, weights




class AestheticModel:
    def __init__(self):
        self.amodel = None


    def get_aesthetic_model(self, clip_model_name):
        """load the aethetic model"""


        cache_folder = "D:/AIrtist/k-diffusion-wrap/aesthetic-predictor"
        path_to_model = cache_folder + "/sa_0_4_"+clip_model_name+"_linear.pth"

        if clip_model_name == "vit_l_14":
            m = nn.Linear(768, 1)
        elif clip_model_name == "vit_b_32":
            m = nn.Linear(512, 1)
        else:
            raise ValueError()
        s = torch.load(path_to_model)
        m.load_state_dict(s)
        m.eval()
        return m

    def CreateModel(self, device):
        self.amodel= self.get_aesthetic_model( "vit_l_14" ).to(device)
        self.amodel.eval()









class FrozenCLIPTextEmbedder(nn.Module):
    """
    Uses the CLIP transformer encoder for text.
    """
    #max length was 77
    def __init__(self, clipModel, device="cuda", max_length=77, n_repeat=1, normalize=True):
        super().__init__()
        self.model = clipModel
        self.device = device
        self.max_length = max_length
        self.n_repeat = n_repeat
        self.normalize = normalize

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        # clip may only use 77 tokens?
        tokens = clip.tokenize(text, self.max_length, True).to(self.device)
        z = self.model.encode_text(tokens)
        if self.normalize:
            z = z / torch.linalg.norm(z, dim=1, keepdim=True)
        return z

    def encode(self, text):
        z = self(text)
        if z.ndim==2:
            z = z[:, None, :]
        z = repeat(z, 'b 1 d -> b k d', k=self.n_repeat)
        return z