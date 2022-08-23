

class ParamsGen():
    def __init__(self):
        super().__init__()
        self.prompts = ['A mysterious orb by Ernst Fuchs']

        # if there are CFG prompts, they are fed into the CFG
        self.CFGprompts = None
        self.CLIPprompts = None

        self.image_prompts = []
        self.n_steps = 500                # 1000 - The number of timesteps to use
        self.clip_guidance_scale = 1000  # 1000 - Controls how much the image should look like the prompt.
        self.tv_scale = 100              # 100 - Controls the smoothness of the final output.
        self.range_scale = 50            # 50 - Controls how far out of range RGB values are allowed to be.
        self.cutn = 16                   # 16 - The number of random crops per step.
                                    # Good values are 16 for 256x256 and 64-128 for 512x512.


        self.overall_clip_scale = 1.0 #scaling for the entire clip calculation ( clip loss, tv scale, range scale)

        self.cut_pow = 0.5               # 0.5 - 
        self.seed = None
        self.num_images_to_sample = 2    # 64 - not sure? -- seems to act as 'number of batches'
        # This can be an URL or Colab local path and must be in quotes.
        self.init_image = None
        self.sigma_start = 10   # The starting noise level when using an init image.
                        # Higher values make the output look more like the init.
        self.init_scale = 1000  # This enhances the effect of the init image, a good value is 1000.

        self.conditioning_scale = 5.0 #??? 


        self.sampleMethod = 'HEUN' #LMS or HEUN

        self.saveEvery = 9000000 #basically, set to never save

        #types of sched: MODEL, KARRAS, 
        self.noiseSchedule = "KARRAS"
        self.sigma_min = -1 #1.4 is a good default
        self.sigma_max = -1 #100.0 is a good default



        self.aesthetics_scale = 0

        self.cutoutMethod = "RANDOM" #EVEN

        self.image_size_x = -1
        self.image_size_y = -1