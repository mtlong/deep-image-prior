## Importing libraries

from __future__ import print_function
import matplotlib.pyplot as plt

import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
from models import *

import torch
import torch.optim

from skimage.measure import compare_psnr
import skimage.io as skio
from models.downsampler import Downsampler

from utils.sr_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor


output_folder_root = "./data/sr/results/"

imsize = -1 
factor_list = [4, 8]
enforse_div32 = 'CROP' # we usually need the dimensions to be divisible by a power of two (32 in this case)
PLOT = True

# To produce images from the paper we took *_GT.png images from LapSRN viewer for corresponding factor,
# e.g. x4/zebra_GT.png for factor=4, and x8/zebra_GT.png for factor=8 
img_idx_list = [1, 2, 3, 4, 5, 6, 7]
noise_type_list = ['noise', 'meshgrid']

for img_idx in img_idx_list:
    path_to_image = 'data/sr/' + str(img_idx) + '.jpg'
    for factor in factor_list:        
        # Starts here
        imgs = load_LR_HR_imgs_sr(path_to_image , imsize, factor, enforse_div32)                
        
        ### Set up paramters and network
        for noise_type in noise_type_list:
            output_folder = output_folder_root + "img_" + str(img_idx) + "_factor_" + str(factor) + "_input_" + noise_type
            os.system("mkdir " + output_folder)
            
            INPUT =     noise_type
            pad   =     'reflection'
            OPT_OVER =  'net'
            KERNEL_TYPE='lanczos2'
            
            if INPUT == "noise":
                input_depth = 32
            else:
                input_depth = 2
            
            LR = 0.01
            tv_weight = 0.0
            
            OPTIMIZER = 'adam'
            
            if factor == 4: 
                num_iter = 4000
                reg_noise_std = 0.03
            elif factor == 8:
                num_iter = 4000
                reg_noise_std = 0.05
            else:
                assert False, 'We did not experiment with other factors'
            
            
            net_input = get_noise(input_depth, INPUT, (imgs['HR_pil'].size[1], imgs['HR_pil'].size[0])).type(dtype).detach()
            
            NET_TYPE = 'skip' # UNet, ResNet
            net = get_net(input_depth, 'skip', pad,
                          skip_n33d=128, 
                          skip_n33u=128, 
                          skip_n11=4, 
                          num_scales=5,
                          upsample_mode='bilinear').type(dtype)
            
            # Losses
            mse = torch.nn.MSELoss().type(dtype)
            
            img_LR_var = np_to_var(imgs['LR_np']).type(dtype)
            
            downsampler = Downsampler(n_planes=3, factor=factor, kernel_type=KERNEL_TYPE, phase=0.5, preserve_size=True).type(dtype)
            
            
            ## Define closure and optimize
            
            def closure():
                global i
                
                if reg_noise_std > 0:
                    net_input.data = net_input_saved + (noise.normal_() * reg_noise_std)
                    
                out_HR = net(net_input)
                out_LR = downsampler(out_HR)
            
                total_loss = mse(out_LR, img_LR_var) 
                
                if tv_weight > 0:
                    total_loss += tv_weight * tv_loss(out_HR)
                    
                total_loss.backward()
            
                # Log
                psnr_LR = compare_psnr(imgs['LR_np'], var_to_np(out_LR))
                psnr_HR = compare_psnr(imgs['HR_np'], var_to_np(out_HR))
                print ('Iteration %05d    PSNR_LR %.3f   PSNR_HR %.3f' % (i, psnr_LR, psnr_HR), '\r', end='')
                                  
                # History
                psnr_history.append([psnr_LR, psnr_HR])
                
                if PLOT and i % 20 == 0:
                    out_HR_np = var_to_np(out_HR)
                    skio.imsave(output_folder + str(i) + ".jpg", np.clip(out_HR_np, 0, 1))
            
                i += 1
                
                return total_loss
            
            
            psnr_history = [] 
            net_input_saved = net_input.data.clone()
            
            noise = net_input.data.clone()
            
            i = 0
            p = get_params(OPT_OVER, net, net_input)
            optimize(OPTIMIZER, p, closure, LR, num_iter)
                        
            out_HR_np = np.clip(var_to_np(net(net_input)), 0, 1)
            skio.imsave(output_folder + "Final.jpg", out_HR_np)
 
            upsampler2 = torch.nn.Upsample(scale_factor = 2, mode = 'nearest')
            net_input_2x = upsampler2(net_input)
            out_HR_np_2x = np.clip(var_to_np(net(net_input_2x)), 0, 1)
            skio.imsave(output_folder + "Final_2x.jpg", out_HR_np_2x)
            
            upsampler4 = torch.nn.Upsample(scale_factor = 4, mode = 'nearest')
            net_input_4x = upsampler4(net_input)
            out_HR_np_4x = np.clip(var_to_np(net(net_input_4x)), 0, 1)
            skio.imsave(output_folder + "Final_4x.jpg", out_HR_np_4x)

            upsampler8 = torch.nn.Upsample(scale_factor = 8, mode = 'nearest')
            net_input_8x = upsampler8(net_input)
            out_HR_np_8x = np.clip(var_to_np(net(net_input_8x)), 0, 1)
            skio.imsave(output_folder + "Final_8x.jpg", out_HR_np_8x)
            



