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
img_idx_list = list(range(1, 18))
#img_idx_list = [9]
noise_type_list = ['noise', 'meshgrid']
num_iter = 15000
#num_iter = 1000
reg_noise_std_list = [0.03, 0.05, 0.075]

for img_idx in img_idx_list:
    path_to_image = 'data/sr/' + str(img_idx) + '.jpg'
    for factor in factor_list:        
        for noise_level_idx in range(len(reg_noise_std_list)):
            reg_noise_std = reg_noise_std_list[noise_level_idx]
            # Starts here
            imgs = load_LR_HR_imgs_sr(path_to_image , imsize, factor, enforse_div32)     
            imgs['bicubic_np'], imgs['sharp_np'], imgs['nearest_np'] = get_baselines(imgs['LR_pil'], imgs['HR_pil'])
            img_bicubics_np = imgs['bicubic_np']         
            img_sharp_np = imgs['sharp_np']
            img_nearest_np = imgs['nearest_np']
            
            ### Set up paramters and network
            for noise_type in noise_type_list:
                output_folder = output_folder_root + "img_" + str(img_idx) + "_factor_" + str(factor) + "_input_" + noise_type + "_noise_level_" + str(noise_level_idx) + "/"
                os.system("mkdir " + output_folder)
                skio.imsave(output_folder + "bicubic.jpg", np.clip(img_bicubics_np, 0, 1).transpose(1, 2, 0))
                skio.imsave(output_folder + "unsharp_mask.jpg", np.clip(img_sharp_np, 0, 1).transpose(1, 2, 0))
                skio.imsave(output_folder + "nearest.jpg", np.clip(img_nearest_np, 0, 1).transpose(1, 2, 0))
                
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
                    
                    if PLOT and i % 200 == 0:
                        out_HR_np = np.clip(var_to_np(out_HR), 0, 1).transpose(1,2,0)
                        skio.imsave(output_folder + str(i) + ".jpg", out_HR_np)
                
                    i += 1
                    
                    return total_loss
                
                
                psnr_history = [] 
                net_input_saved = net_input.data.clone()
                
                noise = net_input.data.clone()
                
                i = 0
                p = get_params(OPT_OVER, net, net_input)
                optimize(OPTIMIZER, p, closure, LR, num_iter)
                            
                out_HR_np = np.clip(var_to_np(net(net_input)), 0, 1).transpose(1,2,0)
                skio.imsave(output_folder + "Final.jpg", out_HR_np)
                
                try:
                    if img_idx > 8:
                        upsampler2 = torch.nn.Upsample(scale_factor = 2, mode = 'nearest')
                        net_input_2x = upsampler2(net_input)
                        out_HR_np_2x = np.clip(var_to_np(net(net_input_2x)), 0, 1).transpose(1,2,0)
                        skio.imsave(output_folder + "Final_2x.jpg", out_HR_np_2x)
                except:
                    print("Cannot generate higher resolution")
            
            



