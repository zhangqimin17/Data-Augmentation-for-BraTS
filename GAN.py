import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import gan_config as cf


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.ReLU(inplace = True),
            nn.Conv2d(ch_out, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.ReLU(inplace = True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.ReLU(inplace = True)
        )

    def forward(self,x):
        x = self.up(x)
        return x
    
# Generator Code
class Generator(nn.Module):
    '''
    This is the Pytorch version of U-Net Architecture.

    The input and output of this network is of the same shape.
    Input Size of Network - (310,240,240). 
        Note that the input size here is just for our dataset in this notebook, but if you use this network for other projects, any input size that is a multiple of 2 ** 5 will work.
    Output Size of Network - (310,240,240).
        Shape Format :  (Channel, Width, Height)
    '''
    def __init__(self, ngpu, img_ch = 155 * 2, output_ch = 155, first_layer_numKernel = 64):
        '''
        Constructor for UNet class.
        Parameters:
            img_ch(int): Input channels for the network. Default: 1
            output_ch(int): Output channels for the final network. Default: 1
            first_layer_numKernel(int): Number of kernels uses in the first layer of our unet.
        '''
        super(Generator, self).__init__()
        self.ngpu = ngpu
        
        self.Maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.Conv1 = conv_block(ch_in = img_ch, ch_out = first_layer_numKernel)
        self.Conv2 = conv_block(ch_in = first_layer_numKernel, ch_out = 2 * first_layer_numKernel)
        self.Conv3 = conv_block(ch_in = 2 * first_layer_numKernel, ch_out = 4 * first_layer_numKernel)
        self.Conv4 = conv_block(ch_in = 4 * first_layer_numKernel, ch_out = 8 * first_layer_numKernel)
        self.Conv5 = conv_block(ch_in = 8 * first_layer_numKernel, ch_out = 16 * first_layer_numKernel)

        self.Up5 = up_conv(ch_in = 16 * first_layer_numKernel, ch_out = 8 * first_layer_numKernel)
        self.Up_conv5 = conv_block(ch_in = 16 * first_layer_numKernel, ch_out = 8 * first_layer_numKernel)

        self.Up4 = up_conv(ch_in = 8 * first_layer_numKernel, ch_out = 4 * first_layer_numKernel)
        self.Up_conv4 = conv_block(ch_in = 8 * first_layer_numKernel, ch_out = 4 * first_layer_numKernel)
        self.Up3 = up_conv(ch_in = 4 * first_layer_numKernel, ch_out = 2 * first_layer_numKernel)
        self.Up_conv3 = conv_block(ch_in = 4 * first_layer_numKernel, ch_out = 2 * first_layer_numKernel)
        
        self.Up2 = up_conv(ch_in = 2 * first_layer_numKernel, ch_out = first_layer_numKernel)
        self.Up_conv2 = conv_block(ch_in = 2 * first_layer_numKernel, ch_out = first_layer_numKernel)

        self.Conv_1x1 = nn.Conv2d(first_layer_numKernel, output_ch, kernel_size = 1, stride = 1, padding = 0)

    def forward(self, x):
        '''
        Method for forward propagation in the network.
        Parameters:
            x(torch.Tensor): Input for the network of size (155, 240, 240).

        Returns:
            output(torch.Tensor): Output after the forward propagation 
                                    of network on the input.
        '''
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        
        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim = 1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim = 1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim = 1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim = 1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        tanh = nn.Tanh()
        output = tanh(d1)

        return output



# class Generator(nn.Module):
#     def __init__(self, ngpu):
#         super(Generator, self).__init__()
#         self.ngpu = ngpu
#         self.main = nn.Sequential(
#             # input is Z, going into a convolution
#             nn.ConvTranspose2d(cf.nz, cf.ngf * 16, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(cf.ngf * 16),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(cf.ngf * 16, cf.ngf * 8, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(cf.ngf * 8),
#             nn.ReLU(True),
#             # state size. (ngf*8) x 4 x 4
#             nn.ConvTranspose2d(cf.ngf * 8, cf.ngf * 4, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(cf.ngf * 4),
#             nn.ReLU(True),
#             # state size. (ngf*4) x 8 x 8
#             nn.ConvTranspose2d(cf.ngf * 4, cf.ngf * 2, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(cf.ngf * 2),
#             nn.ReLU(True),
#             # state size. (ngf*2) x 16 x 16
#             nn.ConvTranspose2d(cf.ngf * 2, cf.ngf, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(cf.ngf),
#             nn.ReLU(True),
#             # state size. (ngf) x 32 x 32
#             nn.ConvTranspose2d(cf.ngf, cf.nc, 3, 1, 1, bias=False),
#             nn.Tanh()
#             # state size. (nc) x 240 x 240
#         )

#     def forward(self, input):
#         return self.main(input)
    
    
# Discriminator code    

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is 155 x 240 x 240
            nn.Conv2d(cf.nc, cf.ndf, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),
            # state size. (ndf) x 120 x 120
            nn.Conv2d(cf.ndf, cf.ndf * 2, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.BatchNorm2d(cf.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),
            # state size. (ndf*2) x 60 x 60
            nn.Conv2d(cf.ndf * 2, cf.ndf * 4, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.BatchNorm2d(cf.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),
            # state size. (ndf*4) x 30 x 30
            nn.Conv2d(cf.ndf * 4, cf.ndf * 8, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.BatchNorm2d(cf.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),
            # state size. (ndf*8) x 15 x 15
            nn.Flatten(),
            nn.Linear(cf.ndf * 8 * 15 * 15, 1, bias=False),
        )

    def forward(self, input):
        return self.main(input)