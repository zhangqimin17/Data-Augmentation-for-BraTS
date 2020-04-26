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
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init



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
    
'''    
 two-layer residual unit: two conv with BN/relu and identity mapping
'''
class residualUnit(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3,stride=1, padding=1, activation=F.relu):
        super(residualUnit, self).__init__()
        self.conv1 = nn.Conv3d(in_size, out_size, kernel_size, stride=1, padding=1)
        init.xavier_uniform(self.conv1.weight, gain = np.sqrt(2.0)) #or gain=1
        init.constant(self.conv1.bias, 0)
        self.conv2 = nn.Conv3d(out_size, out_size, kernel_size, stride=1, padding=1)
        init.xavier_uniform(self.conv2.weight, gain = np.sqrt(2.0)) #or gain=1
        init.constant(self.conv2.bias, 0)
        self.activation = activation
        self.bn1 = nn.BatchNorm3d(out_size)
        self.bn2 = nn.BatchNorm3d(out_size)
        self.in_size = in_size
        self.out_size = out_size
        if in_size != out_size:
            self.convX = nn.Conv3d(in_size, out_size, kernel_size=1, stride=1, padding=0)
            self.bnX = nn.BatchNorm3d(out_size)

    def forward(self, x):
        out1 = self.activation(self.bn1(self.conv1(x)))
        out2 = self.activation(self.bn1(self.conv2(out1)))
        if self.in_size!=self.out_size:
            bridge = self.activation(self.bnX(self.convX(x)))
        output = torch.add(out2, bridge)

        return output
    
'''
    Ordinary UNet Conv Block
'''
class UNetConBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConBlock, self).__init__()
        self.conv = nn.Conv3d(in_size, out_size, kernel_size, stride=1, padding=1)
        self.bn = nn.BatchNorm3d(out_size)
        self.conv2 = nn.Conv3d(out_size, out_size, kernel_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_size)
        self.activation = activation


        init.xavier_uniform(self.conv.weight, gain = np.sqrt(2.0))
        init.constant(self.conv.bias,0)
        init.xavier_uniform(self.conv2.weight, gain = np.sqrt(2.0))
        init.constant(self.conv2.bias,0)
    def forward(self, x):
        out = self.activation(self.bn(self.conv(x)))
        out = self.activation(self.bn2(self.conv2(out)))

        return out

'''
    Ordinary Residual UNet-Up Conv Block
'''
class UNetUpResBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu, space_dropout=False):
        super(UNetUpResBlock, self).__init__()
        self.up = nn.ConvTranspose3d(in_size, out_size, 2, stride=2)
        self.bnup = nn.BatchNorm3d(out_size)

        init.xavier_uniform(self.up.weight, gain = np.sqrt(2.0))
        init.constant(self.up.bias,0)

        self.activation = activation

        self.resUnit = residualUnit(in_size, out_size, kernel_size = kernel_size)

    def center_crop(self, layer, target_size):
        batch_size, n_channels, layer_width, layer_height, layer_depth = layer.size()
        xy1 = (layer_width - target_size) // 2
        return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size), xy1:(xy1 + target_size)]

    def forward(self, x, bridge):
        #print 'x.shape: ',x.shape
        up = self.activation(self.bnup(self.up(x)))
        #crop1 = self.center_crop(bridge, up.size()[2])
        #print 'up.shape: ',up.shape, ' crop1.shape: ',crop1.shape
        crop1 = bridge
        out = torch.cat([up, crop1], 1)

        out = self.resUnit(out)
        # out = self.activation(self.bn2(self.conv2(out)))

        return out
    
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
    

class Generator2(nn.Module):
    def __init__(self, ngpu, in_channel:int=1, channel:int=64, out_channel:int=1):
        super(Generator2, self).__init__()
        self.ngpu = ngpu
        _c = channel

        self.relu = nn.ReLU()
        self.in_channel = in_channel
        self.tp_conv1 = nn.ConvTranspose3d(in_channel, _c*8, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(_c*8)
        
        self.tp_conv2 = nn.Conv3d(_c*8, _c*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(_c*4)
        
        self.tp_conv3 = nn.Conv3d(_c*4, _c*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(_c*2)
        
        self.tp_conv4 = nn.Conv3d(_c*2, _c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm3d(_c)
        
        self.tp_conv5 = nn.Conv3d(_c, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        
    def forward(self, x):

        #noise = noise.view(-1,self.noise,1,1,1)
        h = self.tp_conv1(x)
        h = self.relu(self.bn1(h))
        
        h = F.upsample(h,scale_factor = 2)
        h = self.tp_conv2(h)
        h = self.relu(self.bn2(h))
     
        h = F.upsample(h,scale_factor = 2)
        h = self.tp_conv3(h)
        h = self.relu(self.bn3(h))

        h = F.upsample(h,scale_factor = 2)
        h = self.tp_conv4(h)
        h = self.relu(self.bn4(h))

        h = F.upsample(h,scale_factor = 2)
        h = self.tp_conv5(h)

        h = F.tanh(h)

        return h
    
'''
    ResUNet (lateral connection) with long-skip residual connection (from 1st to last layer)
'''


class ResUNet_LRes(nn.Module):
    def __init__(self, ngpu, in_channel=310, n_classes=155, dp_prob=0):
        super(ResUNet_LRes, self).__init__()
        self.ngpu = ngpu
        #         self.imsize = imsize

        self.activation = F.relu

        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)
        self.pool3 = nn.MaxPool3d(2)
        # self.pool4 = nn.MaxPool3d(2)

        self.conv_block1_64 = UNetConBlock(in_channel, 32)
        self.conv_block64_128 = residualUnit(32, 64)
        self.conv_block128_256 = residualUnit(64, 128)
        self.conv_block256_512 = residualUnit(128, 256)
        # self.conv_block512_1024 = residualUnit(512, 1024)
        # this kind of symmetric design is awesome, it automatically solves the number of channels during upsamping
        # self.up_block1024_512 = UNetUpResBlock(1024, 512)
        self.up_block512_256 = UNetUpResBlock(256, 128)
        self.up_block256_128 = UNetUpResBlock(128, 64)
        self.up_block128_64 = UNetUpResBlock(64, 32)
        self.Dropout = nn.Dropout3d(p=dp_prob)
        self.last = nn.Conv3d(32, n_classes, 1, stride=1)

    def forward(self, x, res_x):
        #         print 'line 70 ',x.size()
        block1 = self.conv_block1_64(x)
        # print 'block1.shape: ', block1.shape
        pool1 = self.pool1(block1)
        # print 'pool1.shape: ', block1.shape
        pool1_dp = self.Dropout(pool1)
        # print 'pool1_dp.shape: ', pool1_dp.shape
        block2 = self.conv_block64_128(pool1_dp)
        pool2 = self.pool2(block2)

        pool2_dp = self.Dropout(pool2)

        block3 = self.conv_block128_256(pool2_dp)
        pool3 = self.pool3(block3)

        pool3_dp = self.Dropout(pool3)

        block4 = self.conv_block256_512(pool3_dp)
        # pool4 = self.pool4(block4)
        #
        # pool4_dp = self.Dropout(pool4)
        #
        # # block5 = self.conv_block512_1024(pool4_dp)
        #
        # up1 = self.up_block1024_512(block5, block4)

        up2 = self.up_block512_256(block4, block3)

        up3 = self.up_block256_128(up2, block2)

        up4 = self.up_block128_64(up3, block1)

        last = self.last(up4)
        # print 'res_x.shape is ',res_x.shape,' and last.shape is ',last.shape
        if len(res_x.shape) == 3:
            res_x = res_x.unsqueeze(1)
        out = torch.add(last, res_x)

        # print 'out.shape is ',out.shape
        return out




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
    
class Discriminator2(nn.Module):
    def __init__(self, ngpu, channel=512,out_class=1,is_dis =True):
        super(Discriminator2, self).__init__()
        self.ngpu = ngpu
        self.is_dis=is_dis
        self.channel = channel
        n_class = out_class 
        
        self.conv1 = nn.Conv3d(155, channel//8, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(channel//8, channel//4, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(channel//4)
        self.conv3 = nn.Conv3d(channel//4, channel//2, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(channel//2)
        self.conv4 = nn.Conv3d(channel//2, channel, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(channel)
        self.conv5 = nn.Conv3d(channel, n_class, kernel_size=4, stride=1, padding=0)
        
    def forward(self, x, _return_activations=False):
        h1 = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        h2 = F.leaky_relu(self.bn2(self.conv2(h1)), negative_slope=0.2)
        h3 = F.leaky_relu(self.bn3(self.conv3(h2)), negative_slope=0.2)
        h4 = F.leaky_relu(self.bn4(self.conv4(h3)), negative_slope=0.2)
        h5 = self.conv5(h4)
        flatten = nn.Flatten()
        #h5 = flatten(h5)
        linear = nn.Linear(cf.ndf * 8 * 15 * 15, 1, bias=False)
        output = h5
        
        return output