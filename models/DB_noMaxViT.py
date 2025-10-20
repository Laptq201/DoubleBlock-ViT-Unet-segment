import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch import optim
import math 
from torch.autograd import Variable
from torch import nn, einsum



class ProjectExciteLayer(nn.Module):
    """
        Project & Excite Module, specifically designed for 3D inputs
        *quote*
    """

    def __init__(self, num_channels, reduction_ratio=4):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ProjectExciteLayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.relu = nn.GELU()
        self.conv_c = nn.Conv3d(in_channels=num_channels, out_channels=num_channels_reduced, kernel_size=1, stride=1)
        self.conv_cT = nn.Conv3d(in_channels=num_channels_reduced, out_channels=num_channels, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()

        # Project:
        # Average along channels and different axes
        squeeze_tensor_w = F.adaptive_avg_pool3d(input_tensor, (1, 1, W))

        squeeze_tensor_h = F.adaptive_avg_pool3d(input_tensor, (1, H, 1))

        squeeze_tensor_d = F.adaptive_avg_pool3d(input_tensor, (D, 1, 1))

        # tile tensors to original size and add:
        final_squeeze_tensor = sum([squeeze_tensor_w.view(batch_size, num_channels, 1, 1, W),
                                    squeeze_tensor_h.view(batch_size, num_channels, 1, H, 1),
                                    squeeze_tensor_d.view(batch_size, num_channels, D, 1, 1)])

        # Excitation:
        final_squeeze_tensor = self.sigmoid(self.conv_cT(self.relu(self.conv_c(final_squeeze_tensor))))
        output_tensor = torch.mul(input_tensor, final_squeeze_tensor)

        return output_tensor

class DoubleConv(nn.Module):
    """(Conv3D -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels, num_groups=16, kernel_size=3, padding=1, stride=1, bias=True):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias),
            #nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.InstanceNorm3d(out_channels, affine = True),
            nn.LeakyReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias),
            nn.InstanceNorm3d(out_channels, affine = True),
            #nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.LeakyReLU()
        )
    
    def forward(self, x):
        return self.double_conv(x)
class Downsampling(nn.Module):
    def __init__(self, in_channels, out_channels, 
        kernel_size, stride=1, padding=0, 
        norm=None):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding)
        
        self.norm = nn.InstanceNorm3d(out_channels, affine = True)#nn.GroupNorm(num_groups=8, num_channels=out_channels)

    def forward(self, x): #downsample(conv) --> post norm
        x = self.conv(x)
        x = self.norm(x)
        return x   



class Decoder1(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_kernel_size, lka_size, kernel_size = 3, spatial_dims=3, test = True, i = 0):
        super().__init__()        

        upsample_stride = upsample_kernel_size
        self.upsample = nn.Upsample(scale_factor=upsample_kernel_size, mode='trilinear', align_corners=True)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.Gnorm = nn.InstanceNorm3d(out_channels, affine = True)#nn.GroupNorm(num_groups=8, num_channels=out_channels)

        self.conv_block = nn.Conv3d(out_channels + out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.point_wise_conv = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm3d(out_channels, affine = True),
            #nn.GroupNorm(num_groups=8, num_channels=out_channels)
        )

        self.depth_wise_conv = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels),
            nn.InstanceNorm3d(out_channels, affine = True),
            #nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.LeakyReLU(negative_slope=0.01 )
        )
        self.conv_2 = nn.Conv3d(in_channels=1, out_channels=2, kernel_size=1)
        self.sigmoid = nn.Sigmoid() 
        self.Leaky = nn.LeakyReLU(negative_slope=0.01)
        self.PE = ProjectExciteLayer(num_channels = out_channels)
    def forward(self, x1, x2): #x2 = the skip connection
        skip = x2
        up1 = self.upsample(x1) #same as skip
        up1 = self.conv(up1)
        up1 = self.Gnorm(up1)
        x2 = self.depth_wise_conv(x2)
        x2 = torch.amax(x2, dim=1, keepdim=True) # 2, 1, 16, 16, 16   
        x1 = torch.amax(up1, dim=1, keepdim=True) # 2, 1, 16, 16, 16
        x1 = x1 + x2 #2 1 16 16 16 bang elementwise addition
        x1 = self.conv_2(x1) #-> 2 2 16 16 16
        x1 = F.softmax(x1, dim = 1) # 2 2 16 16 16
        x1,x2 = x1.split(1, dim = 1) #each = 2 1 16 16 16
        skip_channels = skip.size(1)
        x2 = x2.repeat(1,skip_channels,1,1,1)*skip # 2 128 16 16 16
        x2 = x2 + skip
        x1 = x1.repeat(1,skip_channels,1,1,1)*up1 + up1

        x1 = x1 * self.sigmoid(x2)
        x2 = x2 * self.sigmoid(x1)
       

        x1 = x1+x2 #co the thu cac phep toan khac o day
        x1 = self.point_wise_conv(x1) #out
        x1 = self.sigmoid(x1)
        x1 = skip * x1
        x1 = self.point_wise_conv(x1)#out
        x1 = self.PE(x1)
        x1 = self.sigmoid(x1)
        x1 = up1 * x1
        out = torch.cat((x1, skip), dim=1) #channel_x1 + channel_up1
        out = self.conv_block(out) #out_channel here 
        out = self.Leaky(out)
        return out

class Up2(nn.Module):
    def __init__(self, in_channels, out_channels, trilinear=False, test = True):
        super().__init__()
        spatial_dims = 3
        #self.upT = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.convLK_in = nn.Sequential(
            nn.InstanceNorm3d(in_channels, affine = True),
            #nn.GroupNorm(num_groups=8, num_channels=in_channels),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv3d(in_channels, 2*out_channels, kernel_size=3, padding=1, stride=1, bias=False),
        )
        self.convLK_out = nn.Sequential(
            nn.InstanceNorm3d(2*out_channels, affine = True),#nn.GroupNorm(num_groups=8, num_channels=2*out_channels),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv3d(2*out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        )

        #self.channel_wise_maxpool = torch.amax(input_tensor, dim=1, keepdim=True)
    def forward(self, x1, x2): #x2 = the skip connection
        x1 = self.conv(self.upsample(x1))
        x1 = self.convLK_in(x1)#16 128 128 128
        x1 = self.convLK_out(x1)
        return x1

#Downsampling + ProjectExciteLayer + LayerNormGeneral + LKA
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, loop):
        super().__init__()
        self.loop = loop
        self.encode1 = Downsampling(in_channels, out_channels, 
                                    kernel_size=3, stride=2, padding=1)
        self.conv = DoubleConv(out_channels, out_channels, num_groups=8)
    def forward(self, x):
        x = self.encode1(x)
        x1 = self.conv(x)
        skip = x1
        return x1, skip


class Bneck(nn.Module):
    def __init__(self, in_channels, out_channels, trilinear=False, test = True):
        super().__init__()
        self.encode1 = Downsampling(in_channels, out_channels,kernel_size=3, stride=2, padding=1)
        self.net = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 1),
            nn.InstanceNorm3d(out_channels, affine = True),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, 3, stride = 1, padding = 1, groups = out_channels),
            nn.InstanceNorm3d(out_channels, affine = True),
            nn.GELU(),
        )

    def forward(self, x1): #x2 = the skip connection
        x1 = self.encode1(x1)
        x1 = self.net(x1)
        return x1
    
class Unet(nn.Module):
    def __init__(self, in_channels, n_channels, n_classes): 
        super().__init__()
        self.in_channels = in_channels #4
        self.n_classes = n_classes 
        self.n_channels = n_channels


        self.conv = DoubleConv(in_channels, 2*n_channels, num_groups=8)
        self.enc1 = Encoder(2*n_channels, 4*n_channels, lka_size = 21) #64
        self.enc2 = Encoder(4*n_channels,  8*n_channels, lka_size = 15) #128
        self.enc3 = Encoder(8*n_channels, 16*n_channels, lka_size = 10) #256

        self.bottleneck = Bneck(16*n_channels, 32*n_channels) #512

        
        self.dec1 = Decoder1(32*n_channels, 16*n_channels, upsample_kernel_size=2, lka_size = 10) #concat(256|8 -> 128|16 (u1),skip) Out: 256|16 
        self.dec2 = Decoder1(16*n_channels, 8*n_channels, upsample_kernel_size=2, lka_size = 15)  #concat(128|16 -> 64|32 (u2),skip) Out: 128|32
        self.dec3 = Decoder1(8*n_channels, 4*n_channels, upsample_kernel_size=2, lka_size = 21)   #concat(64|32 -> 32|64 (u3),skip) Out: 64|64
        
        
        self.dec4 = Up2(4*n_channels, n_channels)    #32|64 -> 16|128 (u4)
        self.out = nn.Conv3d(in_channels=n_channels, out_channels=n_classes, kernel_size=1)
        self.apply(self.initialize_weights)
    def forward(self, x):
        x = self.conv(x)
        #---layer 1
        x_1,skip_conv1 = self.enc1(x) #Embedding
        #---layer 2
        x_1,skip_conv2 = self.enc2(x_1)
        #---layer 3 
        x_1, skip_conv3 = self.enc3(x_1) 
        
        #---bottleneck 
        x_1 = self.bottleneck(x_1)

        #---layer 3 
        x_out = self.dec1(x_1, skip_conv3) 
        #---layer 2 
        x_out = self.dec2(x_out, skip_conv2) 
        #---layer 1 
        x_out = self.dec3(x_out, skip_conv1) 
        #---layer output
        x_out = self.dec4(x_out,x) # 16|128 (u4)
        out = self.out(x_out) #3|128
        return out
    def initialize_weights(self, module):
            name = module.__class__.__name__.lower()
            if name in ["conv2d", "conv3d"]:
                nn.init.kaiming_normal_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.constant_(module.bias, 0)



model = Unet(in_channels=4, n_classes=3, n_channels=16).to('cuda')
print('Number of network parameters:', sum(param.numel() for param in model.parameters()))