import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch import optim
import math 
from torch.autograd import Variable
import einops
from einops import rearrange, repeat 
from einops.layers.torch import Rearrange, Reduce
from torch import nn, einsum
from monai.networks.blocks.dynunet_block import get_conv_layer
from monai.networks.blocks import UnetOutBlock




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
    
def MBConv(
    dim_in,
    dim_out,
    *,
    expansion_rate = 2,
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 1

    net = nn.Sequential(
        nn.Conv3d(dim_in, hidden_dim, 1),
        #nn.GroupNorm(num_groups=8, num_channels=hidden_dim),
        nn.InstanceNorm3d(hidden_dim, affine = True),
        nn.GELU(),
        nn.Conv3d(hidden_dim, hidden_dim, 3, stride = stride, padding = 1, groups = hidden_dim),
        #nn.GroupNorm(num_groups=8, num_channels=hidden_dim),
        nn.InstanceNorm3d(hidden_dim, affine = True),
        nn.GELU(),
        ProjectExciteLayer(hidden_dim),
        nn.Conv3d(hidden_dim, dim_out, 1),
        #nn.GroupNorm(num_groups=8, num_channels=dim_out)
        nn.InstanceNorm3d(dim_out, affine = True),
    )
    return net

class GLUMBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        expand_ratio=2
    ):
        super().__init__()

        mid_channels = round(in_channels * expand_ratio)

        self.act = nn.SiLU()
        self.norm = nn.InstanceNorm3d(out_channels, affine = True)#nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.Leaky = nn.LeakyReLU()
        self.inverted_conv = nn.Conv3d(in_channels,
                                       mid_channels,
                                       1,
                                       padding = 0,
                                       bias=True)
        self.depth_conv = nn.Conv3d(
            mid_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            padding=1,
            groups=mid_channels,
            bias=True,
        )
        self.point_conv = nn.Conv3d(
            mid_channels//2,
            out_channels,
            1,
            padding = 0,
            bias=False
        )
        self.Gnorm = nn.InstanceNorm3d(mid_channels, affine = True)
        self.PE = ProjectExciteLayer(num_channels = mid_channels//2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted_conv(x)
        x = self.Gnorm(x)
        x = self.Leaky(x)
        
        x = self.depth_conv(x)
        x = self.Gnorm(x)
        x = self.Leaky(x)
        x, gate = torch.chunk(x, 2, dim=1)
        gate = self.act(gate)
        x = x * gate
        x = self.PE(x)
        x = self.point_conv(x)
        x = self.norm(x)
        return x
class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class PreNormResidual2(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.InstanceNorm3d(dim, affine = True)#nn.GroupNorm(num_groups=8, num_channels=dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 3, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        dropout = 0.,
        window_size = (7,7,7)
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias = False),
            nn.Dropout(dropout)
        )
        # relative positional bias
        w1,w2,w3 = window_size
        # 初始化相对位置索引矩阵[2*H-1,2*W-1,2*D-1,num_heads]
        self.rel_pos_bias = nn.Embedding((2 * w1 - 1) *(2 * w2 - 1)*(2 * w3 - 1), self.heads)
        pos1 = torch.arange(w1)
        pos2 = torch.arange(w2)
        pos3 = torch.arange(w3)
        # 首先我们利用torch.arange和torch.meshgrid函数生成对应的坐标，[3,H,W,D] 然后堆叠起来，展开为一个二维向量，得到的是绝对位置索引。
        grid = torch.stack(torch.meshgrid(pos1, pos2, pos3, indexing = 'ij'))
        grid = rearrange(grid, 'c i j k -> (i j k) c')
        # 广播机制，分别在第一维，第二维，插入一个维度，进行广播相减，得到 3, whd*ww, whd*ww的张量
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...') 
        rel_pos[...,0] += w1 - 1
        rel_pos[...,1] += w2 - 1
        rel_pos[...,2] += w3 - 1
        # 做了个乘法操作，以进行区分,最后一维上进行求和，展开成一个一维坐标   a*x1 + b*x2 + c*x3  (a= hd b=d c =1) 
        rel_pos_indices = (rel_pos * torch.tensor([(2 *w2 - 1)*(2 *w3 - 1), (2 *w3 - 1), 1])).sum(dim = -1)
        # 注册为一个不参与网络学习的变量
        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)
               

    def forward(self, x):
        batch, height, width, depth, window_height, window_width, window_depth ,_, device, h = *x.shape, x.device, self.heads
        # flatten
        x = rearrange(x, 'b x y z w1 w2 w3 d -> (b x y z) (w1 w2 w3) d')
        # project for queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d ) -> b h n d', h = h), (q, k, v))
        # scale
        q = q * self.scale
        # sim
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        # add positional bias
        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')
        # attention
        attn = self.attend(sim)
        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # merge heads
        out = rearrange(out, 'b h (w1 w2 w3) d -> b w1 w2 w3 (h d)', w1 = window_height, w2 = window_width, w3 = window_depth)
        # combine heads out
        out = self.to_out(out)
        return rearrange(out, '(b x y z) ... -> b x y z ...', x = height, y = width, z = depth)
    
class MaxViT_Block(nn.Module):
    def __init__(
        self,
        *,
        dim = 512,
        dim_head = 32,
        window_size = (8,8,8),
        dropout = 0.1,
    ):
        super().__init__()
        w1,w2,w3 = window_size

        self.net = nn.Sequential(
            MBConv(dim, dim), # 1, 1, 1, 1, 8, 8, 8, 256
            Rearrange('b d (x w1) (y w2) (z w3) -> b x y z w1 w2 w3 d', w1 = w1, w2 = w2, w3 = w3),  # block-like attention -> [2, 1, 1, 1, 8, 8, 8, 256]
            PreNormResidual(dim, Attention(dim = dim, dim_head = dim_head, dropout = dropout, window_size = window_size)), #2, 1, 1, 1, 8, 8, 8, 256]
            PreNormResidual(dim, FeedForward(dim = dim, dropout = dropout)),
            Rearrange('b x y z w1 w2 w3 d -> b d (x w1) (y w2) (z w3)'),
            
            Rearrange('b d (x w1) (y w2) (z w3) -> b x y z w1 w2 w3 d', w1 = w1, w2 = w2, w3 = w3),  # block-like attention -> [2, 1, 1, 1, 8, 8, 8, 256]
            PreNormResidual(dim, Attention(dim = dim, dim_head = dim_head, dropout = dropout, window_size = window_size)), #2, 1, 1, 1, 8, 8, 8, 256]
            PreNormResidual(dim, FeedForward(dim = dim, dropout = dropout)),
            Rearrange('b x y z w1 w2 w3 d -> b d (x w1) (y w2) (z w3)'),
            )
    def forward(self, x):
        x = self.net(x)
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
        # up1 = self.Gnorm(up1)

        out = torch.cat((up1, skip), dim=1) #channel_x1 + channel_up1
        out = self.conv_block(out) #out_channel here 
        # out = self.Leaky(out)
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

    def forward(self, x1, x2): #x2 = the skip connection
        x1 = self.conv(self.upsample(x1))
        x1 = self.convLK_in(x1)#16 128 128 128
        x1 = self.convLK_out(x1)
        return x1

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, lka_size, year = 2021):
        super().__init__()
        self.loop = lka_size
        self.year = year
        self.encode1 = Downsampling(in_channels, out_channels, 
                                    kernel_size=3, stride=2, padding=1)
        self.MaxViT_Block = MaxViT_Block(dim = out_channels, #128                        # dimension of first layer, doubles every layer
                                         dim_head = 16,                    # dimension of attention heads, kept at 32 in paper
                                         window_size = (4,4,4),            # window size for block and grids 8 8 8 = out | 4 4 4 -> khong out
        )
    def forward(self, x):
        x = self.encode1(x)
        if self.year == 2021:
            if self.loop == 21:
                x1 = self.MaxViT_Block(x)
            elif self.loop == 15:
                x1 = self.MaxViT_Block(x)
                # x1 = self.MaxViT_Block(x1)
            else:
                x1 = self.MaxViT_Block(x)
                x1 = self.MaxViT_Block(x1)
                # x1 = self.MaxViT_Block(x1)
        else:
            if self.loop == 21:
                x1 = self.MaxViT_Block(x)
            elif self.loop == 15:
                x1 = self.MaxViT_Block(x)
                x1 = self.MaxViT_Block(x1)
            else:
                x1 = self.MaxViT_Block(x)
                x1 = self.MaxViT_Block(x1)
                x1 = self.MaxViT_Block(x1)

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