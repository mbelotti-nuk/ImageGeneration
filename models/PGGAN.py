import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F 

class GeneratorPGGAN(nn.Module):
    def __init__(self, latent_dim=512, out_scale=256):
        super().__init__()

        self.depth = 1
        self.alpha = 1
        self.step_alpha = 0

        self.upscale_blocks = nn.ModuleList()
        self.to_rgb = nn.ModuleList()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # add first block to get 4x4 image
        self.upscale_blocks.append(GblockPGGAN(in_channels=latent_dim, out_channels=latent_dim, first_block=True))
        self.to_rgb.append( EqualizedConv2d(in_channels=latent_dim, out_channels=3, kernel_size=1, padding=0), )

        start_scale = np.log2(4)
        end_scale = np.log2(out_scale)

        start_scale = int(np.log2(4))
        end_scale   = int(np.log2(out_scale))

        in_ch, out_ch = latent_dim, latent_dim
        for i in range(start_scale+1, end_scale+1):
            in_ch = out_ch
            if i >= 6:
                out_ch = in_ch//2
            else:
                out_ch = in_ch
            self.upscale_blocks.append(GblockPGGAN(in_channels=in_ch, out_channels=out_ch, first_block=False))
            self.to_rgb.append( EqualizedConv2d(in_channels=out_ch, out_channels=3, kernel_size=1, padding=0), )
    
    def forward(self, x):
        # go until second last layer
        for block in self.upscale_blocks[:self.depth-1]:
            x = block(x)
        # compute last layer
        x_rgb = self.upscale_blocks[self.depth-1](x)
        x_rgb = self.to_rgb[self.depth-1](x_rgb)
        # smooth change
        if self.alpha < 1 and self.depth > 1: # for the first layer cannot be applied
            x_old_rgb = self.to_rgb[self.depth-2](x)
            x_old_rgb = self.upsample(x_old_rgb)
            
            x_rgb = self.alpha * x_rgb + (1-self.alpha) * x_old_rgb
        return x_rgb
    
    def increase_net(self, n_iterations):
        self.step_alpha = 1/n_iterations
        self.alpha = 1/n_iterations
        self.depth += 1
    
    def step(self):
        # increase alpha
        self.alpha += self.step_alpha
    

class DiscriminatorPGGAN(nn.Module):
    def __init__(self, latent_dim=512, out_scale=256):
        super().__init__()

        self.depth = 1
        self.alpha = 1
        self.step_alpha = 0

        self.downscale_blocks = nn.ModuleList()
        self.from_rgb = nn.ModuleList()
        self.downscale = nn.AvgPool2d(kernel_size=2, stride=2)
        self.from_rgb.append( nn.Sequential( EqualizedConv2d(in_channels=3, out_channels=latent_dim, kernel_size=1), nn.LeakyReLU(0.2) ) )
        # 4x4 -> 1x1
        self.downscale_blocks.append(DblockPGGAN(in_channels=latent_dim, out_channels=1, last_block=True))

        start_scale = int(np.log2(4))
        end_scale   = int(np.log2(out_scale))

        out_ch, in_ch = latent_dim, latent_dim
        for i in range(start_scale+1, end_scale+1):
            out_ch = in_ch
            if i >= 6:
                in_ch = out_ch//2
            else:
                in_ch = out_ch
            self.from_rgb.append( nn.Sequential( EqualizedConv2d(in_channels=3, out_channels=in_ch, kernel_size=1), nn.LeakyReLU(0.2) ) )
            self.downscale_blocks.append(DblockPGGAN(in_channels=in_ch, out_channels=out_ch, last_block=False))

    def forward(self,x_rgb):
        x = self.from_rgb[self.depth-1](x_rgb)
        x = self.downscale_blocks[self.depth-1](x)

        if self.alpha < 1.0 and self.depth > 1:
            x_rgb = self.downscale(x_rgb)
            x_old = self.from_rgb[self.depth-2](x_rgb)
            x = self.alpha * x + (1-self.alpha) * x_old 

        for block in reversed(self.downscale_blocks[:self.depth-1]):
            x = block(x)
        return x       

    def increase_net(self, n_iterations):
        self.step_alpha = 1/n_iterations
        self.alpha = 1/n_iterations
        self.depth += 1

    def step(self):
        # increase alpha
        self.alpha += self.step_alpha


class GblockPGGAN(nn.Module):
    def __init__(self, in_channels, out_channels, first_block=False):
        super().__init__()
        if first_block:
            self.block = nn.Sequential(
                    EqualizedConv2d(in_channels, out_channels, kernel_size=4, padding=3),
                    nn.LeakyReLU(0.2),
                    PixelWiseNorm(),
                    EqualizedConv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2),
                    PixelWiseNorm(),
                    )
        else:
            self.block = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    EqualizedConv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2),
                    PixelWiseNorm(),
                    EqualizedConv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2),
                    PixelWiseNorm(),
                    )
    def forward(self, x):
        return self.block(x)           



class DblockPGGAN(nn.Module):
    def __init__(self, in_channels, out_channels, last_block=False):
        super().__init__()
        if last_block:
            self.block = nn.Sequential(
                    MinibatchStd(),
                    EqualizedConv2d(in_channels+1, out_channels, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2),
                    EqualizedConv2d(out_channels, out_channels, kernel_size=4, padding=0),
                    nn.LeakyReLU(0.2),
                    nn.Sequential(nn.Flatten(), nn.Linear(out_channels, 1))
                    )
        else:
            self.block = nn.Sequential(
                    EqualizedConv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2),
                    EqualizedConv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2),
                    nn.AvgPool2d(kernel_size=2, stride=2) # down sampling
                    )
            
    def forward(self, x):
        return self.block(x)         



class EqualizedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=True):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True)

        # initialize weights to N(0,1)
        self.conv.weight.data.normal_(0,1)
        # set biases to zero
        self.conv.bias.data.fill_(0)
    
    def forward(self,x):
        x = self.conv(x)*self.get_scale()
        return x
    
    def get_scale(self):
        size_w = self.conv.weight.size()
        fan_in = size_w[1:].numel()
        return np.sqrt(2/fan_in)



class PixelWiseNorm(nn.Module):
    def __init__(self,):
        super().__init__()
        self.epsilon = 1E-8

    def forward(self, x):
        norm = torch.sqrt( torch.mean(x**2, dim=1, keepdim=True) + self.epsilon ) 
        return x/norm
    
class MinibatchStd(nn.Module):
	def __init__(self):
		super().__init__()
	def forward(self, x):
		size = list(x.size())
		size[1] = 1
		
		std = torch.std(x, dim=0)
		mean = torch.mean(std)
		return torch.cat((x, mean.repeat(size)),dim=1)