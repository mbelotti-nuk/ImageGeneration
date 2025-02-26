import torch
import torch.nn as nn
import torch.nn.functional as F
import math

EMBED_T = 256
NUM_GROUPS = 16


class DiffusionUNet(nn.Module):
    def __init__(self, in_channels, resolution, attn_resolutions, ch_mult, channels, time_steps=1000, device="mps"):
        super().__init__()
        self.device = device
        self.time_steps = time_steps
        self.schedule = LinearNoiseScheduler(time_steps, beta_start=0.1, beta_end=0.99)

        self.sinusoidal_embedding = SinusoidalEmbeddings(time_steps)
        self.in_conv = nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=3, stride=1, padding=1)

        self.d_blocks = nn.ModuleList()
        d_resolutions = []
        for c in ch_mult:
            in_ch, out_ch = channels, channels*c
            d_resolutions.append((in_ch, out_ch))
            attn = resolution in attn_resolutions
            self.d_blocks.append( DownBlock(in_ch, out_ch, attn) )
            channels = out_ch
            resolution = resolution // 2

        self.res_bone = nn.ModuleList([ResBlock(channels, channels), 
                                      AttnBlock(channels),
                                      ResBlock(channels,channels)])
        
        self.u_blocks = nn.ModuleList()
        for res in reversed(d_resolutions):
            out_ch, in_ch = res
            attn = resolution in attn_resolutions
            self.u_blocks.append(UpBlock(in_ch*2, out_ch, attn))
            resolution = resolution *2

        self.last_conv = nn.Conv2d(in_channels=out_ch, out_channels=in_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x, t):
        t = self.sinusoidal_embedding(x, t) 
        x = self.in_conv(x)
        skips = []
        for d_b in self.d_blocks:
            x, d_skip = d_b(x,t)
            skips.append(d_skip)
        for m_bone in self.res_bone:
            if isinstance(m_bone, ResBlock):
                x = m_bone(x,t)
            else:
                x = m_bone(x)

        for u_b in self.u_blocks:
            x = u_b(x, t, skips.pop())
        return self.last_conv(x)
    

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        num_images = initial_noise.shape[0]
        xt = initial_noise
        with torch.no_grad():
            for t in reversed( range(0, diffusion_steps) ):
                time = torch.ones((num_images,), dtype=torch.int32)*t
                noise_pred = self(xt.to(self.device), time).cpu()
                xt, x_0_pred = self.schedule.sample_prev_timestep(xt.cpu(), noise_pred, t)
        return xt

    
    

class LinearNoiseScheduler:
    r"""
    Class for the linear noise scheduler that is used in DDPM.
    """
    
    def __init__(self, num_timesteps, beta_start, beta_end):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        # Mimicking how compvis repo creates schedule
        self.betas = (
                torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps) ** 2
        )
        self.alphas = 1. - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)
    
    def add_noise(self, original, noise, t):
        r"""
        Forward method for diffusion
        :param original: Image on which noise is to be applied
        :param noise: Random Noise Tensor (from normal dist)
        :param t: timestep of the forward process of shape -> (B,)
        :return:
        """
        original_shape = original.shape
        batch_size = original_shape[0]
        
        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(original.device)[t].reshape(batch_size)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(original.device)[t].reshape(batch_size)
        
        # Reshape till (B,) becomes (B,1,1,1) if image is (B,C,H,W)
        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
        for _ in range(len(original_shape) - 1):
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)
        
        # Apply and Return Forward process equation
        return (sqrt_alpha_cum_prod.to(original.device) * original
                + sqrt_one_minus_alpha_cum_prod.to(original.device) * noise)
    
    def sample_prev_timestep(self, xt, noise_pred, t):
        r"""
            Use the noise prediction by model to get
            xt-1 using xt and the nosie predicted
        :param xt: current timestep sample
        :param noise_pred: model noise prediction
        :param t: current timestep we are at
        :return:
        """

        x0 = ((xt - (self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t] * noise_pred)) /
              torch.sqrt(self.alpha_cum_prod.to(xt.device)[t]))
        x0 = torch.clamp(x0, -1., 1.)
        
        mean = xt - ((self.betas.to(xt.device)[t]) * noise_pred) / (self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t])
        mean = mean / torch.sqrt(self.alphas.to(xt.device)[t])
        
        if t == 0:
            return mean, x0
        else:
            variance = (1 - self.alpha_cum_prod.to(xt.device)[t - 1]) / (1.0 - self.alpha_cum_prod.to(xt.device)[t])
            variance = variance * self.betas.to(xt.device)[t]
            sigma = variance ** 0.5
            z = torch.randn(xt.shape).to(xt.device)
            
            # OR
            # variance = self.betas[t]
            # sigma = variance ** 0.5
            # z = torch.randn(xt.shape).to(xt.device)
            return mean + sigma * z, x0


    

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, attention=False):
        super().__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode='nearest')
        self.res_block1 = ResBlock(in_channels, out_channels)
        self.res_block2 = ResBlock(in_channels//2+ out_channels, out_channels)
        self.attn = None
        if attention: self.attn = AttnBlock(out_channels)
    def forward(self, x, embeddings, skips):
        x = self.up_sample(x)
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.res_block1(x, embeddings)
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.res_block2(x, embeddings)
        if self.attn != None:
            x = self.attn(x)
        return x

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, attention=False):
        super().__init__()
        self.res_block1 = ResBlock(in_channels, out_channels)
        self.res_block2 = ResBlock(out_channels, out_channels)
        self.attn = None
        if attention: self.attn = AttnBlock(out_channels)
        self.avg_pool = nn.AvgPool2d(kernel_size=2)
        

    def forward(self, x, embeddings):
        skips = []
        x = self.res_block1(x, embeddings)
        skips.append(x)
        x = self.res_block2(x, embeddings)
        skips.append(x)
        if self.attn != None:
            x = self.attn(x)
        x = self.avg_pool(x)
        return x, skips 
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.res_conv = None
        if in_channels != out_channels:
            # this is used to make residuals out channels the same
            self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.gnorm = nn.GroupNorm(num_groups=NUM_GROUPS, num_channels=out_channels, eps=1e-6, affine=True)
        self.t_emb_proj = nn.Linear(EMBED_T, out_channels)
        self.silu = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, embeddings):
        residual = x
        if self.res_conv != None:
            residual = self.res_conv(x)
        x = self.conv1(x)
        x = self.gnorm(x)
        x = x + self.t_emb_proj(embeddings)[:, :, None, None]
        x = self.silu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x + residual
    

# class Conv2d(nn.Conv2d):
#     """
#     apply weight standardization  https://arxiv.org/abs/1903.10520
#     """ 
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1, bias=True):
#         super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
#                  padding, dilation, groups, bias)

#     def forward(self, x):
#         weight = self.weight
#         weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
#                                   keepdim=True).mean(dim=3, keepdim=True)
#         weight = weight - weight_mean
#         std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
#         weight = weight / std.expand_as(weight)
#         return F.conv2d(x, weight, self.bias, self.stride,
#                         self.padding, self.dilation, self.groups)


class SinusoidalEmbeddings(nn.Module):
    def __init__(self, time_steps:int, embed_dim=EMBED_T):
        super().__init__()
        position = torch.arange(time_steps).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        embeddings = torch.zeros(time_steps, embed_dim, requires_grad=False)
        embeddings[:, 0::2] = torch.sin(position * div)
        embeddings[:, 1::2] = torch.cos(position * div)
        self.embeddings = embeddings

    def forward(self, x, t):
        embeds = self.embeddings[t].to(x.device)
        return embeds[:, :]



class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.gnorm = nn.GroupNorm(num_groups=NUM_GROUPS, num_channels=in_channels, eps=1e-6, affine=True)
        self.Q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.K = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.V = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor):
        h_ = self.gnorm(x)
        query = self.Q(h_)
        key = self.K(h_)
        value = self.V(h_)

        # Get tensor dimensions
        b, c, h, w = query.shape
        hw = h * w  # Flatten height and width

        # Attention computation
        query = query[:, :, :, :].contiguous()  # (b, c, h, w)
        query = query.reshape(b, c, hw).swapaxes(1, 2)  # (b, hw, c)

        key = key[:, :, :, :].contiguous()  # (b, c, h, w)
        key = key.reshape(b, c, hw)  # (b, c, hw)

        w_ = torch.bmm(query, key) * (c ** -0.5)  # (b, hw, hw)
        w_ = F.softmax(w_, dim=-1)

        # Attend to values
        value = value[:, :, :, :].contiguous()  # (b, c, h, w)
        value = value.reshape(b, c, hw)  # (b, c, hw)

        h_ = torch.bmm(value, w_.swapaxes(1, 2))  # (b, c, hw)
        h_ = h_.reshape(b, c, h, w)  # Restore spatial dimensions

        h_ = self.proj_out(h_)
        return x + h_
