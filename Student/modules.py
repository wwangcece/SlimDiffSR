import math
from typing import Tuple
import warnings
from einops import rearrange
import torch.nn as nn
import torch
import torch.nn.functional as F
from skimage.segmentation import slic
from skimage.util import img_as_float


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.relu(x)
        out = self.conv2(x)
        return identity + out * self.res_scale


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


# ConvBlock
class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        zero_conv=False,
    ):
        super(ConvBlock, self).__init__()
        if zero_conv:
            self.conv_in = zero_module(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
        else:
            self.conv_in = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        self.acti = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        out = self.conv_in(x)
        out = self.acti(out)
        return out


# MlpBlock
class MlpBlock(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, hidden_channels=32, num_layers=5):
        super(MlpBlock, self).__init__()
        fuse_mlp = [nn.Linear(in_channels, hidden_channels)]
        for _ in range(num_layers):
            fuse_mlp.append(nn.ReLU(True))
            fuse_mlp.append(nn.Linear(hidden_channels, hidden_channels))
        fuse_mlp.append(nn.Linear(hidden_channels, out_channels))
        fuse_mlp.append(nn.Sigmoid())
        self.fuse_mlp = nn.Sequential(*fuse_mlp)

    def forward(self, x):
        # [B, in_channels] -> [B, 1]
        return self.fuse_mlp(x)


class TimeMapping(nn.Module):

    def __init__(self, in_channels=4, out_channels=1, lower_limit=10, upper_limit=200):
        super(TimeMapping, self).__init__()
        noise_esti = [ConvBlock(in_channels, 32)]
        for _ in range(6):
            noise_esti.append(ResidualBlockNoBN(32))
        self.noise_esti = nn.Sequential(*noise_esti)
        self.time_predict = MlpBlock(64, out_channels, 512, 4)
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

    def forward(self, x):
        # [B, C, H, W] -> [B, out], whose value follows [lower_limit, upper_limit]
        noise_embed = self.noise_esti(x)
        embed_mean = torch.mean(noise_embed, dim=[-1, -2])
        embed_std = torch.std(noise_embed, dim=[-1, -2])
        noise_embed = torch.cat([embed_mean, embed_std], dim=1)
        noise_embed = self.time_predict(noise_embed)
        if noise_embed.shape[-1] > 1:
            time_embed = (
                noise_embed[:, 0] * (self.upper_limit - self.lower_limit)
                + self.lower_limit
            )
            sqrt_alpha_hat = noise_embed[:, 1] * 0.5 + 0.5
            return time_embed, sqrt_alpha_hat
        else:
            time_embed = (
                noise_embed[:, 0] * (self.upper_limit - self.lower_limit)
                + self.lower_limit
            )
            return time_embed


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [low, up], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * low - 1, 2 * up - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution.

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py

    The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class QGAM(nn.Module):
    def __init__(self, dim, num_queries, squeeze_dim=64):
        super(QGAM, self).__init__()

        # (1) Learnable queries
        self.num_queries = num_queries
        self.query = nn.Parameter(torch.randn(num_queries, dim))  # N learnable queries

        # (2) Linear projections for Q, K, V
        self.q_proj = nn.Linear(
            dim, squeeze_dim, bias=False
        )  # Project input features to Q
        self.k_proj = nn.Linear(
            dim, squeeze_dim, bias=False
        )  # Project input features to K
        self.v_proj = nn.Linear(
            dim, squeeze_dim, bias=False
        )  # Project input features to V

        self.q_proj_c = nn.Linear(
            dim, squeeze_dim, bias=False
        )  # Project input features to Q
        self.k_proj_c = nn.Linear(
            squeeze_dim, squeeze_dim, bias=False
        )  # Project input features to K
        self.v_proj_c = nn.Linear(
            squeeze_dim, squeeze_dim, bias=False
        )  # Project input features to V

        self.out_proj = nn.Linear(squeeze_dim, dim)  # Output projection

        self.scale = dim**-0.5  # Scaling factor for attention
        self.last_attn_weights = None
        self.last_attn_weights2 = None

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        """
        Args:
            hidden_states: input tensor with shape (batch_size, seq_len, dim)
        """
        batch_size, seq_len, dim = hidden_states.size()

        # (4) First cross-attention: query with the input features
        # Project input features to Q, K, V
        q = self.q_proj(self.query)  # (N, dim) learnable query, no batch size
        k = self.k_proj(hidden_states)  # (batch_size, seq_len, dim)
        v = self.v_proj(hidden_states)  # (batch_size, seq_len, dim)

        # Reshape queries for broadcasting (N, batch_size, seq_len)
        q = q.unsqueeze(1).expand(-1, batch_size, -1)  # (N, batch_size, dim)

        # Compute attention scores: (N, batch_size, seq_len) × (batch_size, seq_len, dim)
        attn_scores = (
            torch.bmm(q.transpose(0, 1), k.transpose(1, 2)) * self.scale
        )  # (batch_size, N, seq_len)

        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch_size, N, seq_len)
        self.last_attn_weights = attn_weights.detach()

        # Attention weighted sum of values
        context = torch.bmm(attn_weights, v)  # (batch_size, N, dim)

        # (5) Second cross-attention: project context to K, V and reapply attention
        q2 = self.q_proj_c(hidden_states)  # (batch_size, N, dim)
        k2 = self.k_proj_c(context)  # (batch_size, N, dim)
        v2 = self.v_proj_c(context)  # (batch_size, N, dim)

        # Compute attention scores: (batch_size, N, dim) × (batch_size, N, dim)
        attn_scores2 = (
            torch.bmm(q2, k2.transpose(1, 2)) * self.scale
        )  # (batch_size, N, N)

        # Apply softmax to get attention weights
        attn_weights2 = F.softmax(attn_scores2, dim=-1)  # (batch_size, N, N)
        self.last_attn_weights2 = attn_weights2.detach()

        # Attention weighted sum of values
        output = torch.bmm(attn_weights2, v2)  # (batch_size, N, dim)

        # Project to the original feature space
        output = self.out_proj(output)  # (batch_size, N, dim)

        return output


class DirectionalConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        distillation_rate: float = 0.25,
    ):
        super(DirectionalConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.distillation_rate = distillation_rate

        # （2）横向3x1、纵向1x3可分离卷积
        self.h_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=(3, 1),
                padding=(1, 0),
                groups=in_channels,
                bias=False,
            ),
        )
        self.v_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=(1, 3),
                padding=(0, 1),
                groups=in_channels,
                bias=False,
            ),
        )

        # （2）瓶颈层：C → rC → C
        mid_channels = int(in_channels * distillation_rate)
        self.conv_reduce = nn.Conv2d(
            in_channels, mid_channels, kernel_size=1, bias=False
        )
        self.conv_expand = nn.Conv2d(
            mid_channels, out_channels, kernel_size=1, bias=False
        )

        # （6）全局池化 + 小型MLP（输出1x3）
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // 8, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 8, 3, bias=True),
            nn.Sigmoid(),  # 将权重归一化到[0,1]
        )

    def forward(self, F):
        # （3）横向卷积
        Fh = self.h_conv(F)
        # （4）横向结果再纵向卷积
        Fhv = self.v_conv(Fh)
        # （5）直接纵向卷积
        Fv = self.v_conv(F)

        # （6）全局池化 -> MLP -> 1x3 权重
        g = self.global_pool(F).view(F.size(0), -1)  # [B, C]
        weights = self.mlp(g)  # [B, 3]
        weights = weights.unsqueeze(-1).unsqueeze(-1)  # [B, 3, 1, 1]

        # （7）加权平均融合
        F_combined = weights[:, 0:1] * Fh + weights[:, 1:2] * Fhv + weights[:, 2:3] * Fv

        # （8）瓶颈1x1卷积输出
        out = self.conv_expand(self.conv_reduce(F_combined))
        return out


class FrequencySeparationConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        distillation_rate: float = 0.25,
    ):
        super(FrequencySeparationConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        mid_channels = int(in_channels * distillation_rate)

        # ------------------------------
        # 1️⃣ 低频支路（保持通道数不变）
        # ------------------------------
        self.low_branch = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
        )

        # ------------------------------
        # 2️⃣ 高频支路（保持通道数不变）
        # ------------------------------
        self.high_branch = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
        )

        # ------------------------------
        # 3️⃣ 自适应高低频融合
        # ------------------------------
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 8, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(8, 2, bias=True),
            nn.Sigmoid(),
        )

        # ------------------------------
        # 4️⃣ 瓶颈层：融合后再降维
        # ------------------------------
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
        )

    def forward(self, F):
        # 分支输出
        F_low = self.low_branch(F)
        F_high = self.high_branch(F - F_low)  # 高频支路建模残差信息

        # 自适应加权融合
        g = self.global_pool(F).view(F.size(0), -1)
        weights = self.mlp(g).unsqueeze(-1).unsqueeze(-1)
        w_low, w_high = weights[:, 0:1], weights[:, 1:2]

        F_fused = w_low * F_low + w_high * F_high

        # 瓶颈降维输出
        out = self.bottleneck(F_fused)
        return out
