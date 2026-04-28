import torch, types, copy
from torch import nn
import torch.nn.functional as F
from diffusers.models.unets.unet_2d_blocks import (
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    DownBlock2D,
    UpBlock2D,
    UNetMidBlock2DCrossAttn,
)
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.transformers.transformer_2d import Transformer2DModel
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.downsampling import Downsample2D
from diffusers.models.upsampling import Upsample2D
from forward import (
    MyUNet2DConditionModel_SD_forward,
    MyCrossAttnDownBlock2D_SD_forward,
    MyDownBlock2D_SD_forward,
    MyUNetMidBlock2DCrossAttn_SD_forward,
    MyCrossAttnUpBlock2D_SD_forward,
    MyUpBlock2D_SD_forward,
    MyResnetBlock2D_SD_forward,
    MyTransformer2DModel_SD_forward,
)


def find_parent(model, module_name):
    components = module_name.split(".")
    parent = model
    for comp in components[:-1]:
        parent = getattr(parent, comp)
    return parent, components[-1]


def halve_channels(model):
    for name, module in model.named_modules():
        if hasattr(module, "pruned"):
            continue
        if isinstance(module, nn.Conv2d):
            in_channels = int(module.in_channels * 0.75)
            out_channels = int(module.out_channels * 0.75)
            groups = 1 if module.groups == 1 else int(module.groups * 0.75)
            new_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=groups,
                bias=module.bias is not None,
            )
            with torch.no_grad():
                new_conv.weight.copy_(module.weight[:out_channels, :in_channels])
                if module.bias is not None:
                    new_conv.bias.copy_(module.bias[:out_channels])
            parent, last_name = find_parent(model, name)
            setattr(parent, last_name, new_conv)
            new_conv.pruned = True
        elif isinstance(module, nn.Linear):
            in_features = int(module.in_features * 0.75)
            out_features = int(module.out_features * 0.75)
            new_linear = nn.Linear(
                in_features=in_features,
                out_features=out_features,
                bias=module.bias is not None,
            )
            with torch.no_grad():
                new_linear.weight.copy_(module.weight[:out_features, :in_features])
                if module.bias is not None:
                    new_linear.bias.copy_(module.bias[:out_features])
            parent, last_name = find_parent(model, name)
            setattr(parent, last_name, new_linear)
            new_linear.pruned = True
        elif isinstance(module, nn.GroupNorm):
            num_channels = int(module.num_channels * 0.75)
            for num_groups in [32, 24, 16, 12, 8, 6, 4, 2, 1]:
                if num_channels % num_groups == 0:
                    break
            new_gn = nn.GroupNorm(
                num_groups=num_groups,
                num_channels=num_channels,
                eps=module.eps,
                affine=module.affine,
            )
            with torch.no_grad():
                new_gn.weight.copy_(module.weight[:num_channels])
                new_gn.bias.copy_(module.bias[:num_channels])
            parent, last_name = find_parent(model, name)
            setattr(parent, last_name, new_gn)
            new_gn.pruned = True
        elif isinstance(module, nn.LayerNorm):
            normalized_shape = int(module.normalized_shape[0] * 0.75)
            new_ln = nn.LayerNorm(
                normalized_shape,
                eps=module.eps,
                elementwise_affine=module.elementwise_affine,
            )
            with torch.no_grad():
                new_ln.weight.copy_(module.weight[:normalized_shape])
                new_ln.bias.copy_(module.bias[:normalized_shape])
            parent, last_name = find_parent(model, name)
            setattr(parent, last_name, new_ln)
            new_ln.pruned = True
        elif isinstance(module, Downsample2D) or isinstance(module, Upsample2D):
            module.channels = int(module.channels * 0.75)


class Net(nn.Module):
    def __init__(self, unet, decoder):
        super().__init__()
        del unet.time_embedding
        new_conv_in = nn.Conv2d(16, 320, 3, padding=1)
        new_conv_in.weight.data = unet.conv_in.weight.data.repeat(1, 4, 1, 1)
        new_conv_in.bias.data = unet.conv_in.bias.data
        unet.conv_in = new_conv_in
        new_conv_out = nn.Conv2d(320, 342, 3, padding=1)
        new_conv_out.weight.data = unet.conv_out.weight.data.repeat(86, 1, 1, 1)[:342]
        new_conv_out.bias.data = unet.conv_out.bias.data.repeat(
            86,
        )[:342]
        unet.conv_out = new_conv_out

        def ResnetBlock2D_remove_time_emb_proj(module):
            if isinstance(module, ResnetBlock2D):
                del module.time_emb_proj

        unet.apply(ResnetBlock2D_remove_time_emb_proj)

        def BasicTransformerBlock_remove_cross_attn(module):
            if isinstance(module, BasicTransformerBlock):
                del module.attn2, module.norm2

        unet.apply(BasicTransformerBlock_remove_cross_attn)

        def set_inplace_to_true(module):
            if isinstance(module, nn.Dropout) or isinstance(module, nn.SiLU):
                module.inplace = True

        unet.apply(set_inplace_to_true)

        def replace_forward_methods(module):
            if isinstance(module, CrossAttnDownBlock2D):
                module.forward = types.MethodType(
                    MyCrossAttnDownBlock2D_SD_forward, module
                )
            elif isinstance(module, DownBlock2D):
                module.forward = types.MethodType(MyDownBlock2D_SD_forward, module)
            elif isinstance(module, UNetMidBlock2DCrossAttn):
                module.forward = types.MethodType(
                    MyUNetMidBlock2DCrossAttn_SD_forward, module
                )
            elif isinstance(module, UpBlock2D):
                module.forward = types.MethodType(MyUpBlock2D_SD_forward, module)
            elif isinstance(module, CrossAttnUpBlock2D):
                module.forward = types.MethodType(
                    MyCrossAttnUpBlock2D_SD_forward, module
                )
            elif isinstance(module, ResnetBlock2D):
                module.forward = types.MethodType(MyResnetBlock2D_SD_forward, module)
            elif isinstance(module, Transformer2DModel):
                module.forward = types.MethodType(
                    MyTransformer2DModel_SD_forward, module
                )

        unet.apply(replace_forward_methods)
        unet.forward = types.MethodType(MyUNet2DConditionModel_SD_forward, unet)
        halve_channels(unet)
        unet.body = nn.Sequential(
            *unet.down_blocks,
            unet.mid_block,
            *unet.up_blocks,
            unet.conv_norm_out,
            unet.conv_act,
            unet.conv_out,
        )
        del (
            decoder.conv_in,
            decoder.up_blocks,
            decoder.conv_norm_out,
            decoder.conv_act,
            decoder.conv_out,
        )
        self.body = nn.Sequential(
            nn.PixelUnshuffle(2),
            unet,
            decoder.mid_block,
        )

    def forward(self, x):
        return self.body(x)
