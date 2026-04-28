import torch
from typing import Any, Dict, List, Optional, Tuple, Union


def MyUNet2DConditionModel_SD_forward(self, x):
    global skip
    x = self.conv_in(x)
    skip = [x]
    body = torch.nn.Sequential(
        *self.down_blocks,
        self.mid_block,
        *self.up_blocks,
        self.conv_norm_out,
        self.conv_act,
        self.conv_out,
    )
    x = body(x)
    return x


def MyUNet2DConditionModel_SD_forward_penult(self, x):
    global skip
    skip = [x]
    body = torch.nn.Sequential(
        *self.down_blocks,
        self.mid_block,
        *self.up_blocks,
        self.conv_norm_out,
        self.conv_act,
    )
    x = body(x)
    return x


def MyUNet2DConditionModel_SD_forward_penult_embed(self, x, embed):
    global skip
    global temb
    temb = embed
    skip = [x]
    body = torch.nn.Sequential(
        *self.down_blocks,
        self.mid_block,
        *self.up_blocks,
        self.conv_norm_out,
        self.conv_act,
    )
    x = body(x)
    return x


def MyUNet2DConditionModel_SD_forward_penult_t_embed(self, x, t):
    global skip
    global temb
    t_emb = self.get_time_embed(sample=x, timestep=t)
    temb = self.time_embedding(t_emb, None)
    skip = [x]
    body = torch.nn.Sequential(
        *self.down_blocks,
        self.mid_block,
        *self.up_blocks,
        self.conv_norm_out,
        self.conv_act,
    )
    x = body(x)
    return x


def MyUNet2DConditionModel_SD_forward_penult_spa(self, x, index):
    global skip
    global super_index
    super_index = index
    skip = [x]
    body = torch.nn.Sequential(
        *self.down_blocks,
        self.mid_block,
        *self.up_blocks,
        self.conv_norm_out,
        self.conv_act,
    )
    x = body(x)
    return x


def MyCrossAttnDownBlock2D_SD_forward(self, x):
    for i in range(2):
        x = self.resnets[i](x)
        x = self.attentions[i](x)
        skip.append(x)
    if self.downsamplers is not None:
        x = self.downsamplers[0](x)
        skip.append(x)
    return x


def MyCrossAttnUpBlock2D_SD_forward(self, x):
    for i in range(3):
        x = self.resnets[i](torch.cat([x, skip.pop()], dim=1))
        x = self.attentions[i](x)
    if self.upsamplers is not None:
        x = self.upsamplers[0](x)
    return x


def MyDownBlock2D_SD_forward(self, x):
    for i in range(2):
        x = self.resnets[i](x)
        skip.append(x)
    return x


def MyUNetMidBlock2DCrossAttn_SD_forward(self, x):
    x = self.resnets[0](x)
    x = self.attentions[0](x)
    x = self.resnets[1](x)
    return x


def MyUpBlock2D_SD_forward(self, x):
    for i in range(3):
        x = self.resnets[i](torch.cat([x, skip.pop()], dim=1))
    x = self.upsamplers[0](x)
    return x


def MyResnetBlock2D_SD_forward(self, x_in):
    x = self.norm1(x_in)
    x = self.nonlinearity(x)
    x = self.conv1(x)
    x = self.norm2(x)
    x = self.nonlinearity(x)
    x = self.conv2(x)
    if self.in_channels == self.out_channels:
        return x + x_in
    return x + self.conv_shortcut(x_in)


def MyResnetBlock2D_SD_forward_embed(self, x_in):
    temb_curr = self.time_emb_proj(temb)[:, :, None, None]
    x = self.norm1(x_in)
    x = self.nonlinearity(x)
    x = self.conv1(x) + temb_curr
    x = self.norm2(x)
    x = self.nonlinearity(x)
    x = self.conv2(x)
    if self.in_channels == self.out_channels:
        return x + x_in
    return x + self.conv_shortcut(x_in)


def MyTransformer2DModel_SD_forward(self, x_in):
    b, c, h, w = x_in.shape
    x = self.norm(x_in)
    x = x.permute(0, 2, 3, 1).reshape(b, h * w, c).contiguous()
    x = self.proj_in(x)
    for block in self.transformer_blocks:
        x = x + block.attn1(block.norm1(x))
        x = x + block.ff(block.norm3(x))
    x = self.proj_out(x)
    x = x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
    return x + x_in


def MyTransformer2DModel_SD_forward_no_sa(self, x_in):
    b, c, h, w = x_in.shape
    x = self.norm(x_in)
    x = x.permute(0, 2, 3, 1).reshape(b, h * w, c).contiguous()
    x = self.proj_in(x)
    for block in self.transformer_blocks:
        # x = x + block.attn1(block.norm1(x))
        # x = x + block.attn2(block.norm2(x))
        x = x + block.ff(block.norm3(x))
    x = self.proj_out(x)
    x = x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
    return x + x_in


def MyTransformer2DModel_SD_forward_sa2rcan(self, x_in):
    b, c, h, w = x_in.shape
    x = self.norm(x_in)
    x = x.permute(0, 2, 3, 1).reshape(b, h * w, c).contiguous()
    x = self.proj_in(x)
    for block in self.transformer_blocks:
        x = x + block.attn1(block.norm1(x))
        # x = x + block.attn2(block.norm2(x))
        x = x + block.ff(block.norm3(x))
    x = self.proj_out(x)
    x = x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
    return x + x_in


def MyTransformer2DModel_SD_forward_sa2spa(self, x_in):
    b, c, h, w = x_in.shape
    x = self.norm(x_in)
    x = x.permute(0, 2, 3, 1).reshape(b, h * w, c).contiguous()
    x = self.proj_in(x)
    for block in self.transformer_blocks:
        x = x + block.attn1(block.norm1(x), super_index)
        # x = x + block.attn2(block.norm2(x))
        x = x + block.ff(block.norm3(x))
    x = self.proj_out(x)
    x = x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
    return x + x_in
