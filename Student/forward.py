import torch

def MyUNet2DConditionModel_SD_forward(self, x):
    global skip
    x = self.conv_in(x)
    skip = [x]
    x = self.body(x)
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