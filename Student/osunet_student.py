from collections import OrderedDict
from diffusers import (
    UNet2DConditionModel,
    DDIMScheduler,
    AutoencoderKL,
    AutoencoderTiny,
)

from model import halve_channels
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
    MyUNet2DConditionModel_SD_forward_penult,
    MyCrossAttnDownBlock2D_SD_forward,
    MyDownBlock2D_SD_forward,
    MyUNetMidBlock2DCrossAttn_SD_forward,
    MyCrossAttnUpBlock2D_SD_forward,
    MyUpBlock2D_SD_forward,
    MyResnetBlock2D_SD_forward,
    MyTransformer2DModel_SD_forward,
    MyTransformer2DModel_SD_forward_no_sa,
    MyTransformer2DModel_SD_forward_sa2rcan,
    MyUNet2DConditionModel_SD_forward,
    MyUNet2DConditionModel_SD_forward_penult_embed,
    MyResnetBlock2D_SD_forward_embed,
    MyTransformer2DModel_SD_forward_sa2spa,
    MyUNet2DConditionModel_SD_forward_penult_spa,
    MyUNet2DConditionModel_SD_forward_penult_t_embed,
)
from peft import LoraConfig, get_peft_model
import torch.nn as nn
import torch, types
from .modules import (
    DirectionalConvBlock,
    FrequencySeparationConvBlock,
    QGAM,
)


def make_1step_sched(pretrained_path):
    noise_scheduler_1step = DDIMScheduler.from_pretrained(
        pretrained_path, subfolder="scheduler"
    )
    noise_scheduler_1step.set_timesteps(1, device="cuda")
    noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod.cuda()
    return noise_scheduler_1step


class CrossAttnMidBlock2D(nn.Module):
    def __init__(self, down_block):
        super(CrossAttnMidBlock2D, self).__init__()
        self.mid_block = down_block.resnets[0]
        self.mid_block.downsamplers = None
        self.has_cross_attention = True

    def forward(self, sample, *args, **kwargs):
        sample = self.mid_block(sample, *args, **kwargs)
        return sample


class LEDSR(nn.Module):

    def __init__(self, args, text_embeddings, pretrained_path=None):
        super(LEDSR, self).__init__()

        self.args = args
        self.text_embeddings = text_embeddings.to("cuda")
        self.text_embeddings.requires_grad_(False)

        self.unet = UNet2DConditionModel.from_pretrained(args.sd_path, subfolder="unet")
        if pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu")

            unet_lora_config = LoraConfig(
                r=sd["rank_unet"],
                init_lora_weights="gaussian",
                target_modules=sd["unet_lora_target_modules"],
            )

            # 1️⃣ 临时挂 LoRA
            self.unet.add_adapter(unet_lora_config)

            # 2️⃣ 加载 LoRA 权重
            self.unet.load_state_dict(
                sd["state_dict_unet"],
                strict=False,
            )

            # 3️⃣ 融合到 base 权重
            self.unet.merge_and_unload()

        def replace_sd_self_attn_with_rcan_and_remove_cross_attn(unet: nn.Module):
            """
            在 Stable Diffusion 的 UNet 中：
            - 将 BasicTransformerBlock.attn1 替换为 RCAN 通道注意力（序列版）
            - 移除 cross-attention (attn2) 以及其对应的 norm2
            """

            def _transform(module: nn.Module):
                if isinstance(module, BasicTransformerBlock):
                    # 取通道维 C（norm1 的 normalized_shape 通常就是 [C]）
                    # 兼容 tuple/list 的情况
                    if hasattr(module, "norm1") and hasattr(
                        module.norm1, "normalized_shape"
                    ):
                        norm_shape = module.norm1.normalized_shape
                        channels = (
                            norm_shape[0]
                            if isinstance(norm_shape, (tuple, list))
                            else norm_shape
                        )
                    else:
                        # 兜底：从原 attn1 的投影层上取 in_features
                        try:
                            channels = (
                                module.attn1.to_q.in_features
                            )  # diffusers<=0.27 大多可行
                        except Exception:
                            raise RuntimeError("无法自动推断通道数 C，请手动指定。")

                    # 替换自注意力为通道注意力（保持接口一致）
                    module.attn1 = QGAM(dim=channels, num_queries=32)

                    # 可保留 norm1，不动；移除 cross-attn（以及它的 LayerNorm）
                    if hasattr(module, "attn2"):
                        del module.attn2
                    if hasattr(module, "norm2"):
                        del module.norm2

            unet.apply(_transform)

        replace_sd_self_attn_with_rcan_and_remove_cross_attn(self.unet)

        def replace_conv_with_dcb(module: nn.Module):
            if isinstance(module, ResnetBlock2D):
                module.conv1 = FrequencySeparationConvBlock(
                    module.conv1.in_channels, module.conv1.out_channels
                )
                module.conv2 = DirectionalConvBlock(
                    module.conv2.in_channels, module.conv2.out_channels, 0.25
                )

        self.unet.apply(replace_conv_with_dcb)

        def ResnetBlock2D_remove_time_emb_proj(module):
            if isinstance(module, ResnetBlock2D):
                del module.time_emb_proj

        self.unet.apply(ResnetBlock2D_remove_time_emb_proj)

        def set_inplace_to_true(module):
            if isinstance(module, nn.Dropout) or isinstance(module, nn.SiLU):
                module.inplace = True

        self.unet.apply(set_inplace_to_true)

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
                    MyTransformer2DModel_SD_forward_sa2rcan, module
                )

        self.unet.apply(replace_forward_methods)
        self.unet.forward = types.MethodType(
            MyUNet2DConditionModel_SD_forward_penult, self.unet
        )

        self.vae = AutoencoderTiny.from_pretrained(args.taesd_path)
        self.merge_conv1 = nn.Conv2d(64, 320, 3, 1, 1)
        self.merge_conv2 = nn.Conv2d(320, 64, 3, 1, 1)

        self.unet_depth = len(self.unet.config.block_out_channels)
        mid_block = CrossAttnMidBlock2D(self.unet.down_blocks[args.prune_depth])
        self.unet.down_blocks = self.unet.down_blocks[: args.prune_depth]
        self.unet.down_blocks[-1].downsamplers = None
        self.unet.up_blocks = self.unet.up_blocks[self.unet_depth - args.prune_depth :]

        self.vae.requires_grad_(True)
        self.unet.requires_grad_(True)
        self.merge_conv1.requires_grad_(True)
        self.merge_conv2.requires_grad_(True)
        self.sched = make_1step_sched(args.sd_path)
        self.timestamps = self.sched.timesteps

        self.unet.mid_block = mid_block

        self.unet.to("cuda")
        self.vae.to("cuda")
        self.merge_conv1.to("cuda")
        self.merge_conv2.to("cuda")

        torch.cuda.empty_cache()

        vae_param = sum(p.numel() for p in self.vae.parameters())
        unet_param = sum(p.numel() for p in self.unet.parameters())
        print("vae parameters M:", vae_param / 1e6)
        print("unet parameters M:", unet_param / 1e6)
        print("total parameters M:", (vae_param + unet_param) / 1e6)

    def _get_first_cross_attention_module(self):
        for module in self.unet.modules():
            if isinstance(module, QGAM):
                return module
        raise RuntimeError("未在当前 UNet 中找到 CrossAttentionModule。")

    def set_train(self):
        self.vae.requires_grad_(True)
        self.unet.requires_grad_(True)
        self.merge_conv1.requires_grad_(True)
        self.merge_conv2.requires_grad_(True)

    def set_eval(self):
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.merge_conv1.requires_grad_(False)
        self.merge_conv2.requires_grad_(False)

    def forward(self, lq):
        encode_lq = self.vae.encoder.layers[:-1](lq)
        model_input = self.merge_conv1(encode_lq)
        noise_pred = self.unet(
            model_input,
        )

        z_denoised = self.merge_conv2(noise_pred)
        image = self.vae.decoder.layers[1:](z_denoised)
        return z_denoised, image

    def forward_first_attn_weights(self, lq):
        first_cross_attn = self._get_first_cross_attention_module()
        first_cross_attn.last_attn_weights = None

        encode_lq = self.vae.encoder.layers[:-1](lq)
        model_input = self.merge_conv1(encode_lq)
        _ = self.unet(
            model_input,
        )

        if first_cross_attn.last_attn_weights is None:
            raise RuntimeError("未捕获到第一个 CrossAttentionModule 的 attn_weights。")

        return first_cross_attn.last_attn_weights
