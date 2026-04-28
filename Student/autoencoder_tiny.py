# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2024 Ollin Boer Bohan and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.autoencoders.vae import AutoencoderTinyBlock
from diffusers.models.autoencoders.autoencoder_tiny import AutoencoderTiny
from diffusers.models.autoencoders.vae import *
import torch.nn.functional as F


@dataclass
class AutoencoderTinyOutput(BaseOutput):
    """
    Output of AutoencoderTiny encoding method.

    Args:
        latents (`torch.Tensor`): Encoded outputs of the `Encoder`.

    """

    latents: torch.Tensor


class Zero_conv(nn.Module):
    def __init__(self, in_channels, out_channels, preprocess=False):
        super(Zero_conv, self).__init__()
        if preprocess:
            self.preBlock = AutoencoderTinyBlock(
                in_channels, in_channels, act_fn="relu"
            )
        else:
            self.preBlock = None
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        nn.init.constant_(self.conv.weight, 0)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        if self.preBlock is not None:
            x = self.preBlock(x)
        x = self.conv(x)
        return x


class Identity_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Identity_conv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        if in_channels >= out_channels:
            weight = torch.zeros(out_channels, in_channels, 1, 1)
            for i in range(out_channels):
                weight[i, i, 0, 0] = 1
            self.conv.weight = nn.Parameter(weight)
        else:
            nn.init.constant_(self.conv.weight, 0)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        return x


class AutoencoderSkipTiny(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, taesd_path: str = "madebyollin/taesd", prerpocess: bool = False):
        super().__init__()
        self.taesd = AutoencoderTiny.from_pretrained(taesd_path)
        skip_layers = []
        skip_fusion_layers = []
        for i in range(len(self.taesd.encoder_block_out_channels) - 1):
            skip_layers.append(
                Zero_conv(
                    self.taesd.encoder_block_out_channels[i],
                    self.taesd.decoder_block_out_channels[
                        len(self.taesd.encoder_block_out_channels) - i - 1
                    ],
                    prerpocess,
                )
            )
            skip_fusion_layers.append(
                Identity_conv(
                    self.taesd.decoder_block_out_channels[
                        len(self.taesd.encoder_block_out_channels) - i - 1
                    ]
                    * 2,
                    self.taesd.decoder_block_out_channels[
                        len(self.taesd.encoder_block_out_channels) - i - 1
                    ],
                )
            )
        self.skip_layers = nn.ModuleList(skip_layers)
        self.skip_fusion_layers = nn.ModuleList(skip_fusion_layers)

    def skip_encode(self, x: torch.FloatTensor) -> torch.FloatTensor:
        r"""Skip encoding of the input tensor.

        This method is used to skip the encoding of the input tensor and directly return the input tensor as the output.

        Args:
            x (`torch.FloatTensor`): Input tensor.

        Returns:
            `torch.FloatTensor`: Output tensor.
        """
        x = (x + 1) / 2
        skip_feats = []
        for i in range(len(self.taesd.encoder.layers)):
            x = self.taesd.encoder.layers[i](x)
            if (
                i < len(self.taesd.encoder.layers) - 1
                and self.taesd.encoder.layers[i + 1].__class__.__name__ == "Conv2d"
                and self.taesd.encoder.layers[i].__class__.__name__
                == "AutoencoderTinyBlock"
                and len(skip_feats) < len(self.skip_layers)
            ):
                skip_feats.append(x)
        return x, skip_feats

    def skip_encode_penult(self, x: torch.FloatTensor) -> torch.FloatTensor:
        r"""Skip encoding of the input tensor.

        This method is used to skip the encoding of the input tensor and directly return the input tensor as the output.

        Args:
            x (`torch.FloatTensor`): Input tensor.

        Returns:
            `torch.FloatTensor`: Output tensor.
        """
        x = (x + 1) / 2
        skip_feats = []
        for i in range(len(self.taesd.encoder.layers)):
            if i == (len(self.taesd.encoder.layers) - 1):
                continue
            x = self.taesd.encoder.layers[i](x)
            if (
                i < len(self.taesd.encoder.layers) - 1
                and self.taesd.encoder.layers[i + 1].__class__.__name__ == "Conv2d"
                and self.taesd.encoder.layers[i].__class__.__name__
                == "AutoencoderTinyBlock"
                and len(skip_feats) < len(self.skip_layers)
            ):
                skip_feats.append(x)
        return x, skip_feats

    def skip_decode(
        self, x: torch.FloatTensor, skip_feats: list, uncer: torch.FloatTensor = None
    ) -> torch.FloatTensor:
        r"""Skip decoding of the input tensor.

        This method is used to skip the decoding of the input tensor and directly return the input tensor as the output.

        Args:
            x (`torch.FloatTensor`): Input tensor.

        Returns:
            `torch.FloatTensor`: Output tensor.
        """
        index = len(skip_feats) - 1
        x = torch.tanh(x / 3) * 3
        for i in range(len(self.taesd.decoder.layers)):
            x = self.taesd.decoder.layers[i](x)
            if (
                i < len(self.taesd.decoder.layers) - 1
                and self.taesd.decoder.layers[i].__class__.__name__ == "Conv2d"
                and self.taesd.decoder.layers[i + 1].__class__.__name__
                == "AutoencoderTinyBlock"
            ):
                skip_layer = self.skip_layers[index]
                skip_fusion_layer = self.skip_fusion_layers[index]
                skip_feat = skip_feats.pop()
                skip_feat = skip_layer(skip_feat)
                if uncer is not None:
                    uncer = F.interpolate(
                        uncer,
                        size=(skip_feat.shape[2], skip_feat.shape[3]),
                        mode="nearest",
                    )
                    skip_feat *= uncer
                x = skip_fusion_layer(torch.cat([x, skip_feat], dim=1))
                index -= 1
        return x * 2 - 1

    def skip_decode_penult(
        self, x: torch.FloatTensor, skip_feats: list, uncer: torch.FloatTensor = None
    ) -> torch.FloatTensor:
        r"""Skip decoding of the input tensor.

        This method is used to skip the decoding of the input tensor and directly return the input tensor as the output.

        Args:
            x (`torch.FloatTensor`): Input tensor.

        Returns:
            `torch.FloatTensor`: Output tensor.
        """
        index = len(skip_feats) - 1
        x = torch.tanh(x / 3) * 3
        for i in range(len(self.taesd.decoder.layers)):
            if i == 0:
                continue
            x = self.taesd.decoder.layers[i](x)
            if (
                i < len(self.taesd.decoder.layers) - 1
                and self.taesd.decoder.layers[i].__class__.__name__ == "Conv2d"
                and self.taesd.decoder.layers[i + 1].__class__.__name__
                == "AutoencoderTinyBlock"
            ):
                skip_layer = self.skip_layers[index]
                skip_fusion_layer = self.skip_fusion_layers[index]
                skip_feat = skip_feats.pop()
                skip_feat = skip_layer(skip_feat)
                if uncer is not None:
                    uncer = F.interpolate(
                        uncer,
                        size=(skip_feat.shape[2], skip_feat.shape[3]),
                        mode="nearest",
                    )
                    skip_feat *= uncer
                x = skip_fusion_layer(torch.cat([x, skip_feat], dim=1))
                index -= 1
        return x * 2 - 1

    def forward(self, x):
        enc, skip_feats = self.skip_encode(x)
        dec = self.skip_decode(enc, skip_feats)
        return dec


class AutoencoderTinySkipNoFusion(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, taesd_path: str = "madebyollin/taesd", prerpocess: bool = False):
        super().__init__()
        self.taesd = AutoencoderTiny.from_pretrained(taesd_path)
        skip_layers = []
        for i in range(len(self.taesd.encoder_block_out_channels) - 1):
            skip_layers.append(
                Zero_conv(
                    self.taesd.encoder_block_out_channels[i],
                    self.taesd.decoder_block_out_channels[
                        len(self.taesd.encoder_block_out_channels) - i - 1
                    ],
                    prerpocess,
                )
            )
        self.skip_layers = nn.ModuleList(skip_layers)

    def skip_encode(self, x: torch.FloatTensor) -> torch.FloatTensor:
        r"""Skip encoding of the input tensor.

        This method is used to skip the encoding of the input tensor and directly return the input tensor as the output.

        Args:
            x (`torch.FloatTensor`): Input tensor.

        Returns:
            `torch.FloatTensor`: Output tensor.
        """
        x = (x + 1) / 2
        skip_feats = []
        for i in range(len(self.taesd.encoder.layers)):
            x = self.taesd.encoder.layers[i](x)
            if (
                i < len(self.taesd.encoder.layers) - 1
                and self.taesd.encoder.layers[i + 1].__class__.__name__ == "Conv2d"
                and self.taesd.encoder.layers[i].__class__.__name__
                == "AutoencoderTinyBlock"
                and len(skip_feats) < len(self.skip_layers)
            ):
                skip_feats.append(x)
        return x, skip_feats

    def skip_encode_penult(self, x: torch.FloatTensor) -> torch.FloatTensor:
        r"""Skip encoding of the input tensor.

        This method is used to skip the encoding of the input tensor and directly return the input tensor as the output.

        Args:
            x (`torch.FloatTensor`): Input tensor.

        Returns:
            `torch.FloatTensor`: Output tensor.
        """
        x = (x + 1) / 2
        skip_feats = []
        for i in range(len(self.taesd.encoder.layers)):
            if i == (len(self.taesd.encoder.layers) - 1):
                continue
            x = self.taesd.encoder.layers[i](x)
            if (
                i < len(self.taesd.encoder.layers) - 1
                and self.taesd.encoder.layers[i + 1].__class__.__name__ == "Conv2d"
                and self.taesd.encoder.layers[i].__class__.__name__
                == "AutoencoderTinyBlock"
                and len(skip_feats) < len(self.skip_layers)
            ):
                skip_feats.append(x)
        return x, skip_feats

    def skip_decode(
        self, x: torch.FloatTensor, skip_feats: list, uncer: torch.FloatTensor = None
    ) -> torch.FloatTensor:
        r"""Skip decoding of the input tensor.

        This method is used to skip the decoding of the input tensor and directly return the input tensor as the output.

        Args:
            x (`torch.FloatTensor`): Input tensor.

        Returns:
            `torch.FloatTensor`: Output tensor.
        """
        index = len(skip_feats) - 1
        x = torch.tanh(x / 3) * 3
        for i in range(len(self.taesd.decoder.layers)):
            x = self.taesd.decoder.layers[i](x)
            if (
                i < len(self.taesd.decoder.layers) - 1
                and self.taesd.decoder.layers[i].__class__.__name__ == "Conv2d"
                and self.taesd.decoder.layers[i + 1].__class__.__name__
                == "AutoencoderTinyBlock"
            ):
                skip_layer = self.skip_layers[index]
                skip_feat = skip_feats.pop()
                skip_feat = skip_layer(skip_feat)
                if uncer is not None:
                    uncer = F.interpolate(
                        uncer,
                        size=(skip_feat.shape[2], skip_feat.shape[3]),
                        mode="nearest",
                    )
                    skip_feat = (1 - uncer) * skip_feat
                x += skip_feat
                index -= 1
        return x * 2 - 1

    def skip_decode_penult(
        self, x: torch.FloatTensor, skip_feats: list, uncer: torch.FloatTensor = None
    ) -> torch.FloatTensor:
        r"""Skip decoding of the input tensor.

        This method is used to skip the decoding of the input tensor and directly return the input tensor as the output.

        Args:
            x (`torch.FloatTensor`): Input tensor.

        Returns:
            `torch.FloatTensor`: Output tensor.
        """
        index = len(skip_feats) - 1
        x = torch.tanh(x / 3) * 3
        for i in range(len(self.taesd.decoder.layers)):
            if i == 0:
                continue
            x = self.taesd.decoder.layers[i](x)
            if (
                i < len(self.taesd.decoder.layers) - 1
                and self.taesd.decoder.layers[i].__class__.__name__ == "Conv2d"
                and self.taesd.decoder.layers[i + 1].__class__.__name__
                == "AutoencoderTinyBlock"
            ):
                skip_layer = self.skip_layers[index]
                skip_feat = skip_feats.pop()
                skip_feat = skip_layer(skip_feat)
                if uncer is not None:
                    uncer = F.interpolate(
                        uncer,
                        size=(skip_feat.shape[2], skip_feat.shape[3]),
                        mode="nearest",
                    )
                    skip_feat *= uncer
                x += skip_feat
                index -= 1
        return x * 2 - 1

    def forward(self, x):
        enc, skip_feats = self.skip_encode(x)
        dec = self.skip_decode(enc, skip_feats)
        return dec
