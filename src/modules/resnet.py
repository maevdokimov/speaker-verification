from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types.elements import AcousticEncodedRepresentation, LengthsType, MelSpectrogramType
from nemo.core.neural_types.neural_type import NeuralType

COMPRESSION_FACTOR = 8


class DilatedConvNorm(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, forward_padding=0, side_padding=0, pad_value=0.0, **kwargs
    ):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

        self.pad_value = pad_value
        self.pad_size = forward_padding, 0, side_padding, side_padding

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): batch x channels x D x T
        """
        x = F.pad(x, self.pad_size, "constant", self.pad_value)

        return self.conv(x)


class ResNetBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        apply_downsample: bool = False,
        norm_layer: nn.Module = nn.BatchNorm2d,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.conv1 = DilatedConvNorm(
            in_features, out_features, kernel_size=3, forward_padding=2, side_padding=1, bias=False
        )
        self.bn1 = norm_layer(out_features)
        self.relu = nn.ReLU()
        self.conv2 = DilatedConvNorm(
            out_features, out_features, kernel_size=3, forward_padding=2, side_padding=1, bias=False
        )
        self.bn2 = norm_layer(out_features)

        self.apply_downsample = apply_downsample
        self.different_depth = self.in_features != self.out_features
        if self.apply_downsample:
            self.maxpool = nn.MaxPool2d(2)
        if self.different_depth:
            self.conv3 = nn.Sequential(
                DilatedConvNorm(in_features, out_features, kernel_size=1, bias=False), norm_layer(out_features)
            )

    def forward(self, x: torch.Tensor):
        if self.apply_downsample:
            x = self.maxpool(x)
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.different_depth:
            identity = self.conv3(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckBlock(nn.Module):
    pass


class ResNet(NeuralModule):
    """
    ResNet module for spectrogram representation extraction
    Args:
        block (str): classname of ResNet building block
        layers (List[int]): classical ResNet architecture consists of 4 main layers.
            Number of blocks in each layer. E.g. ResNet34 - [3, 4, 6, 3]
        num_filters (List[int]): model depth at each layer. Length must be equal to layers
        n_mels (int): height of spectrogram
        output_dim (int): dimension of output embeddings
    """

    def __init__(self, block: str, layers: List[int], num_filters: List[int], n_mels: int, output_dim: int):
        super().__init__()

        if len(num_filters) != len(layers):
            raise ValueError(f"Incorrect layers <-> num_filters lengths: " f"{len(num_filters)}, {len(layers)}")

        if block == "ResNetBlock":
            self.block = ResNetBlock
        elif block == "BottleneckBlock":
            self.block = BottleneckBlock
        else:
            raise ValueError(f"Unknown block type: {block}")

        self.norm_layer = nn.BatchNorm2d
        self.layers = layers
        self.num_filters = num_filters
        self.n_mels = n_mels
        self.output_dim = output_dim

        self.conv1 = DilatedConvNorm(1, num_filters[0], kernel_size=3, forward_padding=2, side_padding=1, bias=False)
        self.bn1 = self.norm_layer(num_filters[0])
        self.relu = nn.ReLU()

        self.layer1 = self._make_layer(
            self.block,
            in_features=num_filters[0],
            out_features=num_filters[0],
            n_blocks=layers[0],
            apply_downsample=False,
        )
        self.layer2 = self._make_layer(
            self.block,
            in_features=num_filters[0],
            out_features=num_filters[1],
            n_blocks=layers[1],
            apply_downsample=True,
        )
        self.layer3 = self._make_layer(
            self.block,
            in_features=num_filters[1],
            out_features=num_filters[2],
            n_blocks=layers[2],
            apply_downsample=True,
        )
        self.layer4 = self._make_layer(
            self.block,
            in_features=num_filters[2],
            out_features=num_filters[3],
            n_blocks=layers[3],
            apply_downsample=True,
        )

        self.conv2 = nn.Conv1d((n_mels // COMPRESSION_FACTOR) * num_filters[3], output_dim, kernel_size=1)

    def _make_layer(
        self,
        block: Union[ResNetBlock, BottleneckBlock],
        in_features: int,
        out_features: int,
        n_blocks: int,
        apply_downsample: bool,
    ):
        layers = []
        layers.append(block(in_features, out_features, apply_downsample=apply_downsample))

        for _ in range(1, n_blocks):
            layers.append(block(out_features, out_features, apply_downsample=False))

        return nn.Sequential(*layers)

    @property
    def input_types(self):
        return {
            "audio_signal": NeuralType(("B", "D", "T"), MelSpectrogramType()),
            "length": NeuralType(tuple("B"), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "encoder_outputs": NeuralType(("B", "D", "T"), AcousticEncodedRepresentation()),
            "encoded_lengths": NeuralType(tuple("B"), LengthsType()),
        }

    @typecheck()
    def forward(self, audio_signal, length):
        audio_signal = audio_signal.unsqueeze(1)
        x = self.conv1(audio_signal)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.reshape(x.shape[0], -1, x.shape[-1])
        encoder_outputs = self.conv2(x)
        length = torch.div(length, COMPRESSION_FACTOR, rounding_mode="floor")

        return encoder_outputs, length
