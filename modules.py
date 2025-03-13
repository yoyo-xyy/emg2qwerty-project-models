# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence

import torch
from torch import nn

import math  # Add this
import torch.nn.functional as F
from torch import Tensor

class SpectrogramNorm(nn.Module):
    """A `torch.nn.Module` that applies 2D batch normalization over spectrogram
    per electrode channel per band. Inputs must be of shape
    (T, N, num_bands, electrode_channels, frequency_bins).

    With left and right bands and 16 electrode channels per band, spectrograms
    corresponding to each of the 2 * 16 = 32 channels are normalized
    independently using `nn.BatchNorm2d` such that stats are computed
    over (N, freq, time) slices.

    Args:
        channels (int): Total number of electrode channels across bands
            such that the normalization statistics are calculated per channel.
            Should be equal to num_bands * electrode_chanels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape  # (T, N, bands=2, C=16, freq)
        assert self.channels == bands * C

        x = inputs.movedim(0, -1)  # (N, bands=2, C=16, freq, T)
        x = x.reshape(N, bands * C, freq, T)
        x = self.batch_norm(x)
        x = x.reshape(N, bands, C, freq, T)
        return x.movedim(-1, 0)  # (T, N, bands=2, C=16, freq)


class RotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that takes an input tensor of shape
    (T, N, electrode_channels, ...) corresponding to a single band, applies
    an MLP after shifting/rotating the electrodes for each positional offset
    in ``offsets``, and pools over all the outputs.

    Returns a tensor of shape (T, N, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input of
            shape (T, N, C, ...), this should be equal to C * ... (that is,
            the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        mlp: list[nn.Module] = []
        for out_features in mlp_features:
            mlp.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                ]
            )
            in_features = out_features
        self.mlp = nn.Sequential(*mlp)

        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"
        self.pooling = pooling

        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, C, ...)

        # Create a new dim for band rotation augmentation with each entry
        # corresponding to the original tensor with its electrode channels
        # shifted by one of ``offsets``:
        # (T, N, C, ...) -> (T, N, rotation, C, ...)
        x = torch.stack([x.roll(offset, dims=2) for offset in self.offsets], dim=2)

        # Flatten features and pass through MLP:
        # (T, N, rotation, C, ...) -> (T, N, rotation, mlp_features[-1])
        x = self.mlp(x.flatten(start_dim=3))

        # Pool over rotations:
        # (T, N, rotation, mlp_features[-1]) -> (T, N, mlp_features[-1])
        if self.pooling == "max":
            return x.max(dim=2).values
        else:
            return x.mean(dim=2)


class MultiBandRotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that applies a separate instance of
    `RotationInvariantMLP` per band for inputs of shape
    (T, N, num_bands, electrode_channels, ...).

    Returns a tensor of shape (T, N, num_bands, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input
            of shape (T, N, num_bands, C, ...), this should be equal to
            C * ... (that is, the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
        num_bands (int): ``num_bands`` for an input of shape
            (T, N, num_bands, C, ...). (default: 2)
        stack_dim (int): The dimension along which the left and right data
            are stacked. (default: 2)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim

        # One MLP per band
        self.mlps = nn.ModuleList(
            [
                RotationInvariantMLP(
                    in_features=in_features,
                    mlp_features=mlp_features,
                    pooling=pooling,
                    offsets=offsets,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands

        inputs_per_band = inputs.unbind(self.stack_dim)
        outputs_per_band = [
            mlp(_input) for mlp, _input in zip(self.mlps, inputs_per_band)
        ]
        return torch.stack(outputs_per_band, dim=self.stack_dim)


class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.width = width

        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        # TNC -> NCT -> NcwT
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

        # Skip connection after downsampling
        T_out = x.shape[0]
        x = x + inputs[-T_out:]

        # Layer norm over C
        return self.layer_norm(x)  # TNC


class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # TNC
        x = self.fc_block(x)
        x = x + inputs
        return self.layer_norm(x)  # TNC

class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0
        tds_conv_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, num_features // channels, kernel_width),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)

# Hybrid CNN-GRU Encoder
class HybridConvGRUEncoder(nn.Module):
    """A hybrid encoder combining CNN layers for spatial feature extraction and GRU for temporal modeling.

    Args:
        num_features (int): Number of input features (e.g., num_bands * mlp_features[-1]).
        conv_channels (list): Number of channels for each CNN layer.
        kernel_size (int): Kernel size for temporal convolutions.
        gru_hidden_size (int): Hidden size of the GRU.
        gru_num_layers (int): Number of GRU layers.
        dropout (float): Dropout rate for GRU.
    """
    def __init__(
        self,
        num_features: int,
        conv_channels: Sequence[int] = [64, 64],
        kernel_size: int = 5,
        gru_hidden_size: int = 256,
        gru_num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # CNN layers
        conv_layers = []
        in_channels = num_features
        for out_channels in conv_channels:
            conv_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,  # Preserve sequence length
                )
            )
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.BatchNorm1d(out_channels))
            in_channels = out_channels
        self.conv_block = nn.Sequential(*conv_layers)

        # GRU layer
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=False,
            bidirectional=True,
            dropout=dropout if gru_num_layers > 1 else 0.0,
        )
        # Linear layer to match output to num_features
        self.linear = nn.Linear(gru_hidden_size * 2, num_features)  # *2 for bidirectional

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, num_features)
        x = inputs.movedim(1, 0)  # (N, T, num_features)
        x = x.movedim(2, 1)       # (N, num_features, T)
        x = self.conv_block(x)     # (N, conv_channels[-1], T)
        x = x.movedim(2, 1)       # (N, T, conv_channels[-1])
        x = x.movedim(1, 0)       # (T, N, conv_channels[-1])
        x, _ = self.gru(x)         # (T, N, gru_hidden_size * 2)
        x = self.linear(x)         # (T, N, num_features)
        return x

# Transformer-based Encoder
class TransformerEncoder(nn.Module):
    """Transformer-based encoder replacing TDSConvEncoder.

    Args:
        num_features (int): Input features.
        nhead (int): Number of attention heads.
        num_layers (int): Number of transformer layers.
        dim_feedforward (int): Feed-forward network dimension.
        dropout (float): Dropout rate.
    """
    def __init__(
        self,
        num_features: int,
        nhead: int = 8,
        num_layers: int = 2,  # Increase depth
        dim_feedforward: int = 512,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_features,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(num_features)

    def _generate_pos_encoding(self, T):
        position = torch.arange(T, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.num_features, 2).float() * (-math.log(10000.0) / self.num_features))
        pos_encoding = torch.zeros(T, self.num_features)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding.unsqueeze(1)  # (T, 1, num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, num_features), e.g., (8000, batch_size, 768)
        T, N, _ = inputs.shape
        # Add positional encoding
        pos_encoding = self._generate_pos_encoding(T).to(inputs.device)
        x = inputs + pos_encoding
        # Transformer expects (T, N, F)
        x = self.transformer(x)
        return self.layer_norm(x)

# LSTM-based Encoder
class LSTMEncoder(nn.Module):
    """LSTM-based encoder for EMG sequence modeling.

    Args:
        num_features (int): Input features (e.g., num_bands * mlp_features[-1]).
        hidden_size (int): LSTM hidden state size.
        num_layers (int): Number of LSTM layers.
        dropout (float): Dropout rate between LSTM layers.
        bidirectional (bool): If True, use bidirectional LSTM.
    """
    def __init__(
        self,
        num_features: int,
        hidden_size: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,  # Input: [T, N, F]
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output size from LSTM: hidden_size * 2 if bidirectional
        lstm_out_features = hidden_size * 2 if bidirectional else hidden_size
        # Fully connected block to process LSTM output
        self.fc_block = TDSFullyConnectedBlock(lstm_out_features)
        # Output layer to project back to num_features
        self.out_layer = nn.Linear(lstm_out_features, num_features)
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm(inputs)  # [T, N, hidden_size * (2 if bidirectional)], e.g., [8000, 64, 256]
        x = self.fc_block(x)      # [T, N, hidden_size * 2], applies FC block with residual connection
        x = self.out_layer(x)     # [T, N, num_features], e.g., [8000, 64, 384]
        return self.layer_norm(x) # [T, N, num_features]

# Pooling layer for temporal dimension
class TimePool(nn.Module):
    """Pools over the time dimension (dim=0) for [T, N, F] inputs."""
    def __init__(self, kernel_size: int, stride: int = None):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(1, 2, 0)  # [T, N, F] -> [N, F, T]
        x = self.pool(x)        # [N, F, T//stride]
        return x.permute(2, 0, 1)  # [T//stride, N, F]
