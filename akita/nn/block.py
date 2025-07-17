import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal, Union


class AkitaConvBlock1D(nn.Module):
    """ConvBlock from Akita translated from Keras to PyTorch"""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int = 1,
        activation: str = "relu",
        activation_end: Union[str, None] = None,
        stride: int = 1,
        dilation_rate: int = 1,
        dropout: float = 0,
        conv_type: Literal["separable", "standard"] = "standard",
        pool_size: int = 1,
        pool_type: Literal["softmax", "max"] = "max",
        norm_type: Union[Literal["batch", "batch-sync", "layer"], None] = None,
        bn_momentum: float = 0.99,
        residual: bool = False,
    ):
        super(AkitaConvBlock1D, self).__init__()

        self.residual = residual
        self.activation_end = activation_end
        self.pool_size = pool_size
        self.pool_type = pool_type

        # Convolution layer
        if conv_type == "separable":
            self.conv = nn.Conv1d(
                in_channels=in_dim,
                out_channels=out_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding="same",
                dilation=dilation_rate,
                bias=(norm_type is None),
            )
        else:
            self.conv = nn.Conv1d(
                in_channels=in_dim,
                out_channels=out_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding="same",
                dilation=dilation_rate,
                bias=(norm_type is None),
            )

        # Normalization layer
        if norm_type == "batch":
            self.norm = nn.BatchNorm1d(out_dim, momentum=bn_momentum)
        elif norm_type == "batch-sync":
            self.norm = nn.SyncBatchNorm(out_dim, momentum=bn_momentum)
        elif norm_type == "layer":
            # self.norm = nn.LayerNorm([out_dim, seq_dim])
            raise NotImplementedError
        else:
            self.norm = None

        # Dropout layer
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Activation function
        self.activation = getattr(F, activation) if activation else None

        # End activation function
        self.activation_end_fn = getattr(F, activation_end) if activation_end else None

        # Pooling layer
        if pool_size > 1:
            if pool_type == "softmax":
                self.pool = nn.Softmax(dim=-1)
            else:
                self.pool = nn.MaxPool1d(kernel_size=pool_size)
        else:
            self.pool = None

    def forward(self, x: torch.Tensor):
        residual = x

        x = self.activation(x)
        x = self.conv(x)

        if self.norm:
            x = self.norm(x)

        if self.dropout:
            x = self.dropout(x)

        if self.residual:
            x = x + residual

        if self.activation_end_fn:
            x = self.activation_end_fn(x)

        if self.pool:
            x = self.pool(x)

        return x


class AkitaConvTower1D(nn.Module):
    """ConvTower from Akita translated from Keras to PyTorch"""

    def __init__(
        self,
        in_dim: int,
        out_dim_init: int,
        out_dim: Union[int, None] = None,
        out_dim_mult: int = None,
        divisible_by: int = 1,
        repeat: int = 1,
        **conv_block_1d_kwargs
    ):
        pass
        super(AkitaConvTower1D, self).__init__()

        def _round(x):
            return int(np.round(x / divisible_by) * divisible_by)

        # determine multiplier
        if out_dim_mult is None:
            assert out_dim is not None
            out_dim_mult = np.exp(np.log(out_dim / out_dim_init) / (repeat - 1))

        # initialize kernel dim
        rep_in_dim = in_dim
        rep_out_dim = out_dim_init
        self.blocks = nn.ModuleList()
        for _ in range(repeat):
            # convolution
            _rep_out_dim = _round(rep_out_dim)
            self.blocks.append(
                AkitaConvBlock1D(
                    in_dim=rep_in_dim, out_dim=_rep_out_dim, **conv_block_1d_kwargs
                )
            )

            # update kernel dim
            rep_in_dim = _rep_out_dim
            rep_out_dim *= out_dim_mult

    def forward(self, x: torch.Tensor):
        for block in self.blocks:
            x = block(x)
        return x


class AkitaDialatedResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_dim: int,
        mid_dim: int,
        kernel_size: int = 3,
        rate_mult: int = 2,
        dropout: float = 0,
        repeat: int = 1,
        norm_type=None,
        round: bool = False,
        **conv_block_1d_kwargs
    ):
        super(AkitaDialatedResidualBlock1D, self).__init__()
        self.norm_type = norm_type
        self.blocks = nn.ModuleList()

        dilation_rate = 1.0
        for _ in range(repeat):
            sub_block = nn.Sequential(
                AkitaConvBlock1D(
                    in_dim=in_dim,
                    out_dim=mid_dim,
                    kernel_size=kernel_size,
                    dilation_rate=int(np.round(dilation_rate)),
                    norm_type=norm_type,
                    **conv_block_1d_kwargs
                ),
                AkitaConvBlock1D(
                    in_dim=mid_dim,
                    out_dim=in_dim,
                    dropout=dropout,
                    norm_type=norm_type,
                    **conv_block_1d_kwargs
                ),
            )
            self.blocks.append(sub_block)

            dilation_rate *= rate_mult
            if round:
                dilation_rate = np.round(dilation_rate)

        if norm_type is None:
            self.residual_scale = nn.Parameter(torch.zeros(repeat, 1, in_dim, 1))

    def forward(self, x: torch.Tensor):
        for idx, block in enumerate(self.blocks):
            block_in = x
            x = block(x)

            # scale residual
            if self.norm_type is None:
                x = x * self.residual_scale[idx]

            # residual add
            x = x + block_in

        return x


class AkitaConvBlock2D(nn.Module):
    """ConvBlock from Akita translated from Keras to PyTorch"""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int = 1,
        activation: str = "relu",
        stride: int = 1,
        dilation_rate: int = 1,
        dropout: float = 0,
        conv_type: Literal["separable", "standard"] = "standard",
        pool_size: int = 1,
        pool_type: Literal["softmax", "max"] = "max",
        norm_type: Union[Literal["batch", "batch-sync", "layer"], None] = None,
        bn_momentum: float = 0.99,
        symmetric: bool = False,
    ):
        super(AkitaConvBlock2D, self).__init__()

        self.symmetric = symmetric
        self.pool_size = pool_size
        self.pool_type = pool_type

        # Convolution layer
        if conv_type == "separable":
            self.conv = nn.Conv2d(
                in_channels=in_dim,
                out_channels=out_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding="same",
                dilation=dilation_rate,
                bias=(norm_type is None),
            )
        else:
            self.conv = nn.Conv2d(
                in_channels=in_dim,
                out_channels=out_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding="same",
                dilation=dilation_rate,
                bias=(norm_type is None),
            )

        # Normalization layer
        if norm_type == "batch":
            self.norm = nn.BatchNorm2d(out_dim, momentum=bn_momentum)
        elif norm_type == "batch-sync":
            self.norm = nn.SyncBatchNorm(out_dim, momentum=bn_momentum)
        elif norm_type == "layer":
            # self.norm = nn.LayerNorm([out_dim, seq_dim])
            raise NotImplementedError
        else:
            self.norm = None

        # Dropout layer
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Activation function
        self.activation = getattr(F, activation) if activation else None

        # Pooling layer
        if pool_size > 1:
            if pool_type == "softmax":
                self.pool = nn.Softmax(dim=-1)
            else:
                self.pool = nn.MaxPool2d(kernel_size=pool_size)
        else:
            self.pool = None

    def forward(self, x: torch.Tensor):
        x = self.activation(x)
        x = self.conv(x)

        if self.norm:
            x = self.norm(x)

        if self.dropout:
            x = self.dropout(x)

        if self.pool:
            x = self.pool(x)

        if self.symmetric:
            x_t = x.transpose(-2, -1)
            x = (x + x_t) * 0.5

        return x


class AkitaDialatedResidualBlock2D(nn.Module):
    def __init__(
        self,
        in_dim: int,
        mid_dim: int,
        kernel_size: int = 3,
        rate_mult: int = 2,
        dropout: float = 0,
        repeat: int = 1,
        symmetric: bool = True,
        round: bool = False,
        **conv_block_2d_kwargs
    ):
        super(AkitaDialatedResidualBlock2D, self).__init__()
        self.symmetric = symmetric
        self.blocks = nn.ModuleList()

        dilation_rate = 1.0
        for _ in range(repeat):
            sub_block = nn.Sequential(
                AkitaConvBlock2D(
                    in_dim=in_dim,
                    out_dim=mid_dim,
                    kernel_size=kernel_size,
                    dilation_rate=int(np.round(dilation_rate)),
                    **conv_block_2d_kwargs
                ),
                AkitaConvBlock2D(
                    in_dim=mid_dim,
                    out_dim=in_dim,
                    dropout=dropout,
                    **conv_block_2d_kwargs
                ),
            )
            self.blocks.append(sub_block)

            dilation_rate *= rate_mult
            if round:
                dilation_rate = np.round(dilation_rate)

    def forward(self, x: torch.Tensor):
        for _, block in enumerate(self.blocks):
            block_in = x
            x = block(x)

            # residual add
            x = x + block_in

            if self.symmetric:
                x_t = x.transpose(-2, -1)
                x = (x + x_t) * 0.5

        return x
