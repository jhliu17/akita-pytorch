import torch

from torch import nn
from dataclasses import dataclass
from .nn.block import (
    AkitaConvBlock1D,
    AkitaConvBlock2D,
    AkitaConvTower1D,
    AkitaDialatedResidualBlock1D,
    AkitaDialatedResidualBlock2D,
)
from .nn.module import (
    StochasticReverseComplement,
    StochasticShift,
    OneToTwo,
    ConcatDist2D,
    Cropping2D,
    UpperTri,
    FFNForUpperTri,
    SwitchReverseTriu,
)


@dataclass
class AkitaConfig:
    output_head_num: int = 5
    target_crop: int = 32
    diagonal_offset: int = 2
    augment_rc: bool = True
    augment_shift: int = 11
    activation: str = "relu"
    norm_type: str = "batch"
    bn_momentum: float = 0.9265


class Akita(nn.Module):
    def __init__(self, config: AkitaConfig):
        super(Akita, self).__init__()

        if config.augment_rc:
            self.augment_rc = StochasticReverseComplement()
        else:
            self.augment_rc = None

        if config.augment_shift > 0:
            self.augment_sh = StochasticShift(config.augment_shift)
        else:
            self.augment_sh = None

        self.common_kwargs = {
            "activation": config.activation,
            "norm_type": config.norm_type,
            "bn_momentum": config.bn_momentum,
        }
        self.trunk = nn.Sequential(
            AkitaConvBlock1D(
                in_dim=4, out_dim=96, kernel_size=11, pool_size=2, **self.common_kwargs
            ),
            AkitaConvTower1D(
                96,
                out_dim_init=96,
                out_dim_mult=1,
                kernel_size=5,
                pool_size=2,
                repeat=10,
                **self.common_kwargs,
            ),
            AkitaDialatedResidualBlock1D(
                in_dim=96, mid_dim=48, rate_mult=1.75, repeat=8, dropout=0.4, **self.common_kwargs
            ),
            AkitaConvBlock1D(in_dim=96, out_dim=64, kernel_size=5, **self.common_kwargs),
        )

        self.head = nn.Sequential(
            OneToTwo(operation="mean"),
            ConcatDist2D(),
            AkitaConvBlock2D(
                in_dim=65, out_dim=48, kernel_size=3, symmetric=True, **self.common_kwargs
            ),
            AkitaDialatedResidualBlock2D(
                in_dim=48,
                mid_dim=24,
                kernel_size=3,
                rate_mult=1.75,
                repeat=6,
                dropout=0.1,
                **self.common_kwargs,
            ),
            Cropping2D(config.target_crop),
            UpperTri(config.diagonal_offset),
            FFNForUpperTri(48, config.output_head_num, activation="linear"),
        )
        self.switch = SwitchReverseTriu(diagonal_offset=config.diagonal_offset)

    def forward(self, x: torch.Tensor):
        # the x is in shape of (batch_size, input_seq_length, 4)
        if self.augment_rc is not None:
            x, reverse_bool = self.augment_rc(x)
        if self.augment_sh is not None:
            x = self.augment_sh(x)

        # transpose to (1, 4, input_seq_length) for conv
        x = x.transpose(1, 2)
        x = self.trunk(x)
        x = self.head(x)
        x = self.switch(x, reverse_bool)
        return x
