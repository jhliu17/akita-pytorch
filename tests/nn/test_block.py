import pytest
import torch

from akita_torch.nn.block import AkitaConvBlock1D
from akita_torch.nn.block import AkitaConvTower1D
from akita_torch.nn.block import AkitaDialatedResidualBlock1D
from akita_torch.nn.block import AkitaConvBlock2D
from akita_torch.nn.block import AkitaDialatedResidualBlock2D


@pytest.mark.parametrize(
    "in_dim, out_dim, kernel_size, activation, activation_end, stride, dilation_rate, dropout, conv_type, pool_size, pool_type, norm_type, bn_momentum, residual",
    [
        (
            16,
            32,
            3,
            "relu",
            None,
            1,
            1,
            0.0,
            "standard",
            1,
            "max",
            "batch",
            0.99,
            False,
        ),
        (
            32,
            32,
            3,
            "relu",
            "sigmoid",
            1,
            1,
            0.0,
            "standard",
            1,
            "max",
            "batch",
            0.99,
            True,
        ),
        (
            16,
            32,
            3,
            "relu",
            None,
            1,
            1,
            0.0,
            "standard",
            1,
            "softmax",
            None,
            0.99,
            False,
        ),
    ],
)
def test_akita_conv_block_1d(
    in_dim,
    out_dim,
    kernel_size,
    activation,
    activation_end,
    stride,
    dilation_rate,
    dropout,
    conv_type,
    pool_size,
    pool_type,
    norm_type,
    bn_momentum,
    residual,
):
    block = AkitaConvBlock1D(
        in_dim=in_dim,
        out_dim=out_dim,
        kernel_size=kernel_size,
        activation=activation,
        activation_end=activation_end,
        stride=stride,
        dilation_rate=dilation_rate,
        dropout=dropout,
        conv_type=conv_type,
        pool_size=pool_size,
        pool_type=pool_type,
        norm_type=norm_type,
        bn_momentum=bn_momentum,
        residual=residual,
    )

    x = torch.randn(1, in_dim, 64)
    y = block(x)

    assert y.shape[1] == out_dim
    assert y.shape[2] == (64 // stride)


@pytest.mark.parametrize(
    "in_dim, out_dim_init, out_dim, out_dim_mult, divisible_by, repeat, conv_block_1d_kwargs",
    [
        (
            16,
            32,
            None,
            2,
            1,
            3,
            {
                "kernel_size": 3,
                "activation": "relu",
                "activation_end": None,
                "stride": 1,
                "dilation_rate": 1,
                "dropout": 0.0,
                "conv_type": "standard",
                "pool_size": 1,
                "pool_type": "max",
                "norm_type": "batch",
                "bn_momentum": 0.99,
                "residual": False,
            },
        ),
        (
            16,
            32,
            64,
            None,
            1,
            2,
            {
                "kernel_size": 3,
                "activation": "relu",
                "activation_end": "sigmoid",
                "stride": 1,
                "dilation_rate": 1,
                "dropout": 0.0,
                "conv_type": "standard",
                "pool_size": 1,
                "pool_type": "max",
                "norm_type": "batch",
                "bn_momentum": 0.99,
                "residual": False,
            },
        ),
    ],
)
def test_akita_conv_tower_1d(
    in_dim,
    out_dim_init,
    out_dim,
    out_dim_mult,
    divisible_by,
    repeat,
    conv_block_1d_kwargs,
):
    tower = AkitaConvTower1D(
        in_dim=in_dim,
        out_dim_init=out_dim_init,
        out_dim=out_dim,
        out_dim_mult=out_dim_mult,
        divisible_by=divisible_by,
        repeat=repeat,
        **conv_block_1d_kwargs,
    )

    x = torch.randn(1, in_dim, 64)
    y = tower(x)

    assert y.shape[1] == (out_dim if out_dim else out_dim_init * (out_dim_mult ** (repeat - 1)))
    assert y.shape[2] == 64


@pytest.mark.parametrize(
    "in_dim, mid_dim, kernel_size, rate_mult, dropout, repeat, norm_type, round, conv_block_1d_kwargs",
    [
        (
            16,
            32,
            3,
            2,
            0.0,
            3,
            None,
            False,
            {
                "activation": "relu",
                "activation_end": None,
                "stride": 1,
                "conv_type": "standard",
                "pool_size": 1,
                "pool_type": "max",
                "bn_momentum": 0.99,
                "residual": False,
            },
        ),
        (
            16,
            32,
            3,
            2,
            0.0,
            2,
            "batch",
            True,
            {
                "activation": "relu",
                "activation_end": "sigmoid",
                "stride": 1,
                "conv_type": "standard",
                "pool_size": 1,
                "pool_type": "max",
                "bn_momentum": 0.99,
                "residual": False,
            },
        ),
    ],
)
def test_akita_dialated_residual_block_1d(
    in_dim,
    mid_dim,
    kernel_size,
    rate_mult,
    dropout,
    repeat,
    norm_type,
    round,
    conv_block_1d_kwargs,
):
    block = AkitaDialatedResidualBlock1D(
        in_dim=in_dim,
        mid_dim=mid_dim,
        kernel_size=kernel_size,
        rate_mult=rate_mult,
        dropout=dropout,
        repeat=repeat,
        norm_type=norm_type,
        round=round,
        **conv_block_1d_kwargs,
    )

    x = torch.randn(1, in_dim, 64)
    y = block(x)

    assert y.shape[1] == in_dim
    assert y.shape[2] == 64


@pytest.mark.parametrize(
    "in_dim, out_dim, kernel_size, activation, stride, dilation_rate, dropout, conv_type, pool_size, pool_type, norm_type, bn_momentum, symmetric",
    [
        (
            16,
            32,
            3,
            "relu",
            1,
            1,
            0.0,
            "standard",
            1,
            "max",
            "batch",
            0.99,
            False,
        ),
        (
            32,
            32,
            3,
            "relu",
            1,
            1,
            0.0,
            "standard",
            1,
            "max",
            "batch",
            0.99,
            True,
        ),
        (
            16,
            32,
            3,
            "relu",
            1,
            1,
            0.0,
            "standard",
            1,
            "softmax",
            None,
            0.99,
            False,
        ),
    ],
)
def test_akita_conv_block_2d(
    in_dim,
    out_dim,
    kernel_size,
    activation,
    stride,
    dilation_rate,
    dropout,
    conv_type,
    pool_size,
    pool_type,
    norm_type,
    bn_momentum,
    symmetric,
):
    block = AkitaConvBlock2D(
        in_dim=in_dim,
        out_dim=out_dim,
        kernel_size=kernel_size,
        activation=activation,
        stride=stride,
        dilation_rate=dilation_rate,
        dropout=dropout,
        conv_type=conv_type,
        pool_size=pool_size,
        pool_type=pool_type,
        norm_type=norm_type,
        bn_momentum=bn_momentum,
        symmetric=symmetric,
    )

    x = torch.randn(1, in_dim, 64, 64)
    y = block(x)

    assert y.shape[1] == out_dim
    assert y.shape[2] == 64 // stride
    assert y.shape[3] == 64 // stride


@pytest.mark.parametrize(
    "in_dim, mid_dim, kernel_size, rate_mult, dropout, repeat, symmetric, round, conv_block_2d_kwargs",
    [
        (
            16,
            32,
            3,
            2,
            0.0,
            3,
            True,
            False,
            {
                "activation": "relu",
                "stride": 1,
                "conv_type": "standard",
                "pool_size": 1,
                "pool_type": "max",
                "bn_momentum": 0.99,
            },
        ),
        (
            16,
            32,
            3,
            2,
            0.0,
            2,
            False,
            True,
            {
                "activation": "relu",
                "stride": 1,
                "conv_type": "standard",
                "pool_size": 1,
                "pool_type": "max",
                "bn_momentum": 0.99,
            },
        ),
    ],
)
def test_akita_dialated_residual_block_2d(
    in_dim,
    mid_dim,
    kernel_size,
    rate_mult,
    dropout,
    repeat,
    symmetric,
    round,
    conv_block_2d_kwargs,
):
    block = AkitaDialatedResidualBlock2D(
        in_dim=in_dim,
        mid_dim=mid_dim,
        kernel_size=kernel_size,
        rate_mult=rate_mult,
        dropout=dropout,
        repeat=repeat,
        symmetric=symmetric,
        round=round,
        **conv_block_2d_kwargs,
    )

    x = torch.randn(1, in_dim, 64, 64)
    y = block(x)

    assert y.shape[1] == in_dim
    assert y.shape[2] == 64
    assert y.shape[3] == 64
