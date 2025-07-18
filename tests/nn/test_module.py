import torch

from torch import nn
from akita_torch.nn.module import StochasticReverseComplement, StochasticShift, OneHot2Indices
from akita_torch.nn.module import (
    OneToTwo,
    ConcatDist2D,
    Cropping2D,
    UpperTri,
    FFNForUpperTri,
    SwitchReverseTriu,
)


def test_stochastic_reverse_complement_training(sample_seq_1hot):
    model = StochasticReverseComplement(p=1.0)
    model.train()
    output, reverse_bool = model(sample_seq_1hot)

    assert output.shape == sample_seq_1hot.shape
    assert isinstance(reverse_bool, bool)
    if reverse_bool:
        expected_output = torch.tensor(
            [
                [
                    [0, 1, 0, 0],  # C
                    [0, 0, 0, 1],  # T
                    [0, 0, 1, 0],  # G
                    [1, 0, 0, 0],  # A
                ]
            ],
            dtype=torch.long,
        )
        assert torch.equal(output, expected_output)
    else:
        assert torch.equal(output, sample_seq_1hot)


def test_stochastic_reverse_complement_inference(sample_seq_1hot):
    model = StochasticReverseComplement(p=1.0)
    model.eval()
    output, reverse_bool = model(sample_seq_1hot)

    assert output.shape == sample_seq_1hot.shape
    assert reverse_bool == False
    assert torch.equal(output, sample_seq_1hot)


def test_stochastic_shift_training(sample_seq_1hot):
    # shift right
    model = StochasticShift(shift_max=2, symmetric=True, given_shift=2, deterministic=True)
    model.train()
    output = model(sample_seq_1hot)
    expected_output = torch.tensor(
        [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]]],  # PAD PAD T C
        dtype=torch.long,
    )
    assert output.shape == sample_seq_1hot.shape
    assert torch.equal(output, expected_output)  # Output should be different due to shifting

    # shift left
    model = StochasticShift(shift_max=2, symmetric=True, given_shift=-2, deterministic=True)
    model.train()
    output = model(sample_seq_1hot)
    expected_output = torch.tensor(
        [
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        ],  # A G PAD PAD
        dtype=torch.long,
    )
    assert output.shape == sample_seq_1hot.shape
    assert torch.equal(output, expected_output)  # Output should be different due to shifting


def test_stochastic_shift_inference(sample_seq_1hot):
    model = StochasticShift(shift_max=2, symmetric=True, given_shift=2, deterministic=True)
    model.eval()
    output = model(sample_seq_1hot)

    assert output.shape == sample_seq_1hot.shape
    assert torch.equal(
        output, sample_seq_1hot
    )  # Output should be the same as input in inference mode


def test_one_hot_to_indices(sample_seq_1hot):
    model = OneHot2Indices()
    output = model(sample_seq_1hot)

    expected_output = torch.tensor(
        [[3, 1, 0, 2]],
        dtype=torch.long,
    )
    assert output.shape == sample_seq_1hot.shape[:-1]
    assert torch.equal(output, expected_output)


def test_one_to_two_concat(sample_channel_input):
    model = OneToTwo(operation="concat")
    output = model(sample_channel_input)

    b, c, s = sample_channel_input.shape
    expected_shape = (b, 2 * c, s, s)
    assert output.shape == expected_shape


def test_one_to_two_mean(sample_channel_input):
    model = OneToTwo(operation="mean")
    output = model(sample_channel_input)

    b, c, s = sample_channel_input.shape
    expected_shape = (b, c, s, s)
    assert output.shape == expected_shape


def test_one_to_two_max(sample_channel_input):
    model = OneToTwo(operation="max")
    output = model(sample_channel_input)

    b, c, s = sample_channel_input.shape
    expected_shape = (b, c, s, s)
    assert output.shape == expected_shape


def test_one_to_two_multiply(sample_channel_input):
    model = OneToTwo(operation="multiply")
    output = model(sample_channel_input)

    b, c, s = sample_channel_input.shape
    expected_shape = (b, c, s, s)
    assert output.shape == expected_shape


def test_one_to_two_multiply1(sample_channel_input):
    model = OneToTwo(operation="multiply1")
    output = model(sample_channel_input)

    b, c, s = sample_channel_input.shape
    expected_shape = (b, c, s, s)
    assert output.shape == expected_shape


def test_concat_dist_2d(sample_channel_input):
    model = nn.Sequential(OneToTwo(operation="mean"), ConcatDist2D())
    output = model(sample_channel_input)

    b, c, s = sample_channel_input.shape
    expected_shape = (b, c + 1, s, s)  # One additional channel for distance matrix
    assert output.shape == expected_shape


def test_cropping_2d():
    model = Cropping2D(cropping=((1, 1), (1, 1)))
    input_tensor = torch.randn((2, 3, 5, 5))  # (batch_size, channels, height, width)
    output = model(input_tensor)

    expected_shape = (2, 3, 3, 3)  # After cropping 1 from each side
    assert output.shape == expected_shape

    # Check specific values to ensure correct cropping
    assert torch.equal(output, input_tensor[:, :, 1:-1, 1:-1])


def test_upper_tri():
    model = UpperTri(diagonal_offset=2)
    input_tensor = torch.tensor(
        [
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                ]
            ]
        ],
        dtype=torch.float,
    )  # Shape: (1, 1, 4, 4)
    output = model(input_tensor)

    expected_output = torch.tensor(
        [[[3, 4, 8]]],
        dtype=torch.float,
    )  # Shape: (1, 1, 3)
    assert output.shape == expected_output.shape
    assert torch.equal(output, expected_output)

    model = UpperTri(diagonal_offset=1)
    input_tensor = torch.tensor(
        [
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                ]
            ]
        ],
        dtype=torch.float,
    )  # Shape: (1, 1, 4, 4)
    output = model(input_tensor)

    expected_output = torch.tensor(
        [[[2, 3, 4, 7, 8, 12]]],
        dtype=torch.float,
    )  # Shape: (1, 1, 6)
    assert output.shape == expected_output.shape
    assert torch.equal(output, expected_output)


def test_ffn_for_upper_tri():
    model = FFNForUpperTri(in_dim=1, out_dim=2, activation="relu")
    input_tensor = torch.tensor(
        [
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                ]
            ]
        ],
        dtype=torch.float,
    )  # Shape: (1, 1, 4, 4)

    # Extract upper triangular part
    upper_tri_model = UpperTri(diagonal_offset=1)
    upper_tri_output = upper_tri_model(input_tensor)  # Shape: (1, 1, 6)

    # Apply FFNForUpperTri
    output = model(upper_tri_output)  # Shape: (1, 6, 2)

    assert output.shape == (1, 6, 2)
    assert torch.all(output >= 0)  # Check ReLU activation

    # Test with linear activation
    model = FFNForUpperTri(in_dim=1, out_dim=2, activation="linear")
    output = model(upper_tri_output)  # Shape: (1, 6, 2)

    assert output.shape == (1, 6, 2)


def test_switch_reverse_triu():
    model = SwitchReverseTriu(diagonal_offset=2)
    input_tensor: torch.Tensor = torch.tensor(
        [
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12],
                [13, 14, 15],
                [16, 17, 18],
            ]
        ],
        dtype=torch.float,
    )  # Shape: (1, 6, 3)

    # Test without reverse
    output = model(input_tensor, reverse=False)
    expected_output = input_tensor
    assert output.shape == expected_output.shape
    assert torch.equal(output, expected_output)

    # Test with reverse
    output = model(input_tensor, reverse=True)
    expected_output = torch.tensor(
        [
            [
                [16, 17, 18],
                [13, 14, 15],
                [7, 8, 9],
                [10, 11, 12],
                [4, 5, 6],
                [1, 2, 3],
            ]
        ],
        dtype=torch.float,
    )  # Shape: (1, 6, 3)
    assert output.shape == expected_output.shape
    assert torch.equal(output, expected_output)
