import torch
import numpy as np
import torch.nn as nn

from torch.nn import functional as F
from typing import Literal, Union


class StochasticReverseComplement(nn.Module):
    """Stochastically reverse complement a one-hot encoded DNA sequence.

    Expected Input
        •	The input seq_1hot is expected to be a one-hot encoded DNA sequence of shape (batch_size, sequence_length, 4).
        •	The last dimension represents the four DNA bases: A (0), C (1), G (2), and T (3) in a one-hot format.

    Behaviour
        1.	Reverse Complementing the Sequence
        •	DNA has complementary base pairs:
        •	A ↔ T
        •	C ↔ G
        •	A one-hot encoded DNA sequence can be reverse-complemented by:
        •	Swapping the last axis (A <-> T, C <-> G)
        •	Reversing the sequence order along the sequence length axis
        2.	Stochastic Selection
        •	During training, the function randomly decides whether to return the original sequence or its reverse complement.
        •	This decision is made by generating a random number (tf.random.uniform(shape=[]) > 0.5).
        •	If True, the function returns the reverse complement; otherwise, it returns the original sequence.
        •	It also returns a boolean flag (reverse_bool), indicating whether the sequence was reversed.
        3.	Inference Mode
        •	During inference (training=False), the function always returns the original sequence (no random flipping).

    Example Input (seq_1hot)
        A one-hot encoded DNA sequence for "ATCG":
        seq_1hot = [
            [[1, 0, 0, 0],  # A
            [0, 1, 0, 0],  # C
            [0, 0, 1, 0],  # G
            [0, 0, 0, 1]]  # T
        ] # Shape: (1, 4, 4)

    Possible Output
        If the sequence is randomly reversed and complemented, the output would correspond to "CGAT":
        rc_seq_1hot = [
            [[0, 1, 0, 0],  # C
            [0, 0, 1, 0],  # G
            [1, 0, 0, 0],  # A
            [0, 0, 0, 1]]  # T
        ]
    """

    def __init__(self, p: float = 0.5):
        super(StochasticReverseComplement, self).__init__()
        self.reverse_prob = 1 - p

    def forward(self, seq_1hot: torch.Tensor):
        if self.training:
            rc_seq_1hot = seq_1hot[..., [3, 2, 1, 0]]  # Reverse complement channel-wise
            rc_seq_1hot = torch.flip(rc_seq_1hot, dims=(1,))  # Reverse sequence order
            reverse_bool = torch.rand(1).item() > self.reverse_prob
            src_seq_1hot = rc_seq_1hot if reverse_bool else seq_1hot
            return src_seq_1hot, reverse_bool
        else:
            return seq_1hot, False


class StochasticShift(nn.Module):
    """Stochastically shift a one-hot encoded DNA sequence.

    Expected Input
        •	seq_1hot: A one-hot encoded DNA sequence of shape (batch_size, sequence_length, 4).
        •	The last dimension (4) represents the A, C, G, T bases in one-hot encoding.

    Behaviour
        1.	Setting the Shift Range (self.augment_shifts)
        •	The shift range is determined by shift_max and whether symmetric shifting is enabled:
        •	If symmetric=True: Shifts can be left or right (from -shift_max to +shift_max).
        •	If symmetric=False: Shifts only occur in the rightward direction (from 0 to +shift_max).
        •	Example:
        •	shift_max=2, symmetric=True → possible shifts: [-2, -1, 0, 1, 2]
        •	shift_max=2, symmetric=False → possible shifts: [0, 1, 2]
        2.	Randomly Selecting a Shift (shift_i)
        •	During training, a random shift value is selected using tf.random.uniform().
        •	This index is used to retrieve the actual shift amount from self.augment_shifts.
        3.	Applying the Shift (shift_sequence)
        •	If the shift is not zero, the function shift_sequence(seq_1hot, shift) is called to shift the sequence.
        •	If the shift is zero, the sequence remains unchanged.
        4.	Handling Padding (self.pad)
        •	The pad parameter determines how the missing bases (due to shifting) are handled. In this implementation, it is set to 'uniform', but its specific behavior depends on how shift_sequence is implemented.
        5.	Inference Mode (training=False)
        •	If not in training mode, the function returns the original sequence unchanged.
    """

    def __init__(
        self,
        shift_max: int = 0,
        symmetric: bool = True,
        pad: int = 0,
        given_shift: int = 0,
        deterministic: bool = False,
    ):
        super(StochasticShift, self).__init__()
        self.shift_max = shift_max
        self.symmetric = symmetric
        self.pad = pad
        self.given_shift = given_shift
        self.deterministic = deterministic

        if self.symmetric:
            self.augment_shifts = torch.arange(-self.shift_max, self.shift_max + 1)
            if not (self.given_shift >= -self.shift_max and self.given_shift <= self.shift_max):
                raise ValueError(
                    f"Given shift {self.given_shift} is out of range for symmetric shifts [-{self.shift_max}, {self.shift_max}]"
                )
        else:
            self.augment_shifts = torch.arange(0, self.shift_max + 1)
            if not (self.given_shift >= 0 and self.given_shift <= self.shift_max):
                raise ValueError(
                    f"Given shift {self.given_shift} is out of range for non-symmetric shifts [0, {self.shift_max}]"
                )

    def shift_sequence(self, seq_1hot: torch.Tensor, shift: int):
        """Shift the sequence along the sequence length axis."""
        pad = seq_1hot.new_zeros((seq_1hot.shape[0], abs(shift), seq_1hot.shape[2]))

        if shift > 0:
            shifted_seq = torch.cat([pad, seq_1hot[:, :-shift, :]], dim=1)
        elif shift < 0:
            shifted_seq = torch.cat([seq_1hot[:, -shift:, :], pad], dim=1)
        else:
            shifted_seq = seq_1hot

        return shifted_seq

    def forward(self, seq_1hot: torch.Tensor):
        if self.training:
            shift_idx = torch.randint(0, len(self.augment_shifts), (1,)).item()
            shift = (
                self.augment_shifts[shift_idx].item()
                if not self.deterministic
                else self.given_shift
            )
            sseq_1hot = self.shift_sequence(seq_1hot, shift) if shift != 0 else seq_1hot
            return sseq_1hot
        else:
            return seq_1hot


class OneHot2Indices(nn.Module):
    """Convert one-hot encoded DNA sequence to indices.

    Expected Input
        •	seq_1hot: A one-hot encoded DNA sequence of shape (batch_size, sequence_length, 4).
        •	The last dimension (4) represents the A, C, G, T bases in one-hot encoding.

    Behaviour
        1.	Conversion to Indices
        •	The function converts the one-hot encoded sequence into a sequence of indices.
        •	The indices represent the DNA bases: A (0), C (1), G (2), T (3).
        •	The conversion is done by taking the argmax along the last dimension of the one-hot encoded sequence.

    Example Input (seq_1hot)
        A one-hot encoded DNA sequence for "ATCG":
        seq_1hot = [
            [[1, 0, 0, 0],  # A
            [0, 1, 0, 0],  # C
            [0, 0, 1, 0],  # G
            [0, 0, 0, 1]]  # T
        ] # Shape: (1, 4, 4)

    Possible Output (seq_indices)
        The corresponding indices for the sequence "ATCG":
        seq_indices = [0, 1, 2, 3]  # Shape: (1, 4)
    """

    def forward(self, seq_1hot: torch.Tensor):
        return torch.argmax(seq_1hot, dim=-1)


class OneToTwo(nn.Module):
    """Transform 1D to 2D with i,j vectors operated on. Input shape: [b, c, s]"""

    def __init__(
        self,
        operation: Literal["concat", "mean", "max", "multiply", "multiply1"] = "mean",
    ):
        super(OneToTwo, self).__init__()
        self.operation = operation.lower()

    def forward(self, oned):
        b, c, s = oned.shape  # [batch, channels, seq_len]

        # Expand along the sequence dimension
        twod1 = oned.unsqueeze(-2).expand(-1, -1, s, -1)  # Shape: [b, c, s, s]
        twod2 = twod1.transpose(-2, -1)  # Swap the last two dims to create i-j interaction

        if self.operation == "concat":
            twod = torch.cat(
                [twod1, twod2], dim=1
            )  # Concatenate along the channel dimension [b, 2c, s, s]

        elif self.operation == "multiply":
            twod = twod1 * twod2  # Element-wise multiplication [b, c, s, s]

        elif self.operation == "multiply1":
            twod = (twod1 + 1) * (twod2 + 1) - 1  # Offset multiplication [b, c, s, s]

        else:
            twod1 = twod1.unsqueeze(1)  # [b, 1, c, s, s]
            twod2 = twod2.unsqueeze(1)  # [b, 1, c, s, s]
            twod = torch.cat([twod1, twod2], dim=1)  # [b, 2, c, s, s]

            if self.operation == "mean":
                twod = twod.mean(dim=1)  # Reduce mean along the new dim [b, c, s, s]
            elif self.operation == "max":
                twod = twod.max(dim=1)[0]  # Reduce max along the new dim [b, c, s, s]

        return twod


class ConcatDist2D(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor):
        batch_size, _, seq_len, _ = x.shape

        # Generate pairwise distance matrix
        pos = torch.arange(seq_len, dtype=x.dtype, device=x.device).view(1, -1)
        matrix_repr1 = pos.expand((-1, seq_len))
        matrix_repr2 = matrix_repr1.t()
        dist = (
            torch.abs(matrix_repr1 - matrix_repr2).unsqueeze(0).unsqueeze(0)
        )  # Shape: (1, 1, seq_len, seq_len)

        # Expand to batch size
        dist = dist.expand((batch_size, -1, -1, -1))  # Shape: (batch_size, seq_len, seq_len, 1)

        return torch.cat([x, dist], dim=1)


class Cropping2D(nn.Module):
    """Crops (or slices) a 2D feature map (height, width) along the spatial dimensions."""

    def __init__(self, cropping: Union[int, tuple] = 0):
        """
        cropping: tuple of 2 tuples ((top_crop, bottom_crop), (left_crop, right_crop))
        """
        super(Cropping2D, self).__init__()
        if not isinstance(cropping, tuple):
            cropping = ((cropping, cropping), (cropping, cropping))
        self.top_crop, self.bottom_crop = cropping[0]
        self.left_crop, self.right_crop = cropping[1]

    def forward(self, x: torch.Tensor):
        return x[:, :, self.top_crop : -self.bottom_crop, self.left_crop : -self.right_crop]


class UpperTri(nn.Module):
    """Extracts the upper triangular portion of a square matrix (or batch of matrices) starting from a given diagonal offset."""

    def __init__(self, diagonal_offset: int = 2):
        super(UpperTri, self).__init__()
        self.diagonal_offset = diagonal_offset

    def forward(self, inputs: torch.Tensor):
        """Assumes input shape is (batch_size, channel_dim, seq_len, seq_len)."""
        batch_size, channel_dim, seq_len, _ = inputs.shape

        # Get upper triangular indices
        triu_indices = torch.triu_indices(
            seq_len, seq_len, offset=self.diagonal_offset, device=inputs.device
        )

        # Flatten the last two dimensions
        inputs_flat = inputs.reshape(batch_size, channel_dim, seq_len * seq_len)

        # Convert 2D indices to 1D flattened indices
        triu_index = triu_indices[0] * seq_len + triu_indices[1]

        # Gather the upper triangular elements
        return torch.gather(
            inputs_flat,
            2,
            triu_index.unsqueeze(0).unsqueeze(0).expand(batch_size, channel_dim, -1),
        )


class FFNForUpperTri(nn.Module):
    """A feed-forward network (FFN) for processing upper triangular matrices."""

    def __init__(self, in_dim: int, out_dim: int, activation: str = "linear"):
        super(FFNForUpperTri, self).__init__()
        self.ffn = nn.Conv1d(in_dim, out_dim, kernel_size=1)
        self.activation = activation

    def forward(self, x: torch.Tensor):
        # Apply the FFN (1D convolution) instead of a linear layer
        x = self.ffn(x)

        out = x
        if self.activation == "linear":
            out = x
        else:
            act_fn = getattr(F, self.activation)
            out = act_fn(x)

        # transpose
        out = out.transpose(1, 2)
        return out


class SwitchReverseTriu(nn.Module):
    def __init__(self, diagonal_offset: int):
        super(SwitchReverseTriu, self).__init__()
        self.diagonal_offset = diagonal_offset

    def forward(self, x: torch.Tensor, reverse: bool):
        """
        x: Tensor of shape (batch_size, ut_len, head_num)
        reverse: booleaning whether the x is reversed complement the sequence
        """
        out = x
        if reverse:
            batch_size, ut_len, head_num = x.shape

            # Infer original sequence length
            seq_len = int(np.sqrt(2 * ut_len + 0.25) - 0.5) + self.diagonal_offset

            # Get upper triangular indices
            ut_indices = torch.triu_indices(seq_len, seq_len, self.diagonal_offset, device=x.device)
            if len(ut_indices[0]) != ut_len:
                raise ValueError(
                    f"Failed to infer sequence length from upper triangular indices. Expected {ut_len} but got {len(ut_indices[0])}."
                )

            # Construct a matrix of upper triangular indexes
            mat_ut_indexes = x.new_zeros((seq_len, seq_len), dtype=torch.long)
            mat_ut_indexes[ut_indices[0], ut_indices[1]] = torch.arange(ut_len, device=x.device)

            # Create masks for upper and lower triangular parts
            mask_ut = x.new_zeros((seq_len, seq_len), dtype=torch.bool)
            mask_ut[ut_indices[0], ut_indices[1]] = True
            mask_ld = ~mask_ut

            # Construct symmetric index matrix
            mat_indexes = mat_ut_indexes + (mask_ld * mat_ut_indexes.T)

            # Reverse complement
            # mat_rc_indexes = mat_indexes[::-1, ::-1]
            mat_rc_indexes = torch.flip(mat_indexes, dims=(0, 1))

            # Extract reordered upper triangular indices
            rc_ut_order = mat_rc_indexes[ut_indices[0], ut_indices[1]]

            # Gather the elements based on the reordered indices
            out = x.gather(
                1,
                rc_ut_order.unsqueeze_(0).unsqueeze_(-1).expand(batch_size, -1, head_num),
            )
        else:
            out = x

        return out
