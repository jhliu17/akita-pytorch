import torch
import pytest


@pytest.fixture
def sample_channel_input():
    # A sample channel (batch_size, channel, sequence_length)
    return torch.randn((2, 4, 10))


@pytest.fixture
def sample_seq_1hot():
    # A sample one-hot encoded DNA sequence of shape (batch_size, sequence_length, 4)
    return torch.tensor(
        [[[0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]]],  # T C A G
        dtype=torch.long,
    )


@pytest.fixture
def sample_seq():
    # A sample DNA sequence of shape (sequence_length,)
    return torch.tensor(
        [1, 2, 3, 1, 2, 1],
        dtype=torch.long,
    )


@pytest.fixture
def sample_dna_seq():
    # A sample DNA sequence of shape (sequence_length,)
    return "ACTGGGTCATACG", "CGTATGACCCAGT"


@pytest.fixture
def sample_seq_emb():
    # A sample DNA sequence of shape (batch_size, sequence_length, 64)
    return torch.randn((2, 10, 64))


@pytest.fixture
def sample_1m_seq_1hot():
    # A sample one-hot encoded DNA sequence of shape (batch_size, sequence_length, 4)
    one_million_seq = 1048576
    t = torch.tensor(
        [[[0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]]],  # T C A G
        dtype=torch.float,
    )

    return t.tile((1, one_million_seq // 4, 1))
