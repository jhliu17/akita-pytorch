from akita.model import AkitaConfig, Akita


def test_model_forward(sample_1m_seq_1hot):
    config = AkitaConfig()
    model = Akita(config)
    output = model(sample_1m_seq_1hot)
    assert output.shape == (1, 99681, config.output_head_num)
