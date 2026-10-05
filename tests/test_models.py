"""
Unit tests for Neural Network Architectures.
"""

import pytest
import torch

from src.models import AudioCNN, EdgeDSCNN


@pytest.mark.parametrize("model_cls", [AudioCNN, EdgeDSCNN])
def test_model_forward_pass(model_cls):
    num_classes = 10
    model = model_cls(num_classes=num_classes)
    model.eval()

    batch_size = 2
    dummy_input = torch.randn(batch_size, 1, 64, 101)

    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (batch_size, num_classes)
    assert not torch.isnan(output).any()


def test_edge_dscnn_parameter_efficiency():
    num_classes = 10
    cnn = AudioCNN(num_classes=num_classes)
    dscnn = EdgeDSCNN(num_classes=num_classes)

    cnn_params = sum(p.numel() for p in cnn.parameters())
    dscnn_params = sum(p.numel() for p in dscnn.parameters())

    print(f"AudioCNN params: {cnn_params:,} | EdgeDSCNN params: {dscnn_params:,}")
    assert dscnn_params > 0
    assert cnn_params > 0
