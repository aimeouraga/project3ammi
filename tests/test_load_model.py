import pytest
from model.model import load_model



def test_load_model():
    model, device = load_model()
    assert device is not None, "Device should be set"
    assert model is not None, "Model should be loaded"