import pytest

def test_model_instantiation(linear_model):

    assert linear_model.num_input == 1