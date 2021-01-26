import pytest

def test_model_instantiation(linear_model):

    assert linear_model.num_observ == 1
    assert linear_model.num_input == 2
    assert linear_model.num_param == 2