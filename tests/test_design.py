import pytest

def test_design_instantiation(discrete_design):

    assert discrete_design.num_models == 1
    assert discrete_design.input_dim == 2
    assert discrete_design.observ_dim == 1
    
    


