import pytest
import nloed as nl
import casadi as cs


def test_model_instantiation():
    #define input and parameters
    x = cs.SX.sym('x')
    xnames = ['Input']
    p = cs.SX.sym('p')
    pnames = ['Parameter']

    #define predictor symbol
    predictor = p*x
    #define stats
    stats = cs.vertcat(predictor, 0.1)

    #create a casadi function
    y = cs.Function('y',[x,p],[stats])
    #observ struct
    observ = [(y,'Normal')]

    #instantiate nloed model class
    model = nl.Model(observ,xnames,pnames)
    assert model.num_input == 1