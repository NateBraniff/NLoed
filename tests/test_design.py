import nloed

import pytest
import nloed as nl
import casadi as cs


def test_design_instantiation():
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
    #instantiate model class
    model = nl.Model(observ,xnames,pnames)

    #define input struct
    input_struct={'Inputs':['Input'],'Grid':[[-1,1]]}

    # generate the optimal approximate (relaxed) design
    design = nl.Design(model,[1],'D',discrete_inputs=input_struct)

    assert design.num_models == 1


