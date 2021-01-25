import pytest
import nloed as nl 
import casadi as cs 

@pytest.fixture
def linear_model():
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
    linear_model = nl.Model(observ,xnames,pnames)

    return linear_model

@pytest.fixture
def linear_design(linear_model):
    #define input struct
    input_struct={'Inputs':['Input'],'Grid':[[-1,1]]}

    # generate the optimal approximate (relaxed) design
    linear_design = nl.Design(linear_model,[1],'D',discrete_inputs=input_struct)

    return linear_design