import pytest
import nloed as nl 
import casadi as cs 

@pytest.fixture
def linear_model():
    #define input and parameters
    x = cs.SX.sym('x',2)
    xnames = ['x1','x2']
    p = cs.SX.sym('p',2)
    pnames = ['p1','p2']
    #define predictor symbol
    predictor = 1 + p[0]*x[0] + p[1]*x[1]
    #define stats
    stats = cs.vertcat(predictor, 0.1*predictor**2)
    #create a casadi function
    y = cs.Function('y', [x,p], [stats])
    #observ struct
    observ = [(y,'Normal')]
    #instantiate model class
    linear_model = nl.Model(observ,xnames,pnames)

    return linear_model


@pytest.fixture(params=["Bounds","Grid", "Candidates"])
def discrete_input_struct(request):
    if request.param == "Bounds":
        input_struct={'Inputs':['x1','x2'],
                      'Bounds':[[-1,1],[-1,1]]}
    elif request.param == "Grid":
        input_struct={'Inputs':['x1','x2'],
                      'Grid':[[-1,-1],[1,-1],[-1,1],[1,1]]}
    elif request.param == "Candidates":
        input_struct={'Inputs':['x1','x2'],
                      'Candidates':[[-1,0,1],[-1,0,1]]}

    return input_struct

@pytest.fixture
def discrete_design(linear_model,discrete_input_struct):
    #define input struct
    nominal_param = [1,1]
    objective = 'D'

    # generate the optimal approximate (relaxed) design
    linear_design = nl.Design(linear_model,
                              nominal_param,
                              objective,
                              discrete_inputs=discrete_input_struct)

    return linear_design