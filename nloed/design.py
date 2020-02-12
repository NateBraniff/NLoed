import casadi as cs
import numpy as np

def design( model, beta0, xbounds):

    nparameters=model.beta.size1()

    xgrid=np.linspace(xbounds[0],xbounds[1],101)

    xi0=np.ones(np.shape(xgrid))/xgrid.size
    xi_list=[]
    xi_sum=0

    fim_sum=np.zeros((nparameters,nparameters) )

    constr=[]
    lowerboundxi = []
    upperboundxi = []
    lowerboundconstr = []
    upperboundconstr = []

    for k in range(xgrid.size):

        xi_k=cs.MX.sym('xi_'+ str(k))
        lowerboundxi += [0]
        upperboundxi += [1]
        xi_list += [xi_k]
            
        fim_sum= fim_sum + xi_k*model.fim[0](beta0,xgrid[k])
        xi_sum=xi_sum+xi_k

    constr+=[xi_sum-1]
    lowerboundconstr += [0]
    upperboundconstr += [0]
    
    M = cs.SX.sym('M',nparameters, nparameters)
    R = cs.qr(M)[1]
    det = cs.exp(cs.trace(cs.log(R)))
    qrdeterminant = cs.Function('qrdeterminant',[M],[-det])

    #objective deff (include ds as well)
    #objective=-cs.log(cs.det(fim_sum))
    objective=qrdeterminant(fim_sum)

    # Create an NLP solver
    prob = {'f': objective, 'x': cs.vertcat(*xi_list), 'g': cs.vertcat(*constr)}
    solver = cs.nlpsol('solver', 'ipopt', prob)

    # Solve the NLP
    sol = solver(x0=xi0, lbx=lowerboundxi, ubx=upperboundxi, lbg=lowerboundconstr, ubg=upperboundconstr)
    xi_opt = sol['x'].full().flatten()

    return xi_opt


#Should this be a class?? would make rounding easier

# rough specs
# inputs: model list ,grouping matrix, beta info, past data
# return: ys, xs, weights (not ready for actual experiment because you need to do power/rounding analysis)


# Goals
# can enter either past data or maybe fisher info from past data as starting point to improve an experiment
# grouping response variables that need to be sampled together, paritioning overall sample number across groups
#       (ie 80% and 20%, or if 0% will find optimal weights, perhaps fast weight to specifiy equal weights)
# having x split into two types, grided values with weights xi vs continuous ranges which are constant within a group
# constraints on x, linear and nonlinear
# grid refinment for grided x, starting values for non-grided
# optional optimality criteria, D, Ds (need to group parameters), A, custom (but simple casadi function of fisher info)
# otimality measures for curvature/bias measure, also constrained measure (maybe not possible with bayesian???)
# average designs for multiple models, with priors over the models
# bayesian (normal prior) using sigma points for all or subset of the parameters
# possible support for rounding design into an experiment


# Questions
# Do we scale the FIM?, only D maybe Ds is invariant under linear re-scaling (maybe all rescaling), but not others or custom, however is more numerically stable
# Also do we log the opt(FIM)? for numerical stability, this maybe generally useful
# how to select sigma points
# how to do constrained and curvature/bias optimality measures
