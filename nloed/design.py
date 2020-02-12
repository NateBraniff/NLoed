import casadi as cs
import numpy as np

def design( model, beta0, xbounds):

    nparameters=model.beta.size1()

    xgrid=np.linspace(xbounds[0],xbounds[1],11)

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