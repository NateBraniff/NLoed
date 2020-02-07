import casadi as cs
#import math as mt
import numpy as np

class model:

    def __init__(self, response,beta, x):
        
        self.beta=beta
        self.x=x
        self.dist=[]
        self.loglik=[]
        self.fim=[]

        #this is a counter to ensure different names for casadi symbols in each response
        counter=0

        for r in response:

            if r[0] =='normal':

                #extract theta's from response list
                mu=r[1][0]
                sigma=r[1][1]

                #create a response symbol
                y=cs.SX.sym('y_'+str(counter),1)
                #create loglikelihood symbolics and function 
                logLikSymbol= -0.5*cs.log(2*cs.pi*sigma) - (y-mu)**2/(2*sigma)
                self.loglik.append( cs.Function('ll_'+str(counter), [y,beta,x], [logLikSymbol]) )
                #create FIM symbolics and function
                dmu_dbeta=cs.jacobian(mu,beta)
                dsigma_dbeta=cs.jacobian(sigma,beta)
                fimSymbol=(dmu_dbeta.T @ dmu_dbeta)/sigma+(dsigma_dbeta.T @ dsigma_dbeta)/sigma**2
                self.fim.append(cs.Function('fim_'+str(counter), [beta,x], [fimSymbol]) )
            else:

                print('error')

        counter=counter+1

    def design(self, beta0, xbounds):

        nparameters=self.beta.size1()

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

            xi_k=cs.SX.sym('xi_'+ str(k))
            lowerboundxi += [0]
            upperboundxi += [1]
            xi_list += [xi_k]
                
            fim_sum= fim_sum + xi_k*self.fim[0](beta0,xgrid[k])
            xi_sum=xi_sum+xi_k

        constr+=[xi_sum-1]
        lowerboundconstr += [0]
        upperboundconstr += [0]
        
        #objective deff (include ds as well)
        objective=-cs.log(cs.det(fim_sum))

        # Create an NLP solver
        prob = {'f': objective, 'x': cs.vertcat(*xi_list), 'g': cs.vertcat(*constr)}
        solver = cs.nlpsol('solver', 'ipopt', prob)

        # Solve the NLP
        sol = solver(x0=xi0, lbx=lowerboundxi, ubx=upperboundxi, lbg=lowerboundconstr, ubg=upperboundconstr)
        xi_opt = sol['x'].full().flatten()

        return xi_opt
