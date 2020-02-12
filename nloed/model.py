import casadi as cs
import numpy as np

class model:

    def __init__(self, response,beta, x):
        
        self.beta=beta
        self.x=x
        self.dist=[]
        self.theta=[]
        self.loglik=[]
        self.fim=[]

        #this is a counter to ensure different names for casadi symbols in each response
        counter=0

        for r in response:
            #create a response symbol
            y=cs.MX.sym('y_'+str(counter),1)

            if r[0] =='normal':
                #extract theta's from response list
                mu=r[1][0]
                sigma=r[1][1]
                #create a function for the model (links response distribution parameters to the parameters-of-interest)
                self.theta.append( cs.Function('theta_'+str(counter), [beta,x], [mu,sigma]))
                #create loglikelihood symbolics and function 
                logLikSymbol= -0.5*cs.log(2*cs.pi*sigma) - (y-mu)**2/(2*sigma)
                self.loglik.append( cs.Function('ll_'+str(counter), [y,beta,x], [logLikSymbol]) )
                #generate derivatives of distribution parameters, theta (here mu and sigma) with respect to parameters-of-interest, beta
                dmu_dbeta=cs.jacobian(mu,beta)
                dsigma_dbeta=cs.jacobian(sigma,beta)
                #create FIM symbolics and function
                fimSymbol=(dmu_dbeta.T @ dmu_dbeta)/sigma+(dsigma_dbeta.T @ dsigma_dbeta)/sigma**2
                self.fim.append(cs.Function('fim_'+str(counter), [beta,x], [fimSymbol]) )
            elif r[0]=='poisson':
                #extract theta's from response list
                #lambd=cs.Function('lambd_'+str(counter), [beta,x], [r[1][0]]) 
                lambd=r[1][0]
                #create a function for the model (links response distribution parameters to the parameters-of-interest)
                self.theta.append( cs.Function('theta_'+str(counter), [beta,x], [lambd]))
                #create a custom casadi function for doing factorials (needed in poisson loglikelihood and fim)
                fact=factorial('fact')
                #store the function in the class so it doesn't go out of scope
                self.___factorialFunc=fact
                #create loglikelihood symbolics and function 
                logLikSymbol= y*cs.log(lambd)+fact(y)-lambd
                self.loglik.append( cs.Function('ll_'+str(counter), [y,beta,x], [logLikSymbol]) )
                #generate derivatives of distribution parameters, theta (here mu and sigma) with respect to parameters-of-interest, beta
                dlambd_dbeta=cs.jacobian(lambd,beta)
                #create FIM symbolics and function
                fimSymbol=(dlambd_dbeta.T @ dlambd_dbeta)/lambd
                self.fim.append(cs.Function('fim_'+str(counter), [beta,x], [fimSymbol]) )
            elif r[0]=='lognormal':    
                print('Not Implemeneted')
            elif r[0]=='binomial': 
                print('Not Implemeneted')
            elif r[0]=='exponential': 
                print('Not Implemeneted')
            elif r[0]=='gamma': 
                print('Not Implemeneted')
            else:
                print('Unknown Distribution: '+r[0])

        counter=counter+1


class factorial(cs.Callback):
    def __init__(self, name, opts={}):
        cs.Callback.__init__(self)
        self.construct(name, opts)

    # Number of inputs and outputs
    def get_n_in(self): return 1
    def get_n_out(self): return 1

    # Initialize the object
    def init(self):
        print('initializing object')

    # Evaluate numerically
    def eval(self, arg):
        k = arg[0]
        cnt=1
        f=k
        while (k-cnt)>0:
            f=f*(k-cnt)
            cnt=cnt+1
        return [f]