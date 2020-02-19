import casadi as cs
import numpy as np

class model:

    def __init__(self, response,xnames,betanames):
        
        #names must be unique
        #must enforce ordering of parameters in theta function
        #check for unique names
        if not(len(set(xnames)) == len(xnames)):
            raise Exception('X names must be unique!')
        if not(len(set(betanames)) == len(betanames)):
            raise Exception('Parameter names must be unique!')
        # extract and store dimensions of the model
        self.Ny=len(response)
        self.Nb=max(response[0][2].size_in(0)) #somewhat unsafe, using max assumes its nx1 or 1xn
        self.Nx=max(response[0][2].size_in(1))
        if not(len(set(xnames)) == len(xnames)):
            raise Exception('Model depends on '+str(self.Nx)+' inputs but there are '+str(len(xnames))+' input names!')
        if not(self.Nb == len(betanames)):
            raise Exception('Model depends on '+str(self.Nb)+' parameters but there are '+str(len(betanames))+' parameter names!')
        #read names into a dictionary, can be used to link names to index of list functions
        self.xdict={}
        self.betadict={}
        for i in range(self.Nx):
            self.xdict[xnames[i]]=i
        for i in range(self.Nb):
            self.betadict[betanames[i]]=i
        self.ydict={}
        #lists to contains needed Casadi functions for evaluation, design and fitting
        self.dist=[]
        self.theta=[]
        self.loglik=[]
        self.fim=[]

        #create symbols for parameters and inputs, needed for function defs below
        beta=cs.MX.sym('beta',self.Nb)
        x=cs.MX.sym('x',self.Nx)

        for i in range(self.Ny):
            r=response[i]
            #extract names of response variables
            if not(r[0] in self.ydict):
                self.ydict[r[0]]=i
            else:
                raise Exception('Y names must be unique!')
            #create a response symbol
            y=cs.MX.sym(r[0],1)
            #store the function for the model (links response distribution parameters to the parameters-of-interest)
            self.theta.append(r[2])
            if r[1] =='normal':
                mu=r[2](beta,x)[0]
                sigma=r[2](beta,x)[1]
                #create loglikelihood symbolics and function 
                logLikSymbol= -0.5*cs.log(2*cs.pi*sigma) - (y-mu)**2/(2*sigma)
                self.loglik.append( cs.Function('ll_'+r[0], [y,beta,x], [logLikSymbol]) )
                #generate derivatives of distribution parameters, theta (here mu and sigma) with respect to parameters-of-interest, beta
                dmu_dbeta=cs.jacobian(mu,beta)
                dsigma_dbeta=cs.jacobian(sigma,beta)
                #create FIM symbolics and function
                fimSymbol=(dmu_dbeta.T @ dmu_dbeta)/sigma+(dsigma_dbeta.T @ dsigma_dbeta)/sigma**2
                self.fim.append(cs.Function('fim_'+r[0], [beta,x], [fimSymbol]) )
            elif r[1]=='poisson':
                #extract theta's from response list
                lambd=r[2](beta,x)[0]
                #create a custom casadi function for doing factorials (needed in poisson loglikelihood and fim)
                fact=factorial('fact')
                #store the function in the class so it doesn't go out of scope
                self.___factorialFunc=fact
                #create loglikelihood symbolics and function 
                logLikSymbol= y*cs.log(lambd)+fact(y)-lambd
                self.loglik.append( cs.Function('ll_'+r[0], [y,beta,x], [logLikSymbol]) )
                #generate derivatives of distribution parameters, theta (here mu and sigma) with respect to parameters-of-interest, beta
                dlambd_dbeta=cs.jacobian(lambd,beta)
                #create FIM symbolics and function
                fimSymbol=(dlambd_dbeta.T @ dlambd_dbeta)/lambd
                self.fim.append(cs.Function('fim_'+r[0], [beta,x], [fimSymbol]) )
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


# IRWLS fitting
# fit assesment, with standardized/weighted residual output, confidence regions via asymptotics (with beale bias), likelihood  basins, profile liklihood, sigma point (choose one or maybe two)
# function to generate a prior covariance (that can be fed into design)
# function for easy simulation studies (generate data, with given experiment)