import casadi as cs
import numpy as np

class model:
    """ 
    Add a docstring
    """

    #NOTE: need to rename variables here!!!

    def __init__(self, observationlist, inputnames, paramnames):
        
        #names mustst be unique
        #mustt enforce ordering of parameters in statistics function
        #check for unique names
        if not(len(set(inputnames))  ==  len(inputnames)):
            raise Exception('Model input names must be unique!')
        if not(len(set(paramnames))  ==  len(paramnames)):
            raise Exception('Parameter names must be unique!')
        # extract and store dimensions of the model
        self.NumObserv = len(observationlist)
        self.NumParams = max(observationlist[0][2].size_in(0)) #somewhat unsafe, using max assumes its nx1 or 1xn
        self.NumInputs = max(observationlist[0][2].size_in(1))
        if not(len(set(inputnames))  ==  len(inputnames)):
            raise Exception('Model depends on '+str(self.NumInputs)+' inputs but there are '+str(len(inputnames))+' input names!')
        if not(self.NumParams  ==  len(paramnames)):
            raise Exception('Model depends on '+str(self.NumParams)+' parameters but there are '+str(len(paramnames))+' parameter names!')
        #read names into a dictionary, can be used to link names to index of list functions
        self.InputNameDict = {}
        self.ParamNameDict = {}
        for i in range(self.NumInputs):
            self.InputNameDict[inputnames[i]] = i
        for i in range(self.NumParams):
            self.ParamNameDict[paramnames[i]] = i
        self.ObservNameDict = {}
        #lists to contains needed Casadi functions for evaluation, design and fitting
        self.dist = []
        self.ObservStatistics = []
        self.LogLik = []
        self.FIM = []

        #create symbols for parameters and inputs, needed for function defs below
        ParamSymbols = cs.MX.sym('ParamSymbols',self.NumParams)
        InputSymbols = cs.MX.sym('InputSymbols',self.NumInputs)

        for i in range(self.NumObserv):
            Observation = observationlist[i]
            #extract names of observationlist variables
            if not(Observation[0] in self.ObservNameDict):
                self.ObservNameDict[Observation[0]] = i
            else:
                raise Exception('Observation names must be unique!')
            #create a observationlist symbol
            ObervSymbol = cs.MX.sym(Observation[0],1)
            #store the function for the model (links observationlist distribution parameters to the parameters-of-interest)
            self.ObservStatistics.append(Observation[2])
            if Observation[1]  == 'normal':
                Mean = Observation[2](ParamSymbols,InputSymbols)[0]
                Variance = Observation[2](ParamSymbols,InputSymbols)[1]
                #create LogLikelihood symbolics and function 
                LogLikSymbol =  -0.5*cs.log(2*cs.pi*Variance) - (ObervSymbol-Mean)**2/(2*Variance)
                self.LogLik.append( cs.Function('ll_'+Observation[0], [ObervSymbol,ParamSymbols,InputSymbols], [LogLikSymbol]) )
                #generate derivatives of distribution parameters, ObservStatistics (here Mean and Variance) with respect to parameters-of-interest, Params
                dMean_dParams = cs.jacobian(Mean,ParamSymbols)
                dVariance_dParams = cs.jacobian(Variance,ParamSymbols)
                #create FIM symbolics and function
                FIMSymbol = (dMean_dParams.T @ dMean_dParams)/Variance+(dVariance_dParams.T @ dVariance_dParams)/Variance**2
                self.FIM.append(cs.Function('FIM_'+Observation[0], [ParamSymbols,InputSymbols], [FIMSymbol]) )
            elif Observation[1] == 'poisson':
                #extract ObservStatistics's from observationlist list
                Lambda = Observation[2](ParamSymbols,InputSymbols)[0]
                #create a custom casadi function for doing factorials (needed in poisson LogLikelihood and FIM)
                fact = factorial('fact')
                #store the function in the class so it doesn't go out of scope
                self.___factorialFunc = fact
                #create LogLikelihood symbolics and function 
                LogLikSymbol =  ObervSymbol*cs.log(Lambda)+fact(ObervSymbol)-Lambda
                self.LogLik.append( cs.Function('ll_'+Observation[0], [ObervSymbol,ParamSymbols,InputSymbols], [LogLikSymbol]) )
                #generate derivatives of distribution parameters, ObservStatistics (here Mean and Variance) with respect to parameters-of-interest, Params
                dLambda_dParams = cs.jacobian(Lambda,ParamSymbols)
                #create FIM symbolics and function
                FIMSymbol = (dLambda_dParams.T @ dLambda_dParams)/Lambda
                self.FIM.append(cs.Function('FIM_'+Observation[0], [ParamSymbols,InputSymbols], [FIMSymbol]) )
            elif Observation[0] == 'lognormal':    
                print('Not Implemeneted')
            elif Observation[0] == 'binomial': 
                print('Not Implemeneted')
            elif Observation[0] == 'exponential': 
                print('Not Implemeneted')
            elif Observation[0] == 'gamma': 
                print('Not Implemeneted')
            else:
                print('Unknown Distribution: '+Observation[0])

    def fit(self):
        """
        fit the model to a dataset using maximum likelihood and casadi/IPOPT
        """
        #NOTE: should return, param fit,  beal bias, (profile) logliklihood CI's (not regions)
        #NOTE: leave it to use (and show in docs) how to use loglike to get profile liklihood region
        #NOTE: multiplex this, so it fits multiple datasets
        #NOTE: option for penalized likelihood (not at this time)
        print('Not Implemeneted')

    def sample(self):
        # generate a data sample fromt the model according to a specific design
        #NOTE: multiplex this, so it generate multiple datasets from multiple designs, multiple parameter values
        print('Not Implemeneted')

    #NOTE: should maybe rename this
    def assess(self):
        # assess model/design, returns various estimates of cov, bias, confidence regions/intervals
        # no data: asymptotic: covaraince, beale bias, maybe MSE
        #          sigma point: covariance, bias (using mean) (need to figure out how to do sigma for non-normal data), maybe MSE
        #          monte carlo: covariance, bias, MSE
        # with data: bootstrap; covariance, mean, MSE
        #            (profile) likelihood: intervals/basins not sure how to return
    
    #NOTE: maybe add a basic residual computatino method for goodness of fit assesment?? Or maybe better show how in tutorial but not here

        print('Not Implemeneted')
    
    # UTILITY FUNCTIONS
    def evalFIM(self):
        #NOTE: eval fim at given inputs and dataset
        #NOTE: should this even be here??? how much in model, this isn't data dependent, only design dependent
        print('Not Implemeneted')

    def evalloglik(self):
        #eval the logliklihood with given params and dataset
        print('Not Implemeneted')

    def evalsensitivity(self):
        #eval observation distribution statistic sensitivities at given input and parameter values
        print('Not Implemeneted')


class factorial(cs.Callback):
    def __init__(self, name, opts = {}):
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
        k  =  arg[0]
        cnt = 1
        f = k
        while (k-cnt)>0:
            f = f*(k-cnt)
            cnt = cnt+1
        return [f]


# Full ML fitting, perhaps with penalized likelihood???
# fit assesment, with standardized/weighted residual output, confidence regions via asymptotics (with beale bias), likelihood basins, profile liklihood, sigma point (choose one or maybe two)
# function to generate a prior covariance (that can be fed into design)
# function for easy simulation studies (generate data, with given experiment)