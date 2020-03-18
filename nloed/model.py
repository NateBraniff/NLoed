import casadi as cs
import numpy as np
import copy as cp
from scipy import stats as st
import matplotlib.pyplot as plt


class model:
    """ 
    Add a docstring
    """

    #Moderate:  implement ML fitting
    #           implement data sampling
    #Difficult: implement various cov/bias assesment methods, also (profile) likelihood intervals/plots
    #           add other distributions binomial/bernouli, lognormal, gamma, exponential, weibull etc., negative binomial
    #           implement plotting function (move to utility.py?)
    #NOTE: How to handle profile that goes negative for strictly positive parameter values???? perhaps error out and suggest reparameterization???
    #NOTE: can we allow custom pdf's (with some extra work by the user)
    #NOTE: could to A and E optimality (E using SDP form: min t subject to I<t*I, or maybe even just a max(eig(A)) if its smooth)
    #NOTE: https://math.stackexchange.com/questions/269723/largest-eigenvalue-semidefinite-programming
    #NOTE: Data/design/experiment objects may need deep copies so internal lists aren't shared??
    #NOTE: do we really need scipy and numpy?
    #names must be unique
    #must enforce ordering of parameters in statistics function

    def __init__(self, observationlist, inputnames, paramnames):
        
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
        self.InputNameList = inputnames
        self.ParamNameList= paramnames
        for i in range(self.NumInputs):
            self.InputNameDict[inputnames[i]] = i
        for i in range(self.NumParams):
            self.ParamNameDict[paramnames[i]] = i
        self.ObservNameDict = {}
        self.ObservNameList = []
        #lists to contains needed Casadi functions for evaluation, design and fitting
        self.Dist = []
        self.StatisticModel = []
        self.Model = []
        self.LogLik = []
        self.FIM = []

        #create symbols for parameters and inputs, needed for function defs below
        ParamSymbols = cs.SX.sym('ParamSymbols',self.NumParams)
        InputSymbols = cs.SX.sym('InputSymbols',self.NumInputs)

        for i in range(self.NumObserv):
            Observation = observationlist[i]
            #store the distribution type for later
            self.Dist.append(Observation[1])
            #extract names of observationlist variables
            if not(Observation[0] in self.ObservNameDict):
                self.ObservNameDict[Observation[0]] = i
                self.ObservNameList.append(Observation[0])
            else:
                raise Exception('Observation names must be unique!')
            #create a observationlist symbol
            ObervSymbol = cs.SX.sym(Observation[0],1)
            #store the function for the model (links observationlist distribution parameters to the parameters-of-interest)
            self.StatisticModel.append(Observation[2])
            if Observation[1]  == 'Normal':
                #get the distribution statistics
                Mean = Observation[2](ParamSymbols,InputSymbols)[0]
                Variance = Observation[2](ParamSymbols,InputSymbols)[1]
                #create LogLikelihood symbolics and function 
                LogLikSymbol =  -0.5*cs.log(2*cs.pi*Variance) - (ObervSymbol-Mean)**2/(2*Variance)
                self.LogLik.append( cs.Function('ll_'+Observation[0], [ObervSymbol,ParamSymbols,InputSymbols], [LogLikSymbol]) )
                #generate derivatives of distribution parameters, StatisticModel (here Mean and Variance) with respect to parameters-of-interest, Params
                dMean_dParams = cs.jacobian(Mean,ParamSymbols)
                dVariance_dParams = cs.jacobian(Variance,ParamSymbols)
                #create FIM symbolics and function
                FIMSymbol = (dMean_dParams.T @ dMean_dParams)/Variance+(dVariance_dParams.T @ dVariance_dParams)/Variance**2
                self.FIM.append(cs.Function('FIM_'+Observation[0], [ParamSymbols,InputSymbols], [FIMSymbol]) )
            elif Observation[1] == 'Poisson':
                #get the distribution statistic
                Lambda = Observation[2](ParamSymbols,InputSymbols)[0]
                #create a custom casadi function for doing factorials (needed in poisson LogLikelihood and FIM)
                fact = factorial('fact')
                #store the function in the class so it doesn't go out of scope
                self.___factorialFunc = fact
                #create LogLikelihood symbolics and function 
                LogLikSymbol =  ObervSymbol*cs.log(Lambda)+fact(ObervSymbol)-Lambda
                self.LogLik.append( cs.Function('ll_'+Observation[0], [ObervSymbol,ParamSymbols,InputSymbols], [LogLikSymbol]) )
                #generate derivatives of distribution parameters, StatisticModel (here Mean and Variance) with respect to parameters-of-interest, Params
                dLambda_dParams = cs.jacobian(Lambda,ParamSymbols)
                #create FIM symbolics and function
                FIMSymbol = (dLambda_dParams.T @ dLambda_dParams)/Lambda
                self.FIM.append(cs.Function('FIM_'+Observation[0], [ParamSymbols,InputSymbols], [FIMSymbol]) )
            elif Observation[1] == 'Lognormal':    
                print('Not Implemeneted')
            elif Observation[1] == 'Binomial': 
                print('Not Implemeneted')
            elif Observation[1] == 'Exponential': 
                print('Not Implemeneted')
            elif Observation[1] == 'Gamma': 
                print('Not Implemeneted')
            else:
                raise Exception('Unknown Distribution: '+Observation[1])

    def fit(self,datasets,paramstart,paramconstraints=None,opts={'Intervals':False,'Contours':False,'Profiles':False}):
        """
        fit the model to a dataset using maximum likelihood and casadi/IPOPT
        """
        #NOTE: need to check for asymptotets/neg constraint violation in profile plot, set max steps/through error suggesting transformation respectively
        #NOTE: code write this so that ML is a casadi function of inputs, params, and observations just once, then pass datasets to generate symbols within loop and solve, may be faster
        #NOTE: add some print statments to provide user with progress status
        #NOTE: could solve multiplex simultaneosly (one big opt problem) or sequentially (more flexible, perhaps slower, leaning towards this for now)

        #NOTE: (NOT NOW) perhaps allow for bias-reducing penalized likelihood (based on Firth 1993, not sure if generalizes to fully nonlinear problems, seems to use FIM
        #NOTE: (NOT NOW) allow for parameter constraints to be passed (once FIM supports constraints)

        #NOTE: (NO WILL INTEGRATE)leave it to user (and show in docs) how to use loglike to get profile liklihood region

        #NOTE: (DONE BUT WILL BE LESS OFTEN USED DUE TO MONTE CARLOE COV/BIAS/MSE) multiplex this, so it fits multiple datasets

        #profile liks:
        # use bisection condition to find F level boundary
        # store LL value at grid point, endpoints=CI's store in list
        # plotting: LL profiles per parameter go on diagonal plots, profile traces for each parameter go on off-diagonal lower plots

        if not(isinstance(datasets, list)):
            DesignSet=[[datasets]]
        elif not(isinstance(datasets[0], list)):
            DesignSet=[datasets]
        else:
            DesignSet=datasets


        #loop over different designs
        if opts['Type']=='old':
            DesignFitParameterList=[]
        else:
            ParamSymbolsList=[]
            StartParams=[]
            DesignLogLikFunctionList=[]
            TotalLogLik=0
        for e in range(len(DesignSet)):
            Datasets=DesignSet[e]
            if opts['Type']=='old':
                ReplicateFitParameterList=[]
            else:
                ReplicateLogLikFunctionList=[]
            #loop over replicates within each design
            for r in range(len(Datasets)):
                #set up the logliklihood symbols for given design and replicate
                Data=Datasets[r]
                ParamFitSymbols = cs.SX.sym('ParamFitSymbols'+'_'+str(e)+str(r),self.NumParams)
                LogLik=0
                for i in range(len(Data['Inputs'])):
                    InputRow = Data['Inputs'][i]
                    for j in range(self.NumObserv):
                        Observations = Data['Observation'][i][j]
                        for k in range(len(Observations)):
                            LogLik-=self.LogLik[j](Observations[k],ParamFitSymbols,InputRow)

                if opts['Type']=='old':
                    #NOTE: should be checking solution for convergence, should allow use to pass options to ipopt
                    # Create an IPOPT solver for maximum likelihood problem
                    IPOPTProblemStructure = {'f': LogLik, 'x': ParamFitSymbols}#, 'g': cs.vertcat(*OptimConstraints)
                    IPOPTSolver = cs.nlpsol('solver', 'ipopt', IPOPTProblemStructure,{'ipopt.print_level':0,'print_time':False})
                    # Solve the NLP fitting problem with IPOPT call
                    #print('Begining optimization...')
                    IPOPTSolutionStruct = IPOPTSolver(x0=paramstart)#, lbx=[], ubx=[], lbg=[], ubg=[]
                    FitParameters = list(IPOPTSolutionStruct['x'].full().flatten())
                    ReplicateFitParameterList.append(FitParameters)
                else:
                    LogLikFunc = cs.Function('LogLik_'+str(e)+'_'+str(r), [ParamFitSymbols], [LogLik])
                    ReplicateLogLikFunctionList.append(LogLikFunc)
                    #set up the logliklihood symbols for given design and replicate
                    ParamSymbolsList.append(ParamFitSymbols)
                    StartParams.extend(paramstart)
                    TotalLogLik+=LogLik

            if opts['Type']=='old': 
                DesignFitParameterList.append(ReplicateFitParameterList)
            else:
                DesignLogLikFunctionList.append(ReplicateLogLikFunctionList)

        if opts['Type']=='new':
            #NOTE: should be checking solution for convergence, should allow use to pass options to ipopt
            # Create an IPOPT solver for maximum likelihood problem
            IPOPTProblemStructure = {'f': TotalLogLik, 'x': cs.vertcat(*ParamSymbolsList)}#, 'g': cs.vertcat(*OptimConstraints)
            IPOPTSolver = cs.nlpsol('solver', 'ipopt', IPOPTProblemStructure,{'ipopt.print_level':0,'print_time':False})
            # Solve the NLP fitting problem with IPOPT call
            #print('Begining optimization...')
            IPOPTSolutionStruct = IPOPTSolver(x0=StartParams)#, lbx=[], ubx=[], lbg=[], ubg=[]
            FitParameters = list(IPOPTSolutionStruct['x'].full().flatten())

            DesignFitParameterList=[]
            for e in range(len(DesignSet)):
                ReplicateFitParameterList=[]
                for r in range(len(Datasets)):
                    FitPramaSet= FitParameters[:self.NumParams]
                    del FitParameters[:self.NumParams]
                    ReplicateFitParameterList.append(FitPramaSet)
                DesignFitParameterList.append(ReplicateFitParameterList)


        # #NOTE: things could be do here to speed up, bates/watts 1988 interpolation of contours, kademan/bates 1990 adaptive profile stepping (derivative of ll and delta theta from last step)
        # if opts['Confidence']=='Intervals' or opts['Confidence']=='Profiles' or opts['Confidence']=='Contours':
        #     #get the required confidence level #NOTE: make this a user passed option
        #     ConfidenceLevel = 0.95 
        #     #compute the corresponding chi-sqr percentile
        #     ChiSquaredLevel = st.chi2.ppf(ConfidenceLevel, self.NumParams) #NOTE: unsure of degree's of freedom here!!
        #     #set the (very) approximate increment size for steps in likelihood to reach desired precentile starting from LR=0
        #     MaxLikStep=ChiSquaredLevel/100 #NOTE: default to 100 but could be user provided
        #     #set the step size search tolerance (relative to size of parameter)
        #     ProfileTol=1e-2


        #     #set up a Casadi funciton for the total loglikelihood
        #     TotalLogLikFunc = cs.Function('TotalLogLik_'+str(e)+'_'+str(r), [ParamFitSymbols], [TotalLogLik])
        #     #compute the total likelihood value at the fit parameters
        #     LogLikAtEstimate=TotalLogLikFunc(FitParameters)
        #     #creat lists to store the profile curves, loglik profiles and intervals for each parameter
        #     CurveList=[]
        #     ProfileList=[]
        #     CIList=[]
        #     #loop over each parameter
        #     for p in range(self.NumParams):
        #         #create a symbole for the loglik ratio
        #         LikRatioSymbol = 2*(TotalLogLikFunc(ParamFitSymbols)-LogLikAtEstimate)
        #         #create a symbole for the loglik ratio dervative with respect to parameters, used to check for two big a step size
        #         LikeRatioJacobianSymbol=cs.jacobian(LikRatioSymbol,ParamFitSymbols)
        #         #create a function for the loglik ratio and its derivative
        #         LikRatioFunc = cs.Function('ProfileLikRatioFunc_'+str(p), [ParamFitSymbols], [LikRatioSymbol])
        #         LikRatioJacobianFunc = cs.Function('ProfileLikeRatioJacobianFunc_'+str(p), [ParamFitSymbols], [LikeRatioJacobianSymbol])

        #         (UpperBound,UpperCurve,UpperProfile)=self.__profiletrace(p,True,FitParameters,ParamFitSymbols,LikRatioFunc,LikRatioJacobianFunc,ChiSquaredLevel,ProfileTol,MaxLikStep)
        #         (LowerBound,LowerCurve,LowerProfile)=self.__profiletrace(p,False,FitParameters,ParamFitSymbols,LikRatioFunc,LikRatioJacobianFunc,ChiSquaredLevel,ProfileTol,MaxLikStep)

        #         CIList.append([LowerBound, UpperBound])
        #         CurveList.append(list(reversed(LowerCurve))+[FitParameters]+UpperCurve)
        #         ProfileList.append(list(reversed(LowerProfile))+[0]+UpperProfile)

        #     if opts['Confidence']=='Profiles':
        #         plt.figure()
        #         for p1 in range(self.NumParams):
        #             for p2 in range(p1,self.NumParams):
        #                 print(str(p1)+'_'+str(p2))
        #                 if p1==p2:
        #                     X=[CurveList[p1][ind][p1] for ind in range(len(CurveList[p1]))]
        #                     Y=ProfileList[p1]

        #                     plt.subplot(self.NumParams, self.NumParams, p2*self.NumParams+p1+1)
        #                     plt.plot(X, Y)
        #                     plt.xlabel(self.ParamNameList[p1])
        #                     plt.ylabel('LogLik Ratio')
        #                 else:

        #                     if opts['Confidence']=='Profiles':
        #                         plt.subplot(self.NumParams, self.NumParams, p2*self.NumParams+p1+1)
        #                         X1=[CurveList[p1][ind][p1] for ind in range(len(CurveList[p1]))]
        #                         Y1=[CurveList[p1][ind][p2] for ind in range(len(CurveList[p1]))]
        #                         plt.plot(X1, Y1,label=self.ParamNameList[p1]+'profile')

        #                         X2=[CurveList[p2][ind][p1] for ind in range(len(CurveList[p2]))]
        #                         Y2=[CurveList[p2][ind][p2] for ind in range(len(CurveList[p2]))]
        #                         plt.plot(X2, Y2,label=self.ParamNameList[p2]+'profile')

        #                         plt.legend()
        #                         plt.xlabel(self.ParamNameList[p1])
        #                         plt.ylabel(self.ParamNameList[p2])

        #                     if opts['Confidence']=='Contours':
        #                         NumGridPoints=50
        #                         X=list(np.linspace(CIList[p1][0], CIList[p1][1],NumGridPoints))
        #                         Y=list(np.linspace(CIList[p2][0], CIList[p2][1],NumGridPoints))
        #                         Z=[]
        #                         for yval in Y:
        #                             Zrow=[]
        #                             for xval in X:
        #                                 Zrow.append()
                                                
                    # plt.show()

        if not(isinstance(datasets, list)):
            return DesignFitParameterList[0][0]
        elif not(isinstance(datasets[0], list)):
            return DesignFitParameterList[0]
        else:
            return DesignFitParameterList

    def __profiletrace(self,paramind,increasebool,parametervalues,parametersymbols,loglikratiofunc,dloglikratiofunc,chisqrlevel,profiletol,maxlikstep):
        
        #set the trace direction
        if increasebool:
            Direction=1
        else:
            Direction=-1
        #creart a list for the parameter's profile curve, contains parameter points along the profile
        CurrentProfileCurve=[]
        #create a list to store loglik ratio values along the profile curve
        CurrentProfile=[]
        #set the starting step size relative to the parameter value magnitude
        StepSize=abs(parametervalues[paramind])*profiletol
        #set the current LR to 0, as we start at the fitted value
        CurrentRatio=0
        #set the last ratio to 0, this is used in step size computation
        LastRatio=0
        #set the profile fixed value of the target parameter to the fit value, this is incremented in the loops below
        CurrentFixedParam=parametervalues[paramind]
        #set the current point on the profile curve to the fit point
        CurrentCurvePoint=parametervalues
        #compute one half of the profile, desending from the fitted value
        while CurrentRatio<chisqrlevel:
            #increment the current fixed value of the target parameter
            CurrentFixedParam=CurrentFixedParam+Direction*StepSize
            #create a list casadi symbols for the parameter vector with the target parameter fixed
            LeaveOneFixedParamVec=cs.vertcat(*[parametersymbols[ind] if not(ind==paramind) else cs.SX(CurrentFixedParam) for ind in range(self.NumParams)])
            #create a list of casadi symbols just of the nuisance parameters
            NuisanceParamSymbols=cs.vertcat(*[parametersymbols[ind] for ind in range(self.NumParams) if not(ind==paramind)])
            #generate symbols for the LR
            ProfileOptimSymbol=loglikratiofunc(LeaveOneFixedParamVec)
            # Create an IPOPT solver to optimize the nuisance parameters
            IPOPTProblemStructure = {'f': ProfileOptimSymbol, 'x': NuisanceParamSymbols}#, 'g': cs.vertcat(*OptimConstraints)
            IPOPTSolver = cs.nlpsol('solver', 'ipopt', IPOPTProblemStructure,{'ipopt.print_level':0,'print_time':False})
            # Solve the NLP problem with IPOPT call
            StartingNuisanceParams=[CurrentCurvePoint[ind] for ind in range(self.NumParams) if not(ind==paramind)]
            IPOPTSolutionStruct = IPOPTSolver(x0=StartingNuisanceParams)#, lbx=[], ubx=[], lbg=[], ubg=[]
            #update profile curve
            OptimNuisanceParams= list(IPOPTSolutionStruct['x'].full().flatten())
            LastCurvePoint=CurrentCurvePoint
            CurrentCurvePoint=cp.deepcopy(OptimNuisanceParams)
            CurrentCurvePoint.insert(paramind,CurrentFixedParam)
            CurrentProfileCurve.append(CurrentCurvePoint)
            #uptdate profile
            LastRatio=CurrentRatio
            CurrentRatio= IPOPTSolutionStruct['f'].full()[0][0]
            CurrentProfile.append(CurrentRatio)
            #compute step
            dLikRatio_dParams=list(dloglikratiofunc(CurrentCurvePoint).full().flatten())
            NormedDeltaParams=[(CurrentCurvePoint[ind]-LastCurvePoint[ind])/abs(CurrentCurvePoint[paramind]-LastCurvePoint[paramind]) for ind in range(self.NumParams)]
            StepSize =min( maxlikstep/sum(NormedDeltaParams[ind] * dLikRatio_dParams[ind] for ind in range(self.NumParams)),abs(parametervalues[paramind])*profiletol)
        IterpolatedBound=CurrentFixedParam+StepSize*(CurrentRatio-chisqrlevel)/(CurrentRatio-LastRatio)

        return [IterpolatedBound,CurrentProfileCurve,CurrentProfile]


    def sample(self,experiments,parameters,replicates=1):
        # generate a data sample fromt the model according to a specific design
        #NOTE: multiplex multiple parameter values??
        #NOTE: actually maybe more important to be able to replicate designs N times

        if not(isinstance(experiments, list)):
            ExperimentList=[experiments]
        else:
            ExperimentList=experiments

        Designset=[]
        for e in range(len(ExperimentList)):
            CurrentExperiment=ExperimentList[e]
            DataFormat=cp.deepcopy(CurrentExperiment)
            del DataFormat['Count']
            DataFormat['Observation']=[]
            Datasets=[]
            for r in range(replicates):
                CurrentData=cp.deepcopy(DataFormat)
                for i in range(len(CurrentExperiment['Inputs'])):
                    InputRow = CurrentExperiment['Inputs'][i]
                    ObservRow=[]
                    for j in range(self.NumObserv):
                        ObservCount = CurrentExperiment['Count'][i][j]
                        Statistics=self.StatisticModel[j](parameters,InputRow)
                        if self.Dist[j] == 'Normal':
                            CurrentDataBlock = np.random.normal(Statistics[0], np.sqrt(Statistics[1]), ObservCount).tolist() 
                        elif self.Dist[j] == 'Poisson':
                            CurrentDataBlock = np.random.poisson(Statistics[0]).tolist() 
                        elif self.Dist[j] == 'Lognormal':
                            print('Not Implemeneted')
                        elif self.Dist[j] == 'Binomial':
                            print('Not Implemeneted')
                        elif self.Dist[j] == 'Exponential':
                            print('Not Implemeneted')
                        elif self.Dist[j] == 'Gamma':
                            print('Not Implemeneted')
                        else:
                            raise Exception('Unknown error encountered selecting observation distribution, contact developers')
                        ObservRow.append(CurrentDataBlock)
                    CurrentData['Observation'].append(ObservRow)
                Datasets.append(CurrentData)
            Designset.append(Datasets)
                
        if not(isinstance(experiments, list)):
            if replicates==1:
                return Designset[0][0]
            else:
                return Designset[0]
        else:
            return Designset

    #NOTE: should maybe rename this
    def evaluate(self):
        #maybe this should move to the design class(??)
        #For D (full cov/bias), Ds (partial cov/bias), T separation using the delta method?! but need two models
        # assess model/design, returns various estimates of cov, bias, confidence regions/intervals
        # no data: asymptotic: covaraince, beale bias, maybe MSE
        #          sigma point: covariance, bias (using mean) (need to figure out how to do sigma for non-normal data), maybe MSE
        #          monte carlo: covariance, bias, MSE
        
        print('Not Implemeneted')
        
    def plots(self):
        #FDS plot, standardized variance (or Ds, bayesian equivlant), residuals
        print('Not Implemeneted')
        #NOTE: maybe add a basic residual computation method for goodness of fit assesment?? Or maybe better show how in tutorial but not here
    
    # UTILITY FUNCTIONS
    def getstatistics(self):
        #NOTE: evaluate model, predict y
        #NOTE: also mabye predict error bars based on par cov or past dataset, delta method vs something based on likelihood CI's??
        print('Not Implemeneted')

    def getFIM(self):
        #NOTE: eval fim at given inputs and dataset
        #NOTE: should this even be here??? how much in model, this isn't data dependent, only design dependent
        print('Not Implemeneted')

    def getloglik(self):
        #eval the logliklihood with given params and dataset
        print('Not Implemeneted')

    def getsensitivity(self):
        #eval observation distribution statistic sensitivities at given input and parameter values
        print('Not Implemeneted')



# --------------- Private functions and subclasses ---------------------------------------------



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