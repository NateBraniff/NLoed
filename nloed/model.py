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

        ParamSymbolsList=[]
        StartParams=[]
        DesignLogLikFunctionList=[]
        TotalLogLik=0

        ParamArchetypeSymbols = cs.SX.sym('ParamArchetypeSymbols',self.NumParams)
        #loop over different designs
        for e in range(len(DesignSet)):
            Datasets=DesignSet[e]
            ReplicateLogLikFunctionList=[]

            Data=Datasets[0]
            LogLik=0
            SampleCount=0
            Observations = [element for row in Data['Observation'] for group in row for element in group]
            ObservSymbol = cs.SX.sym('ObservSymbol_'+str(e),len(Observations))
            for i in range(len(Data['Inputs'])):
                InputRow = Data['Inputs'][i]
                for j in range(self.NumObserv):
                    for k in range(len(Data['Observation'][i][j])):
                        LogLik-=self.LogLik[j](ObservSymbol[SampleCount],ParamArchetypeSymbols,InputRow)
                        SampleCount+=1
            ArchetypeLogLikFunc = cs.Function('ArchetypeLogLikFunc', [ObservSymbol,ParamArchetypeSymbols], [LogLik])
                
            #loop over replicates within each design
            for r in range(len(Datasets)):
                #NOTE: could abstract below into a Casadi function to avoid input/observ loop on each dataset and replicate
                Data=Datasets[r]
                ParamFitSymbols = cs.SX.sym('ParamFitSymbols'+'_'+str(e)+str(r),self.NumParams)
                Observations = cs.vertcat(*[cs.SX(element) for row in Data['Observation'] for group in row for element in group])
                LogLik=ArchetypeLogLikFunc(Observations,ParamFitSymbols)

                LogLikFunc = cs.Function('LogLik_'+str(e)+'_'+str(r), [ParamFitSymbols], [LogLik])
                ReplicateLogLikFunctionList.append(LogLikFunc)
                #set up the logliklihood symbols for given design and replicate
                ParamSymbolsList.append(ParamFitSymbols)
                StartParams.extend(paramstart)
                TotalLogLik+=LogLik
            DesignLogLikFunctionList.append(ReplicateLogLikFunctionList)

        #NOTE: this approach is much more fragile to separation (glms with discrete response), randomly weakly identifiable datasets
        #NOTE: should be checking solution for convergence, should allow user to pass options to ipopt
        #NOTE: allow bfgs for very large nonlinear fits, may be faster
        # Create an IPOPT solver for maximum likelihood problem
        IPOPTProblemStructure = {'f': TotalLogLik, 'x': cs.vertcat(*ParamSymbolsList)}#, 'g': cs.vertcat(*OptimConstraints)
        IPOPTSolver = cs.nlpsol('solver', 'ipopt', IPOPTProblemStructure,{'ipopt.print_level':5,'print_time':False})
        # Solve the NLP fitting problem with IPOPT call
        #print('Begining optimization...')
        IPOPTSolutionStruct = IPOPTSolver(x0=StartParams)#, lbx=[], ubx=[], lbg=[], ubg=[]
        FitParameters = list(IPOPTSolutionStruct['x'].full().flatten())

        DesignFitParameterList=[]
        DesignCIList=[]
        for e in range(len(DesignSet)):
            ReplicateFitParameterList=[]
            ReplicateCIList=[]
            for r in range(len(Datasets)):
                FitParamSet= FitParameters[:self.NumParams]

                if "Confidence" in opts.keys():
                    if opts['Confidence']=="Contours" or opts['Confidence']=="Profiles":
                        Figure = plt.figure()
                        ReplicateCIList.append(self.__profileplot(FitParamSet,DesignLogLikFunctionList[e][r],Figure,opts))
                        if opts['Confidence']=="Contours":
                            self.__contourplot(FitParamSet,DesignLogLikFunctionList[e][r],Figure,opts)
                    elif opts['Confidence']=="Intervals":
                        ReplicateCIList.append(self.__confidenceintervals(FitParamSet,DesignLogLikFunctionList[e][r],opts))

                del FitParameters[:self.NumParams]
                ReplicateFitParameterList.append(FitParamSet)
            DesignFitParameterList.append(ReplicateFitParameterList)
            DesignCIList.append(ReplicateCIList)

        if "Confidence" in opts.keys() and (opts['Confidence']=="Contours" or opts['Confidence']=="Profiles"):
            plt.show()

        if not(isinstance(datasets, list)):
            DesignFitParameterList = DesignFitParameterList[0][0]
            DesignCIList = DesignCIList[0][0]
        elif not(isinstance(datasets[0], list)):
            DesignFitParameterList = DesignFitParameterList[0]
            DesignCIList = DesignCIList[0]
        else:
            DesignFitParameterList = DesignFitParameterList
            DesignCIList = DesignCIList

        if "Confidence" in opts.keys() and (opts['Confidence']=="Intervals" or opts['Confidence']=="Contours" or opts['Confidence']=="Profiles"):
            ReturnValue = [DesignFitParameterList, DesignCIList]
        else:
            ReturnValue = DesignFitParameterList

        return ReturnValue

    def __confidenceintervals(self,mleparams,loglikfunc,opts):

        CIList=[]
        for p in range(self.NumParams):
            
            Direction=[1 if i==p else 0 for i in range(self.NumParams) ]
            SolverList=self.__profilesetup(mleparams,loglikfunc,Direction,opts)
            NuisanceParams=[mleparams[i] for i in range(self.NumParams) if not Direction[i]==0]
            
            UpperGamma=self.__logliksearch(NuisanceParams,SolverList,opts,True)
            UpperBound=mleparams[p]+Direction[p]*UpperGamma
            LowerGamma=self.__logliksearch(NuisanceParams,SolverList,opts,False)
            LowerBound=mleparams[p]+Direction[p]*LowerGamma

            CIList.append([LowerBound,UpperBound])

        return CIList

    def __profileplot(self,mleparams,loglikfunc,figure,opts):
        Alpha = opts['ConfidenceLevel']
        ChiSquaredLevel = st.chi2.ppf(Alpha, self.NumParams)

        [CIList,TraceList,ProfileList]=self.__profiletrace(mleparams,loglikfunc,opts)

        for p1 in range(self.NumParams):
            for p2 in range(p1,self.NumParams):
                #print(str(p1)+'_'+str(p2))
                if p1==p2:
                    X=[TraceList[p1][ind][p1] for ind in range(len(TraceList[p1]))]
                    Y=ProfileList[p1]

                    X0=[X[0],X[-1]]
                    Y0=[ChiSquaredLevel,ChiSquaredLevel]

                    plt.subplot(self.NumParams, self.NumParams, p2*self.NumParams+p1+1)
                    plt.plot(X, Y)
                    plt.plot(X0, Y0, 'r--')
                    plt.xlabel(self.ParamNameList[p1])
                    plt.ylabel('LogLik Ratio')
                else:
                    plt.subplot(self.NumParams, self.NumParams, p2*self.NumParams+p1+1)
                    X1=[TraceList[p1][ind][p1] for ind in range(len(TraceList[p1]))]
                    Y1=[TraceList[p1][ind][p2] for ind in range(len(TraceList[p1]))]
                    plt.plot(X1, Y1,label=self.ParamNameList[p1]+'profile')

                    X2=[TraceList[p2][ind][p1] for ind in range(len(TraceList[p2]))]
                    Y2=[TraceList[p2][ind][p2] for ind in range(len(TraceList[p2]))]
                    plt.plot(X2, Y2,label=self.ParamNameList[p2]+'profile')

                    plt.legend()
                    plt.xlabel(self.ParamNameList[p1])
                    plt.ylabel(self.ParamNameList[p2])

        return CIList

    def __profiletrace(self,mleparams,loglikfunc,opts):

        Alpha = opts['ConfidenceLevel']
        ChiSquaredLevel = st.chi2.ppf(Alpha, self.NumParams)
        NumPoints = opts['SampleNumber']

        CIList=[]
        ProfileList=[]
        TraceList=[]
        for p in range(self.NumParams):
            
            Direction=[1 if i==p else 0 for i in range(self.NumParams)]
            SolverList=self.__profilesetup(mleparams,loglikfunc,Direction,opts)
            NuisanceParams=[mleparams[i] for i in range(self.NumParams) if not Direction[i]==0]
            
            UpperGamma=self.__logliksearch(NuisanceParams,SolverList,opts,True)
            UpperBound=mleparams[p]+Direction[p]*UpperGamma

            LowerGamma=self.__logliksearch(NuisanceParams,SolverList,opts,False)
            LowerBound=mleparams[p]+Direction[p]*LowerGamma

            CIList.append([LowerBound,UpperBound])

            #NOTE: this is inefficient as we could return LR value and NuisanceParams from search and pre/append them to trace/profiles
            GammaList=list(np.linspace(LowerGamma, UpperGamma, num=NumPoints))
            #GammaList=list(np.linspace(LowerGamma, UpperGamma, num=NumPoints+1,endpoint=False)[1:])
            IPOPTSolver = SolverList[0]

            ParameterTrace=[]
            LRProfile=[]
            for g in GammaList:

                # Solve the NLP problem with IPOPT call
                IPOPTSolutionStruct = IPOPTSolver(x0=NuisanceParams,p=g)#, lbx=[], ubx=[], lbg=[], ubg=[]
                CurrentRatioGap= IPOPTSolutionStruct['f'].full()[0][0]
                NuisanceParams=list(IPOPTSolutionStruct['x'].full().flatten())
                
                ParamList=cp.deepcopy(NuisanceParams)
                ParamList.insert(p,Direction[p]*g+mleparams[p])
                # ParamList=[]
                # ParamCounter=0
                # for i in range(self.NumParams):
                #     if not Direction[i]==0:
                #         ParamList.append(Direction[i]*g+mleparams[i])
                #     else:
                #         ParamList.append(NuisanceParams[ParamCounter])
                #         ParamCounter+=1
                ParameterTrace.append(ParamList)
                LRProfile.append(ChiSquaredLevel-CurrentRatioGap)
            ProfileList.append(LRProfile)
            TraceList.append(ParameterTrace)

        return [CIList,TraceList,ProfileList]

    def __contourplot(self,mleparams,loglikfunc,figure,opts):
        print("Not implemented yet")

    def __contourtrace(self,mleparams,loglikfunc,opts):

        # Alpha = opts['ConfidenceLevel']
        # ChiSquaredLevel = st.chi2.ppf(Alpha, self.NumParams)
        # NumPoints = opts['SampleNumber']

        # CIList=[]
        # ProfileList=[]
        # TraceList=[]
        # for p in range(self.NumParams):
            
        #     Direction=[1 if i==p else 0 for i in range(self.NumParams)]
        #     SolverList=self.__profilesetup(mleparams,loglikfunc,Direction,opts)
        #     NuisanceParams=[mleparams[i] for i in range(self.NumParams) if not Direction[i]==0]
            
        #     UpperGamma=self.__logliksearch(NuisanceParams,SolverList,opts,True)
        #     UpperBound=mleparams[p]+Direction[p]*UpperGamma

        #     LowerGamma=self.__logliksearch(NuisanceParams,SolverList,opts,False)
        #     LowerBound=mleparams[p]+Direction[p]*LowerGamma

        #     CIList.append([LowerBound,UpperBound])

        #     #NOTE: this is inefficient as we could return LR value and NuisanceParams from search and pre/append them to trace/profiles
        #     GammaList=list(np.linspace(LowerGamma, UpperGamma, num=NumPoints))
        #     #GammaList=list(np.linspace(LowerGamma, UpperGamma, num=NumPoints+1,endpoint=False)[1:])
        #     IPOPTSolver = SolverList[0]

        #     ParameterTrace=[]
        #     LRProfile=[]
        #     for g in GammaList:

        #         # Solve the NLP problem with IPOPT call
        #         IPOPTSolutionStruct = IPOPTSolver(x0=NuisanceParams,p=g)#, lbx=[], ubx=[], lbg=[], ubg=[]
        #         CurrentRatioGap= IPOPTSolutionStruct['f'].full()[0][0]
        #         NuisanceParams=list(IPOPTSolutionStruct['x'].full().flatten())
                
        #         ParamList=cp.deepcopy(NuisanceParams)
        #         ParamList.insert(p,Direction[p]*g+mleparams[p])
        #         # ParamList=[]
        #         # ParamCounter=0
        #         # for i in range(self.NumParams):
        #         #     if not Direction[i]==0:
        #         #         ParamList.append(Direction[i]*g+mleparams[i])
        #         #     else:
        #         #         ParamList.append(NuisanceParams[ParamCounter])
        #         #         ParamCounter+=1
        #         ParameterTrace.append(ParamList)
        #         LRProfile.append(ChiSquaredLevel-CurrentRatioGap)
        #     ProfileList.append(LRProfile)
        #     TraceList.append(ParameterTrace)


        print("Not implemented yet")

    def __profilesetup(self,mleparams,loglikfunc,direction,opts):
        
        Alpha = opts['ConfidenceLevel']
        ChiSquaredLevel = st.chi2.ppf(Alpha, self.NumParams)

        ZeroDimensions=len([d for d in direction if d == 0]) 
        MarginalParamSymbols = cs.SX.sym('ParamSymbols',ZeroDimensions)
        GammaSymbol = cs.SX.sym('GammaSymbol')
        ParamList=[]
        ParamCounter=0
        for i in range(self.NumParams):
            if not direction[i]==0:
                ParamList.append(direction[i]*GammaSymbol+mleparams[i])
            else:
                ParamList.append(MarginalParamSymbols[ParamCounter])
                ParamCounter+=1
        ParamVec=cs.vertcat(*ParamList)
        MarginalLogLikRatioSymbol = 2*(loglikfunc(ParamVec)+loglikfunc(mleparams))

        # Create an IPOPT solver to optimize the nuisance parameters
        IPOPTProblemStructure = {'f': MarginalLogLikRatioSymbol+ChiSquaredLevel, 'x': MarginalParamSymbols,'p':GammaSymbol}#, 'g': cs.vertcat(*OptimConstraints)
        IPOPTSolver = cs.nlpsol('solver', 'ipopt', IPOPTProblemStructure,{'ipopt.print_level':0,'print_time':False})
        IPOPTJacobianSolver = IPOPTSolver.factory('Jsolver', IPOPTSolver.name_in(), ['sym:jac:f:p'])
        IPOPTHessianSolver = IPOPTSolver.factory('Hsolver', IPOPTSolver.name_in(), ['sym:hess:f:p:p'])

        return [IPOPTSolver,IPOPTJacobianSolver,IPOPTHessianSolver]

    def __logliksearch(self,nuisanceparams,solverlist,opts,forward=True):

        Tolerance=opts['Tolerance']
        if forward:
            Gamma=opts['InitialStep']
        else:
            Gamma=-opts['InitialStep']

        IPOPTSolver = solverlist[0]
        IPOPTJacobianSolver = solverlist[1]
        IPOPTHessianSolver = solverlist[2]

        NuisanceParams=nuisanceparams
        CurrentRatioGap=9999
        #NOTE: should check to see if we go negative, loop too many time, take massive steps, want to stay in the domain of the MLE
        #NOTE:need max step check, if steps are all in same direction perhaps set bound at inf and return warning, if oscillating error failure to converge
        while  abs(CurrentRatioGap)>Tolerance:
            # Solve the NLP problem with IPOPT call
            IPOPTSolutionStruct = IPOPTSolver(x0=NuisanceParams,p=Gamma)#, lbx=[], ubx=[], lbg=[], ubg=[]
            IPOPTJacobian = IPOPTJacobianSolver(x0=IPOPTSolutionStruct['x'], lam_x0=IPOPTSolutionStruct['lam_x'], lam_g0=IPOPTSolutionStruct['lam_g'],p=Gamma)
            IPOPTHessian = IPOPTHessianSolver(x0=IPOPTSolutionStruct['x'], lam_x0=IPOPTSolutionStruct['lam_x'], lam_g0=IPOPTSolutionStruct['lam_g'],p=Gamma)
            #update profile curve
            CurrentRatioGap= IPOPTSolutionStruct['f'].full()[0][0]
            dCurrentRatioGap_dGamma=IPOPTJacobian['sym_jac_f_p'].full()[0][0]
            d2CurrentRatioGap_dGamma2=IPOPTHessian['sym_hess_f_p_p'].full()[0][0]
            NuisanceParams=list(IPOPTSolutionStruct['x'].full().flatten())
            #Halley's method
            Gamma=Gamma-(2*CurrentRatioGap*dCurrentRatioGap_dGamma)/(2*dCurrentRatioGap_dGamma**2 -CurrentRatioGap*d2CurrentRatioGap_dGamma2)

        return Gamma

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