import casadi as cs
import numpy as np
import math as mt
import copy as cp
from scipy import stats as st
from scipy.interpolate import splev, splrep
import matplotlib.pyplot as plt

class model:
    """ 
    A class for statisical models in the NLoed package
    """
    #Easy:      [DONE, for normal data] implement data sampling
    #Difficult: implement various cov/bias assesment methods, also (profile) likelihood intervals/plots
    #           add other distributions binomial/bernouli, lognormal, gamma, exponential, weibull etc., negative binomial
    #           implement plotting function (move to utility.py?)
    #NOTE: [maybe a gentle check, i.e. did you mean this? no constraints]How to handle profile that goes negative for strictly positive parameter values???? perhaps error out and suggest reparameterization???
    #NOTE: can we allow custom pdf's (with some extra work by the user)
    #NOTE: [Yes] Add A-opt, D-s Opt, A-s Opt???, Bias??? for exponential family (and log normal?) 
    #NOTE: https://math.stackexchange.com/questions/269723/largest-eigenvalue-semidefinite-programming
    #NOTE: Data/design/experiment objects may need deep copies so internal lists aren't shared??
    #NOTE: all names must be unique
    #NOTE: must enforce ordering of parameters in statistics function
    #NOTE: we need checks on inputs of fit and sample functions!!!
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
        #check if we have the same number of input names as inputs
        if not(len(set(inputnames))  ==  len(inputnames)):
            raise Exception('Model depends on '+str(self.NumInputs)+' inputs but there are '+str(len(inputnames))+' input names!')
        #check if we have same number of parameter names as parameters
        if not(self.NumParams  ==  len(paramnames)):
            raise Exception('Model depends on '+str(self.NumParams)+' parameters but there are '+str(len(paramnames))+' parameter names!')
        #create lists of input/param/param names, can be used to look up names with indices (observ names filled in below)
        self.InputNameList = inputnames
        self.ParamNameList= paramnames
        self.ObservNameList = [] 
        #create dicts for input/param names, can be used to look up indices with names
        self.InputNameDict = {}
        self.ParamNameDict = {}
        self.ObservNameDict = {}
        #populate the input name dict
        for i in range(self.NumInputs):
            self.InputNameDict[inputnames[i]] = i
        #populate the param name dict
        for i in range(self.NumParams):
            self.ParamNameDict[paramnames[i]] = i
        #create a list to store the observation variable distribution type
        self.Dist = []
        #create a list to store casadi functions predicting the observ. var sampling distirbution parameters
        self.Model = []
        #create a list to store casadi functions computing the elemental loglikelihood for each observation variable
        self.LogLik = []
        #create a list to store casadi functions computing the elemental fisher info. matrix for each observation variable
        self.FIM = []
        #create symbols for parameters and inputs, needed for function defs below
        ParamSymbols = cs.SX.sym('ParamSymbols',self.NumParams)
        InputSymbols = cs.SX.sym('InputSymbols',self.NumInputs)
        #loop over the observation variables
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
            self.Model.append(Observation[2])
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

    def fit(self,datasets,paramstart,opts=None):
        """
        This function fits the model to a dataset using maximum likelihood
        it also provides optional marginal confidence intervals
        and plots of logliklihood profiles/traces and marginal projected confidence contours

        Args:
            datasets: either; a dictionary for one dataset OR a list of dictionaries, each design replicats OR a list of lists of dict's where each index in the outer lists has a unique design
            paramstart: a list of starting values for the parameters
            opts: optional, a dictionary of user defined options

        Return:
            ParameterFitStruct: either; a list of lists structure with parameter fit lists, has the same shape/order as the input, OR, the same strcture but with fits and intervals as the leaf object
        """
        #NOTE: needs checks on inputs
        #NOTE: NEED testing for multiple observation input structures, multiple dimensions of parameters ideally, 1,2,3 and 7+
        #NOTE: add some print statments to provide user with progress status
        #NOTE: currently solve multiplex simultaneosly (one big opt problem) but sequentially may be more robust to separation (discrete data), or randomly non-identifiable datasets (need to test)
        #this block allows the user to pass a dataset, list of datasets, list of lists etc. for Design x Replicat fitting
        if not(isinstance(datasets, list)):
            #if a single dataset is passed vis the dataset input, wrap it in two lists so it matches general case
            DesignSet = [[datasets]]
        elif not(isinstance(datasets[0], list)):
            #else if dataset input is a list of replciated datasets, wrap it in a single list to match general case
            DesignSet = [datasets]
        else:
            #else if input dataset input is a list of design, each with a list of replicats, just pass on th input
            DesignSet = datasets
        #create a list to store parameter casadi symbols used for ML optimization
        ParamSymbolsList = []
        #create a list to store starting parameters (they are all determined by paramstart, but dim. depends on design x replicat size)
        StartParams = []
        #create a list to store casadi generic loglikelihood functions for each design
        DesignLogLikFunctionList = []
        #create a total loglikelihood summation store, initialize to zero
        TotalLogLik = 0
        #create an archetypal vector of paramter symbols, used to build casadi loglikelihood functions for each design
        ParamArchetypeSymbols = cs.SX.sym('ParamArchetypeSymbols',self.NumParams)
        #loop over different designs (outer most list)
        for e in range(len(DesignSet)):
            #get the set of replicats for this design
            ReplicatSet = DesignSet[e]
            #create a list to store loglikelihood functions for each specific replicat of the design
            ReplicatLogLikFunctionList = []
            #for each design use the first replicat 
            Data = ReplicatSet[0]
            #create a summation variable for the loglikelihood for a dataset of the current design
            LogLik = 0
            #create a vector of all observations in the design
            Observations = [element for row in Data['Observation'] for group in row for element in group]
            #create a vector of casadi symbols for the observations
            ObservSymbol = cs.SX.sym('ObservSymbol_'+str(e),len(Observations))
            #create a counter to index the total number of observations
            SampleCount = 0
            #loop over the dataset inputs
            for i in range(len(Data['Inputs'])):
                #get the curren input settings
                InputRow = Data['Inputs'][i]
                #loop over the observation varaibles
                for j in range(self.NumObserv):
                    #for the given dataset loop over (the potentially replicated) observations for the given observation variable 
                    #if no observations are taken at the given observation variable len=0 and we skip
                    #NOTE: NEED TO CHECK THIS WORKS FOR MULTI OBSERV DATA
                    for k in range(len(Data['Observation'][i][j])):
                        #create a symbol for the loglikelihood for the given input and observation variable
                        LogLik += self.LogLik[j](ObservSymbol[SampleCount],ParamArchetypeSymbols,InputRow)
                        #increment the observation counter
                        SampleCount += 1
            #create a casadi function for the loglikelihood of the current design (observations are free/input symbols)
            ArchetypeLogLikFunc = cs.Function('ArchetypeLogLikFunc', [ObservSymbol,ParamArchetypeSymbols], [LogLik])
            #loop over replicats within each design
            for r in range(len(ReplicatSet)):
                #NOTE: could abstract below into a Casadi function to avoid input/observ loop on each dataset and replicat
                #get the dataset from the replicat list
                Dataset = ReplicatSet[r]
                #create a vector of parameter symbols for this specific dataset, each dataset gets its own, these are used for ML optimization
                ParamFitSymbols = cs.SX.sym('ParamFitSymbols'+'_'+str(e)+str(r),self.NumParams)
                #extract the vector of observations in the same format as in the ArchetypeLogLikFunc function input
                Observations = cs.vertcat(*[cs.SX(element) for row in Dataset['Observation'] for group in row for element in group])
                #create a symbol for the datasets loglikelihood function by pass in the observations for the free symbols in ObservSymbol
                LogLik = ArchetypeLogLikFunc(Observations,ParamFitSymbols)
                #create a function for
                LogLikFunc = cs.Function('LogLik_'+str(e)+'_'+str(r), [ParamFitSymbols], [LogLik])
                ReplicatLogLikFunctionList.append(LogLikFunc)
                #set up the logliklihood symbols for given design and replicat
                ParamSymbolsList.append(ParamFitSymbols)
                #record the starting parameters for the given replicat and dataset
                StartParams.extend(paramstart)
                #add the loglikelihood to the total 
                #NOTE: this relies on the distirbutivity of serperable optimization problem, should confirm
                TotalLogLik += LogLik
            #append the list of replciate loglik functions to the design list
            DesignLogLikFunctionList.append(ReplicatLogLikFunctionList)
        #NOTE: this approach is much more fragile to separation (glms with discrete response), randomly weakly identifiable datasets
        #NOTE: should be checking solution for convergence, should allow user to pass options to ipopt
        #NOTE: allow bfgs for very large nonlinear fits, may be faster
        # Create an IPOPT solver for overall maximum likelihood problem, we pass negative TotalLogLik because IPOPT minimizes
        TotalParamFitStructure = {'f': -TotalLogLik, 'x': cs.vertcat(*ParamSymbolsList)}#, 'g': cs.vertcat(*OptimConstraints)
        ParamFitSolver = cs.nlpsol('solver', 'ipopt', TotalParamFitStructure,{'ipopt.print_level':5,'print_time':False})
        # Solve the NLP fitting problem with IPOPT call
        ParamFitSolutionStruct = ParamFitSolver(x0=StartParams)#, lbx=[], ubx=[], lbg=[], ubg=[]
        #extract the fit parameters from the solution structure
        FitParameters = list(ParamFitSolutionStruct['x'].full().flatten())
        #NOTE: this is (very slightly) in efficient to do this loop twice but it makes it more readable
        #NOTE: some extra space taken up by CI lists if not requested but not a big deal
        #create a list to store fits params for each design
        DesignFitParameterList = []
        #create a list to store param CI's for each design
        DesignCIList = []
        #loop over each design
        for e in range(len(DesignSet)):
            #create a list to store fits params for each replicat
            ReplicatFitParameterList=[]
            #create a list to store param CI's for each replicat
            ReplicatCIList=[]
            #loop over each replicat of the given design
            for r in range(len(DesignSet[e])):
                #extract the fit parameters from the optimization solution vector
                FitParamSet= FitParameters[:self.NumParams]
                #check if 'confidence' key word is passed in the options dictionary
                if "Confidence" in opts.keys():
                    #check if graphing options; contour plots or profile trace plots, are requested
                    if opts['Confidence']=="Contours" or opts['Confidence']=="Profiles":
                        #if so, create a figure to plot on
                        Figure = plt.figure()
                        #run profileplots to plot the profile traces and return CI's
                        CIList = self.__profileplot(FitParamSet,DesignLogLikFunctionList[e][r],Figure,opts)[0]
                        #add the CI's to the replicate CI list
                        ReplicatCIList.append(CIList)
                        #if contour plots are requested specifically, run contour function to plot the projected confidence contours
                        if opts['Confidence']=="Contours":
                            self.__contourplot(FitParamSet,DesignLogLikFunctionList[e][r],Figure,opts)
                    elif opts['Confidence']=="Intervals":
                        #if confidence intervals are requested, run confidenceintervals to get CI's and add them to replicat list
                        ReplicatCIList.append(self.__confidenceintervals(FitParamSet,DesignLogLikFunctionList[e][r],opts))
                #remove the current extracted parameters from the solution vector
                del FitParameters[:self.NumParams]
                #add the extracted parameters to the replicat parameter lsit
                ReplicatFitParameterList.append(FitParamSet)
            #add the replicat list to the design list
            DesignFitParameterList.append(ReplicatFitParameterList)
            #add replicat CI list to design CI list
            DesignCIList.append(ReplicatCIList)
        #if we need to plot profile traces or contours 
        if "Confidence" in opts.keys() and (opts['Confidence']=="Contours" or opts['Confidence']=="Profiles"):
            plt.show()
        #depending on dimension of input datasets (i.e. single, replcated design, multiple rep'd designs), package output to match
        if not(isinstance(datasets, list)):
            #if a single dataset dict. was passed, return a param vec and CI list accordingly
            DesignFitParameterList = DesignFitParameterList[0][0]
            DesignCIList = DesignCIList[0][0]
        elif not(isinstance(datasets[0], list)):
            #if a list a replicated design was passed, return a a list of param vecs and CI lists accordingly
            DesignFitParameterList = DesignFitParameterList[0]
            DesignCIList = DesignCIList[0]
        else:
            #if a list designs, each listing replicats, was passed, return a a list of lists of param vecs and CI lists accordingly
            DesignFitParameterList = DesignFitParameterList
            DesignCIList = DesignCIList
        #check if CI's were generated, is so return both param fits and CI's in a list, else a just param fits
        #NOTE: this may be a bit awkward, should possibly keep it standardized
        if "Confidence" in opts.keys() and (opts['Confidence']=="Intervals" or opts['Confidence']=="Contours" or opts['Confidence']=="Profiles"):
            ParameterFitStruct = [DesignFitParameterList, DesignCIList]
        else:
            ParameterFitStruct = DesignFitParameterList
        #return the appropriate return structure
        return ParameterFitStruct

    def sample(self,designs,parameters,replicats=1):
        """
        This function generates datasets for a given design or list of designs, with a given set of parameters, with an optional number of replicates

        Args:
            designs: either; a single design dictionary, OR, a list of design dictionaries
            parameters: the parameter values at which to generate the data
            replicates: optional, an integer indicating the number of datasets to generate for each design, default is 1

        Return:
            DesignDataSets: either; a singel dataset for the given design, OR, a list of dataset replicats for the given design, OR, a list of list of dataset replicats for each design
        """
        #NOTE: needs checks on inputs
        #NOTE: multiplex multiple parameter values??
        #NOTE: actually maybe more important to be able to replicat designs N times
        #check if designs is a single design or a list of designs
        if not(isinstance(designs,list)):
            #if single, wrap it in a list to match general case
            DesignList = [designs]
        else:
            #else pass it as it is
            DesignList = designs
        #create a list to store lists of datasets for each design
        DesignDataSets = []
        #loop over the designs
        for e in range(len(DesignList)):
            #extract the current design
            CurrentDesign = DesignList[e]
            #deep copy the current design to create an archetype for that design's datasets
            DatasetArchetype = cp.deepcopy(CurrentDesign)
            #delete the 'count' field in the design, as we are using it as a template for datasets
            del DatasetArchetype['Count']
            #create 'observations' field in the archetype datset
            DatasetArchetype['Observation'] = []
            #create a list for replciated datasets of the current design
            ReplicatDataSets = []
            #loop over the number of replicates
            for r in range(replicats):
                #create a deep copy of the dataset archetype for the current dataset
                CurrentDataSet = cp.deepcopy(DatasetArchetype)
                #loop over the (unique) input rows in the design
                for i in range(len(CurrentDesign['Inputs'])):
                    #get the input row
                    InputRow = CurrentDesign['Inputs'][i]
                    #create a list for the current input row's observations
                    ObservRow = []
                    #loop over the observation variables
                    for j in range(self.NumObserv):
                        #get the count of observations for the given observ. var at the given input row
                        ObservCount = CurrentDesign['Count'][i][j]
                        #compute the sample distirbution statistics using the model
                        SampleStatistics = self.Model[j](parameters,InputRow)
                        #select the appropriate distribution, and generate the sample vec of appropriate length
                        if self.Dist[j] == 'Normal':
                            CurrentObservVec = np.random.normal(SampleStatistics[0], np.sqrt(SampleStatistics[1]), ObservCount).tolist() 
                        elif self.Dist[j] == 'Poisson':
                            CurrentObservVec = np.random.poisson(SampleStatistics[0]).tolist() 
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
                        #add the current observation vector, for the current observ. var., to the observ row
                        ObservRow.append(list(CurrentObservVec))
                    #add the observation row to the current dataset
                    CurrentDataSet['Observation'].append(ObservRow)
                #add the current dataset to the replicat list
                ReplicatDataSets.append(CurrentDataSet)
            #add the replicat list to the design list
            DesignDataSets.append(ReplicatDataSets)
        #check if a single design was passed
        if not(isinstance(designs, list)):
            if replicats==1:
                #if a single design was passed and there replicate count is 1, return a single dataset
                return DesignDataSets[0][0]
            else:
                #else if a single design was passed, but with >1 reps, return a list of datasets
                return DesignDataSets[0]
        else:
            #else if multiple designs were passed (with/without reps), return a list of list of datasets
            return DesignDataSets

    #NOTE: should maybe rename this
    def simulatedesign(self):
        #maybe this should move to the design class(??)
        #For D (full cov/bias), Ds (partial cov/bias), T separation using the delta method?! but need two models
        # assess model/design, returns various estimates of cov, bias, confidence regions/intervals
        # no data: asymptotic: covaraince, beale bias, maybe MSE
        #          sigma point: covariance, bias (using mean) (need to figure out how to do sigma for non-normal data), maybe MSE
        #          monte carlo: covariance, bias, MSE
        
        print('Not Implemeneted')
        
    # UTILITY FUNCTIONS
    def evalmodel(self,parameter,inputs,observ=None,covariance=None,opts=None):
        #NOTE: evaluate model, predict y
        #NOTE: optional pass cov matrix, for use with delta method/MC error bars on predictions
        if not observ:
            ObservList = list(range(self.NumObserv))
        else:
            ObservList = observ

        StatisticList = []
        for o in ObservList:
            StatisticList.append( self.Model[o](parameter,inputs) )
            if 'ErrorBars' in opts:
                if opts['ErrorBars'] == 'Delta':
                    print('Not Implemeneted')
                elif opts['ErrorBars'] == 'MonteCarlo':
                    print('Not Implemeneted')

        return StatisticList

    def evalsensitivity(self):
        #eval observation distribution statistic sensitivities at given input and parameter values
        print('Not Implemeneted')

    # def evalfim(self):
    #     #NOTE: eval fim at given inputs and dataset
    #     #NOTE: should this even be here??? how much in model, this isn't data dependent, only design dependent
    #     print('Not Implemeneted')

    # def evalloglik(self):
    #     #eval the logliklihood with given params and dataset
    #     print('Not Implemeneted')

    # def plots(self):
    #     #FDS plot, standardized variance (or Ds, bayesian equivlant), residuals
    #     print('Not Implemeneted')
    #     #NOTE: maybe add a basic residual computation method for goodness of fit assesment?? Or maybe better show how in tutorial but not here
    



# --------------- Private functions and subclasses ---------------------------------------------

    def __confidenceintervals(self,mleparams,loglikfunc,opts):
        """ 
        This function computes marginal parameter confidence intervals for the model
        around the MLE estimate using the profile likelihood

        Args:
            mleparams: mle parameter estimates, recieved from fitting
            loglikfunc: casadi logliklihood function for the given dataset
            opts: an options dictionary for passing user options

        Returns:
            CIList: list of lists of upper and lower bounds for each parameter
        """
        #create a list to store intervals
        CIList = []
        #loop over parameters in model
        for p in range(self.NumParams):
            #fix parameter along which profile is taken
            FixedParams = [False]*self.NumParams
            FixedParams[p] = True
            #set direction so that it has unit length in profile direction
            Direction = [0]*self.NumParams
            Direction[p] = 1
            #setup the profile solver
            SolverList = self.__profilesetup(mleparams,loglikfunc,FixedParams,Direction,opts)
            #extract starting values for marginal parameters (those to be optimized during profile)
            MarginalParam = [mleparams[i] for i in range(self.NumParams) if Direction[i]==0]
            #search to find the radius length in the specified profile direction, positive search
            UpperRadius = self.__logliksearch(SolverList,MarginalParam,opts,True)[0]
            #compute the location of the upper parameter bound
            UpperBound = mleparams[p] + Direction[p] * UpperRadius
            #search to find the radius length in the specified profile direction, negative search
            LowerRadius  =self.__logliksearch(SolverList,MarginalParam,opts,False)[0]
            #compute the location of the lower parameter bound
            LowerBound = mleparams[p]+Direction[p]*LowerRadius
            CIList.append([LowerBound,UpperBound])
        return CIList

    def __profileplot(self,mleparams,loglikfunc,figure,opts):
        """ 
        This function plots profile parameter traces for each parameter value

        Args:
            mleparams: mle parameter estimates, recieved from fitting
            loglikfunc: casadi logliklihood function for the given dataset
            figure: the figure object on which plotting occurs
            opts: an options dictionary for passing user options

        Returns:
            CIList: list of lists of upper and lower bounds for each parameter
            TraceList: list of list of lists of parameter vector values along profile trace for each parameter
            ProfileList: List of lists of logliklihood ratio values for each parameter along the profile trace
        """
        #extract the alpha level and compute the chisquared threshold, or use default of 0.95
        if "ConfidenceLevel" in opts.keys():
            Alpha = opts['ConfidenceLevel']
        else:
            Alpha=0.95
        ChiSquaredLevel = st.chi2.ppf(Alpha, self.NumParams)
        #run profile trave to get the CI's, parameter traces, and LR profile
        [CIList,TraceList,ProfileList] = self.__profiletrace(mleparams,loglikfunc,opts)
        #loop over each pair of parameters
        for p1 in range(self.NumParams):
            for p2 in range(p1,self.NumParams):
                #check if parameter pair matches
                if p1 == p2:
                    #if on the diagonal, generate a profile plot
                    #get data for the profile
                    X = [TraceList[p1][ind][p1] for ind in range(len(TraceList[p1]))]
                    Y = ProfileList[p1]
                    #get data for the threshold
                    X0 = [X[0],X[-1]]
                    Y0 = [ChiSquaredLevel,ChiSquaredLevel]
                    #plot the profile and threshold
                    plt.subplot(self.NumParams, self.NumParams, p2*self.NumParams+p1+1)
                    plt.plot(X, Y)
                    plt.plot(X0, Y0, 'r--')
                    plt.xlabel(self.ParamNameList[p1])
                    plt.ylabel('LogLik Ratio')
                else:
                    #if off-diagonal generate a pair of parameter profile trace plots
                    #plot the profile parameter trace for p1
                    plt.subplot(self.NumParams, self.NumParams, p2*self.NumParams+p1+1)
                    X1 = [TraceList[p1][ind][p1] for ind in range(len(TraceList[p1]))]
                    Y1 = [TraceList[p1][ind][p2] for ind in range(len(TraceList[p1]))]
                    plt.plot(X1, Y1,label=self.ParamNameList[p1]+'profile')
                    #plot the profile parameter trace for p2
                    X2 = [TraceList[p2][ind][p1] for ind in range(len(TraceList[p2]))]
                    Y2 = [TraceList[p2][ind][p2] for ind in range(len(TraceList[p2]))]
                    plt.plot(X2, Y2,label=self.ParamNameList[p2]+'profile')
                    plt.legend()
                    plt.xlabel(self.ParamNameList[p1])
                    plt.ylabel(self.ParamNameList[p2])
        #return CI, trace and profilem (for extensibility)
        return [CIList,TraceList,ProfileList]

    def __profiletrace(self,mleparams,loglikfunc,opts):
        """ 
        This function compute the profile logliklihood parameter trace for each parameter in the model

        Args:
            mleparams: mle parameter estimates, recieved from fitting
            loglikfunc: casadi logliklihood function for the given dataset
            opts: an options dictionary for passing user options

        Returns:
            CIList: list of lists of upper and lower bounds for each parameter
            TraceList: list of list of lists of parameter vector values along profile trace for each parameter
            ProfileList: List of lists of logliklihood ratio values for each parameter along the profile trace
        """
        #extract the alpha level and compute the chisquared threshold, or use default of 0.95
        if "ConfidenceLevel" in opts.keys():
            Alpha = opts['ConfidenceLevel']
        else:
            Alpha = 0.95
        ChiSquaredLevel = st.chi2.ppf(Alpha, self.NumParams)
        #extract the number of points to compute along the trace, or use default of 10
        if "SampleNumber" in opts.keys():
            NumPoints = opts['SampleNumber']
        else:
            NumPoints = 10
        #create lists to store the CI's, profile logliklkhood values and parameter traces
        CIList = []
        ProfileList = []
        TraceList = []
        #loop over each parameter in the model
        for p in range(self.NumParams):
            #indicate the parameter, along which the profile is taken, is fixed
            FixedParams = [False]*self.NumParams
            FixedParams[p] = True
            #set the direction of the profile so that it has unit length
            Direction = [0]*self.NumParams
            Direction[p] = 1
            #generate the profile solvers
            SolverList = self.__profilesetup(mleparams,loglikfunc,FixedParams,Direction,opts)
            #set the starting values of the marginal parameters from the mle estimates
            MarginalParams = [mleparams[i] for i in range(self.NumParams) if not FixedParams[i]]
            #preform a profile search to find the upper bound on the radius for the profile trace
            [UpperRadius,UpperParamList,UpperLLRGap] = self.__logliksearch(SolverList,MarginalParams,opts,True)
            #compute the parameter upper bound
            UpperBound = mleparams[p] + Direction[p]*UpperRadius
            #insert the profile parameter (upper) in the marginal parameter vector (upper), creates a complete parameter vector
            UpperParamList.insert(p,UpperBound)
            #preform a profile search to find the lower bound on the radius for the profile trace
            [LowerRadius,LowerParamList,LowerLLRGap] = self.__logliksearch(SolverList,MarginalParams,opts,False)
            #compute the parameter lower bound
            LowerBound = mleparams[p] + Direction[p]*LowerRadius
            #insert the profile parameter (lower) in the marginal parameter vector (lower), creates a complete parameter vector
            LowerParamList.insert(p,LowerBound)
            #record the uppper and lower bounds in the CI list
            CIList.append([LowerBound,UpperBound])
            #Create a grid of radia from the lower radius bound to the upper radius bound with the number of points requested in the profile
            RadiusList = list(np.linspace(LowerRadius, UpperRadius, num=NumPoints+1,endpoint=False)[1:])
            #extract the marginal logliklihood solver, to compute the profile
            IPOPTSolver = SolverList[0]
            #insert the lower parameter bound and the logliklihood ratio in the trace list and profile list respectivly 
            ParameterTrace = [LowerParamList]
            LRProfile = [ChiSquaredLevel-LowerLLRGap]
            #loop over the radius grid 
            for r in RadiusList:
                # Solve the for the marginal maximumlikelihood estimate
                ProfileSolutionStruct = IPOPTSolver(x0=MarginalParams,p=r)#, lbx=[], ubx=[], lbg=[], ubg=[]
                #extract the current logliklihood ratio gap (between the chi-sqaured level and current loglik ratio)
                CurrentRatioGap = ProfileSolutionStruct['f'].full()[0][0]
                #extract the marginal parameter vector
                #NOTE: need to test how low dim. (1-3 params) get handled in this code, will cause errors for 1 param models !!
                MarginalParams = list(ProfileSolutionStruct['x'].full().flatten())
                #copy and insert the profile parameter in the marginal vector
                ParamList = cp.deepcopy(MarginalParams)
                ParamList.insert(p,Direction[p]*r+mleparams[p])
                #insert the full parameter vector in the trace lists
                ParameterTrace.append(ParamList)
                #insert the likelihood ratio for the current radius
                LRProfile.append(ChiSquaredLevel-CurrentRatioGap)
            #insert the upper bound in the parameter trace after looping over the grid
            ParameterTrace.append(UpperParamList)
            #insert the upper loglik ratio in the profile list
            LRProfile.append(ChiSquaredLevel-UpperLLRGap)
            #insert the final loglik profile in the profile list , recording the current parameter's trace
            ProfileList.append(LRProfile)
            #insert the final parameter trace into the trace list, recording the current parameter's profile
            TraceList.append(ParameterTrace)
        #return the intervals, parameter trace and loglik ratio profile
        return [CIList,TraceList,ProfileList]

    def __contourplot(self,mleparams,loglikfunc,figure,opts):
        """ 
        This function plots the projections of the confidence volume in a 2d plane for each pair of parameters
        this creates marginal confidence contours for each pair of parameters

        Args:
            mleparams: mle parameter estimates, recieved from fitting
            loglikfunc: casadi logliklihood function for the given dataset
            figure: the figure object on which plotting occurs
            opts: an options dictionary for passing user options
        """
        #loop over each unique pair of parameters 
        for p1 in range(self.NumParams):
            for p2 in range(p1+1,self.NumParams):
                #compute the x and y values for the contour trace
                [Xfit,Yfit] = self.__contourtrace(mleparams,loglikfunc,[p1,p2],opts)
                #plot the contour on the appropriate subplot (passed in from fit function, shared with profileplot)
                plt.subplot(self.NumParams, self.NumParams, p2*self.NumParams+p1+1)
                plt.plot(Xfit, Yfit,label=self.ParamNameList[p1]+' '+self.ParamNameList[p2]+' contour')
                plt.legend()
                plt.xlabel(self.ParamNameList[p1])
                plt.ylabel(self.ParamNameList[p2])

    def __contourtrace(self,mleparams,loglikfunc,coordinates,opts):
        """ 
        This function plots the projections of the confidence volume in a 2d plane for each pair of parameters
        this creates marginal confidence contours for each pair of parameters

        Args:
            mleparams: mle parameter estimates, recieved from fitting
            loglikfunc: casadi logliklihood function for the given dataset
            coordinates: a pair of parameter coordinates specifying the 2d contour to be computed in parameter space
            opts: an options dictionary for passing user options

        Returns:
            [Xfit,Yfit]: x,y-values in parameter space specified by coordinates tracing the projected profile confidence contour outline
        """
        if "RadialNumber" in opts.keys():
            RadialNum = opts['RadialNumber']
        else:
            RadialNum = 30
        #extract the parameter coordinat indicies for the specified trace
        p1 = coordinates[0]
        p2 = coordinates[1]
        #mark extracted indices as fixed for the loglik search
        FixedParams = [False]*self.NumParams
        FixedParams[p1] = True
        FixedParams[p2] = True
        #set the starting values for the marginal parameters based on the mle estimate
        MarginalParams = [mleparams[i] for i in range(self.NumParams) if FixedParams[i]==0]
        #create a list of angles (relative to the mle, in p1-p2 space) overwhich we perform the loglik search to trace the contour
        AngleList = list(np.linspace(-mt.pi, mt.pi,RadialNum))
        #create an empty list to sore the radiai resulting from the search
        RadiusList = []
        #loop over the angles
        for Angle in AngleList:
            #compute the sine and cosine of the angle
            AngleCosine = mt.cos(Angle)
            AngleSine = mt.sin(Angle)
            #compute the direction in p1-p2 space for the search
            Direction = [0]*self.NumParams
            Direction[p1] = AngleCosine
            Direction[p2] = AngleSine
            #setup the solver for the search
            SolverList = self.__profilesetup(mleparams,loglikfunc,FixedParams,Direction,opts)
            #run the profile loglik search and return the found radia for the given angle
            Radius = self.__logliksearch(SolverList,MarginalParams,opts,True)[0]
            #record the radius
            RadiusList.append(Radius)
        #fit a periodic spline to the Radius-Angle data
        RadialSplineFit = splrep(AngleList,RadiusList,per=True)
        #generate a dense grid of angles to perform interpolation on
        AngleInterpolants = np.linspace(-mt.pi, mt.pi,1000)
        #compute the sine and cosine for each interpolation angle
        AngleInterpCosine = [mt.cos(a) for a in AngleInterpolants]
        AngleInterpSine = [mt.sin(a) for a in AngleInterpolants]
        #use the periodic spline to interpolate the radius over the dense interpolation angle grid
        RadialInterpolation = splev(AngleInterpolants,RadialSplineFit)
        #compute the resulting x and y coordinates for the contour in the p1-p2 space
        Xfit = [AngleInterpCosine[i]*RadialInterpolation[i]+mleparams[p1] for i in range(len(AngleInterpolants))]
        Yfit = [AngleInterpSine[i]*RadialInterpolation[i]+mleparams[p2] for i in range(len(AngleInterpolants))]
        #return the contour coordinates
        return [Xfit,Yfit]
        #NOTE: should maybe pass profile extrema from CI's into contours to add to fit points in interpolation
        #        it is unlikely we will 'hit' absolute extrema unless we have very dense sampling, splines don't need an even grid

    def __profilesetup(self,mleparams,loglikfunc,fixedparams,direction,opts):
        """ 
        This function creates function/solver objects for performing a profile likelihood search for the condifence boundary
        in the specified direction, the function/solver objects compute the logliklihood ratio gap
        at a given radius (along the specified direction), along with the LLR gaps 1st and 2nd derivative with respect to the radius.
        marginal (free) parameters (if they exist) are optimized conditional on the fixed parameters specified by the radius and direction
        the likelihood ratio gap is the negative difference between the chi-squared boundary and the loglik ratio at the current radius

        Args:
            mleparams: mle parameter estimates, recieved from fitting
            loglikfunc: casadi logliklihood function for the given dataset
            fixedparams: a boolean vector, same length as the parameters, true means cooresponding parameters fixed by direction and radius, false values are marginal and optimized (if they exist)
            direction: a direction in parameter space, coordinate specified as true in fixedparams are used as the search direction
            opts: an options dictionary for passing user options

        Returns:
            ProfileLogLikSolver: casadi function/ipopt solver that returns the loglik ratio gap for a given radius, after optimizing free/marginal parameters if they exist
            ProfileLogLikJacobianSolver: casadi function/ipopt derived derivative function that returns the derivative of the loglik ratio gap with respect to the radius (jacobian is 1x1)
            ProfileLogLikHessianSolver: casadi function/ipopt derived 2nd derivative function that returns the 2nd derivative of the loglik ratio gap with respect to the radius (hessian is 1x1)
        """
        #check if confidence level is passed if not default to 0.95
        if "ConfidenceLevel" in opts.keys():
            Alpha = opts['ConfidenceLevel']
        else:
            Alpha = 0.95
        #compute the chi-squared level from alpha
        ChiSquaredLevel = st.chi2.ppf(Alpha, self.NumParams)
        #compute the number of fixed parameters (along which we do boundary search, radius direction)
        NumFixedParams = sum(fixedparams)
        #compute the number of free/marginal parameters, which are optimized at each step of the search
        NumMarginalParams = self.NumParams-NumFixedParams
        if NumFixedParams == 0:
            raise Exception('No fixed parameters passed to loglikelihood search, contact developers!')
        #create casadi symbols for the marginal parameters
        MarginalParamSymbols = cs.SX.sym('ParamSymbols',NumMarginalParams)
        #create casadi symbols for the radius (radius measured from mle, in given direction, to chi-squared boundary)
        RadiusSymbol = cs.SX.sym('RadiusSymbol')
        #creat a list to store a complete parameter vector
        #this is a mixture of fixed parameters set by the direction and radius, and marginal parameters which are free symbols
        ParamList = []
        #create a counter to count marginal parameters already added
        MarginalCounter = 0
        #loop over the parameters
        for i in range(self.NumParams):   
            if fixedparams[i]:
                #if the parameter is fixed, add an entry parameterized by the radius from the mle in given direction
                ParamList.append(direction[i]*RadiusSymbol+mleparams[i])
            else:
                #else add marginal symbol to list and increment marginal counter
                ParamList.append(MarginalParamSymbols[MarginalCounter])
                MarginalCounter+=1
        #convert the list of a casadi vector
        ParamVec = cs.vertcat(*ParamList)
        #create a symnol for the loglikelihood ratio gap at the parameter vector
        LogLikRatioGapSymbol = 2*( loglikfunc(mleparams) - loglikfunc(ParamVec)) - ChiSquaredLevel
        #check if any marginal parameters exist
        if not NumMarginalParams == 0:
            #if there are marginal parameters create Ipopt solvers to optimize the marginal params
            # create an IPOPT solver to minimize the loglikelihood for the marginal parameters
            # this solver minimize the logliklihood ratio but has a return objective value of the LLR gap, so its root is on the boundary
            #  it accepts the radius as a fixed parameter
            IPOPTProblemStructure = {'f': LogLikRatioGapSymbol, 'x': MarginalParamSymbols,'p':RadiusSymbol}#, 'g': cs.vertcat(*OptimConstraints)
            ProfileLogLikSolver = cs.nlpsol('PLLSolver', 'ipopt', IPOPTProblemStructure,{'ipopt.print_level':0,'print_time':False})
            #create a casadi function that computes the derivative of the optimal LLR gap solution with respect to the radius parameter
            ProfileLogLikJacobianSolver = ProfileLogLikSolver.factory('PLLJSolver', ProfileLogLikSolver.name_in(), ['sym:jac:f:p'])
            #create a casadi function that computes the 2nd derivative of the optimal LLR gap solution with respect to the radius parameter
            ProfileLogLikHessianSolver = ProfileLogLikSolver.factory('PLLHSolver', ProfileLogLikSolver.name_in(), ['sym:hess:f:p:p'])
        else:
            # else if there are no marginal parameters (i.e. 2d model), create casadi functions emulating the above without optimization (which is not needed)
            ProfileLogLikSolver = cs.Function('PLLSolver', [RadiusSymbol], [LogLikRatioGapSymbol]) 
            #create the 1st derivative function
            ProfileLogLikJacobianSymbol = cs.jacobian(LogLikRatioGapSymbol,RadiusSymbol)
            ProfileLogLikJacobianSolver = cs.Function('PLLJSolver', [RadiusSymbol], [ProfileLogLikJacobianSymbol]) 
            #create the second derivative function
            ProfileLogLikHessianSymbol = cs.jacobian(ProfileLogLikJacobianSymbol,RadiusSymbol) 
            ProfileLogLikHessianSolver = cs.Function('PLLHSolver', [RadiusSymbol], [ProfileLogLikHessianSymbol]) 
            #NOTE: not sure what second returned value of casadi hessian func is, did this to avoid it (it may be gradient, limited docs)
        #return the solvers/functions
        return [ProfileLogLikSolver,ProfileLogLikJacobianSolver,ProfileLogLikHessianSolver]

    def __logliksearch(self,solverlist,marginalparams,opts,forward=True):
        """ 
        This function performs a root finding algorithm using solverlist objects
        It uses halley's method to find the radius value (relative to the mle) where the loglik ratio equals the chi-squared level
        This radius runs along the direction specified in the solverlist when they are created
        Halley's method is a higher order extension of newton's method for finding roots

        Args:
            solverlist: solver/casadi functions for finding the loglikelihood ratio gap at a given radius from mle, and its 1st/2nd derivatives
            marginalparams: starting values (usually the mle) for the marginal parameters
            opts: an options dictionary for passing user options
            forward: boolean, if true search is done in the forward (positive) radius direction (relative to direction specidied in solver list), if false perform search starting with a negative radius

        Returns:
            Radius: returns the radius corresponding to the chi-squared boundary of the loglikelihood region
            MarginalParams: returns the optimal setting of the marginal parameters at the boundary
            CurrentRatioGap: returns the residual loglikelihood ratio gap at the boundary (it should be small, within tolerance)
        """
        #check if root finding tolerance has been passed if not use default
        if "Tolerance" in opts.keys():
            Tolerance=opts['Tolerance']
        else:
            Tolerance=0.001
        #NOTE: this should be a user option
        #set the max number of iterations in the root finding method
        MaxIterations=50
        #check if the search is run in negative or positive direction, set intial step accordingly
        if forward:
            Radius = opts['InitialStep']
        else:
            Radius = -opts['InitialStep']
        #get the number of marginal parameters
        NumMarginalParams = len(marginalparams)
        #get the solver/function objects, loglik ratio gap and derivatives w.r.t. radius
        ProfileLogLikSolver = solverlist[0]
        ProfileLogLikJacobianSolver = solverlist[1]
        ProfileLogLikHessianSolver = solverlist[2]
        #set the initial marginal parameters
        MarginalParams = marginalparams
        #set the initial LLR gap to a very large number
        CurrentRatioGap = 9e9
        #NOTE: should check to see if we go negative, loop too many time, take massive steps, want to stay in the domain of the MLE
        #NOTE:need max step check, if steps are all in same direction perhaps set bound at inf and return warning, if oscillating error failure to converge
        #create a counter to track root finding iterations
        IterationCounter = 0
        #loop until tolerance criteria are met (LLR gap drops to near zero)
        while  abs(CurrentRatioGap)>Tolerance and IterationCounter<MaxIterations:
            if not NumMarginalParams == 0:
                #if there are marginal parameters
                #run the ipopt solver to optimize the marginal parameters, conditional on the current radius
                ProfileLogLikStruct = ProfileLogLikSolver(x0=MarginalParams,p=Radius)#, lbx=[], ubx=[], lbg=[], ubg=[]
                #solver for the LLR gap 1st derivative w.r.t. the radius
                ProfileLogLikJacobianStruct = ProfileLogLikJacobianSolver(x0=ProfileLogLikStruct['x'], lam_x0=ProfileLogLikStruct['lam_x'], lam_g0=ProfileLogLikStruct['lam_g'],p=Radius)
                #solver for the LLR gap 2nd derivative w.r.t. the radius
                ProfileLogLikHessianStruct = ProfileLogLikHessianSolver(x0=ProfileLogLikStruct['x'], lam_x0=ProfileLogLikStruct['lam_x'], lam_g0=ProfileLogLikStruct['lam_g'],p=Radius)
                #update the current optimal values of the marginal parameters
                MarginalParams = list(ProfileLogLikStruct['x'].full().flatten())
                #update the current LLR gap value
                CurrentRatioGap = ProfileLogLikStruct['f'].full()[0][0]
                #extract the LLR gap 1st derivative value
                dCurrentRatioGap_dRadius = ProfileLogLikJacobianStruct['sym_jac_f_p'].full()[0][0]
                #extract the LLR gap 2nd derivative value
                d2CurrentRatioGap_dRadius2 = ProfileLogLikHessianStruct['sym_hess_f_p_p'].full()[0][0]
            else:
                #else if there are no marginal parameters
                #call the appropriate casadi function to get the current LLR gap value
                CurrentRatioGap = ProfileLogLikSolver(Radius).full()[0][0]
                #call the appropriate casadi function to get the LLR gap 1st derivative value
                dCurrentRatioGap_dRadius = ProfileLogLikJacobianSolver(Radius).full()[0][0]
                #call the appropriate casadi function to get the LLR gap 2nd derivative value
                d2CurrentRatioGap_dRadius2 = ProfileLogLikHessianSolver(Radius).full()[0][0]
            #increment the iterations counter
            IterationCounter+=1
            #use Halley's method (higher order extention of newtons method) to compute the new radius value
            Radius = Radius - (2*CurrentRatioGap*dCurrentRatioGap_dRadius)/(2*dCurrentRatioGap_dRadius**2 - CurrentRatioGap*d2CurrentRatioGap_dRadius2)
        # throw error if maximum number of iterations exceeded
        if IterationCounter>=MaxIterations:
            raise Exception('Maximum number of iterations reached in logliklihood boundary search!')
        #return the radius of the root, the optimal marginal parameters at the root and the CurrentRatioGap at the root (should be near 0)
        return [Radius,MarginalParams,CurrentRatioGap]

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