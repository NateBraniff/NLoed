import casadi as cs
import numpy as np
import math as mt
import copy as cp
from scipy import stats as st
from scipy.interpolate import splev,  splrep
import matplotlib.pyplot as plt

class Model:
    """ 
    A class for statisical models in the NLoed package
    """
    #Easy:      [DONE,  for normal data] implement data sampling
    #Difficult: implement various cov/bias assesment methods,  also (profile) likelihood intervals/plots
    #           add other distributions binomial/bernouli,  lognormal,  gamma,  exponential,  weibull etc.,  negative binomial
    #           implement plotting function (move to utility.py?)
    #NOTE: [maybe a gentle check,  i.e. did you mean this? no constraints]How to handle profile that goes negative for strictly positive parameter values???? perhaps error out and suggest reparameterization???
    #NOTE: can we allow custom pdf's (with some extra work by the user)
    #NOTE: [Yes] Add A-opt,  D-s Opt,  A-s Opt???,  Bias??? for exponential family (and log normal?) 
    #NOTE: https://math.stackexchange.com/questions/269723/largest-eigenvalue-semidefinite-programming
    #NOTE: Data/design/experiment objects may need deep copies so internal lists aren't shared??
    #NOTE: all names must be unique
    #NOTE: must enforce ordering of parameters in statistics function
    #NOTE: we need checks on inputs of fit and sample functions!!!
    def __init__(self,  observ_struct,  input_names,  param_names):
        #check for unique names
        if not(len(set(input_names)) == len(input_names)):
            raise Exception('Model input names must be unique!')
        if not(len(set(param_names)) == len(param_names)):
            raise Exception('Parameter names must be unique!')
        # extract and store dimensions of the model
        self.num_observ = len(observ_struct)
        self.num_param = max(observ_struct[0][2].size_in(0)) #somewhat unsafe,  using max assumes its nx1 or 1xn
        self.num_input = max(observ_struct[0][2].size_in(1))
        #check if we have the same number of input names as inputs
        if not(len(set(input_names)) == len(input_names)):
            raise Exception('Model depends on '+str(self.num_input)+' inputs but there are '+str(len(input_names))+' input names!')
        #check if we have same number of parameter names as parameters
        if not(self.num_param == len(param_names)):
            raise Exception('Model depends on '+str(self.num_param)+' parameters but there are '+str(len(param_names))+' parameter names!')
        #create lists of input/param/param names,  can be used to look up names with indices (observ names filled in below)
        self.input_name_list = input_names
        self.param_name_list= param_names
        self.observ_name_list = [] 
        #create dicts for input/param names,  can be used to look up indices with names
        self.input_name_dict = {}
        self.param_name_dict = {}
        self.observ_name_dict = {}
        #populate the input name dict
        for i in range(self.num_input):
            self.input_name_dict[input_names[i]] = i
        #populate the param name dict
        for i in range(self.num_param):
            self.param_name_dict[param_names[i]] = i
        #create a list to store the observation variable distribution type
        self.distribution = []
        #create a list to store casadi functions predicting the observ. variable sampling distirbution statistics
        self.model = []
        #create a list to store casadi functions predicting the sensitivity of the observ. sampling statistics
        self.sensitivity = []
        #create a list to store casadi functions computing the elemental loglikelihood for each observation variable
        self.loglik = []
        #create a list to store casadi functions computing the elemental fisher info. matrix for each observation variable
        self.fisher_info_matrix = []
        #create symbols for parameters and inputs,  needed for function defs below
        param_symbols = cs.SX.sym('param_symbols', self.num_param)
        input_symbols = cs.SX.sym('input_symbols', self.num_input)
        #loop over the observation variables
        for i in range(self.num_observ):
            observ_name = observ_struct[i][0]
            observ_distribution = observ_struct[i][1]
            observ_model= observ_struct[i][2]
            #store the distribution type for later
            self.distribution.append(observ_distribution)
            #extract names of observ_struct variables
            if not(observ_name in self.observ_name_dict):
                self.observ_name_dict[observ_name] = i
                self.observ_name_list.append(observ_name)
            else:
                raise Exception('Observation names must be unique!')
            #create a observ_struct symbol
            observ_symbol = cs.SX.sym(observ_name, 1)
            #store the function for the model (links observ_struct distribution parameters to the parameters-of-interest)
            self.model.append(observ_model)
            if observ_distribution  == 'Normal':
                #get the distribution statistics
                mean_symbol = observ_model(param_symbols, input_symbols)[0]
                variance_symbol = observ_model(param_symbols, input_symbols)[1]
                #create LogLikelihood symbolics and function 
                loglik_symbol = -0.5*cs.log(2*cs.pi*variance_symbol) - (observ_symbol - mean_symbol)**2/(2*variance_symbol)
                self.loglik.append( cs.Function('ll_'+observ_name,  [observ_symbol, param_symbols, input_symbols],  [loglik_symbol]) )
                #generate derivatives of distribution parameters,  StatisticModel (here mean and variance) with respect to parameters-of-interest,  Params
                mean_sensitivity_symbol = cs.jacobian(mean_symbol, param_symbols)
                variance_sensitivity_symbol = cs.jacobian(variance_symbol, param_symbols)
                #create sensitivity functions for the mean and variance
                mean_sensitivity_func = cs.Function('mean_sens_'+observ_name,  [param_symbols, input_symbols],  [mean_sensitivity_symbol]) 
                variance_sensitivity_func = cs.Function('var_sens_'+observ_name,  [param_symbols, input_symbols],  [variance_sensitivity_symbol]) 
                #store sensitivity functions for the the model
                self.sensitivity.append([mean_sensitivity_func, variance_sensitivity_func])
                #create FIM symbolics and function
                fisher_info_symbol = ((mean_sensitivity_symbol.T @ mean_sensitivity_symbol)/variance_symbol 
                                        + (variance_sensitivity_symbol.T @ variance_sensitivity_symbol)/variance_symbol**2)
                self.fisher_info_matrix.append(cs.Function('FIM_'+observ_name,  [param_symbols, input_symbols],  [fisher_info_symbol]) )
            elif observ_distribution == 'Poisson':
                #get the distribution statistic 
                lambda_symbol = observ_model(param_symbols, input_symbols)[0]
                #create a custom casadi function for doing factorials (needed in poisson LogLikelihood and FIM)
                casadi_factorial = factorial('fact')
                #store the function in the class so it doesn't go out of scope
                self.__factorialFunc = casadi_factorial
                #create LogLikelihood symbolics and function 
                loglik_symbol =  observ_symbol*cs.log(lambda_symbol)+casadi_factorial(observ_symbol)-lambda_symbol
                self.loglik.append( cs.Function('ll_'+observ_name,  [observ_symbol, param_symbols, input_symbols],  [loglik_symbol]) )
                #generate derivatives of distribution parameters,  StatisticModel (here Mean and variance) with respect to parameters-of-interest,  Params
                lamba_sensitivity_symbol = cs.jacobian(lambda_symbol, param_symbols)
                #create sensitivity function for lambda
                lambda_sensitivity_func = cs.Function('var_sens_'+observ_name,  [param_symbols, input_symbols],  [variance_sensitivity_symbol]) 
                #store sensitivity function for lambda
                self.sensitivity.append([lambda_sensitivity_func])
                #create FIM symbolics and function
                fisher_info_symbol = (lamba_sensitivity_symbol.T @ lamba_sensitivity_symbol)/lambda_symbol
                self.fisher_info_matrix.append(cs.Function('FIM_'+observ_name,  [param_symbols, input_symbols],  [fisher_info_symbol]) )
            elif observ_distribution == 'Lognormal':    
                print('Not Implemeneted')
            elif observ_distribution == 'Binomial': 
                print('Not Implemeneted')
            elif observ_distribution == 'Exponential': 
                print('Not Implemeneted')
            elif observ_distribution == 'Gamma': 
                print('Not Implemeneted')
            else:
                raise Exception('Unknown Distribution: '+observ_distribution)

    def fit(self, dataset_struct, start_param, options=None):
        """
        This function fits the model to a dataset using maximum likelihood
        it also provides optional marginal confidence intervals
        and plots of logliklihood profiles/traces and marginal projected confidence contours

        Args:
            datasets: either; a dictionary for one dataset OR a list of dictionaries,  each design replicats OR a list of lists of dict's where each index in the outer lists has a unique design
            start_param: a list of starting values for the parameters
            options: optional,  a dictionary of user defined options

        Return:
            ParameterFitStruct: either; a list of lists structure with parameter fit lists,  has the same shape/order as the input,  OR,  the same strcture but with fits and intervals as the leaf object
        """
        #NOTE: needs checks on inputs
        #NOTE: NEED testing for multiple observation input structures,  multiple dimensions of parameters ideally,  1, 2, 3 and 7+
        #NOTE: add some print statments to provide user with progress status
        #NOTE: currently solve multiplex simultaneosly (one big opt problem) but sequentially may be more robust to separation (discrete data),  or randomly non-identifiable datasets (need to test)
        #this block allows the user to pass a dataset, list of datasets,  list of lists etc. for Design x Replicat fitting
        if not(isinstance(dataset_struct,  list)):
            #if a single dataset is passed vis the dataset_struct input,  wrap it in two lists so it matches general case
            design_datasets = [[dataset_struct]]
        elif not(isinstance(dataset_struct[0],  list)):
            #else if dataset_struct input is a list of replciated datasets,  wrap it in a single list to match general case
            design_datasets = [dataset_struct]
        else:
            #else if dataset_struct input is a list of designs, each with a list of replicats,  just pass on th input
            design_datasets = dataset_struct
        #create a list to store parameter casadi symbols used for ML optimization
        param_symbols_list = []
        #create a list to store starting parameters (they are all determined by start_param,  but dim. depends on design x replicat size)
        start_params_list = []
        #create a list to store casadi generic loglikelihood functions for each design
        design_loglik_func_list = []
        #create a total loglikelihood summation store, initialize to zero
        total_loglik_symbol = 0
        #create an archetypal vector of paramter symbols, used to build casadi loglikelihood functions for each design
        archetype_param_symbols = cs.SX.sym('archetype_param_symbols', self.num_param)
        #loop over different designs (outer most list)
        for e in range(len(design_datasets)):
            #get the set of replicats for this design
            replicat_datasets = design_datasets[e]
            #create a list to store loglikelihood functions for each specific replicat of the design
            replicat_loglik_func_list = []
            #for each design use the first replicat 
            archetype_data = replicat_datasets[0]
            #create a summation variable for the loglikelihood for a loglik_symbolset of the current design
            archetype_loglik_symbol = 0
            #create a vector of all observations in the design
            example_observ_list = [element for row in archetype_data['Observation'] for group in row for element in group]
            #create a vector of casadi symbols for the observations
            archetype_observ_symbol = cs.SX.sym('archetype_observ_symbol'+str(e), len(example_observ_list))
            #create a counter to index the total number of observations
            sample_count = 0
            #loop over the dataset inputs
            for i in range(len(archetype_data['Inputs'])):
                #get the curren input settings
                input_row = archetype_data['Inputs'][i]
                #loop over the observation varaibles
                for j in range(self.num_observ):
                    #for the given dataset loop over (the potentially replicated) observations for the given observation variable 
                    #if no observations are taken at the given observation variable len=0 and we skip
                    #NOTE: NEED TO CHECK THIS WORKS FOR MULTI OBSERV DATA
                    for k in range(len(archetype_data['Observation'][i][j])):
                        #create a symbol for the loglikelihood for the given input and observation variable
                        archetype_loglik_symbol += self.loglik[j](archetype_observ_symbol[sample_count], archetype_param_symbols, input_row)
                        #increment the observation counter
                        sample_count += 1
            #create a casadi function for the loglikelihood of the current design (observations are free/input symbols)
            archetype_loglik_func = cs.Function('archetype_loglik_func'+str(e),  [archetype_observ_symbol, archetype_param_symbols],  [archetype_loglik_symbol])
            #loop over replicats within each design
            for r in range(len(replicat_datasets)):
                #NOTE: could abstract below into a Casadi function to avoid input/observ loop on each dataset and replicat
                #get the dataset from the replicat list
                dataset = replicat_datasets[r]
                #create a vector of parameter symbols for this specific dataset,  each dataset gets its own,  these are used for ML optimization
                fit_param_symbols = cs.SX.sym('fit_param_symbols'+'_'+str(e)+str(r), self.num_param)
                #extract the vector of observations in the same format as in the archetype_loglik_func function input
                observ_list = cs.vertcat(*[cs.SX(element) for row in dataset['Observation'] for group in row for element in group])
                #create a symbol for the datasets loglikelihood function by pass in the observations for the free symbols in ObservSymbol
                dataset_loglik_symbol = archetype_loglik_func(observ_list, fit_param_symbols)
                #create a function for
                dataset_loglik_func = cs.Function('dataset_loglik_func_'+str(e)+'_'+str(r),  [fit_param_symbols],  [dataset_loglik_symbol])
                replicat_loglik_func_list.append(dataset_loglik_func)
                #set up the logliklihood symbols for given design and replicat
                param_symbols_list.append(fit_param_symbols)
                #record the starting parameters for the given replicat and dataset
                start_params_list.extend(start_param)
                #add the loglikelihood to the total 
                #NOTE: this relies on the distirbutivity of serperable optimization problem,  should confirm
                total_loglik_symbol += dataset_loglik_symbol
            #append the list of replciate loglik functions to the design list
            design_loglik_func_list.append(replicat_loglik_func_list)
        #NOTE: this approach is much more fragile to separation (glms with discrete response),  randomly weakly identifiable datasets
        #NOTE: should be checking solution for convergence, should allow user to pass options to ipopt
        #NOTE: allow bfgs for very large nonlinear fits, may be faster
        # Create an IPOPT solver for overall maximum likelihood problem,  we pass negative total_loglik_symbol because IPOPT minimizes
        total_loglik_optim_struct = {'f': -total_loglik_symbol,  'x': cs.vertcat(*param_symbols_list)}#,  'g': cs.vertcat(*OptimConstraints)
        param_fitting_solver = cs.nlpsol('solver',  'ipopt',  total_loglik_optim_struct, {'ipopt.print_level':5, 'print_time':False})
        # Solve the NLP fitting problem with IPOPT call
        param_fit_solution_struct = param_fitting_solver(x0=start_params_list)#,  lbx=[],  ubx=[],  lbg=[],  ubg=[]
        #extract the fit parameters from the solution structure
        fit_param = list(param_fit_solution_struct['x'].full().flatten())
        #NOTE: this is (very slightly) in efficient to do this loop twice but it makes it more readable
        #NOTE: some extra space taken up by CI lists if not requested but not a big deal
        #create a list to store fits params for each design
        design_param_fit_list = []
        #create a list to store param CI's for each design
        design_param_interval_list = []
        #loop over each design
        for e in range(len(design_datasets)):
            #create a list to store fits params for each replicat
            replicat_param_fit_list=[]
            #create a list to store param CI's for each replicat
            replicat_param_interval_list=[]
            #loop over each replicat of the given design
            for r in range(len(design_datasets[e])):
                #extract the fit parameters from the optimization solution vector
                fit_param_set= fit_param[:self.num_param]
                #check if 'confidence' key word is passed in the options dictionary
                if "Confidence" in options.keys():
                    #check if graphing options; contour plots or profile trace plots,  are requested
                    if options['Confidence']=="Contours" or options['Confidence']=="Profiles":
                        #if so, create a figure to plot on
                        fig = plt.figure()
                        #run profileplots to plot the profile traces and return CI's
                        interval_list = self.__profileplot(fit_param_set, design_loglik_func_list[e][r], fig, options)[0]
                        #add the CI's to the replicate CI list
                        replicat_param_interval_list.append(interval_list)
                        #if contour plots are requested specifically, run contour function to plot the projected confidence contours
                        if options['Confidence']=="Contours":
                            self.__contourplot(fit_param_set, design_loglik_func_list[e][r], fig, options)
                    elif options['Confidence']=="Intervals":
                        #if confidence intervals are requested, run confidenceintervals to get CI's and add them to replicat list
                        replicat_param_interval_list.append(self.__confidence_intervals(fit_param_set, design_loglik_func_list[e][r], options))
                #remove the current extracted parameters from the solution vector
                del fit_param[:self.num_param]
                #add the extracted parameters to the replicat parameter lsit
                replicat_param_fit_list.append(fit_param_set)
            #add the replicat list to the design list
            design_param_fit_list.append(replicat_param_fit_list)
            #add replicat CI list to design CI list
            design_param_interval_list.append(replicat_param_interval_list)
        #if we need to plot profile traces or contours 
        if "Confidence" in options.keys() and (options['Confidence']=="Contours" or options['Confidence']=="Profiles"):
            plt.show()
        #depending on dimension of input datasets (i.e. single, replcated design, multiple rep'd designs),  package output to match
        if not(isinstance(dataset_struct,  list)):
            #if a single dataset dict. was passed, return a param vec and CI list accordingly
            design_param_fit_list = design_param_fit_list[0][0]
            design_param_interval_list = design_param_interval_list[0][0]
        elif not(isinstance(dataset_struct[0],  list)):
            #if a list a replicated design was passed, return a a list of param vecs and CI lists accordingly
            design_param_fit_list = design_param_fit_list[0]
            design_param_interval_list = design_param_interval_list[0]
        else:
            #if a list designs,  each listing replicats, was passed, return a a list of lists of param vecs and CI lists accordingly
            design_param_fit_list = design_param_fit_list
            design_param_interval_list = design_param_interval_list
        #check if CI's were generated, is so return both param fits and CI's in a list, else a just param fits
        #NOTE: this may be a bit awkward, should possibly keep it standardized
        if "Confidence" in options.keys() and (options['Confidence']=="Intervals" or options['Confidence']=="Contours" or options['Confidence']=="Profiles"):
            param_fit_struct = [design_param_fit_list,  design_param_interval_list]
        else:
            param_fit_struct = design_param_fit_list
        #return the appropriate return structure
        return param_fit_struct

    def sample(self, design_struct, param, replicats=1):
        """
        This function generates datasets for a given design or list of designs,  with a given set of parameters,  with an optional number of replicates

        Args:
            designs: either; a single design dictionary,  OR,  a list of design dictionaries
            parameters: the parameter values at which to generate the data
            replicates: optional,  an integer indicating the number of datasets to generate for each design,  default is 1

        Return:
            design_datasets: either; a singel dataset for the given design,  OR,  a list of dataset replicats for the given design,  OR,  a list of list of dataset replicats for each design
        """
        #NOTE: needs checks on inputs
        #NOTE: multiplex multiple parameter values??
        #NOTE: actually maybe more important to be able to replicat designs N times
        #check if designs is a single design or a list of designs
        if not(isinstance(design_struct, list)):
            #if single,  wrap it in a list to match general case
            design_list = [design_struct]
        else:
            #else pass it as it is
            design_list = design_struct
        #create a list to store lists of datasets for each design
        design_datasets = []
        #loop over the designs
        for e in range(len(design_list)):
            #extract the current design
            design = design_list[e]
            #deep copy the current design to create an archetype for that design's datasets
            archetype_dataset = cp.deepcopy(design)
            #delete the 'count' field in the design,  as we are using it as a template for datasets
            del archetype_dataset['Count']
            #create 'observations' field in the archetype datset
            archetype_dataset['Observation'] = []
            #create a list for replciated datasets of the current design
            replicat_datasets = []
            #loop over the number of replicates
            for r in range(replicats):
                #create a deep copy of the dataset archetype for the current dataset
                dataset = cp.deepcopy(archetype_dataset)
                #loop over the (unique) input rows in the design
                for i in range(len(design['Inputs'])):
                    #get the input row
                    input_row = design['Inputs'][i]
                    #create a list for the current input row's observations
                    observ_row = []
                    #loop over the observation variables
                    for j in range(self.num_observ):
                        #get the count of observations for the given observ. var at the given input row
                        observ_count = design['Count'][i][j]
                        #select the appropriate distribution,  and generate the sample vec of appropriate length
                        if self.distribution[j] == 'Normal':
                            #compute the sample distirbution statistics using the model
                            [mean,variance] = self.model[j](param, input_row)
                            observ = np.random.normal(mean,  np.sqrt(variance),  observ_count).tolist() 
                        elif self.distribution[j] == 'Poisson':
                            lambda_ = self.model[j](param, input_row)
                            observ = np.random.poisson(lambda_).tolist() 
                        elif self.distribution[j] == 'Lognormal':
                            print('Not Implemeneted')
                        elif self.distribution[j] == 'Binomial':
                            print('Not Implemeneted')
                        elif self.distribution[j] == 'Exponential':
                            print('Not Implemeneted')
                        elif self.distribution[j] == 'Gamma':
                            print('Not Implemeneted')
                        else:
                            raise Exception('Unknown error encountered selecting observation distribution,  contact developers')
                        #add the current observation vector,  for the current observ. var.,  to the observ row
                        observ_row.append(list(observ))
                    #add the observation row to the current dataset
                    dataset['Observation'].append(observ_row)
                #add the current dataset to the replicat list
                replicat_datasets.append(dataset)
            #add the replicat list to the design list
            design_datasets.append(replicat_datasets)
        #check if a single design was passed
        if not(isinstance(design_struct,  list)):
            if replicats==1:
                #if a single design was passed and there replicate count is 1,  return a single dataset
                return design_datasets[0][0]
            else:
                #else if a single design was passed,  but with >1 reps,  return a list of datasets
                return design_datasets[0]
        else:
            #else if multiple designs were passed (with/without reps),  return a list of list of datasets
            return design_datasets

    #NOTE: should maybe rename this
    def eval_design(self):
        #maybe this should move to the design class(??)
        #For D (full cov/bias),  Ds (partial cov/bias),  T separation using the delta method?! but need two models
        # assess model/design,  returns various estimates of cov,  bias,  confidence regions/intervals
        # no data: asymptotic: covaraince,  beale bias,  maybe MSE
        #          sigma point: covariance,  bias (using mean) (need to figure out how to do sigma for non-normal data),  maybe MSE
        #          monte carlo: covariance,  bias,  MSE
        
        print('Not Implemeneted')
        
    # UTILITY FUNCTIONS
    def eval_model(self, param, inputs, param_covariance=None, sensitivity=False, observ_indices=None, options=None):
        #NOTE: evaluate model,  predict y
        #NOTE: optional pass cov matrix,  for use with delta method/MC error bars on predictions
        if not observ_indices:
            observ_list = list(range(self.num_observ))
        else:
            observ_list = observ_indices

        if not options:
            options={}

        if 'ErrorMethod' in options.keys():
            error_method = options["ErrorMethod"]
        else:
            error_method='Delta'

        if 'SampleSize' in options.keys():
            num_mc_samples = options["SampleSize"]
        else:
            num_mc_samples=10000

        if 'ErrorBars' in options.keys():
            alpha = options["ConfidenceLevel"]
        else:
            alpha=0.95

        statistic_list = []
        sensitivity_list = []
        error_bounds_list = []
        for o in observ_list:
            stat_row = []
            sensitivity_row = []
            bound_row=[]
            for s in range(len(self.model)):
                statistic = self.model[o][s](param, inputs).full()[0][0]
                stat_row.append(statistic)
                if sensitivity:
                    sensitivity_vec = self.sensitivity[o][s](param, inputs).full()[0][0]
                    sensitivity_row.append(sensitivity_vec)
                if param_covariance:
                    if error_method == "Delta":
                        if not sensitivity:
                            sensitivity_vec = self.sensitivity[o][s](param, inputs).full()[0][0]
                        delta= np.sqrt(sensitivity_vec.T @ param_covariance @sensitivity_vec)
                        bounds=[statistic-delta,statistic+delta]
                    elif error_method == "MonteCarlo":
                        stat_func=self.model[o][s]
                        param_sample = np.random.normal(param,  param_covariance,  num_mc_samples).tolist() 
                        mc_sample = []
                        for par in param_sample:
                            mc_sample.append(stat_func(par, inputs))
                        bounds = np.percentile(mc_sample, [100*(1-alpha)/2, 100*(1.5-alpha/2)])
                    else:
                        raise Exception('No such option; '+str(error_method)+' exists for field \'ErrorMethod\'!')
                    bound_row.append(bounds)
            statistic_list.append(stat_row)
            sensitivity_list.append(sensitivity_row)
            error_bounds_list.append(bound_row)

        prediction_struct=[statistic_list]
        if sensitivity:
            prediction_struct.append(sensitivity_list)
        if param_covariance:
            prediction_struct.append(error_bounds_list)

        return prediction_struct

    # def evalfim(self):
    #     #NOTE: eval fim at given inputs and dataset
    #     #NOTE: should this even be here??? how much in model,  this isn't data dependent,  only design dependent
    #     print('Not Implemeneted')

    # def evalloglik(self):
    #     #eval the logliklihood with given params and dataset
    #     print('Not Implemeneted')

    # def plots(self):
    #     #FDS plot,  standardized variance (or Ds,  bayesian equivlant),  residuals
    #     print('Not Implemeneted')
    #     #NOTE: maybe add a basic residual computation method for goodness of fit assesment?? Or maybe better show how in tutorial but not here

# --------------- Private functions and subclasses ---------------------------------------------

    def __confidence_intervals(self, mle_params, loglik_func, options):
        """ 
        This function computes marginal parameter confidence intervals for the model
        around the MLE estimate using the profile likelihood

        Args:
            mle_params: mle parameter estimates,  recieved from fitting
            loglik_func: casadi logliklihood function for the given dataset
            options: an options dictionary for passing user options

        Returns:
            interval_list: list of lists of upper and lower bounds for each parameter
        """
        #create a list to store intervals
        interval_list = []
        #loop over parameters in model
        for p in range(self.num_param):
            #fix parameter along which profile is taken
            fixed_param = [False]*self.num_param
            fixed_param[p] = True
            #set direction so that it has unit length in profile direction
            direction = [0]*self.num_param
            direction[p] = 1
            #setup the profile solver
            solver_list = self.__profilesetup(mle_params, loglik_func, fixed_param, direction, options)
            #extract starting values for marginal parameters (those to be optimized during profile)
            marginal_param = [mle_params[i] for i in range(self.num_param) if direction[i]==0]
            #search to find the radius length in the specified profile direction,  positive search
            upper_radius = self.__logliksearch(solver_list, marginal_param, options, True)[0]
            #compute the location of the upper parameter bound
            upper_bound = mle_params[p] + direction[p] * upper_radius
            #search to find the radius length in the specified profile direction,  negative search
            lower_radius  =self.__logliksearch(solver_list, marginal_param, options, False)[0]
            #compute the location of the lower parameter bound
            lower_bound = mle_params[p]+direction[p]*lower_radius
            interval_list.append([lower_bound, upper_bound])
        return interval_list

    def __profileplot(self, mle_params, loglik_func, figure, options):
        """ 
        This function plots profile parameter traces for each parameter value

        Args:
            mle_params: mle parameter estimates,  recieved from fitting
            loglik_func: casadi logliklihood function for the given dataset
            figure: the figure object on which plotting occurs
            options: an options dictionary for passing user options

        Returns:
            interval_list: list of lists of upper and lower bounds for each parameter
            trace_list: list of list of lists of parameter vector values along profile trace for each parameter
            profile_list: List of lists of logliklihood ratio values for each parameter along the profile trace
        """
        #extract the alpha level and compute the chisquared threshold,  or use default of 0.95
        if "ConfidenceLevel" in options.keys():
            alpha = options['ConfidenceLevel']
        else:
            alpha=0.95
        chi_squared_level = st.chi2.ppf(alpha,  self.num_param)
        #run profile trave to get the CI's,  parameter traces,  and LR profile
        [interval_list, trace_list, profile_list] = self.__profiletrace(mle_params, loglik_func, options)
        #loop over each pair of parameters
        for p1 in range(self.num_param):
            for p2 in range(p1, self.num_param):
                #check if parameter pair matches
                if p1 == p2:
                    #if on the diagonal,  generate a profile plot
                    #get data for the profile
                    x = [trace_list[p1][ind][p1] for ind in range(len(trace_list[p1]))]
                    y = profile_list[p1]
                    #get data for the threshold
                    x0 = [x[0], x[-1]]
                    y0 = [chi_squared_level, chi_squared_level]
                    #plot the profile and threshold
                    plt.subplot(self.num_param,  self.num_param,  p2*self.num_param+p1+1)
                    plt.plot(x,  y)
                    plt.plot(x0,  y0,  'r--')
                    plt.xlabel(self.param_name_list[p1])
                    plt.ylabel('LogLik Ratio')
                else:
                    #if off-diagonal generate a pair of parameter profile trace plots
                    #plot the profile parameter trace for p1
                    plt.subplot(self.num_param,  self.num_param,  p2*self.num_param+p1+1)
                    x1 = [trace_list[p1][ind][p1] for ind in range(len(trace_list[p1]))]
                    y1 = [trace_list[p1][ind][p2] for ind in range(len(trace_list[p1]))]
                    plt.plot(x1,  y1, label=self.param_name_list[p1]+'profile')
                    #plot the profile parameter trace for p2
                    x2 = [trace_list[p2][ind][p1] for ind in range(len(trace_list[p2]))]
                    y2 = [trace_list[p2][ind][p2] for ind in range(len(trace_list[p2]))]
                    plt.plot(x2,  y2, label=self.param_name_list[p2]+'profile')
                    plt.legend()
                    plt.xlabel(self.param_name_list[p1])
                    plt.ylabel(self.param_name_list[p2])
        #return CI,  trace and profilem (for extensibility)
        return [interval_list, trace_list, profile_list]

    def __profiletrace(self, mle_params, loglik_func, options):
        """ 
        This function compute the profile logliklihood parameter trace for each parameter in the model

        Args:
            mle_params: mle parameter estimates,  recieved from fitting
            loglik_func: casadi logliklihood function for the given dataset
            options: an options dictionary for passing user options

        Returns:
            interval_list: list of lists of upper and lower bounds for each parameter
            trace_list: list of list of lists of parameter vector values along profile trace for each parameter
            profile_list: List of lists of logliklihood ratio values for each parameter along the profile trace
        """
        #extract the alpha level and compute the chisquared threshold,  or use default of 0.95
        if "ConfidenceLevel" in options.keys():
            alpha = options['ConfidenceLevel']
        else:
            alpha = 0.95
        chi_squared_level = st.chi2.ppf(alpha,  self.num_param)
        #extract the number of points to compute along the trace,  or use default of 10
        if "SampleNumber" in options.keys():
            num_points = options['SampleNumber']
        else:
            num_points = 10
        #create lists to store the CI's,  profile logliklkhood values and parameter traces
        interval_list = []
        profile_list = []
        trace_list = []
        #loop over each parameter in the model
        for p in range(self.num_param):
            #indicate the parameter,  along which the profile is taken,  is fixed
            fixed_param = [False]*self.num_param
            fixed_param[p] = True
            #set the direction of the profile so that it has unit length
            direction = [0]*self.num_param
            direction[p] = 1
            #generate the profile solvers
            solver_list = self.__profilesetup(mle_params, loglik_func, fixed_param, direction, options)
            #set the starting values of the marginal parameters from the mle estimates
            marginal_param = [mle_params[i] for i in range(self.num_param) if not fixed_param[i]]
            #preform a profile search to find the upper bound on the radius for the profile trace
            [upper_radius, upper_param_list, upper_loglik_ratio_gap] = self.__logliksearch(solver_list, marginal_param, options, True)
            #compute the parameter upper bound
            upper_bound = mle_params[p] + direction[p]*upper_radius
            #insert the profile parameter (upper) in the marginal parameter vector (upper),  creates a complete parameter vector
            upper_param_list.insert(p, upper_bound)
            #preform a profile search to find the lower bound on the radius for the profile trace
            [lower_radius, lower_param_list, lower_loglik_ratio_gap] = self.__logliksearch(solver_list, marginal_param, options, False)
            #compute the parameter lower bound
            lower_bound = mle_params[p] + direction[p]*lower_radius
            #insert the profile parameter (lower) in the marginal parameter vector (lower),  creates a complete parameter vector
            lower_param_list.insert(p, lower_bound)
            #record the uppper and lower bounds in the CI list
            interval_list.append([lower_bound, upper_bound])
            #Create a grid of radia from the lower radius bound to the upper radius bound with the number of points requested in the profile
            radius_list = list(np.linspace(lower_radius,  upper_radius,  num=num_points+1, endpoint=False)[1:])
            #extract the marginal logliklihood solver,  to compute the profile
            profile_loglik_solver = solver_list[0]
            #insert the lower parameter bound and the logliklihood ratio in the trace list and profile list respectivly 
            param_trace = [lower_param_list]
            loglik_ratio_profile = [chi_squared_level - lower_loglik_ratio_gap]
            #loop over the radius grid 
            for r in radius_list:
                # Solve the for the marginal maximumlikelihood estimate
                profile_solution_struct = profile_loglik_solver(x0=marginal_param, p=r)#,  lbx=[],  ubx=[],  lbg=[],  ubg=[]
                #extract the current logliklihood ratio gap (between the chi-sqaured level and current loglik ratio)
                ratio_gap = profile_solution_struct['f'].full()[0][0]
                #extract the marginal parameter vector
                #NOTE: need to test how low dim. (1-3 params) get handled in this code,  will cause errors for 1 param models !!
                marginal_param = list(profile_solution_struct['x'].full().flatten())
                #copy and insert the profile parameter in the marginal vector
                param_list = cp.deepcopy(marginal_param)
                param_list.insert(p, direction[p]*r+mle_params[p])
                #insert the full parameter vector in the trace lists
                param_trace.append(param_list)
                #insert the likelihood ratio for the current radius
                loglik_ratio_profile.append(chi_squared_level - ratio_gap)
            #insert the upper bound in the parameter trace after looping over the grid
            param_trace.append(upper_param_list)
            #insert the upper loglik ratio in the profile list
            loglik_ratio_profile.append(chi_squared_level - upper_loglik_ratio_gap)
            #insert the final loglik profile in the profile list ,  recording the current parameter's trace
            profile_list.append(loglik_ratio_profile)
            #insert the final parameter trace into the trace list,  recording the current parameter's profile
            trace_list.append(param_trace)
        #return the intervals,  parameter trace and loglik ratio profile
        return [interval_list, trace_list, profile_list]

    def __contourplot(self, mle_params, loglik_func, figure, options):
        """ 
        This function plots the projections of the confidence volume in a 2d plane for each pair of parameters
        this creates marginal confidence contours for each pair of parameters

        Args:
            mle_params: mle parameter estimates,  recieved from fitting
            loglik_func: casadi logliklihood function for the given dataset
            figure: the figure object on which plotting occurs
            options: an options dictionary for passing user options
        """
        #loop over each unique pair of parameters 
        for p1 in range(self.num_param):
            for p2 in range(p1+1, self.num_param):
                #compute the x and y values for the contour trace
                [x_fit, y_fit] = self.__contourtrace(mle_params, loglik_func, [p1, p2], options)
                #plot the contour on the appropriate subplot (passed in from fit function,  shared with profileplot)
                plt.subplot(self.num_param,  self.num_param,  p2*self.num_param+p1+1)
                plt.plot(x_fit,  y_fit, label=self.param_name_list[p1]+' '+self.param_name_list[p2]+' contour')
                plt.legend()
                plt.xlabel(self.param_name_list[p1])
                plt.ylabel(self.param_name_list[p2])

    def __contourtrace(self, mle_params, loglik_func, coordinates, options):
        """ 
        This function plots the projections of the confidence volume in a 2d plane for each pair of parameters
        this creates marginal confidence contours for each pair of parameters

        Args:
            mle_params: mle parameter estimates,  recieved from fitting
            loglik_func: casadi logliklihood function for the given dataset
            coordinates: a pair of parameter coordinates specifying the 2d contour to be computed in parameter space
            options: an options dictionary for passing user options

        Returns:
            [x_fit, y_fit]: x, y-values in parameter space specified by coordinates tracing the projected profile confidence contour outline
        """
        if "RadialNumber" in options.keys():
            num_radial = options['RadialNumber']
        else:
            num_radial = 30
        #extract the parameter coordinat indicies for the specified trace
        p1 = coordinates[0]
        p2 = coordinates[1]
        #mark extracted indices as fixed for the loglik search
        fixed_param = [False]*self.num_param
        fixed_param[p1] = True
        fixed_param[p2] = True
        #set the starting values for the marginal parameters based on the mle estimate
        marginal_param = [mle_params[i] for i in range(self.num_param) if fixed_param[i]==0]
        #create a list of angles (relative to the mle,  in p1-p2 space) overwhich we perform the loglik search to trace the contour
        angle_list = list(np.linspace(-mt.pi,  mt.pi, num_radial))
        #create an empty list to sore the radiai resulting from the search
        radius_list = []
        #loop over the angles
        for angle in angle_list:
            #compute the sine and cosine of the angle
            angle_cosine = mt.cos(angle)
            angle_sine = mt.sin(angle)
            #compute the direction in p1-p2 space for the search
            direction = [0]*self.num_param
            direction[p1] = angle_cosine
            direction[p2] = angle_sine
            #setup the solver for the search
            solver_list = self.__profilesetup(mle_params, loglik_func, fixed_param, direction, options)
            #run the profile loglik search and return the found radia for the given angle
            radius = self.__logliksearch(solver_list, marginal_param, options, True)[0]
            #record the radius
            radius_list.append(radius)
        #fit a periodic spline to the Radius-Angle data
        radial_spline_fit = splrep(angle_list, radius_list, per=True)
        #generate a dense grid of angles to perform interpolation on
        angle_interpolants = np.linspace(-mt.pi,  mt.pi, 1000)
        #compute the sine and cosine for each interpolation angle
        angle_interp_cosine = [mt.cos(a) for a in angle_interpolants]
        angle_interp_sine = [mt.sin(a) for a in angle_interpolants]
        #use the periodic spline to interpolate the radius over the dense interpolation angle grid
        radial_interpolation = splev(angle_interpolants, radial_spline_fit)
        #compute the resulting x and y coordinates for the contour in the p1-p2 space
        x_fit = [angle_interp_cosine[i]*radial_interpolation[i]+mle_params[p1] for i in range(len(angle_interpolants))]
        y_fit = [angle_interp_sine[i]*radial_interpolation[i]+mle_params[p2] for i in range(len(angle_interpolants))]
        #return the contour coordinates
        return [x_fit, y_fit]
        #NOTE: should maybe pass profile extrema from CI's into contours to add to fit points in interpolation
        #        it is unlikely we will 'hit' absolute extrema unless we have very dense sampling,  splines don't need an even grid

    def __profilesetup(self, mle_params, loglik_func, fixedparams, direction, options):
        """ 
        This function creates function/solver objects for performing a profile likelihood search for the condifence boundary
        in the specified direction,  the function/solver objects compute the logliklihood ratio gap
        at a given radius (along the specified direction),  along with the LLR gaps 1st and 2nd derivative with respect to the radius.
        marginal (free) parameters (if they exist) are optimized conditional on the fixed parameters specified by the radius and direction
        the likelihood ratio gap is the negative difference between the chi-squared boundary and the loglik ratio at the current radius

        Args:
            mle_params: mle parameter estimates,  recieved from fitting
            loglik_func: casadi logliklihood function for the given dataset
            fixedparams: a boolean vector,  same length as the parameters,  true means cooresponding parameters fixed by direction and radius,  false values are marginal and optimized (if they exist)
            direction: a direction in parameter space,  coordinate specified as true in fixedparams are used as the search direction
            options: an options dictionary for passing user options

        Returns:
            profile_loglik_solver: casadi function/ipopt solver that returns the loglik ratio gap for a given radius,  after optimizing free/marginal parameters if they exist
            profile_loglik_jacobian_solver: casadi function/ipopt derived derivative function that returns the derivative of the loglik ratio gap with respect to the radius (jacobian is 1x1)
            profile_loglik_hessian_solver: casadi function/ipopt derived 2nd derivative function that returns the 2nd derivative of the loglik ratio gap with respect to the radius (hessian is 1x1)
        """
        #check if confidence level is passed if not default to 0.95
        if "ConfidenceLevel" in options.keys():
            alpha = options['ConfidenceLevel']
        else:
            alpha = 0.95
        #compute the chi-squared level from alpha
        chi_squared_level = st.chi2.ppf(alpha,  self.num_param)
        #compute the number of fixed parameters (along which we do boundary search,  radius direction)
        num_fixed_param = sum(fixedparams)
        #compute the number of free/marginal parameters,  which are optimized at each step of the search
        num_marginal_param = self.num_param-num_fixed_param
        if num_fixed_param == 0:
            raise Exception('No fixed parameters passed to loglikelihood search,  contact developers!')
        #create casadi symbols for the marginal parameters
        marginal_param_symbols = cs.SX.sym('marginal_param_symbols', num_marginal_param)
        #create casadi symbols for the radius (radius measured from mle,  in given direction,  to chi-squared boundary)
        radius_symbol = cs.SX.sym('radius_symbol')
        #creat a list to store a complete parameter vector
        #this is a mixture of fixed parameters set by the direction and radius,  and marginal parameters which are free symbols
        param_list = []
        #create a counter to count marginal parameters already added
        marginal_counter = 0
        #loop over the parameters
        for i in range(self.num_param):   
            if fixedparams[i]:
                #if the parameter is fixed,  add an entry parameterized by the radius from the mle in given direction
                param_list.append( direction[i]*radius_symbol + mle_params[i] )
            else:
                #else add marginal symbol to list and increment marginal counter
                param_list.append(marginal_param_symbols[marginal_counter])
                marginal_counter+=1
        #convert the list of a casadi vector
        param = cs.vertcat(*param_list)
        #create a symnol for the loglikelihood ratio gap at the parameter vector
        loglik_ratio_gap_symbol = 2*( loglik_func(mle_params) - loglik_func(param)) - chi_squared_level
        #check if any marginal parameters exist
        if not num_marginal_param == 0:
            #if there are marginal parameters create Ipopt solvers to optimize the marginal params
            # create an IPOPT solver to minimize the loglikelihood for the marginal parameters
            # this solver minimize the logliklihood ratio but has a return objective value of the LLR gap,  so its root is on the boundary
            #  it accepts the radius as a fixed parameter
            profile_problem_struct = {'f': loglik_ratio_gap_symbol,  'x': marginal_param_symbols, 'p':radius_symbol}#,  'g': cs.vertcat(*OptimConstraints)
            profile_loglik_solver = cs.nlpsol('PLLSolver',  'ipopt',  profile_problem_struct, {'ipopt.print_level':0, 'print_time':False})
            #create a casadi function that computes the derivative of the optimal LLR gap solution with respect to the radius parameter
            profile_loglik_jacobian_solver = profile_loglik_solver.factory('PLLJSolver',  profile_loglik_solver.name_in(),  ['sym:jac:f:p'])
            #create a casadi function that computes the 2nd derivative of the optimal LLR gap solution with respect to the radius parameter
            profile_loglik_hessian_solver = profile_loglik_solver.factory('PLLHSolver',  profile_loglik_solver.name_in(),  ['sym:hess:f:p:p'])
        else:
            # else if there are no marginal parameters (i.e. 2d model),  create casadi functions emulating the above without optimization (which is not needed)
            profile_loglik_solver = cs.Function('PLLSolver',  [radius_symbol],  [loglik_ratio_gap_symbol]) 
            #create the 1st derivative function
            profile_loglik_jacobian_symbol = cs.jacobian(loglik_ratio_gap_symbol, radius_symbol)
            profile_loglik_jacobian_solver = cs.Function('PLLJSolver',  [radius_symbol],  [profile_loglik_jacobian_symbol]) 
            #create the second derivative function
            profile_loglik_hessian_symbol = cs.jacobian(profile_loglik_jacobian_symbol, radius_symbol) 
            profile_loglik_hessian_solver = cs.Function('PLLHSolver',  [radius_symbol],  [profile_loglik_hessian_symbol]) 
            #NOTE: not sure what second returned value of casadi hessian func is,  did this to avoid it (it may be gradient,  limited docs)
        #return the solvers/functions
        return [profile_loglik_solver, profile_loglik_jacobian_solver, profile_loglik_hessian_solver]

    def __logliksearch(self, solver_list, marginal_param, options, forward=True):
        """ 
        This function performs a root finding algorithm using solver_list objects
        It uses halley's method to find the radius value (relative to the mle) where the loglik ratio equals the chi-squared level
        This radius runs along the direction specified in the solver_list when they are created
        Halley's method is a higher order extension of newton's method for finding roots

        Args:
            solver_list: solver/casadi functions for finding the loglikelihood ratio gap at a given radius from mle,  and its 1st/2nd derivatives
            marginal_param: starting values (usually the mle) for the marginal parameters
            options: an options dictionary for passing user options
            forward: boolean,  if true search is done in the forward (positive) radius direction (relative to direction specidied in solver list),  if false perform search starting with a negative radius

        Returns:
            radius: returns the radius corresponding to the chi-squared boundary of the loglikelihood region
            marginal_param: returns the optimal setting of the marginal parameters at the boundary
            ratio_gap: returns the residual loglikelihood ratio gap at the boundary (it should be small,  within tolerance)
        """
        #check if root finding tolerance has been passed if not use default
        if "Tolerance" in options.keys():
            tolerance = options['Tolerance']
        else:
            tolerance = 0.001
        #NOTE:this should maybe be relative to the search params, may need to set in profile_setup
        if "InitialStep" in options.keys():
            init_step = options['InitialStep']
        else:
            init_step = 0.01
        #set the max number of iterations in the root finding method
        #NOTE: this should be a user option
        max_iterations=50
        #check if the search is run in negative or positive direction,  set intial step accordingly
        if forward:
            radius = init_step
        else:
            radius = -init_step
        #get the number of marginal parameters
        num_marginal_param = len(marginal_param)
        #get the solver/function objects,  loglik ratio gap and derivatives w.r.t. radius
        profile_loglik_solver = solver_list[0]
        profile_loglik_jacobian_solver = solver_list[1]
        profile_loglik_hessian_solver = solver_list[2]
        #set the initial marginal parameters
        #marginal_param = marginal_param NOTE:delete this?
        #set the initial LLR gap to a very large number
        ratio_gap = 9e9
        #NOTE: should check to see if we go negative,  loop too many time,  take massive steps,  want to stay in the domain of the MLE
        #NOTE:need max step check,  if steps are all in same direction perhaps set bound at inf and return warning,  if oscillating error failure to converge
        #create a counter to track root finding iterations
        iteration_counter = 0
        #loop until tolerance criteria are met (LLR gap drops to near zero)
        while  abs(ratio_gap)>tolerance and iteration_counter<max_iterations:
            if not num_marginal_param == 0:
                #if there are marginal parameters
                #run the ipopt solver to optimize the marginal parameters,  conditional on the current radius
                profile_solution_struct = profile_loglik_solver(x0=marginal_param, p=radius)#,  lbx=[],  ubx=[],  lbg=[],  ubg=[]
                #solver for the LLR gap 1st derivative w.r.t. the radius
                profile_jacobian_struct = profile_loglik_jacobian_solver(x0=profile_solution_struct['x'],  lam_x0=profile_solution_struct['lam_x'],  lam_g0=profile_solution_struct['lam_g'], p=radius)
                #solver for the LLR gap 2nd derivative w.r.t. the radius
                profile_hessian_struct = profile_loglik_hessian_solver(x0=profile_solution_struct['x'],  lam_x0=profile_solution_struct['lam_x'],  lam_g0=profile_solution_struct['lam_g'], p=radius)
                #update the current optimal values of the marginal parameters
                marginal_param = list(profile_solution_struct['x'].full().flatten())
                #update the current LLR gap value
                ratio_gap = profile_solution_struct['f'].full()[0][0]
                #extract the LLR gap 1st derivative value
                dratiogap_dradius = profile_jacobian_struct['sym_jac_f_p'].full()[0][0]
                #extract the LLR gap 2nd derivative value
                d2ratiogap_dradius2 = profile_hessian_struct['sym_hess_f_p_p'].full()[0][0]
            else:
                #else if there are no marginal parameters
                #call the appropriate casadi function to get the current LLR gap value
                ratio_gap = profile_loglik_solver(radius).full()[0][0]
                #call the appropriate casadi function to get the LLR gap 1st derivative value
                dratiogap_dradius = profile_loglik_jacobian_solver(radius).full()[0][0]
                #call the appropriate casadi function to get the LLR gap 2nd derivative value
                d2ratiogap_dradius2 = profile_loglik_hessian_solver(radius).full()[0][0]
            #increment the iterations counter
            iteration_counter+=1
            #use Halley's method (higher order extention of newtons method) to compute the new radius value
            radius = radius - (2*ratio_gap*dratiogap_dradius)/(2*dratiogap_dradius**2 - ratio_gap*d2ratiogap_dradius2)
        # throw error if maximum number of iterations exceeded
        if iteration_counter>=max_iterations:
            raise Exception('Maximum number of iterations reached in logliklihood boundary search!')
        #return the radius of the root,  the optimal marginal parameters at the root and the ratio_gap at the root (should be near 0)
        return [radius, marginal_param, ratio_gap]

class factorial(cs.Callback):
    def __init__(self,  name,  options = {}):
        cs.Callback.__init__(self)
        self.construct(name,  options)

    # Number of inputs and outputs
    def get_n_in(self): return 1
    def get_n_out(self): return 1

    # Initialize the object
    def init(self):
        print('initializing object')

    # Evaluate numerically
    def eval(self,  arg):
        k  =  arg[0]
        cnt = 1
        f = k
        while (k-cnt)>0:
            f = f*(k-cnt)
            cnt = cnt+1
        return [f]


# Full ML fitting,  perhaps with penalized likelihood???
# fit assesment,  with standardized/weighted residual output,  confidence regions via asymptotics (with beale bias),  likelihood basins,  profile liklihood,  sigma point (choose one or maybe two)
# function to generate a prior covariance (that can be fed into design)
# function for easy simulation studies (generate data,  with given experiment)