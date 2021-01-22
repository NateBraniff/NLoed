import os as os
import sys as sys
import copy as cp
import math as mt
#need to clean up this import style (below)
from contextlib import contextmanager

import casadi as cs
import numpy as np
import pandas as pd
#need to clean up this import style (below x3)
import matplotlib.pyplot as plt
from scipy import stats as st
from scipy.interpolate import splev,  splrep


class Model:
    """ The NLoed Model class implements a mathematical model and provides useful functions for model building.

    This class encodes casadi symbolic structures connecting model inputs and parameters to the
    observation variables. The class also encodes the assumed distirbution for each model observation
    variable.
    
    Upon construction, the class populates a series of casadi function attributes for
    computing various model predictions (i.e. mean, variance, sensitivity), logliklhoods, data
    sampling, and the fisher information. These function attributes are used to implement the class's
    public user-callable functions; fit(), predict(), evaluate() and sample(). The function attributes
    are also used by NLoed's Design class when a Model instance is passsed for experimental design.

    Attributes:
        symbolics_boolean (bool): A boolean indicating if Casadi SX (True) or MX (false) symbolics
            should be used.
        num_observ (integer): An integer indicating the number of observation variables in the model.
        num_input (integer): An integer indicating the number of input variables accepted by the model.
        num_param (integer): An integer indicating the number of parameters accepted by the model.
        input_name_list (list of strings): A list of the input variable names, in the order passed 
            to the model constructor.
        param_name_list (list of strings): A list of the parameter names, in the order passed 
            to the model constructor.
        observ_name_list (list of strings): A list of the observation variable names, in the order
            passed to the model constructor.
        loglik (dictionary of functions): This function attribute consists of a dictionary in which 
            the keys are the model's observation variable names and the values are casadi functions
            computing the loglikelihood of a single observation of the variable at the passed
            observation value with the passed input and parameter settings.

            Call Structure: Model.loglik[obs_name](obs_value, inputs, parameters)
        fisher_info_matrix (dictionary of functions): This function attribute consists of a 
            dictionary in which the keys are the model's observation variable names and the values 
            are casadi functions computing the fisher information matrix for a single observation of
            the specified observation variable at the passed input and parameter settings.
            Model.model_mean(inputs, parameters)

            Call Structure: Model.fisher_info_matrix[obs_name](inputs, parameters)
        model_mean (dictionary of functions): This function attribute consists of a 
            dictionary in which the keys are the model's observation variable names and the values 
            are casadi functions computing the expected mean observation value at the passed input
            and parameter settings.

            Call Structure: Model.model_mean[obs_name](inputs, parameters)
        model_variance (dictionary of functions): This function attribute consists of a 
            dictionary in which the keys are the model's observation variable names and the values 
            are casadi functions computing the expected variance of the observation variable at the
            passed input and parameter settings.

            Call Structure: Model.model_variance[obs_name](inputs, parameters)
        model_sensitivity (dictionary of functions): This function attribute consists of a 
            dictionary in which the keys are the model's observation variable names and the values 
            are casadi functions computing the parameteric sensitivity of the expected mean 
            observation of the specified variable at the passed input and parameter settings.

            Call Structure: Model.model_variance[obs_name](inputs, parameters)
        observation_sampler (dictionary of functions): This function attribute consists of a 
            dictionary in which the keys are the model's observation variable names and the values 
            are casadi-based functions generating random realizations of the observation variable value
            from its expected distribution, conditioned on the passed input and parameter settings.

            Call Structure: Model.model_variance[obs_name](inputs, parameters)
        observation_percentile (dictionary of functions): This function attribute consists of a 
            dictionary in which the keys are the model's observation variable names and the values 
            are casadi-based functions computing the requested percentile of the observation variable's
            distribution conditioned on the passed input and parameter settings.

            Call Structure: Model.model_variance[obs_name](percentile, inputs, parameters)

    """

    # FEATURES TO ADD & TESTING
    #NOTE: Add D-s Opt, A-opt, A-s Opt
    #NOTE: Es-opt (next line)
    #NOTE: https://math.stackexchange.com/questions/269723/largest-eigenvalue-semidefinite-programming
    #NOTE: Data/design/experiment objects may need deep copies so internal lists aren't shared??

    # CHECKS AND MESSAGES
    #NOTE: [maybe a gentle check,  i.e. did you mean this? no constraints] How to handle profile that goes negative for strictly positive parameter values???? perhaps error out and suggest reparameterization???
    #NOTE: all names must be unique
    #NOTE: must enforce ordering of parameters in statistics function
    #NOTE: we need checks on inputs of fit and sample functions!!!
    #NOTE: should rename inputs to cavariates or controls or smthg

    # COMMENTS AND VAR NAMING AND FUNCTION DOCSTRINGS
    # DOCUMENTATION

    #add format for dataframes to docstrings
    #add error classes and document them in docstrings??

    distribution_dict={ 'Normal':       ['Mean','Variance'],
                        'Poisson':      ['Rate'],
                        'Lognormal':    ['GeoMean','GeoVariance'],
                        'Bernoulli':    ['Probability'],
                        'Binomial':     ['Probability'],
                        'Exponential':  ['Rate'],
                        'Gamma':        ['Shape','Scale']}

    def __init__(self, observ_struct, input_names, param_names, options={}):
        """ The class constructor for NLoed's Model class. 

        This function accepts the users Casadi functions, distribution lables, and input and
        parmeter names, and generate an NLoed model object for the specified model. During instantiation
        the Casadi functions are used to generate a variety of function attributes for each model
        observation variable; including functions for computing model predictions, loglikelihood,
        fisher information and data sampling. These function attributes are used both by the NLoed
        Model instance's public function and if/when the Model instance is passed to the NLoed Design
        class for optimizing an experimental design.

        Args:
            observ_struct (list of tuples): A list of tuples, one tuple for each of the models
                observation variables. The first element of the tuple is a Casadi function mapping
                the model inputs and parameters to the given observation's sampling statistics. The
                second element of each tuple is a string indicating the type of parameteric distribution
                assigned to that observation variable. Observation names are extracted from the string
                label of the Casadi function for each obervation variable.
            input_names (list of strings): A list of strings specifying the names of the model inputs
                in the same order that the inputs are expected by the Casadi functions also passed by
                the user.
            param_names (list of strings): A list of strings specifying the names of the model parameters
                in the same order that the parameters are expected by the Casadi functions also passed by
                the user.
            options (dict, optional): A dictionary of user-defined options, possible key-value pairs
                include:

                "ScalarSymbolics" --
                Purpose: Determines whether SX or MX Casadi symbolics are used within the NLoed Model
                class, True implies scalar symbolics via SX.,
                Type: boolean,
                Default Value: True,
                Possible Values: True or False

        """

        default_options = \
          { 'ScalarSymbolics':       [True,       lambda x: isinstance(x,bool) ]}
        options=cp.deepcopy(options)
        for key in options.keys():
            if not key in options.keys():
                raise Exception('Invalid option key; '+key+'!')
            elif not default_options[key][1](options[key]):
                raise Exception('Invalid value; '+str(options[key])+', passed for option key; '+key+'!')
        for key in default_options.keys():
            if not key in options.keys() :
                options[key] = default_options[key][0]

        #check for unique names
        if not(len(set(input_names)) == len(input_names)):
            raise Exception('Model input names must be unique!')
        if not(len(set(param_names)) == len(param_names)):
            raise Exception('Parameter names must be unique!')
        if not isinstance(observ_struct,list):
            raise Exception('Observation structure must be a list of tuples!')
        for o in range(len(observ_struct)):
            obs_tuple = observ_struct[o]
            if not isinstance(obs_tuple,tuple):
                raise Exception('Error in observation entry; '+str(o)+', not a tuple, observation structure must be a list of tuples!')
            if not len(obs_tuple)==2:
                raise Exception('Error in observation entry; '+str(o)+', wronge tuple dimension, length must be two!')
            if not isinstance(obs_tuple[0],cs.Function):
                raise Exception('Error in observation tuple; '+str(o)+', first tuple entry must be a Casadi function!') 
            if not isinstance(obs_tuple[1],str) and obs_tuple[1] in self.distribution_dict.keys():
                raise Exception('Error in observation tuple; '+str(o)+', second tuple entry must be a string naming the distribution type!') 
            if not obs_tuple[0].n_in() == 2:
                raise Exception('Error in observation tuple;  '+str(o)+', Casadi function must take only two input vectors; inputs followed by parameters!')
            if not obs_tuple[0].n_out() == 1:
                raise Exception('Error in observation tuple;  '+str(o)+', Casadi function must return a single vector of observation distribution statistics!')
            # obs_tuple[2].size_out(0)) NOTE: need to check size of inputs to function and if they match input/param names, are columns etc.

        self.symbolics_boolean = options['ScalarSymbolics']
        self.num_observ = len(observ_struct)
        self.num_input = len(input_names)
        self.num_param = len(param_names) 
        #create lists of input/param/param names,  can be used to look up names with indices (observ names filled in below)
        self.input_name_list = input_names
        self.param_name_list= param_names
        self.observ_name_list = [] 
        #create dicts for input/param names,  can be used to look up indices with names
        # self.input_name_dict, self.param_name_dict = {}, {}
        # #self.observ_name_dict ={}
        # #populate the input name dict
        # for i in range(self.num_input):
        #     self.input_name_dict[input_names[i]] = i
        # #populate the param name dict
        # for i in range(self.num_param):
        #     self.param_name_dict[param_names[i]] = i
        #create a list to store the observation variable distribution type and
        #   functions for computing observation sampling statistics, loglikelihood, and fisher info. matrix
        #self.distribution, 
        #self.model,
        self.loglik, self.fisher_info_matrix = {}, {}
        #create a list to store casadi functions for the predicted mean, mean parameteric sensitivity, prediction variance 
        self.model_mean, self.model_variance, self.model_sensitivity = {},{},{}
        self.observation_sampler, self.observation_percentile ={}, {}
        #loop over the observation variables
        for i in range(self.num_observ):
            #fetch the observation name, distribution type and casadi function (maps inputs and params to observation statistics)
            observ_model = observ_struct[i][0]
            observ_distribution = observ_struct[i][1]
            observ_name = observ_model.name()
            #extract names of observ_struct variables
            if not(observ_name in self.observ_name_list):
                #self.observ_name_dict[observ_name] = i
                self.observ_name_list.append(observ_name)
            else:
                raise Exception('Observation names must be unique!')
            #self.distribution[observ_name] = observ_distribution
            #self.model[observ_name] = observ_model
            [loglik, fish_info, model_mean, model_sens, model_var, obs_sampler, obs_percent] = \
                self.__get_distribution_functions(observ_model, observ_distribution)
            self.loglik[observ_name] = loglik
            self.fisher_info_matrix[observ_name] = fish_info
            self.model_mean[observ_name] = model_mean
            self.model_sensitivity[observ_name] = model_sens
            self.model_variance[observ_name] = model_var
            self.observation_sampler[observ_name] = obs_sampler
            self.observation_percentile[observ_name] = obs_percent
            
# --------------- Public (user-callable) functions -------------------------------------------------

    def fit(self, datasets, start_param=None, options={}):
        """ A function for fitting the NLoed model to a dataset contained in a dataframe.

        This function fits the model to a dataset using maximum likelihood. This function can also 
        return marginal confidence intervals as well as plots of liklihood profiles and projections 
        of profiles traces and confidence contours.

        Args:
            datasets (dataframe OR list of dataframes): A dataframe containing the dataset to be fit.
                OR a list of dataframes, each containing a dataset replicate of a given design
                OR a list of lists of dataframes, where each index in the outer list corresponds
                to a unique design and each inner index coresponds to a replicate of the given design
            start_param (array-like, optional): An array of starting parameter values where the 
                local fitting optimization should be started
            options (dict, optional): A dictionary of user-defined options, possible key-value pairs
                include:

                "Confidence" --
                Purpose: Determines confidnece diagnostics to be returned or plotted,
                Type: string,
                Default Value: "None",
                Possible Values:
                "None" = No intervals returned,
                "Intervals" = Marginal intervals returned,
                "Profiles" = Same as "Intervals" but trace projections are plotted using Matplotlib,
                "Contours" = Same as  "Profiles" but confidence contour projections are also plotted.

                "ConfidenceLevel" --
                Purpose: Sets the confidence level for the marginal intervals, traces, profiles and contours,
                Type: float,
                Default Value: 0.95,
                Possible Values: 0<1.

                'RadialNumber' --
                Purpose: Determines the number of radial searches performed out from the fit 
                parameter value used to find the confidence contour projections,
                Type: integer
                Default Value: 30
                Possible Values: >1, but sufficient density is needed for good interpolation

                "SampleNumber" --
                Purpose: 
                Type: integer
                Default Value: 10
                Possible Values: >1

                "Tolerance" --
                Purpose: 
                Type: float
                Default Value: 0.001
                Possible Values: >0

                "InitialStep" --
                Purpose: 
                Type: float
                Default Value: 0.01
                Possible Values: >0

                "MaxSteps" --
                Purpose: 
                Type: integer
                Default Value: 2000
                Possible Values: >1

                "SearchFactor" --
                Purpose: 
                Type: float
                Default Value: 5.0
                Possible Values: >0

                "InitParamBounds" --
                Purpose: 
                Type: array-like
                Default Value: False
                Possible Values:

                "InitSearchNumber" --
                Purpose: 
                Type: integer
                Default Value: 3
                Possible Values: >0

                "Verbose" --
                Purpose: 
                Type: boolean
                Default Value: True
                Possible Values: True or False

        Return:
            dataframe OR list of dataframes: A dataframe containing the fit parameters, 
            and if requested, confidence interval information. If a list of datasets was provided, 
            each row of the dataframe corresponds to the dataset index in the passed list.
            If a list of lists of dataframes was provided (designs by replicates),
            a list of dataframes will be returned with the same length as the outer index of the
            passed list of lists.
        """
        #NOTE: needs checks on inputs, var names, designs must be exact reps
        #NOTE: NEED testing for multiple observation input structures,  multiple dimensions of parameters ideally,  1, 2, 3 and 7+
        #NOTE: add some print statments to provide user with progress status
        #NOTE: currently solve multiplex simultaneosly (one big opt problem) but sequentially may be more robust to separation (discrete data),  or randomly non-identifiable datasets (need to test)
        default_options = \
          { 'Confidence':       ['None',                lambda x: isinstance(x,str) and(x=='None' or x=='Intervals' or x=='Profiles' or x=='Contours')],
            'ConfidenceLevel':  [0.95,                  lambda x: isinstance(x,float) and 0<=x and x<=1],
            'RadialNumber':     [30,                    lambda x: isinstance(x,int) and 1<x],
            'SampleNumber':     [10,                    lambda x: isinstance(x,int) and 1<x],
            'Tolerance':        [0.001,                 lambda x: isinstance(x,float) and 0<x],
            'InitialStep':      [.01,                   lambda x: isinstance(x,float) and 0<x],
            'MaxSteps':         [2000,                  lambda x: isinstance(x,int) and 1<x],
            'SearchFactor':     [5,                     lambda x: isinstance(x,float) and 0<x],
            'SearchBound':      [3.0,                   lambda x: isinstance(x,float) and 0<x],
            'InitParamBounds':  [False,                 lambda x: isinstance(x,list) or isinstance(x,np.ndarray)],
            'InitSearchNumber': [3,                     lambda x: isinstance(x,int) and 0<x],
            'Verbose':          [True,                  lambda x: isinstance(x,bool)]}
        options=cp.deepcopy(options)
        for key in options.keys():
            if not key in options.keys():
                raise Exception('Invalid option key; '+key+'!')
            elif not default_options[key][1](options[key]):
                raise Exception('Invalid value; '+str(options[key])+', passed for option key; '+key+'!')
        for key in default_options.keys():
            if not key in options.keys() :
                options[key] = default_options[key][0]
        if start_param is None:
            start_param = [0]*self.num_param
        # if not len(start_param) == self.num_param:
        #     raise Exception('Starting parameter mismatch, there were; '+str(len(start_param))+', provided but; '+str(self.num_param)+' needed!')
        
        if isinstance(datasets, pd.DataFrame):
            #if a single dataset is passed vis the dataset_struct input
            # wrap it in a single list to match general case
            replicat_datasets = [datasets]
        else:
            #else if dataset_struct input is a list of designs, each with a list of replicates,
            # just pass on the input
            replicat_datasets = datasets
        #set confidence interval boolean
        interval_bool = options['Confidence']=="Intervals" \
                        or options['Confidence']=="Profiles" \
                        or options['Confidence']=="Contours"
        #set plotting boolean
        plot_bool = options['Confidence']=="Contours" or options['Confidence']=="Profiles"
        #get total number of datasets
        num_datasets = len(replicat_datasets)
        #create a list to store parameter casadi optimization symbols, starting parameters values, generic casadi loglikelihood functions
        #param_symbols_list, start_params_list = [], []
        loglik_func_list, fit_param_list = [], []
        #create archetypal param symbol vector , used for casadi loglik functions for each design
        if self.symbolics_boolean:
            archetype_param_symbols = cs.SX.sym('archetype_param_symbols', self.num_param)
        else:
            archetype_param_symbols = cs.MX.sym('archetype_param_symbols', self.num_param)
        
        archetype_loglik_symbol = 0
        #for each design use the first replicate 
        archetype_dataset = replicat_datasets[0]
        # get onbservation count for this design
        num_observations = len(archetype_dataset.index)
        #create a vector of casadi symbols for the observations
        if self.symbolics_boolean:
            archetype_observ_symbol = cs.SX.sym('archetype_observ_symbol', num_observations)
        else:
            archetype_observ_symbol = cs.MX.sym('archetype_observ_symbol', num_observations)
        #loop over the dataset inputs
        for index,row in archetype_dataset.iterrows():
            #get the curren input settings
            input_row = row[self.input_name_list].to_numpy()
            #create a symbol for the loglikelihood for the given input and observation variable
            archetype_loglik_symbol += self.loglik[row['Variable']](archetype_observ_symbol[index],
                                                                    input_row,
                                                                    archetype_param_symbols)
        #create a casadi function for the loglikelihood of the current design (observations are free/input symbols)
        archetype_loglik_func = cs.Function('archetype_loglik_func',
                                            [archetype_observ_symbol, archetype_param_symbols],
                                            [archetype_loglik_symbol])
        #print("at solver creation")
        archetype_loglik_optim_struct = {'f': -archetype_loglik_symbol, 'x': archetype_param_symbols,'p':archetype_observ_symbol}
        #archetype_fitting_solver = cs.nlpsol('solver', 'ipopt', archetype_loglik_optim_struct, {'ipopt.print_level':5, 'print_time':False,'ipopt.hessian_approximation':'limited-memory'}) 
        archetype_fitting_solver = cs.nlpsol('solver', 'ipopt', archetype_loglik_optim_struct, {'ipopt.print_level':5, 'print_time':True }) 
        #print("done solver creation")

        if options['Verbose']:
            progress_counter=0
            self.__progress_bar(progress_counter, num_datasets, prefix = 'Fitting Dataset(s):')

        #loop over replicates within each design
        for r in range(num_datasets):
            #NOTE: could abstract below into a Casadi function to avoid input/observ loop on each dataset and replicat
            #get the dataset from the replicate list
            dataset = replicat_datasets[r]
            #create a vector of parameter symbols for this specific dataset,  each dataset gets its own,  these are used for ML optimization
            if self.symbolics_boolean:
                fit_param_symbols = cs.SX.sym('fit_param_symbols_'+str(r), self.num_param)
            else:
                fit_param_symbols = cs.MX.sym('fit_param_symbols_'+str(r), self.num_param)
            #extract the vector of observations in the same format as in the archetype_loglik_func function input
            observ_vec = dataset['Observation'].to_numpy() 
            #create a symbol for the datasets loglikelihood function by pass in the observations for the free symbols in ObservSymbol
            dataset_loglik_symbol = archetype_loglik_func(observ_vec, fit_param_symbols)

            loglik_func = cs.Function('dataset_loglik_func_'+str(r),
                                                    [fit_param_symbols],[dataset_loglik_symbol])
            #add it to the list of replicate loglik functions
            loglik_func_list.append(loglik_func)

            #crude grid search of staring parameters, if request in options
            #print(options['InitParamBounds'])
            if options['InitParamBounds']:
                candidate_list = [np.linspace(bnds[0],bnds[1],options['InitSearchNumber']) \
                                    for bnds in options['InitParamBounds']]
                param_grid = self.__create_grid(candidate_list)
                param_grid.append(start_param)
                loglik_value = -1e7
                for param in param_grid:
                    new_loglik_value = loglik_func(param)
                    if  new_loglik_value > loglik_value:
                        start_param = param
                        loglik_value = new_loglik_value

            #NOTE: could do a full multi-start here or top N multistart etc. will depen on testing
            #NOTE: should probably make start param search optionally specified by param covaraince
            #especially if we allow laplace-basyian style fitting (actuall only reall makes sense
            # if we do both bayes fitting with bayes covarianse start)
            #print("at solver")
            #fit_param = archetype_fitting_solver(x0=start_param, p=observ_vec)['x'].full().flatten()
            with silence_stdout():
                fit_param = archetype_fitting_solver(x0=start_param, p=observ_vec)['x'].full().flatten()
            fit_param_list.append(fit_param)

            if options['Verbose']:
                progress_counter += 1
                self.__progress_bar(progress_counter, num_datasets, prefix = 'Fitting Dataset(s):')

        #create row index for parameter dataframe export, specifies design index for each datset
        #design_index = [design_indx+1 for design_indx in range(num_designs) for i in range(num_replicat_list[design_indx])]
        #check if a  confidence value was passed as an option, requires confidence intervals at least
        if interval_bool:
            #create a multiindex for the columns of parameter dataframe export
            # first level define estimate and lower/upper bound, second is param values
            column_index = pd.MultiIndex.from_product([['Estimate','Lower','Upper'],self.param_name_list],names=['Value', 'Parameter'])
            #create an empty list to store lower/upper bounds for intervals
            bound_list = []
            #loop over each design
            for d in range(num_datasets):
                if options['Verbose']:
                    print('Computing confidence information for replicat: '+str(d))
                #get the parameter estimates for current dataset
                #param_vec = fit_param_matrix[d,:]
                param_vec = fit_param_list[d]
                #check if graphing options; contour plots or profile trace plots,  are requested
                if options['Confidence']=="Contours" or options['Confidence']=="Profiles":
                    #if so, create a figure to plot on
                    fig = plt.figure()
                    #run profileplots to plot the profile traces and return CI's
                    interval_list = self.__profile_plot(param_vec, loglik_func_list[d], fig, options)[0]
                    #if contour plots are requested specifically, run contour function to plot the projected confidence contours
                    if options['Confidence']=="Contours":
                        self.__contour_plot(param_vec, loglik_func_list[d], fig, options)
                elif options['Confidence']=="Intervals":
                    #if confidence intervals are requested, run confidenceintervals to get CI's and add them to replicate list
                    interval_list = self.__confidence_intervals(param_vec, loglik_func_list[d], options)
                #store the current boundary vector in the bound_list
                bound_list.append( np.asarray(interval_list).T.flatten())
            #if requested, show the plots
            if plot_bool:
                #NOTE: need to give order of plots to user, perhaps with title
                plt.show()
            #combine the bound vectors in boundlist into a single numpy matrix
            bound_param_matrix = np.stack(bound_list)
            #concatenate the fit_param_matrix matrix of fit parameters with the bound_param_matrix of parameter interval boundaries
            param_output_matrix = np.concatenate([np.stack(fit_param_list), bound_param_matrix], axis=1)
        else:
            #if no intervals are requested, set column indext to single level, with just parameter names
            #column_index = self.param_name_list
            column_index = pd.MultiIndex.from_product([['Estimate'],self.param_name_list],names=['Value', 'Parameter'])
            param_output_matrix = np.stack(fit_param_list)
        #create ouput dataframe with all request parameter information
        param_data = pd.DataFrame(param_output_matrix, columns=column_index)
        #return the dataframe
        return param_data

    def sample(self, designs, param, design_replicates=1,options={}):
        """A function for generating simulated data from the NLoed model for a given design passed 
        as a dataframe.
        
        This function generates datasets using the NLoed model and a provided design (or a list of 
        designs) via simulation and Numpy/Scipy's random number generation. This simulation is done
        at a user-provided set of parameter values. The number of replicates of the design that are
        simulated is optional but defaults to one.

        Args:
            designs (dataframe OR list of dataframes): A dataframe containing the design to be simulated,
                OR a list of dataframes containing multiple designs to be simulated
            param (array-like, floats): The parameter values used to generate the data
            design_replicates (integer, optional): An integer indicating the number of dataset 
                replicates to be generated for each design. The default is 1 per design passed.
            options (dict, optional): A dictionary of user-defined options, possible key-value pairs
                include:

                "Verbose" --
                Purpose: Determines the amount of print feedback provided while the function executes
                Type: bool,
                Default Value: True,
                Possible Values: True or False

        Returns:
            dataframe OR list of dataframes:
            A dataframe containg a simulation of the design is returned by default
            OR, if design_replicates was set to >1, a list of dataframes containg simulated 
            replicates is returned for the given design
            OR,  if a list of dataframes containing a set of design was passed,
            a list of lists of dataframes is returned, the outer index corresponds to the design
            list, and the inner index corresponds to the number of replicates requested.
        """
        #NOTE: needs checks on inputs
        #NOTE: should make the returned dataframe a multicolumn index to be consistent
        #NOTE: multiplex multiple parameter values??
        #NOTE: actually maybe more important to be able to replicate designs N times
        default_options = \
          { 'Verbose':          [True,                  lambda x: isinstance(x,int) and 0<x]}
        options=cp.deepcopy(options)
        for key in options.keys():
            if not key in options.keys():
                raise Exception('Invalid option key; '+key+'!')
            elif not default_options[key][1](options[key]):
                raise Exception('Invalid value; '+str(options[key])+', passed for option key; '+key+'!')
        for key in default_options.keys():
            if not key in options.keys() :
                options[key] = default_options[key][0]
        if not len(param) == self.num_param:
            raise Exception('Starting parameter mismatch, there were; '+str(len(param))+', provided but; '+str(self.num_param)+' needed!')

        #NOTE: NEED TO ADD LOOP OVER DESIGNS INPUT

        itemized_design = designs.reindex(designs.index.repeat(designs['Replicates']))
        itemized_design.drop('Replicates',axis=1,inplace=True)
        itemized_design.reset_index(drop=True,inplace=True)
        #create a list for replciated datasets of the current design
        replicat_datasets = []
        if options['Verbose']:
            progress_counter=0
            self.__progress_bar(progress_counter, design_replicates, prefix = 'Sampling Datasets:')
        #loop over the number of replicates
        for r in range(design_replicates):
            dataset=itemized_design.copy()
            observation_list = []
            for index,row in itemized_design.iterrows():
                input_row = row[self.input_name_list].to_numpy()
                observ_name = row['Variable']
                observation_list.append(self.observation_sampler[observ_name](input_row, param).item())
            dataset['Observation'] = observation_list
            replicat_datasets.append(dataset)
            if options['Verbose']:
                progress_counter += 1
                self.__progress_bar(progress_counter, design_replicates, prefix = 'Sampling Datasets:')
                
        if design_replicates==1:
            #if a single design was passed and there replicate count is 1,  return a single dataset
            return replicat_datasets[0]
        else:
            #else if a single design was passed,  but with >1 reps,  return a list of datasets
            return replicat_datasets
        
    def predict(self, input_struct, param, covariance_matrix=None, options={}):
        """ A function for generating prediction information from the NLoed model.
        
        This function can be used to compute predictions from an NLoed model instance given the
        user-provided input conditions and paramater values. Prediction information includes the
        predicted mean response of the model, confidence intervals for the mean response under 
        parameter uncertainty, confidence intervals for the observation distribution, and sensitivity
        analysis of the mean response. The returned intervals can be computed in a number of ways; 
        exactly, with a normal (local and deterministic) approximation, or using Monte Carlo simulation. 

        Args:
            input_struct (dataframe): A dataframe specifying the combination of inputs values and 
                observations at which predictions are desired.
            param (array-like, floats): The parameter vector values at which the predictions are to be made.
            covariance_matrix (array-like, floats, optional): A symetric matrix specifying the parameter's 
                normal covariance matrix, required if parameter uncertainty is to be included. The
                prior mean set by the values passed via the param argument.
            options (dictionary, optional): A dictionary of user-defined options, possible key-value
                pairs include:

                "Method" --
                Purpose: Selects the method used to compute the mean and prediction and observation 
                intervals.
                Type: string
                Default Value: "Delta"
                Possible Values: "Exact"=Compute predictions and intervals exactly; this
                option is only available if no parameter uncertainty information is NOT provided .
                "Delta"=Use a normal, local apporximation to compute the prediction and
                intervals, "MonteCarlo"=Use Monte Carlo simulation of the model to compute the 
                prediction and intervals under any parameter or observation uncertainty.

                "PredictionInterval" --
                Purpose: A boolean to indicat if prediction intervals are to be returned, true if yes.
                Type: bool
                Default Value: False
                Possible Values: True or False

                "ObservationInterval" --
                Purpose: A boolean to indicat if observation intervals are to be returned, true if yes.
                Type: bool
                Default Value: False
                Possible Values: True or False

                "Sensitivity" --
                Purpose: A boolean to indicat if observation intervals are to be returned, true if yes.
                Type: bool
                Default Value: False
                Possible Values: True or False

                "PredictionSampleNumber" --
                Purpose: An integer indicating the number of parameter vectors to be sampled from
                the prior in order to compute the prediction (and observation0 intervals using Monte
                Carlo simulation.
                Type: integer
                Default Value: 10000
                Possible Values:

                "ObservationSampleNumber" --
                Purpose: An integer indicating the number of observation values to be sampled from
                the observation distribution in order to compute the observation interval using 
                Monte Carlo simulation.
                Type: integer
                Default Value: 10
                Possible Values:

                "ConfidenceLevel" --
                Purpose: A float specifying the confidence level desired for any intervals computed.
                Type: float
                Default Value: 0.95
                Possible Values: <1, >0


        Returns:
            dataframe: A dataframe containing the requested prediction quntities computed at the
                input and parameter settings passed.

        """
        #NOTE: evaluate model to predict mean and prediction interval for y
        #NOTE: optional pass cov matrix, for use with delta method/MC error bars on predictions
        #NOTE: should multiplex over inputs
        # if isinstance(input_list, pd.DataFrame):
        #     input_list = [[input_list]]
        # elif isinstance(input_list[0], pd.DataFrame):
        #     input_list = [input_list]

        default_options = { 'Method':                   ['Delta',   lambda x: isinstance(x,str) and ( x=='Exact' or x=='Delta' or x=='MonteCarlo')],
                            'PredictionInterval':       [False,     lambda x: isinstance(x,bool)],
                            'ObservationInterval':      [False,     lambda x: isinstance(x,bool)],
                            'Sensitivity':              [False,     lambda x: isinstance(x,bool)],
                            'PredictionSampleNumber':   [10000,     lambda x: isinstance(x,int) and 1<x],
                            'ObservationSampleNumber':  [10,        lambda x: isinstance(x,int) and 1<x],
                            'ConfidenceLevel':          [0.95,      lambda x: isinstance(x,float) and 0<=x and x<=1]}
                            
        options=cp.deepcopy(options)
        for key in options.keys():
            if not key in options.keys():
                raise Exception('Invalid option key; '+key+'!')
            elif not default_options[key][1](options[key]):
                raise Exception('Invalid value; '+str(options[key])+', passed for option key; '+key+'!')
        for key in default_options.keys():
            if not key in options.keys() :
                options[key] = default_options[key][0]
        if covariance_matrix is None:
            if options['PredictionInterval']:
                raise Exception('Prediction intervals cannot be computed without parameter uncertainty \
                                 specified by a covariance matrix!')
            if options['ObservationInterval'] and not options['Method']=='Exact':
                raise Exception('Observation intervals can only be computed using \'Exact\' method \
                                without a covariance matrix!')

        #NOTE: need to add check for covariance matrix format and dimensions, agnostic to array/list type

        percentiles = np.array([(1-options['ConfidenceLevel'])/2, (0.5+options['ConfidenceLevel']/2)])

        prediction_mean, observation_mean = [], []
        prediction_lower, prediction_upper = [], []
        observation_lower, observation_upper = [], []
        sensitivity_lists = []
        for par_name in self.param_name_list:
            sensitivity_lists.append([])
        for index,row in input_struct.iterrows():
            input_vec = row[self.input_name_list].to_numpy()
            observ_name = row['Variable'] 
            if options['Method']=='Exact':
                prediction_mean.append(self.model_mean[observ_name](input_vec,param))
                if options['PredictionInterval']:
                     raise Exception('Prediction intervals cannot be computed exactly!')
                if options['ObservationInterval']:
                    observation_bounds = self.observation_percentile[observ_name](percentiles,input_vec,param)
                    observation_lower.append(observation_bounds[0])
                    observation_upper.append(observation_bounds[1])
                
            elif options['Method']=='Delta':
                mean = self.model_mean[observ_name](input_vec,param)
                prediction_mean.append(mean)
                stddev_multiplier = -st.norm.ppf((1-options['ConfidenceLevel'])/2)
                if options['PredictionInterval']:
                    stddev = np.sqrt(self.model_sensitivity[observ_name](input_vec,param)\
                                    @ covariance_matrix \
                                    @ self.model_sensitivity[observ_name](input_vec,param))
                    prediction_lower.append(mean - stddev_multiplier*stddev)
                    prediction_upper.append(mean + stddev_multiplier*stddev)

                if options['ObservationInterval']:
                    #observation_mean.append(mean)
                    prediction_var = (self.model_sensitivity[observ_name](input_vec,param)\
                                    @ covariance_matrix\
                                    @ self.model_sensitivity[observ_name](input_vec,param))
                    observation_var = self.model_variance[observ_name](input_vec,param) 
                    stddev_total = np.sqrt(prediction_var + observation_var)
                    observation_lower.append(mean - stddev_multiplier*stddev_total)
                    observation_upper.append(mean + stddev_multiplier*stddev_total)

            elif options['Method']=='MonteCarlo':
                param_sample = np.random.multivariate_normal(param,  covariance_matrix,  options['PredictionSampleNumber'])
                prediction_sample = [self.model_mean[observ_name](input_vec,par) for par in param_sample]
                prediction_mean.append(np.mean(prediction_sample))
                if options['PredictionInterval']:
                    prediction_bounds = np.percentile(prediction_sample,100*percentiles) 
                    prediction_lower.append(prediction_bounds[0])
                    prediction_upper.append(prediction_bounds[1])

                if options['ObservationInterval']:
                    prior_predictive_sample = [value.item()
                                    for par in param_sample
                                        for value in self.observation_sampler[observ_name]\
                                            (input_vec,par,options['ObservationSampleNumber'])]
                    #observation_mean.append(np.mean(prior_predictive_sample))
                    observation_bounds = np.percentile(prior_predictive_sample,100*percentiles) 
                    observation_lower.append(observation_bounds[0])
                    observation_upper.append(observation_bounds[1])

            if options['Sensitivity']:
                sensitivity_vec = self.model_sensitivity[observ_name](input_vec,param)
                for p in range(self.num_param):
                    sensitivity_lists[p].append(sensitivity_vec[p])

        output_data = cp.deepcopy(input_struct)
        column_name_list =output_data.columns.tolist()
        num_col_names = len(column_name_list)
        output_data.columns=pd.MultiIndex(levels=[['Inputs'],column_name_list],
                                          codes=[[0]*num_col_names,[i for i in range(num_col_names)]])
        output_data['Prediction','Mean'] = prediction_mean
        if options['PredictionInterval']:
            output_data['Prediction','Lower'] = prediction_lower
            output_data['Prediction','Upper'] = prediction_upper
        if options['ObservationInterval']:
            #output_data['Observation','Mean'] = observation_mean
            output_data['Observation','Lower'] = observation_lower
            output_data['Observation','Upper'] = observation_upper
        if options['Sensitivity']:
            for p in range(self.num_param):
                output_data['Sensitivity',self.param_name_list[p]] = sensitivity_lists[p]

        return output_data

    #NOTE: should maybe rename this
    def evaluate(self, designs, param, options={}):
        """ A function for evaluating the peformance metrics of an experimental design applied to
         the NLoed Model.

        This function can be used to produce evaluation metrics regarding the expected parameter
        fitting accurach a given experimental design will achieve when used with the given NLoed 
        Model instance. 
        
        Evaluation metrics inclide the expected parameter covariance, bias and mean
        squared error as well as the Fisher information matrix. These metrics can be computed either
        asymptoticall (i.e. via the Fisher information matrix; in this case the bias is zero and the
        covariance and MSE are equal) or using Monte Carlo simulation and fitting (in which case the
        bias, covariance and MSE may be unique). By default this function only returns the parameter
        covariance for the design, computed asymptoticall.
        
        All evaluation is performed at the candidate parameter vector passed, and as such these 
        metrics are only local appoximations valid near the candidate point.

        Args:
            designs (dataframe OR list of dataframes): A dataframe specifying the experimental design
                to be evaluated,
                OR a list of dataframes describing a set ofexperimental designs to be evaluated.
            param (array-like, floats): The parameter vector values at which the predictions are to be made.
            options (dictionary, optional): A dictionary of user-defined options, possible key-value
                pairs include:

                "Method" --
                Purpose: A string indicating which computational method, asymptotic or Monte Carlo,
                is used to compute the evaluation metrics.
                Type: string
                Default Value: "Asymptotic
                Possible Values: "Asymptotic"=Uses a first-order approximation based onvthe FIM 
                matrix to compute the parameter covariance; bias is zero. "MonteCarlo"= Uses 
                repeated data simulation and fitting to compute the evaluation metrics.

                "Covariance" --
                Purpose: A boolean value indicating if the parameter covariance matrix should be
                computed and returned, true implies yes.
                Type: boolean
                Default Value: True
                Possible Values: True or False

                "Bias" --
                Purpose: A boolean value indicating if the parameter bias vector should be
                computed and returned, true implies yes.
                Type: boolean
                Default Value: False
                Possible Values: True or False

                "MSE" --
                Purpose: A boolean value indicating if the parameter mean squated error vector 
                should be computed and returned, true implies yes.
                Type: boolean
                Default Value: False
                Possible Values: True or False

                "FIM" --
                Purpose: A boolean value indicating if the Fisher information matrix 
                should be computed and returned, true implies yes.
                Type: boolean
                Default Value: False
                Possible Values: True or False

                "SampleNumber" --
                Purpose: An integer indicating the number of simulations of the experimental design,
                and subsequent fittings, that should be performed if the Monte Carlo method is used
                to compute the evaluation metrics.
                Type: integer
                Default Value: 1000
                Possible Values: >0, must be large enough for a statistically stable estimate
                

        Return: 
            dataframe: A dataframe containg the design evaluation metrics,
            OR if a list of design dataframes was passed, a list of dataframes is returned
            each containing the design evaluation metrics for its corresponding design.

        """
        #maybe this should move to the design class(??)
        #For D (full cov/bias),  Ds (partial cov/bias),  T separation using the delta method?! but need two models
        # assess model/design,  returns various estimates of cov,  bias,  confidence regions/intervals
        # no data: asymptotic: covaraince,  beale bias,  maybe MSE
        #          sigma point: covariance,  bias (using mean) (need to figure out how to do sigma for non-normal data),  maybe MSE
        #          monte carlo: covariance,  bias,  MSE
        default_options = { 'Method':                   ['Asymptotic',  lambda x: isinstance(x,str) and ( x=='Asymptotic' or x=='MonteCarlo')],
                            'Covariance':               [True,          lambda x: isinstance(x,bool)],
                            'Bias':                     [False,         lambda x: isinstance(x,bool)],
                            'MSE':                      [False,         lambda x: isinstance(x,bool)],
                            'SampleNumber':             [1000,          lambda x: isinstance(x,int) and 1<x],
                            'FIM':                      [False,         lambda x: isinstance(x,bool)]}
        
        options=cp.deepcopy(options)
        for key in options.keys():
            if not key in options.keys():
                raise Exception('Invalid option key; '+key+'!')
            elif not default_options[key][1](options[key]):
                raise Exception('Invalid value; '+str(options[key])+', passed for option key; '+key+'!')
        for key in default_options.keys():
            if not key in options.keys() :
                options[key] = default_options[key][0]

        # itemized_design = design.reindex(design.index.repeat(design['Replicates']))
        # itemized_design.drop('Replicates',axis=1,inplace=True)
        # itemized_design.reset_index(drop=True,inplace=True)

        if isinstance(designs, pd.DataFrame):
            designset_list = [designs]
        else:
            designset_list = designs

        eval_data_list = []
        #loop over the number of replicates
        for i in range(len(designset_list)):
            designset = designset_list[i]

            upper_level = []
            lower_level = []
            eval_mat = np.array([], dtype=np.int64).reshape(self.num_param,0)

            if options['Method'] == 'Asymptotic' or options['FIM']:
                fisher_info_sum = np.zeros((self.num_param,self.num_param))
                for index,row in designset.iterrows():
                    input_row = row[self.input_name_list].to_numpy()
                    observ_name = row['Variable']
                    if 'Replicates' in designset:
                        reps = row['Replicates']
                    else:
                        reps = 1
                    fisher_info_sum += reps * self.fisher_info_matrix[observ_name](input_row, param).full()

                if options['FIM']:
                    upper_level.extend(['FIM']*self.num_param)
                    lower_level.extend(self.param_name_list)
                    eval_mat = np.hstack((eval_mat,fisher_info_sum))
                    #NOTE: warning of error for singular matrix?           

            #NOTE: could add empirical FIM comp (some observed FIM) for montecarlo method

            if options['Method'] == 'MonteCarlo':
                sample_data = self.sample(designset, param,options['SampleNumber'], options={'Verbose':True})
                sample_fits = self.fit(sample_data, start_param=param, options={'Verbose':True})

            if options['Covariance']:
                columns_index = pd.MultiIndex.from_product([['Covariance'],self.param_name_list])
                if options['Method'] == 'Asymptotic':
                    cov_matrix = np.matrix(fisher_info_sum).I
                elif options['Method'] == 'MonteCarlo':
                    cov_matrix = np.cov(sample_fits.to_numpy().T)
                else:
                    print('Error')
                #cov_data = pd.DataFrame(cov_matrix,
                #                        index=self.param_name_list,
                #                       columns= columns_index)
                upper_level.extend(['Covariance']*self.num_param)
                lower_level.extend(self.param_name_list)
                eval_mat = np.hstack((eval_mat,cov_matrix))#pd.concat([eval_data,cov_data],axis=1)
            if options['Bias']:
                columns_index = pd.MultiIndex.from_product([['Bias'],['Bias']])
                if options['Method'] == 'Asymptotic':
                    bias_vec = np.zeros((self.num_param))
                elif options['Method'] == 'MonteCarlo':
                    bias_vec = np.mean(sample_fits.to_numpy(),0) - param
                else:
                    print('Error')
                # bias_data = pd.DataFrame(bias_vec,
                #                         index=self.param_name_list,
                #                         columns=columns_index)
                # eval_data = pd.concat([eval_data,bias_data],axis=1)
                upper_level.extend(['Bias'])
                lower_level.extend(['Bias'])
                eval_mat = np.hstack((eval_mat,bias_vec.reshape(self.num_param,1)))
            if options['MSE']:
                columns_index = pd.MultiIndex.from_product([['MSE'],['MSE']])
                if options['Method'] == 'Asymptotic':
                    mse_vec = np.diagonal(np.matrix(fisher_info_sum).I)
                elif options['Method'] == 'MonteCarlo':
                    mse_vec = np.mean((sample_fits.to_numpy() - np.array([param,]*options['SampleNumber']))**2,0)
                else:
                    print('Error')
                # mse_data = pd.DataFrame(mse_vec,
                #                         index=self.param_name_list,
                #                         columns= columns_index)
                # eval_data = pd.concat([eval_data,mse_data],axis=1)
                upper_level.extend(['MSE'])
                lower_level.extend(['MSE'])
                eval_mat = np.hstack((eval_mat,mse_vec.reshape(self.num_param,1)))

            columns_index = pd.MultiIndex.from_arrays([upper_level,lower_level])
            eval_data = pd.DataFrame(eval_mat,index=self.param_name_list,columns= columns_index)

            eval_data_list.append(eval_data)

                
        if isinstance(designs, pd.DataFrame):
            #if a single design was passed and there replicate count is 1,  return a single dataset
            return eval_data_list[0]
        else:
            #else if a single design was passed,  but with >1 reps,  return a list of datasets
            return eval_data_list

    # def plots(self):
    #     #FDS plot,  standardized variance (or Ds,  bayesian equivlant),  residuals
    #     print('Not Implemeneted')
    #     #NOTE: maybe add a basic residual computation method for goodness of fit assesment?? Or maybe better show how in tutorial but not here

# --------------- Private functions (for constructor) ----------------------------------------------

    def __get_distribution_functions(self, observ_model, observ_distribution ):
        """ A private function that automatically generates function attributes for the provided 
        observation variable information.
        
        This private function is used during the NLoed Model class construction.

        This function accepts the observation name, distribution type
        and observation model and constructs casadi functions to compute
        the logliklihood, fisher information, prediction mean/sensitivity/variance
        and also a numpy/casadi function to return random samples from the the model

        Args:
            observ_model: A symbolic Casadi function mapping the model input and parameter vectors 
                to the observation distribution statistics.
            observ_distribution (string): A string specifying the observation distribution type, 
                must be one of the supported distributions.
            
        Returns:
            list of functions: A list of Casadi(-based) functions in the following order;
                    loglikelihood, fisher information, observation mean , observation variance, 
                    mean sensitivity, observation sampler, observation percentile
        """
        #get the observation name
        observ_name = observ_model.name()
        #create symbols for parameters and inputs,  and observation variable
        if self.symbolics_boolean:
            param_symbols = cs.SX.sym('param_symbols', self.num_param)
            input_symbols = cs.SX.sym('input_symbols', self.num_input)
            observ_symbol = cs.SX.sym(observ_name+'_symbol', 1)
        else:
            param_symbols = cs.MX.sym('param_symbols', self.num_param)
            input_symbols = cs.MX.sym('input_symbols', self.num_input)
            observ_symbol = cs.MX.sym(observ_name+'_symbol', 1)
        #generate the distribution statistics symbols, and prediction sensitivity symbol
        stats = observ_model(input_symbols, param_symbols)
        stats_sensitivity = cs.jacobian(stats, param_symbols)

        if observ_distribution  == 'Normal':
            if not stats.shape == (2,1):
                raise Exception('The Normal distribution accepts only 2 distributional parameters, but dimension '+str(stats)+' provided!')
            #create LogLikelihood symbolics and function 
            loglik_symbol = -0.5*cs.log(cs.sqrt(stats[1])) - (observ_symbol - stats[0])**2/(2*stats[1])
            #create FIM symbolics and function
            if self.symbolics_boolean:
                elemental_fim = cs.vertcat(cs.horzcat(1/stats[1], cs.SX(0)),cs.horzcat(cs.SX(0), 1/(2*stats[1]**2)))
            else:
                elemental_fim = cs.vertcat(cs.horzcat(1/stats[1], cs.MX(0)),cs.horzcat(cs.MX(0), 1/(2*stats[1]**2)))
            #create prediction functions for the mean, sensitivity and variance
            model_mean_symbol = stats[0]
            model_sensitivity_symbol = stats_sensitivity[0,:]
            model_variance_symbol = stats[1]
            #create random sampling function
            var_to_stdev = lambda s: [s[0].full().item(), np.sqrt(s[1]).full().item()]
            observ_sampler_func = lambda inpt,par,n=1: np.random.normal(*var_to_stdev(observ_model(inpt,par)),size=n)
            observation_percentile_func = lambda perc,inpt,par: st.norm.ppf(perc,*var_to_stdev(observ_model(inpt,par)))
        elif observ_distribution == 'Poisson':
            if not stats.shape == (1,1):
                raise Exception('The Poisson distribution accepts only 1 distributional parameters, but dimension '+str(stats)+' provided!')
            #create a custom casadi function for doing factorials, store the function in the class so it doesn't go out of scope
            # self.__factorialFunc = casadi_factorial
            # casadi_factorial = factorial('fact')
            #create LogLikelihood symbolics and function 
            #loglik_symbol = observ_symbol*cs.log(stats[0]) + casadi_factorial(observ_symbol) - stats[0]
            loglik_symbol = observ_symbol*cs.log(stats[0]) - stats[0]
            #create FIM symbolics and function
            elemental_fim = 1/stats[0]
            #create prediction functions for the mean, sensitivity and variance
            model_mean_symbol = stats[0]
            model_sensitivity_symbol = stats_sensitivity[0,:]
            model_variance_symbol = stats[0]
            #create random sampling function
            observ_sampler_func = lambda inpt,par,n=1: np.random.poisson(observ_model(inpt,par).full().item(),size=n)
            observation_percentile_func = lambda inpt,par,perc: st.poisson.ppf(perc,*observ_model(inpt,par).full().item())
        elif observ_distribution == 'Lognormal': 
            if not stats.shape == (2,1):
                raise Exception('The Lognormal distribution accepts only 2 distributional parameters, but dimension '+str(stats)+' provided!')
            #create LogLikelihood symbolics and function 
            #loglik_symbol = -cs.log(observ_symbol) - 0.5*cs.log(2*cs.pi*stats[1])\
            #     - (cs.log(observ_symbol) - stats[0])**2/(2*stats[1])
            loglik_symbol = - 0.5*cs.log(2*cs.pi*stats[1]) - (cs.log(observ_symbol) - stats[0])**2/(2*stats[1])
            #create FIM symbolics and function
            if self.symbolics_boolean:
                elemental_fim = cs.vertcat(cs.horzcat(1/stats[1], cs.SX(0)),cs.horzcat(cs.SX(0), 1/(2*stats[1]**2)))
            else:
                elemental_fim = cs.vertcat(cs.horzcat(1/stats[1], cs.MX(0)),cs.horzcat(cs.MX(0), 1/(2*stats[1]**2)))
            #create prediction functions for the mean, sensitivity and variance
            model_mean_symbol = cs.exp(stats[0] + stats[1]/2)
            model_sensitivity_symbol = cs.jacobian(model_mean_symbol,param_symbols)
            model_variance_symbol = (cs.exp(stats[1])-1) * cs.exp(2*stats[0] + stats[1])
            #create random sampling function
            var_to_stdev = lambda s: [s[0].full().item(), np.sqrt(s[1].full().item())]
            observ_sampler_func = lambda inpt,par,n=1: np.random.lognormal(*var_to_stdev(observ_model(inpt,par)),size=n)
            to_shape_scale = lambda s: [np.sqrt(s[1].full().item()), 0, np.exp(s[0].full().item())]
            observation_percentile_func = lambda perc,inpt,par: st.lognorm.ppf(perc,*to_shape_scale(observ_model(inpt,par)))
        elif observ_distribution == 'Binomial': 
            if not stats.shape == (2,1):
                raise Exception('The Binomial distribution accepts only 2 distributional parameters, but dimension '+str(stats)+' provided!')
            if not np.mod(cs.DM(stats[1]).full().item(),1)==0 :
                raise Exception('The second distributional parameter for Binomial must be a whole number positive integer!')
            #create LogLikelihood symbolics and function 
            loglik_symbol = observ_symbol*cs.log(stats[0]) + (stats[1]-observ_symbol)*cs.log(1-stats[0])
            #create FIM symbolics and function
            if self.symbolics_boolean:
                elemental_fim = cs.vertcat(cs.horzcat(stats[1]/(stats[0]*(1-stats[0])), cs.SX(0)),cs.horzcat(cs.SX(0), cs.SX(0)))
            else:
                elemental_fim = cs.vertcat(cs.horzcat(stats[1]/(stats[0]*(1-stats[0])), cs.MX(0)),cs.horzcat(cs.MX(0), cs.MX(0)))
            #create prediction functions for the mean, sensitivity and variance
            model_mean_symbol = stats[0]*stats[1]
            model_sensitivity_symbol = cs.jacobian(model_mean_symbol,param_symbols)
            model_variance_symbol = stats[1]*stats[0]*(1-stats[0])
            #create random sampling function
            float_to_int = lambda s: [int(s[1].full().item()), s[0].full().item()]
            observ_sampler_func = lambda inpt,par,n=1: np.random.binomial(*float_to_int(observ_model(inpt,par)),size=n)
            observation_percentile_func = lambda perc,inpt,par: st.binom.ppf(perc,*float_to_int(observ_model(inpt,par)))
        elif observ_distribution == 'Bernoulli':
            if not stats.shape == (1,1):
                raise Exception('The Bernoulli distribution accepts only 1 distributional parameters, but dimension '+str(stats)+' provided!')
            #create LogLikelihood symbolics and function 
            loglik_symbol = observ_symbol*cs.log(stats[0]) + (1-observ_symbol)*cs.log(1-stats[0])
            #create FIM symbolics and function
            elemental_fim = 1/(stats[0]*(1-stats[0]))
            #create prediction functions for the mean, sensitivity and variance
            model_mean_symbol = stats[0]
            model_sensitivity_symbol = cs.jacobian(model_mean_symbol,param_symbols)
            model_variance_symbol = stats[0]*(1-stats[0])
            #create random sampling function
            observ_sampler_func = lambda inpt,par,n=1: np.random.binomial(1,observ_model(inpt,par).full().item(),size=n)
            observation_percentile_func = lambda perc,inpt,par: st.bernoulli.ppf(perc,observ_model(inpt,par).full().item())
        elif observ_distribution == 'Exponential': 
            if not stats.shape == (1,1):
                raise Exception('The Exponential distribution accepts only 1 distributional parameters, but dimension '+str(stats)+' provided!')
            #create LogLikelihood symbolics and function 
            loglik_symbol = cs.log(stats[0]) - observ_symbol*stats[0]
            #create FIM symbolics and function
            elemental_fim = 1/(cs.power(stats[0],2))
            #create prediction functions for the mean, sensitivity and variance
            model_mean_symbol = 1/stats[0]
            model_sensitivity_symbol = cs.jacobian(model_mean_symbol,param_symbols)
            model_variance_symbol = 1/(cs.power(stats[0],2))
            #create random sampling function
            observ_sampler_func = lambda inpt,par,n=1: np.random.exponential(1/observ_model(inpt,par).full().item(),size=n)
            observation_percentile_func = lambda perc,inpt,par: st.expon.ppf(perc,1/observ_model(inpt,par).full().item())
        elif observ_distribution == 'Gamma': 
            print('Not Implemeneted')
        else:
            raise Exception('Unknown Distribution: '+observ_distribution)

        loglik_func = cs.Function('loglik_'+observ_name, [observ_symbol, input_symbols, param_symbols], [loglik_symbol])
        fisher_info_symbol = stats_sensitivity.T @ elemental_fim @ stats_sensitivity
        fisher_info_func = cs.Function('fim_'+observ_name, [input_symbols, param_symbols], [fisher_info_symbol])

        mean_casadi_func = cs.Function('pred_mean_'+observ_name,
                                        [input_symbols, param_symbols],
                                        [model_mean_symbol])
        model_mean_func = lambda inpt,par: mean_casadi_func(inpt,par).full().item()
        sensitivity_casadi_func = cs.Function('pred_sens_'+observ_name,
                                                [input_symbols, param_symbols],
                                                [model_sensitivity_symbol])
        model_sensitivity_func = lambda inpt,par: sensitivity_casadi_func(inpt,par).full().flatten()
        variance_casadi_func = cs.Function('pred_var'+observ_name,
                                            [input_symbols, param_symbols],
                                            [model_variance_symbol])
        model_variance_func = lambda inpt,par: variance_casadi_func(inpt,par).full().item()

        function_list = [loglik_func, fisher_info_func,
                         model_mean_func, model_sensitivity_func,model_variance_func,
                         observ_sampler_func, observation_percentile_func]

        return function_list

# --------------- Private functions (used by public functions) -------------------------------------

    def __confidence_intervals(self, mle_params, loglik_func, options):
        """ A private helper function for computing parameter confidence intervals.

        This private function is used in some calls to the public Model.fit() function and is 
        triggered by a user call after instantiation.

        This function computes marginal parameter confidence intervals for the model
        around the MLE estimate using profile likelihoods.

        Args:
            mle_params (array-like, floats): A vector specifying the MLE parameter estimates generated and 
                passed during a call to Model.fit().
            loglik_func: A Casadi function for computing the total data log-likelihood for the
                current dataset being handled within the calling Model.fit() call. The only argument
                is a putative parameter vector.
            options (dictionary): A dictionary of user-defined options encoded as key-value
                pairs. This dictionary is passed through from a call to Model.fit(), see
                the fit() functions documentation for possible key-value pairs.

        Returns:
            list of lists: A list of lists, the outer index corresponds to each parameter coordinate,
                the innder list contains both the upper and lower bounds for each parameter.
        """
        if options['Verbose']:
            progress_counter=0
            self.__progress_bar(progress_counter, self.num_param, prefix = 'Confidence Intervals:')
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
            solver_list = self.__profile_setup(mle_params, loglik_func, fixed_param, direction, options)
            #extract starting values for marginal parameters (those to be optimized during profile)
            marginal_param = [mle_params[i] for i in range(self.num_param) if direction[i]==0]
            #search to find the radius length in the specified profile direction,  positive search
            upper_radius = self.__loglik_search(solver_list, marginal_param, options, True)[0]
            #compute the location of the upper parameter bound
            upper_bound = mle_params[p] + direction[p] * upper_radius
            #search to find the radius length in the specified profile direction,  negative search
            lower_radius  =self.__loglik_search(solver_list, marginal_param, options, False)[0]
            #compute the location of the lower parameter bound
            lower_bound = mle_params[p]+direction[p]*lower_radius
            interval_list.append([lower_bound, upper_bound])
            if options['Verbose']:
                progress_counter += 1
                self.__progress_bar(progress_counter, self.num_param, prefix = 'Confidence Intervals:')
        return interval_list

    def __profile_plot(self, mle_params, loglik_func, figure, options):
        """ A private helper function for plotting both the profile likelihoods and the projections
        of the profiles traces.

        This private function is used in some calls to the public Model.fit() function and is 
        triggered by a user call after instantiation.

        This function generates a square grid of Matplotlib sub-plots, with a row and column for each
        parameter. The logelikelihood profiles of each parameter are plotted on sub-plots along the
        diagonal. These diganol plots also include a horizantal reference line indicating 
        loglikleihood value at which the perscribed confidence level boundaries occur. Projections 
        of the profile likelihood traces for each pair of parameters value are plotted on the lower
        triangular sub-plots. This function also returns lists containing the computed intervals,
        traces and profiles.

        Args:
            mle_params (array-like, floats): A vector specifying the MLE parameter estimates generated and 
                passed during a call to Model.fit().
            loglik_func (function): A Casadi function for computing the total data log-likelihood 
                for the current dataset being handled within the calling Model.fit() call. The only 
                argument is a putative parameter vector.
            figure (integer):The figure object handle on which the plot is generated.
            options (dictionary): A dictionary of user-defined options encoded as key-value
                pairs. This dictionary is passed through from a call to Model.fit(), see
                the fit() functions documentation for possible key-value pairs.


        Returns:
            nested lists: A multi-level list of lists is returned, the outer list contains three elements;
                1 - a list of lists where the outer index corresponds to each parameter and the 
                inner list contains the upper and lower confidence interval bounds for each parameter,
                2 - a list of list of lists where the outer index corresponds to each parameter and
                the middle index corresponds to points along each profile trace and the inner list
                contains the value of the parameter vector at that point in the parameter trace.
                3 - a ;ist of lists where the outer index corresponds to each parameter value and 
                the inner index corresponds to logliklihood ratio values for the given parameter
                at points along the likelihood profile. Note this return is passed up from a call
                to __profile_trace()
        """

        #extract the confidence level and compute the chisquared threshold
        chi_squared_level = st.chi2.ppf( options['ConfidenceLevel'],  self.num_param)
        #run profile trace to get the CI's,  parameter traces,  and LR profile
        [interval_list, trace_list, profile_list] = self.__profile_trace(mle_params, loglik_func, options)
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
                    #plt.legend()
                    plt.xlabel(self.param_name_list[p1])
                    plt.ylabel(self.param_name_list[p2])

        #return CI,  trace and profilem (for extensibility)
        return [interval_list, trace_list, profile_list]

    def __profile_trace(self, mle_params, loglik_func, options):
        """ A private helper function for computing the likelihood profile and trace for each 
        parameter.

        This private function is used in some calls to the public Model.fit() function and is 
        triggered by a user call after instantiation.

        This function accepts the MLE parameter estimate and a Casadi function for the total data
        log-likelihood of the target dataset being processed by the current Model.fit() call
        and uses this to organize computation of a profile likelihood and the corresponding 
        parameter vector trace along the computed profile, for each parameter. In the 
        course of the this computation, this function also computes the likelihood confidence 
        intervals for each parameter as well. This function organizes the profile computation for 
        each parameter and in both increasing and decreasing directions, however it relies on lower
        level helper functions (__loglik_search and __profile_setup) to do the actual computing.

        Args:
            mle_params (array-like, floats): A vector specifying the MLE parameter estimates generated and 
                passed during a call to Model.fit().
            loglik_func (function): A Casadi function for computing the total data log-likelihood 
                for the current dataset being handled within the initating Model.fit() call. The only 
                argument is a putative parameter vector.
            options (dictionary): A dictionary of user-defined options encoded as key-value
                pairs. This dictionary is passed through from a call to Model.fit(), see
                the fit() functions documentation for possible key-value pairs.

        Returns:
            nested lists: A multi-level list of lists is returned, the outer list contains three elements;
                1 - a list of lists where the outer index corresponds to each parameter and the 
                inner list contains the upper and lower confidence interval bounds for each parameter,
                2 - a list of list of lists where the outer index corresponds to each parameter and
                the middle index corresponds to points along each profile trace and the inner list
                contains the value of the parameter vector at that point in the parameter trace.
                3 - a ;ist of lists where the outer index corresponds to each parameter value and 
                the inner index corresponds to logliklihood ratio values for the given parameter
                at points along the likelihood profile.
        """
        if options['Verbose']:
            progress_counter=0
            self.__progress_bar(progress_counter, self.num_param, prefix = 'Profile Traces:')

        #extract the confidence level and compute the chisquared threshold
        chi_squared_level = st.chi2.ppf(options['ConfidenceLevel'],  self.num_param)
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
            profile_loglik_solver = self.__profile_setup(mle_params, loglik_func, fixed_param, direction, options)
            #set the starting values of the marginal parameters from the mle estimates
            marginal_param = [mle_params[i] for i in range(self.num_param) if not fixed_param[i]]
            #preform a profile search to find the upper bound on the radius for the profile trace
            [upper_radius, upper_param_list, upper_loglik_ratio_gap] = self.__loglik_search(profile_loglik_solver, marginal_param, options, True)
            #compute the parameter upper bound
            upper_bound = mle_params[p] + direction[p]*upper_radius
            #insert the profile parameter (upper) in the marginal parameter vector (upper),  creates a complete parameter vector
            upper_param_list.insert(p, upper_bound)
            #preform a profile search to find the lower bound on the radius for the profile trace
            [lower_radius, lower_param_list, lower_loglik_ratio_gap] = self.__loglik_search(profile_loglik_solver, marginal_param, options, False)
            #compute the parameter lower bound
            lower_bound = mle_params[p] + direction[p]*lower_radius
            #insert the profile parameter (lower) in the marginal parameter vector (lower),  creates a complete parameter vector
            lower_param_list.insert(p, lower_bound)
            #record the uppper and lower bounds in the CI list
            interval_list.append([lower_bound, upper_bound])
            #Create a grid of radia from the lower radius bound to the upper radius bound with the number of points requested in the profile
            radius_list = list(np.linspace(lower_radius,  upper_radius,  num=options['SampleNumber']+1, endpoint=False)[1:])
            #extract the marginal logliklihood solver,  to compute the profile
            #profile_loglik_solver = solver_list[0]
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
            if options['Verbose']:
                progress_counter += 1
                self.__progress_bar(progress_counter, self.num_param, prefix = 'Profile Traces:')
        #return the intervals,  parameter trace and loglik ratio profile
        return [interval_list, trace_list, profile_list]

    def __contour_plot(self, mle_params, loglik_func, figure, options):
        """ A private helper function for plotting the likelihood confidence contour plots during 
        calls to Model.fit()

        This private function is used in some calls to the public Model.fit() function and is 
        triggered by a user call after instantiation.

        This function plots the projections of the profile likelihood-based confidence volume in a
        2d plane for each pair of parameters using Matplotlib. This creates marginal confidence 
        contours for each pair of parameters which are added to the trace projection plots generated
        by __profile_plot() in the lower-triangualr sub-plots of the passed figure.

        Args:
            mle_params (array-like, floats): A vector specifying the MLE parameter estimates generated and 
                passed during a call to Model.fit().
            loglik_func (function): A Casadi function for computing the total data log-likelihood 
                for the current dataset being handled within the calling Model.fit() call. The only 
                argument is a putative parameter vector.
            figure (integer):The figure object handle on which the plot is generated.
            options (dictionary): A dictionary of user-defined options encoded as key-value
                pairs. This dictionary is passed through from a call to Model.fit(), see
                the fit() functions documentation for possible key-value pairs.
        """
        if options['Verbose']:
            total_iterations = self.num_param*(self.num_param-1)/2
            progress_counter=0
            self.__progress_bar(progress_counter, total_iterations, prefix = 'Contour Traces:')

        #loop over each unique pair of parameters 
        for p1 in range(self.num_param):
            for p2 in range(p1+1, self.num_param):
                #compute the x and y values for the contour trace
                [x_fit, y_fit] = self.__contour_trace(mle_params, loglik_func, [p1, p2], options)
                #plot the contour on the appropriate subplot (passed in from fit function,  shared with profileplot)
                plt.subplot(self.num_param,  self.num_param,  p2*self.num_param+p1+1)
                plt.plot(x_fit,  y_fit, label=self.param_name_list[p1]+' '+self.param_name_list[p2]+' contour')
                #plt.legend()
                plt.xlabel(self.param_name_list[p1])
                plt.ylabel(self.param_name_list[p2])
                if options['Verbose']:
                    progress_counter += 1
                    self.__progress_bar(progress_counter, total_iterations, prefix = 'Contour Traces:')


    def __contour_trace(self, mle_params, loglik_func, coordinates, options):
        """ A private helper function for computing the profile likelihood confidence contour
        projections for a specified pair of parameters.

        This private function is used in some calls to the public Model.fit() function and is 
        triggered by a user call after instantiation.

        This function computes a projection boundary of the profile likelihood confidence volume in
        a 2D plane for the specified pair of parameters. To do this a series of vectors radiating
        out from the MLE in the target parameter pair plane are searched. Along these vectors the
        marginal (non-target-pair) parameters are re-optimized iteratively. The vectors are extended
        until the conditionally optimized value of the likelihood falls below the selected confidence
        threshold. The 2D coordinates of these boundary points are recorded and their values are used
        to interpolate the projected confidence boundary via a periodic spline. This function
        sets up the radial search and performs the interpolation but relies on lower level helper 
        functions (__profile_setup and __loglik_search) to compute the boundary points.

        Args:
            mle_params (array-like, floats): A vector specifying the MLE parameter estimates generated and 
                passed during a call to Model.fit().
            loglik_func (function): A Casadi function for computing the total data log-likelihood 
                for the current dataset being handled within the calling Model.fit() call. The only 
                argument is a putative parameter vector.
            coordinates (array-like, integers): A pair of parameter indices specifying the specific 
                parameter pair for which a two 2D contour is to be computed in parameter space.
            options (dictionary): A dictionary of user-defined options encoded as key-value
                pairs. This dictionary is passed through from a call to Model.fit(), see
                the fit() functions documentation for possible key-value pairs.

        Returns:
            list of lists: A list of lists is returned, where the outer list contains two entries;
            a list of x coorindates followed by a list of y coordinates. The coordinates fall on
            the contour projection in the 2D target parameter pair plane. The number of of x and y
            coordinates is equal and is set by the "RadialNumber" key in Model.fit()'s options
            dictionary.
        """
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
        angle_list = list(np.linspace(-mt.pi,  mt.pi, options['RadialNumber']))
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
            solver_list = self.__profile_setup(mle_params, loglik_func, fixed_param, direction, options)
            #run the profile loglik search and return the found radia for the given angle
            radius = self.__loglik_search(solver_list, marginal_param, options, True)[0]
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

    def __profile_setup(self, mle_params, loglik_func, fixed_params, direction, options):
        """ A private helper function for constructing profile likelihood optimization solvers.

        This private function is used in some calls to the public Model.fit() function and is 
        triggered by a user call after instantiation.

        This function creates a Casadi optimization solver object for optimizing the model's 
        likelihood (on the target dataset of the iniating Model.fit() call) w.r.t. the marginal
        parameter values while holding a specific subset of parameters fixed at specified values.
        
        There can be a single fixed parameter (for the common profile likelihood computations), or a set
        (currently we only ever need a pair, used for computing confidence contour projections in 2D).
        The fixed parameter values are specified by a radius and direction vector which makes it
        easy to perform radial searches required for the confidence contour trace (the optimizer is
        iteratively run with different radia values in order to find the boundary points). The 
        direction vector is hard-coded into the resulting solver object after a call to 
        __profile_setup() but the radius can be adjust (hence moving the 'fixed' parameters) when 
        the solver object is actually run for optimization. This alterable radius value
        enables the actual profile likelihood search.
        
        The solver object actually optimizes the loglikelihood ratio (LLR) gap, the likelihood ratio
        gap is the negative difference between the chi-squared boundary (specified by the user in 
        the initiating call to Model.fit()) and the loglikelihood ratio at the current marginal
        parameter vector being optimized. This is just a rezeroing of the scale and is equivlant to
        optimizing the loglikelihood but makes some of the numerics and output more convenient to 
        manage.

        Note that limited testing has been done on single parameter models however the resulting
        return object for a 1D model should behave like a regular solver but actually just returns
        the LLR gap value at the radius-specified fixed values of the lone parameter.

        Args:
            mle_params (array-like, floats): A vector specifying the MLE parameter estimates generated and 
                passed during a call to Model.fit().
            loglik_func (function): A Casadi function for computing the total data log-likelihood 
                for the current dataset being handled within the calling Model.fit() call. The only 
                argument is a putative parameter vector.
            fixed_params (array-like, booleans): A boolean vector that is the same length as the 
                model's parameter vector. True values imply the cooresponding parameter (by position)
                should remain fixed in the solver object by the provideddirection and radius,
                False values indicate the corresponding parameter is marginal and should optimized.
            direction (array-like, floats): A vector that is the same length as the parameter vector
                specifying a direction in parameter space. Coordinate specified as True (non-marginal)
                in the fixed_params argument are used as the direction (along which the solver's
                run-time radius argument specifies the value of the fixed parameters)
            options (dictionary): A dictionary of user-defined options encoded as key-value
                pairs. This dictionary is passed through from a call to Model.fit(), see
                the fit() functions documentation for possible key-value pairs.

        Returns:
            casadi solver: A Casadi Ipopt solver object is returned, which can optimize the 
            loglikelihood ratio gap w.r.t. the marginal parameter values, conditioned on the fixed
            parmeter value which are specified by a radius (set at call time) and direction (set in
            __profile_setup() above when the solver is created). 
        """
        #compute the chi-squared level from confidence level
        chi_squared_level = st.chi2.ppf(options['ConfidenceLevel'],  self.num_param)
        #compute the number of fixed parameters (along which we do boundary search,  radius direction)
        num_fixed_param = sum(fixed_params)
        #compute the number of free/marginal parameters,  which are optimized at each step of the search
        num_marginal_param = self.num_param-num_fixed_param
        if num_fixed_param == 0:
            raise Exception('No fixed parameters passed to loglikelihood search,  contact developers!')
        #create casadi symbols for the marginal parameters
        if self.symbolics_boolean:
            marginal_param_symbols = cs.SX.sym('marginal_param_symbols', num_marginal_param)
        else:
            marginal_param_symbols = cs.MX.sym('marginal_param_symbols', num_marginal_param)
        #create casadi symbols for the radius (radius measured from mle,  in given direction,  to chi-squared boundary)
        if self.symbolics_boolean:
            radius_symbol = cs.SX.sym('radius_symbol')
        else:
            radius_symbol = cs.MX.sym('radius_symbol')
        #creat a list to store a complete parameter vector
        #this is a mixture of fixed parameters set by the direction and radius,  and marginal parameters which are free symbols
        param_list = []
        #create a counter to count marginal parameters already added
        marginal_counter = 0
        #loop over the parameters
        for i in range(self.num_param):   
            if fixed_params[i]:
                #if the parameter is fixed,  add an entry parameterized by the radius from the mle in given direction
                param_list.append( direction[i]*radius_symbol + mle_params[i] )
            else:
                #else add marginal symbol to list and increment marginal counter
                param_list.append(marginal_param_symbols[marginal_counter])
                marginal_counter+=1
        #convert the list of a casadi vector
        param = cs.vertcat(*param_list)
        #create a symnol for the loglikelihood ratio gap at the parameter vector
        loglik_ratio_gap_symbol = 2*(loglik_func(mle_params) - loglik_func(param)) - chi_squared_level
        #check if any marginal parameters exist
        if not num_marginal_param == 0:
            #if there are marginal parameters create Ipopt solvers to optimize the marginal params
            # create an IPOPT solver to minimize the loglikelihood for the marginal parameters
            # this solver minimize the logliklihood ratio but has a return objective value of the LLR gap,  so its root is on the boundary
            #  it accepts the radius as a fixed parameter
            profile_problem_struct = {'f': loglik_ratio_gap_symbol,  'x': marginal_param_symbols, 'p':radius_symbol}#,  'g': cs.vertcat(*OptimConstraints)
            profile_loglik_solver = cs.nlpsol('PLLSolver',  'ipopt',  profile_problem_struct, {'ipopt.print_level':0, 'print_time':False})
        else:
            # else if there are no marginal parameters (i.e. 2d model),  create casadi functions emulating the above without optimization (which is not needed)
            profile_loglik_solver = cs.Function('PLLSolver',  [radius_symbol],  [loglik_ratio_gap_symbol]) 
        #return the solvers/functions
        return profile_loglik_solver

    def __loglik_search(self, profile_loglik_solver, marginal_param, options, forward=True):
        """ A private helper function for performing a bisection search for profile likelihood 
        boundary points.

        This private function is used in some calls to the public Model.fit() function and is 
        triggered by a user call after instantiation.

        This function performs a root finding algorithm using the solver object  (profile_loglik_solver)
        generated by __profile_setup(). Currently the function uses a bisection search. The search 
        is performed along the direction vector specified when the solve object is created by 
        __profile_setup(), with the given set of fixed and marginal parameters specified in that 
        call. The search dimensions is the radius argument of the solver object, which specified when
        combined with the direction vector hard-coded in the solver object to specify its magnuted, 
        resulst in a specific set of fixed parmaeter values at which the conditional likelihood 
        optimization is performed. The radius at which a root is found indicates a boundary point in
        the confidence volume at the confidence level specified by the use in their initating call to 
        Model.fit().

        Note a passed version used Halley's method; a higher order extension of Newton's method for
        finding roots. This was possible using Casadi's ability to differentiate optimization solutions
        and was faster, but proved to be numerically unstable.

        Args:
            profile_loglik_solver: The Casadi Ipopt solver object functions for finding the conditionally
                optimized loglikelihood ratio gap at a given radius from MLE. This object is generated
                with a call to __profile_setup().
            marginal_param (array-like, floats): The starting values (usually the MLE) for the 
                marginal parameters that are passed to the Ipopt solver.
            options (dictionary): A dictionary of user-defined options encoded as key-value
                pairs. This dictionary is passed through from a call to Model.fit(), see
                the fit() functions documentation for possible key-value pairs.
            forward (boolean): A boolean indicating the directino of the search, if True the search
                is done in the forward (positive) radius direction (relative to direction specidied
                in solver object), if False the search is performed starting with a negative radius.

        Returns:
            list: The return object is a three element list, the first entry is  contains the radius
            value at which the root was found, the second entry contains a list of the optimzed marginal
            parameter values at the root, and the third entry contains the residual value of the LLR 
            gap at the root (this should be near zero).
        """
        #check if the search is run in negative or positive direction,  set intial step accordingly
        if forward:
            sign = 1
        else:
            sign = -1
        #radius = sign*options['InitialStep']
        #get the number of marginal parameters
        num_marginal_param = len(marginal_param)

        init_radius =  sign*options['InitialStep']
        upper_radius = init_radius / 10**(1/options['SearchFactor'])
        upper_ratio_gap = -st.chi2.ppf(options['ConfidenceLevel'],  self.num_param)
        upper_marginal_param = np.array(marginal_param)
        while not np.sign(upper_ratio_gap)==1 :
            upper_radius = upper_radius * 10**(1/options['SearchFactor'])
            lower_ratio_gap = upper_ratio_gap
            if num_marginal_param > 0:
                lower_marginal_param = upper_marginal_param
                profile_solution_struct = profile_loglik_solver(x0=upper_marginal_param, p=upper_radius)
                #update the current LLR gap value
                upper_ratio_gap = profile_solution_struct['f'].full().item()
                #update the current optimal values of the marginal parameters
                upper_marginal_param = profile_solution_struct['x'].full().flatten()
            else:
                upper_ratio_gap = profile_loglik_solver(upper_radius).full().item()
            #check if search bounds violated
            if np.log10(upper_radius/init_radius) > options['SearchBound']:
                raise Exception('Unable to find loglikelihood ratio boundary in required bounds!')

        lower_radius = upper_radius / 10**(1/options['SearchFactor'])
        
        #NOTE: should check to see if we go negative,  loop too many time,  take massive steps,  want to stay in the domain of the MLE
        #NOTE:need max step check,  if steps are all in same direction perhaps set bound at inf and return warning,  if oscillating error failure to converge
        #set the initial LLR gap to a very large number
        middle_ratio_gap = 9e9
        middle_marginal_param = []
        #create a counter to track root finding iterations
        iteration_counter = 0
        #loop until tolerance criteria are met (LLR gap drops to near zero)
        while  abs(middle_ratio_gap)>options['Tolerance']:
            
            middle_radius = \
                    (upper_ratio_gap*upper_radius + np.abs(lower_ratio_gap)*lower_radius)\
                        /(upper_ratio_gap-lower_ratio_gap)

            if num_marginal_param > 0:
                middle_init_marginal_param = \
                    (upper_ratio_gap*upper_marginal_param + np.abs(lower_ratio_gap)*lower_marginal_param)\
                        /(upper_ratio_gap-lower_ratio_gap)

                profile_solution_struct = profile_loglik_solver(x0=middle_init_marginal_param,
                                                                p=middle_radius)
                #update the current LLR gap value
                middle_ratio_gap = profile_solution_struct['f'].full().item()
                #update the current optimal values of the marginal parameters
                middle_marginal_param = profile_solution_struct['x'].full().flatten()
            else:
                #else if there are no marginal parameters
                #call the appropriate casadi function to get the current LLR gap value
                middle_ratio_gap = profile_loglik_solver(middle_radius).full().item()

            if middle_ratio_gap>0:
                upper_radius = middle_radius
                upper_ratio_gap = middle_ratio_gap
                upper_marginal_param = middle_marginal_param
            else:
                lower_radius = middle_radius
                lower_ratio_gap = middle_ratio_gap
                lower_marginal_param = middle_marginal_param

            #print('Rad: '+str(middle_radius)+', Gap: '+str(middle_ratio_gap)+', Par: '+str(middle_marginal_param))
            #increment the iterations counter
            iteration_counter+=1
            if iteration_counter>options['MaxSteps']:
                raise Exception('Maximum number of iterations reached in logliklihood boundary search!')
            
        #return the radius of the root,  the optimal marginal parameters at the root and the ratio_gap at the root (should be near 0)
        return [middle_radius.item(), list(middle_marginal_param), middle_ratio_gap]



    def __create_grid(self,input_candidate_list):
        """A private helper function to create a grid of candidate parameter points 

        This fucntion creates a grid of candidate parameter points by permuting a lits of candidate
        points. This is used for an a crude initial parmeter search prior to maximum likelihood 
        optimization by the Model.fit function. This function uses recursion to permute the candidate
        grid.

        Args:
            input_candidate_list (list of array-likes): A list, the same length as the model's 
                parameter vector, each element is an array-like vector of candidate parameter
                values for the given parameter.

        Returns:
            list of arrays: The return is a list, the same length as the number of permutations of
            the candidate paramter points passed, each element of the list contains a Numpy parameter
            vector with a unique permutation of the candidate points.  
        """
        new_grid = []
        candidates_current_dim = input_candidate_list.pop()

        if len(input_candidate_list)>0:
            current_grid = self.__create_grid(input_candidate_list)
            for current_grid_point in current_grid:
                for canidate_element in candidates_current_dim:
                    new_grid_point = np.hstack((current_grid_point,canidate_element))
                    new_grid.extend([new_grid_point])
        else:
            new_grid=[d for d in candidates_current_dim]

        return new_grid

    def __progress_bar (self, iteration, total, prefix = ''):
        """A private helper function to print a progress bar in a looped process

        This fucntion prints or update a progress bar on the commandline output indicating, the 
        current progress through a loop.

        Args:
            iteration (integer): Current iteration in the process
            total (integer): Total number of iterations in the process
            prefix (string): A prefix string to name the process
        """
        max_length= 50
        percent = ("{0:.1f}").format(100 * (iteration / float(total)))
        filledLength = int(max_length * iteration // total)
        bar = '█' * filledLength + '-' * (max_length - filledLength)
        print('\r%s |%s| %s%%' % (prefix, bar, percent), end = "\r")
        if iteration == total: 
            print()

#NOTE: Note sure if these can be included in the class somehow
@contextmanager
def silence_stdout():
    old_target = sys.stdout
    try:
        with open(os.devnull, "w") as new_target:
            sys.stdout = new_target
            yield new_target
    finally:
        sys.stdout = old_target

# class factorial(cs.Callback):
#     """A private helper function to print a progress bar in a looped process

#         This fucntion prints or update a progress bar on the commandline output indicating, the 
#         current progress through a loop.

#         Args:
#             iteration (integer): Current iteration in the process
#             total (integer): Total number of iterations in the process
#             prefix (string): A prefix string to name the process
#         """
#     def __init__(self,  name,  options = {}):
#         cs.Callback.__init__(self)
#         self.construct(name,  options)

#     # Number of inputs and outputs
#     def get_n_in(self): return 1
#     def get_n_out(self): return 1

#     # Initialize the object
#     def init(self):
#         print('initializing object')

#     # Evaluate numerically
#     def eval(self,  arg):
#         k  =  arg[0]
#         cnt = 1
#         f = max(k,1)
#         while (k-cnt)>0:
#             f = f*(k-cnt)
#             cnt = cnt+1
#         return [f]


#NOTE: commented stuff is saved from using newton type root search in profile loglik
#NOTE: replaced with grid search for stability in more complex models (currently in code ~June 2020)
    # def __profile_setup(self, mle_params, loglik_func, fixed_params, direction, options):
    #     """ 
    #     This function creates function/solver objects for performing a profile likelihood search for the condifence boundary
    #     in the specified direction,  the function/solver objects compute the logliklihood ratio gap
    #     at a given radius (along the specified direction),  along with the LLR gaps 1st and 2nd derivative with respect to the radius.
    #     marginal (free) parameters (if they exist) are optimized conditional on the fixed parameters specified by the radius and direction
    #     the likelihood ratio gap is the negative difference between the chi-squared boundary and the loglik ratio at the current radius

    #     Args:
    #         mle_params: mle parameter estimates,  recieved from fitting
    #         loglik_func: casadi logliklihood function for the given dataset
    #         fixed_params: a boolean vector,  same length as the parameters,  true means cooresponding parameters fixed by direction and radius,  false values are marginal and optimized (if they exist)
    #         direction: a direction in parameter space,  coordinate specified as true in fixed_params are used as the search direction
    #         options: an options dictionary for passing user options

    #     Returns:
    #         profile_loglik_solver: casadi function/ipopt solver that returns the loglik ratio gap for a given radius,  after optimizing free/marginal parameters if they exist
    #         profile_loglik_jacobian_solver: casadi function/ipopt derived derivative function that returns the derivative of the loglik ratio gap with respect to the radius (jacobian is 1x1)
    #         profile_loglik_hessian_solver: casadi function/ipopt derived 2nd derivative function that returns the 2nd derivative of the loglik ratio gap with respect to the radius (hessian is 1x1)
    #     """
    #     #compute the chi-squared level from confidence level
    #     chi_squared_level = st.chi2.ppf(options['ConfidenceLevel'],  self.num_param)
    #     #compute the number of fixed parameters (along which we do boundary search,  radius direction)
    #     num_fixed_param = sum(fixed_params)
    #     #compute the number of free/marginal parameters,  which are optimized at each step of the search
    #     num_marginal_param = self.num_param-num_fixed_param
    #     if num_fixed_param == 0:
    #         raise Exception('No fixed parameters passed to loglikelihood search,  contact developers!')
    #     #create casadi symbols for the marginal parameters
    #     marginal_param_symbols = cs.SX.sym('marginal_param_symbols', num_marginal_param)
    #     #create casadi symbols for the radius (radius measured from mle,  in given direction,  to chi-squared boundary)
    #     radius_symbol = cs.SX.sym('radius_symbol')
    #     #creat a list to store a complete parameter vector
    #     #this is a mixture of fixed parameters set by the direction and radius,  and marginal parameters which are free symbols
    #     param_list = []
    #     #create a counter to count marginal parameters already added
    #     marginal_counter = 0
    #     #loop over the parameters
    #     for i in range(self.num_param):   
    #         if fixed_params[i]:
    #             #if the parameter is fixed,  add an entry parameterized by the radius from the mle in given direction
    #             param_list.append( direction[i]*radius_symbol + mle_params[i] )
    #         else:
    #             #else add marginal symbol to list and increment marginal counter
    #             param_list.append(marginal_param_symbols[marginal_counter])
    #             marginal_counter+=1
    #     #convert the list of a casadi vector
    #     param = cs.vertcat(*param_list)
    #     #create a symnol for the loglikelihood ratio gap at the parameter vector
    #     loglik_ratio_gap_symbol = 2*(loglik_func(mle_params) - loglik_func(param)) - chi_squared_level
    #     #check if any marginal parameters exist
    #     if not num_marginal_param == 0:
    #         #if there are marginal parameters create Ipopt solvers to optimize the marginal params
    #         # create an IPOPT solver to minimize the loglikelihood for the marginal parameters
    #         # this solver minimize the logliklihood ratio but has a return objective value of the LLR gap,  so its root is on the boundary
    #         #  it accepts the radius as a fixed parameter
            
    #         profile_problem_struct = {'f': loglik_ratio_gap_symbol,  'x': marginal_param_symbols, 'p':radius_symbol}#,  'g': cs.vertcat(*OptimConstraints)
    #         profile_loglik_solver = cs.nlpsol('PLLSolver',  'ipopt',  profile_problem_struct, {'ipopt.print_level':0, 'print_time':False})

    #         #alternate solver, for better derivatives
    #         #opts = dict(qpsol='qrqp', qpsol_options=dict(print_iter=False,error_on_fail=False), print_time=False)
    #         #solver = ca.nlpsol('solver', 'sqpmethod', prob, opts)
    #         #profile_problem_struct = {'f': loglik_ratio_gap_symbol,  'x': marginal_param_symbols, 'p':radius_symbol}#,  'g': cs.vertcat(*OptimConstraints)
    #         #profile_loglik_solver = cs.nlpsol('PLLSolver',  'sqpmethod',  profile_problem_struct, {'qpsol':'qrqp', 'qpsol_options':{'print_iter':False,'print_header':False,'error_on_fail':False},"print_header":False,"print_iteration":False,"print_time":False})

    #         #create a casadi function that computes the derivative of the optimal LLR gap solution with respect to the radius parameter
    #         profile_loglik_jacobian_solver = profile_loglik_solver.factory('PLLJSolver',  profile_loglik_solver.name_in(),  ['sym:jac:f:p'])
    #         #create a casadi function that computes the 2nd derivative of the optimal LLR gap solution with respect to the radius parameter
    #         profile_loglik_hessian_solver = profile_loglik_solver.factory('PLLHSolver',  profile_loglik_solver.name_in(),  ['sym:hess:f:p:p'])
    #     else:
    #         # else if there are no marginal parameters (i.e. 2d model),  create casadi functions emulating the above without optimization (which is not needed)
    #         profile_loglik_solver = cs.Function('PLLSolver',  [radius_symbol],  [loglik_ratio_gap_symbol]) 
    #         #create the 1st derivative function
    #         profile_loglik_jacobian_symbol = cs.jacobian(loglik_ratio_gap_symbol, radius_symbol)
    #         profile_loglik_jacobian_solver = cs.Function('PLLJSolver',  [radius_symbol],  [profile_loglik_jacobian_symbol]) 
    #         #create the second derivative function
    #         profile_loglik_hessian_symbol = cs.jacobian(profile_loglik_jacobian_symbol, radius_symbol) 
    #         profile_loglik_hessian_solver = cs.Function('PLLHSolver',  [radius_symbol],  [profile_loglik_hessian_symbol]) 
    #         #NOTE: not sure what second returned value of casadi hessian func is,  did this to avoid it (it may be gradient,  limited docs)
    #     #return the solvers/functions
    #     return [profile_loglik_solver, profile_loglik_jacobian_solver, profile_loglik_hessian_solver]

    # def __loglik_search(self, solver_list, marginal_param, options, forward=True):
    #     """ 
    #     This function performs a root finding algorithm using solver_list objects
    #     It uses halley's method to find the radius value (relative to the mle) where the loglik ratio equals the chi-squared level
    #     This radius runs along the direction specified in the solver_list when they are created
    #     Halley's method is a higher order extension of newton's method for finding roots

    #     Args:
    #         solver_list: solver/casadi functions for finding the loglikelihood ratio gap at a given radius from mle,  and its 1st/2nd derivatives
    #         marginal_param: starting values (usually the mle) for the marginal parameters
    #         options: an options dictionary for passing user options
    #         forward: boolean,  if true search is done in the forward (positive) radius direction (relative to direction specidied in solver list),  if false perform search starting with a negative radius

    #     Returns:
    #         radius: returns the radius corresponding to the chi-squared boundary of the loglikelihood region
    #         marginal_param: returns the optimal setting of the marginal parameters at the boundary
    #         ratio_gap: returns the residual loglikelihood ratio gap at the boundary (it should be small,  within tolerance)
    #     """
    #     #check if the search is run in negative or positive direction,  set intial step accordingly
    #     if forward:
    #         sign = 1
    #     else:
    #         sign = -1
    #     #radius = sign*options['InitialStep']
    #     #get the number of marginal parameters
    #     num_marginal_param = len(marginal_param)
    #     #get the solver/function objects,  loglik ratio gap and derivatives w.r.t. radius
    #     profile_loglik_solver = solver_list[0]
    #     profile_loglik_jacobian_solver = solver_list[1]
    #     profile_loglik_hessian_solver = solver_list[2]
    #     #set the initial marginal parameters
    #     #marginal_param = marginal_param NOTE:delete this?
    #     #generate initial search starting point

    #     #**********IN PROGRESS#**********
    #     radial_grid = sign*np.logspace(np.log10(options['InitialStep']),np.log10(options['InitialStep'])+2,100)
    #     radius = radial_grid [np.argmin( np.abs([ profile_loglik_solver(x0=marginal_param, p=r)['f'].full().item() for r in radial_grid] ))]
    #     #**********IN PROGRESS#**********

    #     #set the initial LLR gap to a very large number
    #     ratio_gap = 9e9
    #     #NOTE: should check to see if we go negative,  loop too many time,  take massive steps,  want to stay in the domain of the MLE
    #     #NOTE:need max step check,  if steps are all in same direction perhaps set bound at inf and return warning,  if oscillating error failure to converge
    #     #create a counter to track root finding iterations
    #     iteration_counter = 0
    #     #loop until tolerance criteria are met (LLR gap drops to near zero)
    #     while  abs(ratio_gap)>options['Tolerance'] and iteration_counter<options['MaxSteps']:
    #         if not num_marginal_param == 0:
    #             #if there are marginal parameters
    #             #run the ipopt solver to optimize the marginal parameters,  conditional on the current radius
    #             profile_solution_struct = profile_loglik_solver(x0=marginal_param, p=radius)#,  lbx=[],  ubx=[],  lbg=[],  ubg=[]
    #             #solver for the LLR gap 1st derivative w.r.t. the radius
    #             profile_jacobian_struct = \
    #                 profile_loglik_jacobian_solver( x0=profile_solution_struct['x'],
    #                                                 lam_x0=profile_solution_struct['lam_x'],
    #                                                 lam_g0=profile_solution_struct['lam_g'],
    #                                                 p=radius)
    #             #solver for the LLR gap 2nd derivative w.r.t. the radius
    #             profile_hessian_struct = \
    #                 profile_loglik_hessian_solver(  x0=profile_solution_struct['x'],
    #                                                 lam_x0=profile_solution_struct['lam_x'],
    #                                                 lam_g0=profile_solution_struct['lam_g'],
    #                                                 p=radius)
    #             #update the current optimal values of the marginal parameters
    #             marginal_param = list(profile_solution_struct['x'].full().flatten())
    #             #update the current LLR gap value
    #             ratio_gap = profile_solution_struct['f'].full().item()
    #             #extract the LLR gap 1st derivative value
    #             dratiogap_dradius = profile_jacobian_struct['sym_jac_f_p'].full().item()
    #             #extract the LLR gap 2nd derivative value
    #             d2ratiogap_dradius2 = profile_hessian_struct['sym_hess_f_p_p'].full().item()
    #         else:
    #             #else if there are no marginal parameters
    #             #call the appropriate casadi function to get the current LLR gap value
    #             ratio_gap = profile_loglik_solver(radius).full().item()
    #             #call the appropriate casadi function to get the LLR gap 1st derivative value
    #             dratiogap_dradius = profile_loglik_jacobian_solver(radius).full().item()
    #             #call the appropriate casadi function to get the LLR gap 2nd derivative value
    #             d2ratiogap_dradius2 = profile_loglik_hessian_solver(radius).full().item()
    #         #increment the iterations counter
    #         iteration_counter+=1
    #         #use Halley's method (higher order extention of newtons method) to compute the new radius value
    #         #**********IN PROGRESS#**********
    #         newradius = radius - (2*ratio_gap*dratiogap_dradius) \
    #                     /(2*dratiogap_dradius**2 - ratio_gap*d2ratiogap_dradius2)
    #         if radius*sign<0:
    #             tt=0
    #             #print('WARNING! Search direction has reversed')
    #         #**********IN PROGRESS#**********
    #         print('Rad: '+str(radius)+', Gap: '+str(ratio_gap)+', Par: '+str(marginal_param)+', df: '+str(dratiogap_dradius)+', ddf: '+str(d2ratiogap_dradius2))
    #         radius = newradius
    #     # throw error if maximum number of iterations exceeded
    #     if iteration_counter>=options['MaxSteps']:
    #         raise Exception('Maximum number of iterations reached in logliklihood boundary search!')
    #     #return the radius of the root,  the optimal marginal parameters at the root and the ratio_gap at the root (should be near 0)
    #     return [radius, marginal_param, ratio_gap]

    # def fit(self, dataset_struct, start_param, options={}):
    #     """
    #     This function fits the model to a dataset using maximum likelihood
    #     it also provides optional marginal confidence intervals
    #     and plots of logliklihood profiles/traces and marginal projected confidence contours

    #     Args:
    #         datasets: either; a dictionary for one dataset OR a list of dictionaries,  each design replicates OR a list of lists of dict's where each index in the outer lists has a unique design
    #         start_param: a list of starting values for the parameters
    #         options: optional,  a dictionary of user defined options

    #     Return:
    #         ParameterFitStruct: either; a list of lists structure with parameter fit lists,  has the same shape/order as the input,  OR,  the same strcture but with fits and intervals as the leaf object
    #     """
    #     #NOTE: needs checks on inputs, var names, designs must be exact reps
    #     #NOTE: NEED testing for multiple observation input structures,  multiple dimensions of parameters ideally,  1, 2, 3 and 7+
    #     #NOTE: add some print statments to provide user with progress status
    #     #NOTE: currently solve multiplex simultaneosly (one big opt problem) but sequentially may be more robust to separation (discrete data),  or randomly non-identifiable datasets (need to test)
    #     default_options = { 'Confidence':       ['None',    lambda x: isinstance(x,str) and (x=='None' or x=='Interval' or x=='Profile' or x=='Contours')],
    #                         'ConfidenceLevel':  [0.95,      lambda x: isinstance(x,float) and 0<=x and x<=1],
    #                         'RadialNumber':     [30,        lambda x: isinstance(x,int) and 1<x],
    #                         'SampleNumber':     [10,        lambda x: isinstance(x,int) and 1<x],
    #                         'Tolerance':        [0.001,     lambda x: isinstance(x,float) and 0<x],
    #                         'InitialStep':      [.01,         lambda x: isinstance(x,float) and 0<x],
    #                         'MaxSteps':         [1000,      lambda x: isinstance(x,int) and 1<x],
    #                         'SearchFactor':     [5,         lambda x: isinstance(x,float) and 0<x],
    #                         'SearchBound':      [4,         lambda x: isinstance(x,float) and 0<x]}
    #     for key in options.keys():
    #         if not key in default_options.keys():
    #             raise Exception('Invalid option key; '+key+'!')
    #         elif not default_options[key][1](options[key]):
    #             raise Exception('Invalid value; '+str(options[key])+', passed for option key; '+key+'!')
    #     for key in default_options.keys():
    #         if not key in options.keys() :
    #             options[key] = default_options[key][0]
    #     if not len(start_param) == self.num_param:
    #         raise Exception('Starting parameter mismatch, there were; '+str(len(start_param))+', provided but; '+str(self.num_param)+' needed!')
    #     #this block allows the user to pass a dataset, list of datasets,  list of lists etc. for Design x Replicate fitting
    #     if isinstance(dataset_struct, pd.DataFrame):
    #         #if a single dataset is passed vis the dataset_struct input,  wrap it in two lists so it matches general case
    #         design_datasets = [[dataset_struct]]
    #     elif isinstance(dataset_struct[0], pd.DataFrame):
    #         #else if dataset_struct input is a list of replciated datasets,  wrap it in a single list to match general case
    #         design_datasets = [dataset_struct]
    #     else:
    #         #else if dataset_struct input is a list of designs, each with a list of replicates,  just pass on th input
    #         design_datasets = dataset_struct
    #     #set confidence interval boolean
    #     interval_bool = options['Confidence']=="Intervals" or options['Confidence']=="Profiles" or options['Confidence']=="Contours"
    #     #set plotting boolean
    #     plot_bool = options['Confidence']=="Contours" or options['Confidence']=="Profiles"
    #     #get number of designs
    #     num_designs = len(design_datasets)
    #     #get number of replicates for each design
    #     num_replicat_list = [len(rep_list) for rep_list in design_datasets]
    #     #get total number of datasets
    #     num_datasets = sum(num_replicat_list)
    #     #create a list to store parameter casadi optimization symbols, starting parameters values, generic casadi loglikelihood functions
    #     param_symbols_list, start_params_list, loglik_func_list = [], [], []
    #     #create a total loglikelihood summation store, initialize to zero
    #     total_loglik_symbol = 0
    #     #create an archetypal vector of paramter symbols, used to build casadi loglikelihood functions for each design
    #     archetype_param_symbols = cs.SX.sym('archetype_param_symbols', self.num_param)
    #     #loop over different designs (outer most list)
    #     for d in range(num_designs):
    #         #get the set of replicates for this design
    #         replicat_datasets = design_datasets[d]
    #         #create a summation variable for the loglikelihood for a loglik_symbolset of the current design
    #         archetype_loglik_symbol = 0
    #         #for each design use the first replicate 
    #         archetype_dataset = replicat_datasets[0]
    #         # get onbservation count for this design
    #         num_observations = len(archetype_dataset.index)
    #         #create a vector of casadi symbols for the observations
    #         archetype_observ_symbol = cs.SX.sym('archetype_observ_symbol'+str(d), num_observations)
    #         #loop over the dataset inputs
    #         for index,row in archetype_dataset.iterrows():
    #             #get the curren input settings
    #             input_row = row[self.input_name_list].to_numpy()
    #             #get the observation variable index
    #             #observ_var_index = self.observ_name_dict[row['Variable']]
    #             #create a symbol for the loglikelihood for the given input and observation variable
    #             archetype_loglik_symbol += self.loglik[row['Variable']](archetype_observ_symbol[index],
    #                                                                     input_row,
    #                                                                     archetype_param_symbols)
    #         #create a casadi function for the loglikelihood of the current design (observations are free/input symbols)
    #         archetype_loglik_func = cs.Function('archetype_loglik_func'+str(d),
    #                                             [archetype_observ_symbol, archetype_param_symbols],
    #                                             [archetype_loglik_symbol])
    #         #loop over replicates within each design
    #         for r in range(num_replicat_list[d]):
    #             #NOTE: could abstract below into a Casadi function to avoid input/observ loop on each dataset and replicat
    #             #get the dataset from the replicate list
    #             dataset = replicat_datasets[r]
    #             #create a vector of parameter symbols for this specific dataset,  each dataset gets its own,  these are used for ML optimization
    #             fit_param_symbols = cs.SX.sym('fit_param_symbols'+'_'+str(d)+str(r), self.num_param)
    #             #extract the vector of observations in the same format as in the archetype_loglik_func function input
    #             observ_vec = dataset['Observation'].to_numpy() 
    #             #create a symbol for the datasets loglikelihood function by pass in the observations for the free symbols in ObservSymbol
    #             dataset_loglik_symbol = archetype_loglik_func(observ_vec, fit_param_symbols)
    #             #add it to the list of replicate loglik functions
    #             loglik_func_list.append( cs.Function('dataset_loglik_func_'+str(d)+'_'+str(r),
    #                                                  [fit_param_symbols],
    #                                                  [dataset_loglik_symbol]))
    #             #set up the logliklihood symbols for given design and replicat
    #             param_symbols_list.append(fit_param_symbols)
    #             #record the starting parameters for the given replicate and dataset
    #             start_params_list.extend(start_param)
    #             #add the loglikelihood to the total 
    #             #NOTE: this relies on the distirbutivity of serperable optimization problem,  should confirm
    #             total_loglik_symbol += dataset_loglik_symbol
    #     #NOTE: this approach is much more fragile to separation (glms with discrete response),  randomly weakly identifiable datasets
    #     #NOTE: should be checking solution for convergence, should allow user to pass options to ipopt
    #     #NOTE: allow bfgs for very large nonlinear fits, may be faster
    #     # Create an IPOPT solver for overall maximum likelihood problem,  we pass negative total_loglik_symbol because IPOPT minimizes
    #     total_loglik_optim_struct = {'f': -total_loglik_symbol,  'x': cs.vertcat(*param_symbols_list)}#,  'g': cs.vertcat(*OptimConstraints)
    #     param_fitting_solver = cs.nlpsol('solver',  'ipopt',  total_loglik_optim_struct, {'ipopt.print_level':5, 'print_time':False}) #,'ipopt.hessian_approximation':'limited-memory'
    #     # Solve the NLP fitting problem with IPOPT call
    #     param_fit_solution_struct = param_fitting_solver(x0=start_params_list)
    #     #extract the fit parameters from the solution structure
    #     fit_param_matrix = param_fit_solution_struct['x'].full().flatten().reshape((-1,self.num_param))
    #     #create row index for parameter dataframe export, specifies design index for each datset
    #     design_index = [design_indx+1 for design_indx in range(num_designs) for i in range(num_replicat_list[design_indx])]
    #     #check if a  confidence value was passed as an option, requires confidence intervals at least
    #     if interval_bool:
    #         #create a multiindex for the columns of parameter dataframe export
    #         # first level define estimate and lower/upper bound, second is param values
    #         column_index = pd.MultiIndex.from_product([['Estimate','Lower','Upper'],self.param_name_list],names=['Value', 'Parameter'])
    #         #create an empty list to store lower/upper bounds for intervals
    #         bound_list = []
    #         #loop over each design
    #         for d in range(num_datasets):
    #             #get the parameter estimates for current dataset
    #             param_vec = fit_param_matrix[d,:]
    #             #check if graphing options; contour plots or profile trace plots,  are requested
    #             if options['Confidence']=="Contours" or options['Confidence']=="Profiles":
    #                 #if so, create a figure to plot on
    #                 fig = plt.figure()
    #                 #run profileplots to plot the profile traces and return CI's
    #                 interval_list = self.__profile_plot(param_vec, loglik_func_list[d], fig, options)[0]
    #                 #if contour plots are requested specifically, run contour function to plot the projected confidence contours
    #                 if options['Confidence']=="Contours":
    #                     self.__contour_plot(param_vec, loglik_func_list[d], fig, options)
    #             elif options['Confidence']=="Intervals":
    #                 #if confidence intervals are requested, run confidenceintervals to get CI's and add them to replicate list
    #                 interval_list = self.__confidence_intervals(param_vec, loglik_func_list[d], options)
    #             #store the current boundary vector in the bound_list
    #             bound_list.append( np.asarray(interval_list).T.flatten())
    #         #if requested, show the plots
    #         if plot_bool:
    #             #NOTE: need to give order of plots to user, perhaps with title
    #             plt.show()
    #         #combine the bound vectors in boundlist into a single numpy matrix
    #         bound_param_matrix = np.stack(bound_list)
    #         #concatenate the fit_param_matrix matrix of fit parameters with the bound_param_matrix of parameter interval boundaries
    #         param_output_matrix = np.concatenate([fit_param_matrix, bound_param_matrix], axis=1)
    #     else:
    #         #if no intervals are requested, set column indext to single level, with just parameter names
    #         column_index = self.param_name_list
    #         param_output_matrix = fit_param_matrix
    #     #create ouput dataframe with all request parameter information
    #     param_data = pd.DataFrame(param_output_matrix, index=design_index, columns=column_index)
    #     #return the dataframe
    #     return param_data

# Full ML fitting,  perhaps with penalized likelihood???
# fit assesment,  with standardized/weighted residual output,  confidence regions via asymptotics (with beale bias),  likelihood basins,  profile liklihood,  sigma point (choose one or maybe two)
# function to generate a prior covariance (that can be fed into design)
# function for easy simulation studies (generate data,  with given experiment)