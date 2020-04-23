import casadi as cs
import numpy as np
import pandas as pd
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
    #NOTE: should rename inputs to cavariates or controls or smthg

    distribution_dict={ 'Normal':       ['Mean','Variance'],
                        'Poisson':      ['Rate'],
                        'Lognormal':    ['GeoMean','GeoVariance'],
                        'Bernoulli':    ['Probability'],
                        'Binomial':     ['Probability'],
                        'Exponential':  ['Rate'],
                        'Gamma':        ['Shape','Scale']}

    def __init__(self,  observ_struct,  input_names,  param_names):
        
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
            if not len(obs_tuple)==3:
                raise Exception('Error in observation entry; '+str(o)+', wronge tuple dimension, length must be three!')
            if not isinstance(obs_tuple[0],str):
                raise Exception('Error in observation tuple; '+str(o)+', first tuple entry must be a string naming the observation variable!') 
            if not isinstance(obs_tuple[1],str) and obs_tuple[1] in self.distribution_dict.keys():
                raise Exception('Error in observation tuple; '+str(o)+', second tuple entry must be a string naming the distribution type!') 
            if not isinstance(obs_tuple[2],cs.Function):
                raise Exception('Error in observation tuple; '+str(o)+', third tuple entry must be a Casadi function!') 
            if not obs_tuple[2].n_in() == 2:
                raise Exception('Error in observation tuple;  '+str(o)+', Casadi function must take only two input vectors; parameters followed by inputs!')
            if not min(obs_tuple[2].size_in(0)) == 1:
                raise Exception('Error in observation tuple;  '+str(o)+', Casadi function must accept model inputs as a 1xn vector, however current dimensions are '+str(obs_tuple[2].size_in(0))+'!')
            if not len(input_names) == max(obs_tuple[2].size_in(0)):
                raise Exception('Error in observation tuple;  '+str(o)+', Casadi function accepts input '+str(max(obs_tuple[2].size_in(0)))+' but '+str(len(input_names))+' named inputs were passed!')
            if not min(obs_tuple[2].size_in(1)) == 1:
                raise Exception('Casadi function must accept model parameters as a 1xn vector, however current dimensions are '+str(obs_tuple[2].size_in(1))+'!')
            if not len(param_names) == max(obs_tuple[2].size_in(1)):
                raise Exception('Error in observation tuple;  '+str(o)+', Casadi function accepts input '+str(max(obs_tuple[2].size_in(1)))+' but '+str(len(param_names))+' named inputs were passed!')

        # extract and store dimensions of the model
        self.num_observ = len(observ_struct)
        self.num_input = len(input_names)
        self.num_param = len(param_names) 
        #create lists of input/param/param names,  can be used to look up names with indices (observ names filled in below)
        self.input_name_list = input_names
        self.param_name_list= param_names
        self.observ_name_list = [] 
        #create dicts for input/param names,  can be used to look up indices with names
        self.input_name_dict, self.param_name_dict, self.observ_name_dict = {}, {}, {}
        #populate the input name dict
        for i in range(self.num_input):
            self.input_name_dict[input_names[i]] = i
        #populate the param name dict
        for i in range(self.num_param):
            self.param_name_dict[param_names[i]] = i
        #create a list to store the observation variable distribution type and
        #   functions for computing observation sampling statistics, loglikelihood, and fisher info. matrix
        self.distribution, self.model, self.loglik, self.fisher_info_matrix = [],[],[],[]
        #create a list to store casadi functions for the predicted mean, mean parameteric sensitivity, prediction variance 
        self.prediction_mean , self.prediction_variance, self.prediction_sensitivity, self.observation_sampler = [],[],[], []
        #loop over the observation variables
        for i in range(self.num_observ):
            #fetch the observation name, distribution type and casadi function (maps inputs and params to observation statistics)
            observ_name = observ_struct[i][0]
            observ_distribution = observ_struct[i][1]
            observ_model = observ_struct[i][2]
            #extract names of observ_struct variables
            if not(observ_name in self.observ_name_dict):
                self.observ_name_dict[observ_name] = i
                self.observ_name_list.append(observ_name)
            else:
                raise Exception('Observation names must be unique!')
            self.distribution.append(observ_distribution)
            self.model.append(observ_model)
            [loglik, fisher_info, predict_mean, predict_sensitivity, predict_variance, observ_sampler] = \
                self._get_distribution_functions(observ_name, observ_distribution, observ_model)
            self.loglik.append(loglik)
            self.fisher_info_matrix.append(fisher_info)
            self.prediction_mean.append(predict_mean)
            self.prediction_sensitivity.append(predict_sensitivity)
            self.prediction_variance.append(predict_variance)
            self.observation_sampler.append(observ_sampler)
            
    def _get_distribution_functions(self, observ_name, observ_distribution, observ_model):
        """
        This function 

        Args:

        Return:
        """
        #create symbols for parameters and inputs,  and observation variable
        param_symbols = cs.SX.sym('param_symbols', self.num_param)
        input_symbols = cs.SX.sym('input_symbols', self.num_input)
        observ_symbol = cs.SX.sym(observ_name, 1)
        #generate the distribution statistics symbols, and prediction sensitivity symbol
        stats = observ_model(input_symbols, param_symbols)
        stats_sensitivity = cs.jacobian(stats, param_symbols)

        if observ_distribution  == 'Normal':
            if not stats.shape == (2,1):
                raise Exception('The Normal distribution accepts only 2 distributional parameters, but dimension '+str(stats)+' provided!')
            #create LogLikelihood symbolics and function 
            loglik_symbol = -0.5*cs.log(2*cs.pi*stats[1]) - (observ_symbol - stats[0])**2/(2*stats[1])
            #create FIM symbolics and function
            elemental_fim = cs.vertcat(cs.horzcat(1/stats[1], cs.SX(0)),cs.horzcat(cs.SX(0), 1/(2*stats[1]**2)))
            #create prediction functions for the mean, sensitivity and variance
            prediction_mean_symbol = stats[0]
            prediction_sensitivity_symbol = stats_sensitivity[0,:]
            prediction_variance_symbol = stats[1]
            #create random sampling function
            var_to_stdev = lambda s: [s[0].full().item(), np.sqrt(s[1]).full().item()]
            observ_sampler_func = lambda inpt,par: np.random.normal(*var_to_stdev(observ_model(inpt,par)))
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
            prediction_mean_symbol = stats[0]
            prediction_sensitivity_symbol = stats_sensitivity[0,:]
            prediction_variance_symbol = stats[0]
            #create random sampling function
            observ_sampler_func = lambda inpt,par: np.random.poisson(*observ_model(inpt,par).full().flatten())
        elif observ_distribution == 'Lognormal': 
            if not stats.shape == (2,1):
                raise Exception('The Lognormal distribution accepts only 2 distributional parameters, but dimension '+str(stats)+' provided!')
            #create LogLikelihood symbolics and function 
            #loglik_symbol = -cs.log(observ_symbol) - 0.5*cs.log(2*cs.pi*stats[1])\
            #     - (cs.log(observ_symbol) - stats[0])**2/(2*stats[1])
            loglik_symbol = - 0.5*cs.log(2*cs.pi*stats[1]) - (cs.log(observ_symbol) - stats[0])**2/(2*stats[1])
            #create FIM symbolics and function
            elemental_fim = cs.vertcat(cs.horzcat(1/stats[1], cs.SX(0)),cs.horzcat(cs.SX(0), 1/(2*stats[1]**2)))
            #create prediction functions for the mean, sensitivity and variance
            prediction_mean_symbol = cs.exp(stats[0] + stats[1]/2)
            prediction_sensitivity_symbol = cs.jacobian(prediction_mean_symbol,param_symbols)
            prediction_variance_symbol = (cs.exp(stats[1])-1) * cs.exp(2*stats[0] + stats[1])
            #create random sampling function
            var_to_stdev = lambda s: [s[0].full().item(), np.sqrt(s[1]).full().item()]
            observ_sampler_func = lambda inpt,par: np.random.lognormal(*var_to_stdev(observ_model(inpt,par)))
        elif observ_distribution == 'Binomial': 
            if not stats.shape == (2,1):
                raise Exception('The Binomial distribution accepts only 2 distributional parameters, but dimension '+str(stats)+' provided!')
            if not np.mod(cs.DM(stats[1]).full().item(),1)==0 :
                raise Exception('The second distributional parameter for Binomial must be a whole number positive integer!')
            #create LogLikelihood symbolics and function 
            loglik_symbol = observ_symbol*cs.log(stats[0]) + (stats[1]-observ_symbol)*cs.log(1-stats[0])
            #create FIM symbolics and function
            elemental_fim = cs.vertcat(cs.horzcat(stats[1]/(stats[0]*(1-stats[0])), cs.SX(0)),cs.horzcat(cs.SX(0), cs.SX(0)))
            #create prediction functions for the mean, sensitivity and variance
            prediction_mean_symbol = stats[0]*stats[1]
            prediction_sensitivity_symbol = cs.jacobian(prediction_mean_symbol,param_symbols)
            prediction_variance_symbol = stats[1]*stats[0]*(1-stats[0])
            #create random sampling function
            float_to_int = lambda s: [int(s[1].full().item()), s[0].full().item()]
            observ_sampler_func = lambda inpt,par: np.random.binomial(*float_to_int(observ_model(inpt,par)))
        elif observ_distribution == 'Bernoulli':
            if not stats.shape == (1,1):
                raise Exception('The Bernoulli distribution accepts only 1 distributional parameters, but dimension '+str(stats)+' provided!')
            #create LogLikelihood symbolics and function 
            loglik_symbol = observ_symbol*cs.log(stats[0]) + (1-observ_symbol)*cs.log(1-stats[0])
            #create FIM symbolics and function
            elemental_fim = 1/(stats[0]*(1-stats[0]))
            #create prediction functions for the mean, sensitivity and variance
            prediction_mean_symbol = stats[0]
            prediction_sensitivity_symbol = cs.jacobian(prediction_mean_symbol,param_symbols)
            prediction_variance_symbol = stats[0]*(1-stats[0])
            #create random sampling function
            observ_sampler_func = lambda inpt,par: np.random.binomial(1,observ_model(inpt,par).full().item())
        elif observ_distribution == 'Exponential': 
            if not stats.shape == (1,1):
                raise Exception('The Exponential distribution accepts only 1 distributional parameters, but dimension '+str(stats)+' provided!')
            #create LogLikelihood symbolics and function 
            loglik_symbol = cs.log(stats[0]) - observ_symbol*stats[0]
            #create FIM symbolics and function
            elemental_fim = 1/(cs.power(stats[0],2))
            #create prediction functions for the mean, sensitivity and variance
            prediction_mean_symbol = 1/stats[0]
            prediction_sensitivity_symbol = cs.jacobian(prediction_mean_symbol,param_symbols)
            prediction_variance_symbol = 1/(cs.power(stats[0],2))
            #create random sampling function
            observ_sampler_func = lambda inpt,par: np.random.exponential(1/observ_model(inpt,par).full().item())
        elif observ_distribution == 'Gamma': 
            print('Not Implemeneted')
        else:
            raise Exception('Unknown Distribution: '+observ_distribution)

        loglik_func = cs.Function('loglik_'+observ_name, [observ_symbol, input_symbols, param_symbols], [loglik_symbol])
        fisher_info_symbol = stats_sensitivity.T @ elemental_fim @ stats_sensitivity
        fisher_info_func = cs.Function('fim_'+observ_name, [input_symbols, param_symbols], [fisher_info_symbol])
        prediction_mean_func = cs.Function('pred_mean_'+observ_name, [input_symbols, param_symbols], [prediction_mean_symbol])
        prediction_sensitivity_func = cs.Function('pred_sens_'+observ_name, [input_symbols, param_symbols], [prediction_sensitivity_symbol])
        prediction_variance_func = cs.Function('pred_var'+observ_name, [input_symbols, param_symbols], [prediction_variance_symbol])

        return [loglik_func, fisher_info_func, prediction_mean_func, prediction_sensitivity_func, prediction_variance_func, observ_sampler_func]


    def fit(self, dataset_struct, start_param, options={}):
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
        #NOTE: needs checks on inputs, var names, designs must be exact reps
        #NOTE: NEED testing for multiple observation input structures,  multiple dimensions of parameters ideally,  1, 2, 3 and 7+
        #NOTE: add some print statments to provide user with progress status
        #NOTE: currently solve multiplex simultaneosly (one big opt problem) but sequentially may be more robust to separation (discrete data),  or randomly non-identifiable datasets (need to test)
        default_options = { 'Confidence':       ['None',    lambda x: isinstance(x,str) and (x=='None' or x=='Interval' or x=='Profile' or x=='Contours')],
                            'ConfidenceLevel':  [0.95,      lambda x: isinstance(x,float) and 0<=x and x<=1],
                            'RadialNumber':     [30,        lambda x: isinstance(x,int) and 1<x],
                            'SampleNumber':     [10,        lambda x: isinstance(x,int) and 1<x],
                            'Tolerance':        [0.001,     lambda x: isinstance(x,float) and 0<x],
                            'InitialStep':      [0.03,      lambda x: isinstance(x,float) and 0<x],
                            'MaxSteps':         [1000,        lambda x: isinstance(x,int) and 1<x]}
        for key in options.keys():
            if not key in default_options.keys():
                raise Exception('Invalid option key; '+key+'!')
            elif not default_options[key][1](options[key]):
                raise Exception('Invalid value; '+str(options[key])+', passed for option key; '+key+'!')
        for key in default_options.keys():
            if not key in options.keys() :
                options[key] = default_options[key][0]
        if not len(start_param) == self.num_param:
            raise Exception('Starting parameter mismatch, there were; '+str(len(start_param))+', provided but; '+str(self.num_param)+' needed!')
        #this block allows the user to pass a dataset, list of datasets,  list of lists etc. for Design x Replicat fitting
        if isinstance(dataset_struct, pd.DataFrame):
            #if a single dataset is passed vis the dataset_struct input,  wrap it in two lists so it matches general case
            design_datasets = [[dataset_struct]]
        elif isinstance(dataset_struct[0], pd.DataFrame):
            #else if dataset_struct input is a list of replciated datasets,  wrap it in a single list to match general case
            design_datasets = [dataset_struct]
        else:
            #else if dataset_struct input is a list of designs, each with a list of replicats,  just pass on th input
            design_datasets = dataset_struct
        #set confidence interval boolean
        interval_bool = options['Confidence']=="Intervals" or options['Confidence']=="Profiles" or options['Confidence']=="Contours"
        #set plotting boolean
        plot_bool = options['Confidence']=="Contours" or options['Confidence']=="Profiles"
        #get number of designs
        num_designs = len(design_datasets)
        #get number of replicats for each design
        num_replicat_list = [len(rep_list) for rep_list in design_datasets]
        #get total number of datasets
        num_datasets = sum(num_replicat_list)
        #create a list to store parameter casadi optimization symbols, starting parameters values, generic casadi loglikelihood functions
        param_symbols_list, start_params_list, loglik_func_list = [], [], []
        #create a total loglikelihood summation store, initialize to zero
        total_loglik_symbol = 0
        #create an archetypal vector of paramter symbols, used to build casadi loglikelihood functions for each design
        archetype_param_symbols = cs.SX.sym('archetype_param_symbols', self.num_param)
        #loop over different designs (outer most list)
        for d in range(num_designs):
            #get the set of replicats for this design
            replicat_datasets = design_datasets[d]
            #create a summation variable for the loglikelihood for a loglik_symbolset of the current design
            archetype_loglik_symbol = 0
            #for each design use the first replicat 
            archetype_dataset = replicat_datasets[0]
            # get onbservation count for this design
            num_observations = len(archetype_dataset.index)
            #create a vector of casadi symbols for the observations
            archetype_observ_symbol = cs.SX.sym('archetype_observ_symbol'+str(d), num_observations)
            #loop over the dataset inputs
            for index,row in archetype_dataset.iterrows():
                #get the curren input settings
                input_row = row[self.input_name_list].to_numpy()
                #get the observation variable index
                observ_var_index = self.observ_name_dict[row['Variable']]
                #create a symbol for the loglikelihood for the given input and observation variable
                archetype_loglik_symbol += self.loglik[observ_var_index](archetype_observ_symbol[index], input_row, archetype_param_symbols)
            #create a casadi function for the loglikelihood of the current design (observations are free/input symbols)
            archetype_loglik_func = cs.Function('archetype_loglik_func'+str(d),  [archetype_observ_symbol, archetype_param_symbols],  [archetype_loglik_symbol])
            #loop over replicats within each design
            for r in range(num_replicat_list[d]):
                #NOTE: could abstract below into a Casadi function to avoid input/observ loop on each dataset and replicat
                #get the dataset from the replicat list
                dataset = replicat_datasets[r]
                #create a vector of parameter symbols for this specific dataset,  each dataset gets its own,  these are used for ML optimization
                fit_param_symbols = cs.SX.sym('fit_param_symbols'+'_'+str(d)+str(r), self.num_param)
                #extract the vector of observations in the same format as in the archetype_loglik_func function input
                observ_vec = dataset['Observation'].to_numpy() 
                #create a symbol for the datasets loglikelihood function by pass in the observations for the free symbols in ObservSymbol
                dataset_loglik_symbol = archetype_loglik_func(observ_vec, fit_param_symbols)
                #add it to the list of replicate loglik functions
                loglik_func_list.append(cs.Function('dataset_loglik_func_'+str(d)+'_'+str(r),  [fit_param_symbols],  [dataset_loglik_symbol]))
                #set up the logliklihood symbols for given design and replicat
                param_symbols_list.append(fit_param_symbols)
                #record the starting parameters for the given replicat and dataset
                start_params_list.extend(start_param)
                #add the loglikelihood to the total 
                #NOTE: this relies on the distirbutivity of serperable optimization problem,  should confirm
                total_loglik_symbol += dataset_loglik_symbol
        #NOTE: this approach is much more fragile to separation (glms with discrete response),  randomly weakly identifiable datasets
        #NOTE: should be checking solution for convergence, should allow user to pass options to ipopt
        #NOTE: allow bfgs for very large nonlinear fits, may be faster
        # Create an IPOPT solver for overall maximum likelihood problem,  we pass negative total_loglik_symbol because IPOPT minimizes
        total_loglik_optim_struct = {'f': -total_loglik_symbol,  'x': cs.vertcat(*param_symbols_list)}#,  'g': cs.vertcat(*OptimConstraints)
        param_fitting_solver = cs.nlpsol('solver',  'ipopt',  total_loglik_optim_struct, {'ipopt.print_level':5, 'print_time':False})
        # Solve the NLP fitting problem with IPOPT call
        param_fit_solution_struct = param_fitting_solver(x0=start_params_list)
        #extract the fit parameters from the solution structure
        fit_param_matrix = param_fit_solution_struct['x'].full().flatten().reshape((-1,self.num_param))
        #create row index for parameter dataframe export, specifies design index for each datset
        design_index = [design_indx+1 for design_indx in range(num_designs) for i in range(num_replicat_list[design_indx])]
        #check if a  confidence value was passed as an option, requires confidence intervals at least
        if interval_bool:
            #create a multiindex for the columns of parameter dataframe export
            # first level define estimate and lower/upper bound, second is param values
            column_index = pd.MultiIndex.from_product([['Estimate','Lower','Upper'],self.param_name_list],names=['Value', 'Parameter'])
            #create an empty list to store lower/upper bounds for intervals
            bound_list = []
            #loop over each design
            for d in range(num_datasets):
                #get the parameter estimates for current dataset
                param_vec = fit_param_matrix[d,:]
                #check if graphing options; contour plots or profile trace plots,  are requested
                if options['Confidence']=="Contours" or options['Confidence']=="Profiles":
                    #if so, create a figure to plot on
                    fig = plt.figure()
                    #run profileplots to plot the profile traces and return CI's
                    interval_list = self.__profileplot(param_vec, loglik_func_list[d], fig, options)[0]
                    #if contour plots are requested specifically, run contour function to plot the projected confidence contours
                    if options['Confidence']=="Contours":
                        self.__contourplot(param_vec, loglik_func_list[d], fig, options)
                elif options['Confidence']=="Intervals":
                    #if confidence intervals are requested, run confidenceintervals to get CI's and add them to replicat list
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
            param_output_matrix = np.concatenate([fit_param_matrix, bound_param_matrix], axis=1)
        else:
            #if no intervals are requested, set column indext to single level, with just parameter names
            column_index = self.param_name_list
            param_output_matrix = fit_param_matrix
        #create ouput dataframe with all request parameter information
        param_data = pd.DataFrame(param_output_matrix, index=design_index, columns=column_index)
        #return the dataframe
        return param_data

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
        if not len(param) == self.num_param:
            raise Exception('Starting parameter mismatch, there were; '+str(len(param))+', provided but; '+str(self.num_param)+' needed!')
        #check if designs is a single design or a list of designs
        if isinstance(design_struct, pd.DataFrame):
            #if single,  wrap it in a list to match general case
            design_list = [design_struct]
        else:
            #else pass it as it is
            design_list = design_struct
        #create a list to store lists of datasets for each design
        design_datasets = []
        #loop over the designs
        for design in design_list:
            #ensure the design is sorted by inputs and has standard column ordering (inputs, observs)
            unique_design = design.groupby(self.input_name_list,as_index=False).sum()[self.input_name_list+self.observ_name_list]
            #expand the unique design so theire is a row for each; unique input combination X observ variable
            expanded_design = unique_design.melt(id_vars=self.input_name_list,var_name='Variable',value_name='Replicats')
            expanded_design.sort_values(self.input_name_list,inplace=True)
            expanded_design.reset_index(drop=True,inplace=True)
            #expand design further so that design has a row for each; unique input combination X observ var  X replicate
            itemized_design = expanded_design.reindex(expanded_design.index.repeat(expanded_design['Replicats']))
            itemized_design.drop('Replicats',axis=1,inplace=True)
            itemized_design.reset_index(drop=True,inplace=True)
            #create a list for replciated datasets of the current design
            replicat_datasets = []
            #loop over the number of replicates
            for r in range(replicats):
                dataset=itemized_design.copy()
                observation_list = []
                for index,row in itemized_design.iterrows():
                    input_row = row[self.input_name_list].to_numpy()
                    observ_name = row['Variable']
                    observation_list.append(self.observation_sampler[self.observ_name_dict[observ_name]](input_row, param))
                dataset['Observation'] = observation_list
                replicat_datasets.append(dataset)
            design_datasets.append(replicat_datasets)
                
        #check if a single design was passed
        if isinstance(design_struct,  pd.DataFrame):
            if replicats==1:
                #if a single design was passed and there replicate count is 1,  return a single dataset
                return design_datasets[0][0]
            else:
                #else if a single design was passed,  but with >1 reps,  return a list of datasets
                return design_datasets[0]
        else:
            #else if multiple designs were passed (with/without reps),  return a list of list of datasets
            return design_datasets

    #NOTE: replace with constructor defined function !!!!!!!!!!!!
    # def _sample_distribution(self, input_row, observ_name, param):
    #     """
    #     This function samples the appropriate distribution for the observation varibale name passed, at 
    #     the input value passed, with the specified parameter values

    #     Args:
    #         input_row: a vector specifying the input settings
    #         observ_name: the name of the observation variable to be sampled
    #         param: the parameter settings at which to take the sample

    #     Return:
    #         observation: the (random) sample of the observation variable
    #     """
    #     #NOTE: should maybe allow for seed to be passed for debugging
    #     observ_index = self.observ_name_dict[observ_name]
    #     distribution_name = self.distribution[observ_index]

    #     if distribution_name == 'Normal':
    #         #compute the sample distirbution statistics using the model
    #         [mean,variance] = self.model[observ_index](param, input_row)
    #         observation = np.random.normal(mean,  np.sqrt(variance)).item()
    #     elif distribution_name == 'Poisson':
    #         lambda_ = self.model[observ_index](param, input_row)
    #         observation = np.random.poisson(lambda_).item()
    #     elif distribution_name == 'Lognormal':
    #         print('Not Implemeneted')
    #     elif distribution_name == 'Binomial':
    #         print('Not Implemeneted')
    #     elif distribution_name == 'Exponential':
    #         print('Not Implemeneted')
    #     elif distribution_name == 'Gamma':
    #         print('Not Implemeneted')
    #     else:
    #         raise Exception('Unknown error encountered selecting observation distribution,  contact developers')

    #     return observation
        
    def predict(self, input_struct, param, covariance_matrix=None,  options={}):
        #NOTE: evaluate model to predict mean and prediction interval for y
        #NOTE: optional pass cov matrix,  for use with delta method/MC error bars on predictions
        #NOTE: should multiplex over inputs
        # if isinstance(input_list, pd.DataFrame):
        #     input_list = [[input_list]]
        # elif isinstance(input_list[0], pd.DataFrame):
        #     input_list = [input_list]

        default_options = { 'Interval':         ['None',    lambda x: isinstance(x,str) and x=='None' or x=='Mean' or x=='Prediction'],
                            'Method':           ['Delta',   lambda x: isinstance(x,str) and x=='Delta' or x=='MonteCarlo'],
                            'SampleNumber':     [10000,     lambda x: isinstance(x,int) and 1<x],
                            'ConfidenceLevel':  [0.95,      lambda x: isinstance(x,float) and 0<=x and x<=1],
                            'Sensitivity':      [False,     lambda x: isinstance(x,bool)]}
        for key in options.keys():
            if not key in default_options.keys():
                raise Exception('Invalid option key; '+key+'!')
            elif not default_options[key][1](options[key]):
                raise Exception('Invalid value; '+str(options[key])+', passed for option key; '+key+'!')
        for key in default_options.keys():
            if not key in options.keys() :
                options[key] = default_options[key][0]
    #NEW
        # for index,row in input_struct.iterrows():

        #     input_vec = row[self.input_name_list]
        #     observ_name = row['Variable'] 

        #     if options['Method']=='Delta':
                


        #     elif options['Method']=='MonteCarlo':
    
    #NOTE: replace with constructor defined function !!!!!!!!!!!!
    def _predict_mean(self, input_row, observ_name, param, options):
        """
        This function returns the predicted mean of the observation variable 

        Args:
            input_row: a vector specifying the input settings
            observ_name: the name of the observation variable to be sampled
            param: the parameter settings at which to take the sample
            options:

        Return:
            response_info: the (random) sample of the observation variable
        """
        #NOTE: should maybe allow for seed to be passed for debugging
        print('Not Implemeneted')
        # observ_index = self.observ_name_dict[observ_name]
        # distribution_name = self.distribution[observ_index]
        # standard_dev_multiplier = -st.norm.ppf((1-options['Confidence'])/2)

        # if distribution_name == 'Normal' or distribution_name == 'Poisson' or distribution_name == 'Bernoulli':
        #     #compute the sample distirbution statistics using the model
        #     [mean,variance] = self.model[observ_index](param, input_row)
        #     if options['Interval']  == 'Mean' or options['Interval']  == 'Prediction':
        #         mean_sensitivity = self.sensitivity[observ_index](param, input_row)
        #         mean_variance = mean_sensitivity.T * covariance_matrix * mean_sensitivity
        #         mean_standard_dev = standard_dev_multiplier * np.sqrt()
        #     # if options['Interval']  == 'Prediction':
        #     #     prediction_variance = 
        # elif distribution_name == 'Poisson':
        #     mean = self.model[observ_index](param, input_row)
        #     observation = np.random.poisson(lambda_).item()
        # elif distribution_name == 'Lognormal':
        #     print('Not Implemeneted')
        # elif distribution_name == 'Binomial':
        #     print('Not Implemeneted')
        # elif distribution_name == 'Exponential':
        #     print('Not Implemeneted')
        # elif distribution_name == 'Gamma':
        #     print('Not Implemeneted')
        # else:
        #     raise Exception('Unknown error encountered selecting observation distribution,  contact developers')

        # return observation
    

    #NOTE: replace with constructor defined function !!!!!!!!!!!!
    def _predict_interval(self, input_row, observ_name, covariance_matrix, param, options):
        """
        This function returns the predicted mean of the observation variable 

        Args:
            input_row: a vector specifying the input settings
            observ_name: the name of the observation variable to be sampled
            param: the parameter settings at which to take the sample
            options:

        Return:
            response_info: the (random) sample of the observation variable
        """
        #NOTE: should maybe allow for seed to be passed for debugging
        print('Not Implemeneted')
        # observ_index = self.observ_name_dict[observ_name]
        # distribution_name = self.distribution[observ_index]
        # standard_dev_multiplier = -st.norm.ppf((1-options['Confidence'])/2)

        # if distribution_name == 'Normal':
        #     #compute the sample distirbution statistics using the model
        #     # [mean,variance] = self.model[observ_index](param, input_row)
        #     # if options['Interval']  == 'Mean' or options['Interval']  == 'Prediction':
        #     #     mean_sensitivity = self.sensitivity[observ_index](param, input_row)
        #     #     mean_variance = mean_sensitivity.T * covariance_matrix * mean_sensitivity
        #     #     mean_standard_dev = standard_dev_multiplier * np.sqrt()
        #     # if options['Interval']  == 'Prediction':
        #     #     prediction_variance = 
        #     mean = self.model[observ_index](param, input_row)
        # elif distribution_name == 'Poisson':
        #     mean = self.model[observ_index](param, input_row)
        #     observation = np.random.poisson(lambda_).item()
        # elif distribution_name == 'Lognormal':
        #     print('Not Implemeneted')
        # elif distribution_name == 'Binomial':
        #     print('Not Implemeneted')
        # elif distribution_name == 'Exponential':
        #     print('Not Implemeneted')
        # elif distribution_name == 'Gamma':
        #     print('Not Implemeneted')
        # else:
        #     raise Exception('Unknown error encountered selecting observation distribution,  contact developers')

        # return observation
    #OLD
        # prediction_dict = {}
        # prediction_dict['Inputs'] = input_list
        # for o in observ_list:
        #     statistics = [ [stat.full()[0][0]
        #                     for stat in self.model[o](param, inputs)]
        #                     for inputs in input_list]
        #     if error_method == "MonteCarlo":
        #         param_sample = np.random.multivariate_normal(param,  covariance_array,  num_mc_samples).tolist()
        #         mc_sample =  [[[stat.full()[0][0]
        #                         for stat in self.model[o](par, inputs)]
        #                         for par in param_sample]
        #                         for inputs in input_list]
        #     observ_dict = {}
        #     for s in range(len(self.sensitivity)):
        #         stat_dict={}

        #         stat_list = [stat[s] for stat in statistics]
        #         stat_dict['Value'] = stat_list

        #         if sensitivity or error_method == "Delta":
        #             sensitivity_list = [list(self.sensitivity[o][s](param, inputs).full()[0]) for inputs in input_list]
        #             stat_dict['Sensitivity'] = sensitivity_list
        #         if param_covariance:
        #             if error_method == "Delta":
        #                 standard_dev_multiplier = -st.norm.ppf((1-alpha)/2)
        #                 delta_list = [standard_dev_multiplier*(np.sqrt(np.array(sensitivity_vec) @ covariance_array @ np.array(sensitivity_vec).T))
        #                                 for sensitivity_vec in sensitivity_list]
        #                 #NOTE: need to scale this to alpha interval, right now just a single standard dev
        #                 bounds_list = [[stat_list[i]-delta_list[i],stat_list[i]+delta_list[i]]
        #                                 for i in range(len(stat_list))]
        #             elif error_method == "MonteCarlo":
        #                 bounds_list = [np.percentile(
        #                     [mc_sample[i][j][s] for j in range(num_mc_samples)],[100*(1-alpha)/2, 100*(0.5+alpha/2)]
        #                     ) for i in range(len(stat_list))]
        #             else:
        #                 raise Exception('No such option; '+str(error_method)+' exists for field \'ErrorMethod\'!')
        #             stat_dict['Bounds'] = bounds_list
        #         observ_dict[self.stat_name_dict[self.distribution[o]][s]] = stat_dict
        #     prediction_dict[self.observ_name_list[0]]= observ_dict

        # return prediction_dict

    

    #NOTE: should maybe rename this
    def eval_design(self):
        #maybe this should move to the design class(??)
        #For D (full cov/bias),  Ds (partial cov/bias),  T separation using the delta method?! but need two models
        # assess model/design,  returns various estimates of cov,  bias,  confidence regions/intervals
        # no data: asymptotic: covaraince,  beale bias,  maybe MSE
        #          sigma point: covariance,  bias (using mean) (need to figure out how to do sigma for non-normal data),  maybe MSE
        #          monte carlo: covariance,  bias,  MSE
        
        print('Not Implemeneted')

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
        #extract the confidence level and compute the chisquared threshold
        chi_squared_level = st.chi2.ppf( options['ConfidenceLevel'],  self.num_param)
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
            radius_list = list(np.linspace(lower_radius,  upper_radius,  num=options['SampleNumber']+1, endpoint=False)[1:])
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
        #compute the chi-squared level from confidence level
        chi_squared_level = st.chi2.ppf(options['ConfidenceLevel'],  self.num_param)
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
        #check if the search is run in negative or positive direction,  set intial step accordingly
        if forward:
            radius = options['InitialStep']
        else:
            radius = -options['InitialStep']
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
        while  abs(ratio_gap)>options['Tolerance'] and iteration_counter<options['MaxSteps']:
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
        if iteration_counter>=options['MaxSteps']:
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
        f = max(k,1)
        while (k-cnt)>0:
            f = f*(k-cnt)
            cnt = cnt+1
        return [f]


# Full ML fitting,  perhaps with penalized likelihood???
# fit assesment,  with standardized/weighted residual output,  confidence regions via asymptotics (with beale bias),  likelihood basins,  profile liklihood,  sigma point (choose one or maybe two)
# function to generate a prior covariance (that can be fed into design)
# function for easy simulation studies (generate data,  with given experiment)