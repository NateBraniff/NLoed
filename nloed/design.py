import os as os
import sys as sys
import random as rn
import copy as cp
#need to clean up this import style (below)
from contextlib import contextmanager

import casadi as cs
import numpy as np
import pandas as pd

class Design:
    """ The NLoed Design class formulates a design optimization problem and generates an optimized
    relaxed experimental design.

    This class accepts a set of NLoed model instances as well as the users specifications of their
    experimental requirments. This information is used to formulate the 
    experimental design problem as a non-linear programming problem that is passable to the IPOPT
    solver via Casadi's interface.
    
    During construction, optimization problem is structured and passed to IPOPT. The problem solution
    is then extracted and stored within the model object. This solution represents a relaxed solution
    to the design problem and has real valued weights representing the number of samples taken in
    various input conditions. The classes user-callable function can be used to round the relaxed
    solution to an exact design with a specified sample size.

    Attributes:
        num_models (integer): The number of models for which the design was optimized, generally 1.
        input_dim (integer): The dimensions of the model input, this must be consistent across all
            of the models passed.
        observ_dim (integer): The number of observation variables that each model has, all models
            must have the same observation variables. 
        input_name_list (list of strings): A list of names for the model inputs, these must be shared
            by all of the passed models.
        observ_name_list (list of strings): Alist of names for the model observation variables, these
            must match across all models passed.
        model_list (list of models): A list of the NLoed Model instances passed to the design
            constructor.
        param_list (list of array-like): A list of the nominal parameter values at which the design
            is to be optimized.
        objective_list (list of strings): A list of the objectives which are to be optimized (as
            a weighted combination).
        relaxed_design (dataframe): A dataframe containing the optimized relaxed design returned
            by IPOPT.
        
    """

    def __init__(self, models, parameters, objective, discrete_inputs=None, continuous_inputs=None,
                 observ_groups=None, fixed_design=None, options={}):
        """ The class constructor for NLoed's Design class. 

        This function accepts an NLoed Model instance as well as experimental specifications such as the
        nominal parameter values, the objective, input constraints and format, observation groupings
        and past design information passed by the user. This information specifies the objectives 
        constraints the user has for their desired experiment. During construction a Casadi
        symbolic structure is created for the overall design objective. This symbol is used to call
        IPOPT via Casadi's optimization interface. After IPOPT optimization is complete, the constructor
        function parses the returned solution into a dataframe which is stored as a class attribute.

        Args:
            models (list of models): A list of NLoed model isntances for which an optimize design is
                desired.
            parameters (list of array-like): A list of the nominal parameter values at which the
                design should be optimized. 
            objective (list of strings): A list of string names for the objectives to be used for
                design optimization. Multiple objectives are combined as a weighted combination.
            discrete_inputs (dictionary, conditionally optional): A dictionary indicating which 
                model inputs, if any, should be handled discretly. Discret input values are
                fixed within the optimizatio call, according to the discretization chosen. Inputs
                are selected based on adjusting the sampling weights over the discrete grid of values.
                The exact method used for discretization can also be specified in this structure.

                NOTE-- need examples
            continuous_inputs (dictionary, conditionally optional): A dictionary indicating which
                model inputs, if any, should be handled continuously. Continuous inputs are treated
                as optimization variables within the optimization call, and can be adjusted as real
                values within their boundaries. The user needs to specify the number of unique possible
                continuous levels that each input can take within the overall design. The continuous
                bounds and number of unique levels for each continuous input can also be specified in
                this structure.

                NOTE-- need examples
            observ_groups (list of list of strings): A list of lists, the outer list consists of
                an entry for each grouping of observation variables that must be sampled together.
                Each inner list contains the string names of the observation variables that are sampled
                together in each group.
            fixed_design (dictionary):A dictionary specifiying the structure of any fixed aspects
                of the experimental design, or past experimental designs on which the new design
                should be conditioned. This dictionary specifies both the fixed design structure and
                its expected sampleing weight relative to the new design.

                NOTE-- need examples
            options (dict, optional): A dictionary of user-defined options, possible key-value pairs
                include:

                "PrintLevel" --
                Purpose: Sets the amount of printed feedback returned during instantiation
                Type: string,
                Default Value: 'Basic',
                Possible Values: 'None', 'Basic', 'Verbose'

                "IPOPTHessian" --
                Purpose: Determines whether IPOPT uses an exact hessian computed with symbolic 
                derivatives or an approximate BFGS
                Type: string,
                Default Value: 'Exact,
                Possible Values: 'Exact', 'BFGS'

                "LockWeights" --
                Purpose: A boolean, should be used with continuous inputs, if true only input levels
                are optimized input weights kept constant at their inital values. By default the inital
                values are all equal resulting in equal sampling weight for each optimized input condition.
                Type: bool,
                Default Value: False,
                Possible Values: True or False

        """

        default_options = \
                 {  'PrintLevel':          ['Basic',  lambda x: isinstance(x,str) and (x=='None' or x=='Basic' or x=='Verbose')],
                    'IPOPTHessian':        ['Exact',  lambda x: isinstance(x,str) and (x=='Exact' or x=='BFGS')],
                    'LockWeights':         [False,    lambda x: isinstance(x,bool)]}
        options=cp.deepcopy(options)
        for key in options.keys():
            if not key in options.keys():
                raise Exception('Invalid option key; '+key+'!')
            elif not default_options[key][1](options[key]):
                raise Exception('Invalid value; '+str(options[key])+', passed for option key; '+key+'!')
        for key in default_options.keys():
            if not key in options.keys() :
                options[key] = default_options[key][0]

        model_input = models
        if not isinstance(models,list):
            model_input = [models]
        param_input = parameters
        if not isinstance(parameters[0],list):
            param_input = [parameters]    

        #model data init
        self.num_models = len(model_input)
        self.input_dim = model_input[0].num_input
        self.observ_dim = model_input[0].num_observ
        self.input_name_list = model_input[0].input_name_list
        self.observ_name_list = model_input[0].observ_name_list
        
        self.model_list = []
        self.param_list = []
        self.objective_list = []

        #loop over models check dimensions and dicts, build objective functions, create fim and beta lists 
        for m in range(self.num_models):
            model = model_input[m]
            params = param_input[m]

            #need to handle model weights in model, currently will treat as equal
            #if params contains a covariance matrix, generate param list and weights
            #ParamList used for bayesian priors
            #need to handle weights in params at some point, currently will treat as equal

            #NOTE: model D score must be weighted/rooted log-divided according to number of params, need to check this
            if objective =='D':
                matrix = cs.SX.sym('Matrx',model.num_param, model.num_param)
                r_factorial = cs.qr(matrix)[1]
                normed_log_det = cs.trace(cs.log(r_factorial))/model.num_param
                objective_func = cs.Function('ObjFunc'+str(m),[matrix],[-normed_log_det]) 
            #elif objective_type == 'A':
            #    print('Not implemented!')
            #elif objective_type == 'Ds':
            #    print('Not implemented!')
            #elif objective_type == 'Custom':
            #    print('Not implemented!')
            else:
                raise Exception('Unknown objective: '+str(objective)+'!')

            #create the fim list for summing fim symbolics in group loop and parameter symbols for each model 
            self.model_list.append(model)
            self.param_list.append(params)
            self.objective_list.append(objective_func)
            
        #observ struct
        if options['LockWeights']:
            observ_groups = [[obs for obs in self.observ_name_list]]
        elif not(observ_groups):
            observ_groups = [[obs] for obs in self.observ_name_list]
        self.observ_group_list = observ_groups
        self.num_observ_group = len(observ_groups)
        
        #if user has passed discrete inputs
        if  discrete_inputs:
            #discrete_input_flag = True
            [discrete_input_grid,
             discrete_grid_length,
             discrete_input_names,
             discrete_input_num] = self.__discrete_settup(discrete_inputs, options)
        else:
            #discrete_input_flag = False
            discrete_input_grid=[[]]
            discrete_grid_length = 1
            discrete_input_names = []
            discrete_input_num = 0

        #NOTE: CONVERT APPOX GRID TO DICT of LISTS, PASS NAME, GET LIST OF GRID ELEMENTS FOR GIVEN NAME

        #if the user has passed continuous inputs
        if continuous_inputs:
            #continuous_input_flag = True
            [continuous_symbol_archetypes,
             continuous_symbol_list,
             continuous_lowerbounds,
             continuous_upperbounds,
             continuous_archetype_num,
             continuous_symbol_num,
             continuous_input_num,
             continuous_input_names,
             continuous_init] = self.__continuous_settup(continuous_inputs, options)
        else:
            #continuous_input_flag = False
            continuous_symbol_archetypes = [[]]
            continuous_symbol_list =[]
            continuous_lowerbounds = []
            continuous_upperbounds = []
            continuous_archetype_num = 1
            continuous_symbol_num = 0
            continuous_input_num = 0
            continuous_input_names = []
            continuous_init = []

        if not(self.input_dim == continuous_input_num + discrete_input_num):
            raise Exception('All input variables must be passed as either discrete or continuous, and the total number of inputs passed must be the same as recieved by the model(s)!\n'
                        'There are '+str(discrete_input_num)+' discrete inputs and '+str(continuous_input_num)+' continuous inputs passed but model(s) expect '+str(self.input_dim)+'!')

        [fim_list,
         weight_symbol_list,
         weight_sum,
         weight_num,
         weight_init,
         weight_design_map] = self.__weighting_settup(discrete_input_names,
                                                     discrete_input_grid,
                                                     continuous_input_names,
                                                     continuous_symbol_archetypes,
                                                     fixed_design,
                                                     options)

        [ipopt_problem_struct,
         optim_init,
         var_lowerbounds,
         var_upperbounds,
         constraints,
         constraint_lowerbounds,
         constraint_upperbounds] = self.__optim_settup(fim_list,
                                                      continuous_symbol_list,
                                                      continuous_lowerbounds,
                                                      continuous_upperbounds,
                                                      continuous_init,
                                                      weight_symbol_list,
                                                      weight_sum,
                                                      weight_init,
                                                      options)
        
        #NOTE: should test init, and error out if init is nonvalid

        
        if options['IPOPTHessian'] == 'Exact':
            hessian_type = 'exact'
        else:
            hessian_type = 'limited-memory'
        if not options['PrintLevel'] == 'None':
            print('Setting up optimization problem...',end='')
        #Create an IPOPT solver
        ipopt_solver = cs.nlpsol('solver', 'ipopt', ipopt_problem_struct, 
                                    {'ipopt.hessian_approximation':hessian_type,
                                    'ipopt.max_iter':10000})
        if not options['PrintLevel'] == 'None':
            print('problem set-up complete.')
            print('Begining optimization:')
        with silence_stdout(not(options['PrintLevel'] == 'Verbose')):
            # Solve the NLP with IPOPT call
            ipopt_solution_struct = ipopt_solver(x0=optim_init,
                                                lbx=var_lowerbounds,
                                                ubx=var_upperbounds,
                                                lbg=constraint_lowerbounds,
                                                ubg=constraint_upperbounds)
        if not options['PrintLevel'] == 'None':
            if ipopt_solver.stats()['success']:
                print('Optimization succeded!')
            else:
                print('Optimization failed!')
            print('Return status: ',ipopt_solver.stats()['return_status'])
            print('Iteration count: ',ipopt_solver.stats()['iter_count'])

        optim_solution = ipopt_solution_struct['x'].full().flatten()

        optim_continuous_values = optim_solution[:continuous_symbol_num]

        if not options["LockWeights"]:
            optim_weights = optim_solution[continuous_symbol_num:]
        else:
            optim_weights = weight_init

        relaxed_design = pd.DataFrame(columns=self.input_name_list+['Variable', 'Weights'])

        for i in range(len(optim_weights)):
            weight = optim_weights[i]
            if weight>1e-4:
                design_row_prototype = {}
                input_map = weight_design_map[i]['InputMap']
                for x in self.input_name_list:
                    if input_map[x]['Type'] == 'A':
                        design_row_prototype[x] = discrete_input_grid[input_map[x]['Index']][x]
                    elif input_map[x]['Type'] == 'E':
                        design_row_prototype[x] = optim_continuous_values[input_map[x]['Index']]
                design_row_prototype['Weights'] = weight

                observ_map = weight_design_map[i]['ObsMap']
                for obs in observ_map:
                    design_row = cp.deepcopy(design_row_prototype)
                    design_row['Variable'] = obs
                    design_row['Weights'] = design_row['Weights']/len(observ_map)
                    relaxed_design = relaxed_design.append(design_row, ignore_index=True)

        self.relaxed_design = relaxed_design

# --------------- Public (user-callable) functions -------------------------------------------------

    def round(self, sample_size, options={}):
        """ A function to round the relaxed optimal design produced by NLoed into an exact design with
        a discrete sample size.

        This function uses a rounding method to discretize the replicate allocation sampling weights.
        The sampling weights are generated during design optimization in the Design class instantiation.
        The resulting relaxed design is stored in the Design class instance as an attribute but the
        design is not implementable because it uses real valued weights to measure replication fractions
        and has no specific sample size. Rounding methods, the defualt being Adam's apportionment, are
        used to discretize these weights to create a design with a desired sample size by this function.
        This function then returns a data frame containg the implementable exact design generated through
        rounding. Rounding does not always yield a unique design, in these cases this function may
        make an arbitrary selection amongst competeing potential roundings of the same relaxed design. 

        Args:
            sample_size (integer): An integer indicating the desired sample size the user wants.
            options (dictionary, optional): A dictionary of user-defined options, possible key-value
                pairs include:

        Return:
            dataframe: A dataframe is returned containing the exact design with the desired sample size.

        """
        default_options = { }
        options=cp.deepcopy(options)
        for key in options.keys():
            if not key in options.keys():
                raise Exception('Invalid option key; '+key+'!')
            elif not default_options[key][1](options[key]):
                raise Exception('Invalid value; '+str(options[key])+', passed for option key; '+key+'!')
        for key in default_options.keys():
            if not key in options.keys() :
                options[key] = default_options[key][0]

        l = len(self.relaxed_design.index)
        nu = sample_size - l/2

        candidate_design = cp.deepcopy(self.relaxed_design)
        apportion = np.ceil(nu*self.relaxed_design['Weights']).to_numpy().astype(int)
        candidate_design.drop('Weights',axis=1,inplace=True)
        candidate_design['Replicates'] = apportion

        while not sum(candidate_design['Replicates'])==sample_size:
            if sum(candidate_design['Replicates'])<sample_size:
                thresh = np.divide(candidate_design['Replicates'].to_numpy(),self.relaxed_design['Weights'].to_numpy())
                thresh_set = np.where(thresh == thresh.min())[0]
                inc = 1
            elif sum(candidate_design['Replicates'])>sample_size:
                thresh = np.divide(candidate_design['Replicates'].to_numpy()-1,self.relaxed_design['Weights'].to_numpy())
                thresh_set = np.where(thresh == thresh.max())[0]
                inc = -1

            candidate_design.iloc[rn.choice(thresh_set), candidate_design.columns.get_loc('Replicates')] +=inc

            # candidate_list = []
            # for ind in thresh_set:
            #     new_candidate = cp.deepcopy(candidate_design)
            #     new_candidate.iloc[0, new_candidate.columns.get_loc('Replicates')] +=1
            #     for models
            #         self.model_list[0].evaluate(candidate_design,self.param_list[0],{'Covariance':False})['FIM'].to_numpy()
            #     candidate_list.append(new_candidate)
        return candidate_design

    def relaxed(self, options={}):
        """ A function to return the relaxed design dataframe, containing the real-valued sampling
        weights.

        This function returns the archetypal relaxed design as a dataframe.
        The relaxed design is generally not useful for implementing expeirments but may be of value
        for checking optimality conditions, for xample with the general equivlance theorem.

        Args:
            options (dictionary, optional): A dictionary of user-defined options, possible key-value
                pairs include:

        Return:
            dataframe: The returned dataframe contains the relaxed design.
        """
        
        return self.relaxed_design

    def power(self, sample_size, options={}):
        """ A function provides comparisons of performance of differing sample size selection for 
        the rounding of a given relaxed design.

        This function has not been implemented yet.

        Args:
            sample_size (integer):
            options (dictionary):

        Return:

        """
        print('not implemented')
        # function for power analysis with given design and model

        #pre N: asymptotic confidence intervals only 
        #enter an N range Nmin to Nmax (or just single N)
        #discretize with multiple N -> sigma/boostrap CI's/Cov's at each N (jitter at each N?!) [graphical, thershold]
        #                           -> FDS/D-eff(Opt-eff) plots [graphical, thershold]
        #                           -> what about T optimality designs, what about bayesian CI's??

        # Discretize (depends on N):    basic Adam's apportionment, 
        #                               other rounding options (naive...)
        # Sample Size Selection (N?):   assesment of confidence interval size with design replication or N selection
        #                               similar FDS/D-eff/relative D-eff plots for selecting sample size
        # Fine Tuning:                  jitter/tuning of continuous design

        #use sigma points (or if needed monte carlo simulation) to determine 'power curve', shrinking of parameter error (bias +var) with n design replications
        #perhaps combine with rounding methods?? to allow for rounded variations of the same design.
        # rounding should probably happen here

        #include a prediction variance plot for verifying genral equivlance theorem plot (either as a var vs X or a stem plot for support points in high dim)
        #FDS-like plots, prediction variance limit vs fraction of design space with lower variance than limit, used to compare designs and sample sizes, along with CI's (ellipses or intervals)
        #d-efficiency plots for comparing sample size (normed to discrete max for each sample size) vs relative efficiency (normed to lowest sample size discrete or rounded continuous)
        #    shows 'real'improvment from extra samples rather than distance from unachievable ideal, howevver regular d-efficiency may motivate adding a single point to achieve near theoretical max

# --------------- Private functions (for constructor) ----------------------------------------------

    def __discrete_settup(self, discrete_inputs, options):
        """ A private helper function that parses the discrete inputs dictionary.

        This private function is used during the NLoed Design class construction.

        This function accepts the user-provided discrete_inputs dictionary (if one is passed) and
        constructs the grid of candidate inputs for the discrete input subpace of the model inputs.
        Regardless of how the user provides the discrete grid information (i.e. 'Grid', 'Bounds', or
        'Candidates") this function uses that information to construct a list of candidate points
        for the discrete inputs which are used to set up the design optimization problem in later
        stages of the Design constructor.

        Args:
            discrete_inputs (dictionary): The user provided discrete_inputs dictionary passed to the
                Design constructor.
            options (dictionary, optional): A dictionary of user-defined options, possible key-value
                pairs include:

        Returns:
            list: The returned object is a list of data structures in the follow order;
                discrete_input_grid -- a list of dictionaries, each dictionary contains a candidate
                grid point where the dictionary keys are the discrete input string names and
                the dictionary values are the candidates input values for that point,
                discrete_grid_length -- The number of candidate grid point (dictionaries) in the 
                discrete_input_grid list.
                discrete_input_names -- A list of names of the discrete inputs.
                discrete_input_num -- The number of model inputs being handled discretetly.
        """
        #get names, number and indices of discrete inputs
        discrete_input_names =  discrete_inputs['Inputs']
        discrete_input_num = len(discrete_input_names)
        
        #check if inquality OptimConstraintsains have been passed, if so store them 
        discrete_input_constr = []
        if 'Constraints' in  discrete_inputs:
            discrete_input_constr  =  discrete_inputs['Constraints']   

        #create a list for storing possible levels of each discretemate input
        discrete_input_candidates = []
        if not 'Grid' in discrete_inputs:
            if 'Bounds' in   discrete_inputs:
                #set resolution of grid NOTE: this should be able to be specified by the user, will change
                if 'NumPoints' in discrete_inputs:
                    num_points = discrete_inputs['NumPoints']
                else:
                    num_points = 5
                #loop over bounds passed, and use resolution and bounds to populate xlist
                for bound in discrete_inputs['Bounds']:
                    discrete_input_candidates.append(np.linspace(bound[0],bound[1],num_points).tolist())
            elif  'Candidates' in discrete_inputs:
                for grid in discrete_inputs['Candidates']:
                    discrete_input_candidates.append(grid)
            else:
                raise Exception('Discerete inputs must be passed with either input \'Bounds\' or a pre-defined list of \'Grid\' candidats!')
            #call recursive createGrid function to generate ApproxInputGrid, a list of all possible permuations of xlist's that also satisfy inequality OptimConstraintsaints
            #NOTE: currently, createGrid doesn't actually check the inequality OptimConstraintsaints, need to add, and perhaps add points on inequality boundary??!
            discrete_input_grid = self.__create_grid(cp.deepcopy(discrete_input_names), discrete_input_candidates, discrete_input_constr)
        else:
            discrete_input_array = discrete_inputs['Grid']
            discrete_input_grid = []
            for row in discrete_input_array:
                row_dict = {}
                for indx in range(discrete_input_num):
                    row_dict[discrete_input_names[indx]] = row[indx]
                discrete_input_grid.append(row_dict)
        
        discrete_grid_length = len(discrete_input_grid)
        return [discrete_input_grid, discrete_grid_length, discrete_input_names, discrete_input_num]

    def __continuous_settup(self, continuous_inputs, options):
        """ A private helper function that parses the continuous inputs dictionary.

        This private function is used during the NLoed Design class construction.

        This function accepts the user-provided continuous_inputs dictionary (if one is passed) and
        constructs the required data structures needed to formulate the Casadi/IPOPT design optimization
        problem. This includes creating Casadi symbols for each unique value of each continuous input
        specified by the 'Structure' field of the continuous_inputs dictionary. This function also 
        then combines these input symbols into input archetypes (i.e. specifying combination of continuous
        input symbols are grouped together in the same input vector). These arechtypes are latter concatenated 
        with the discrete input grid points to create complete model input vectors. This function also
        parses out and collects the continuous input bounds, needed for formulating the optimization
        problem. This function also handles initial values for the continuous input symbol values 
        which the optimizer uses as its initialization point (these may be user-provided or automatically
        generated).

        Args:
            discrete_inputs (dictionary): The user provided discrete_inputs dictionary passed to the
                Design constructor.
            options (dictionary, optional): A dictionary of user-defined options, possible key-value
                pairs include:

        Return:
            list: The returned object is a list of data structures in the follow order;
                continuous_symbol_archetypes -- A list of dictionaries, one for each archetypal
                continuous input point, the keys are the names of the continuous inputs and the
                values are the corresponding Casadi symbols for that inputs value in that archetypal
                point.
                continuous_symbol_list -- A list of Casadi symbols for every continuous input's possible
                value in the symbol archetypes, as determined by the user-passed 'Structure' field.
                There can be more than one symbol for a given continuous input.
                continuous_lowerbounds -- A list of the lower bounds for each continuous input symbol,
                in the same order as the symbols listed in continuous_symbol_list.
                continuous_upperbounds -- A list of the upper bounds for each continuous input symbol,
                in the same order as the symbols listed in continuous_symbol_list.
                continuous_archetype_num -- The number of continuous input archetypes in continuous_symbol_archetypes.
                continuous_symbol_num -- The number of continuous input symbols in continuous_symbol_list.
                continuous_input_num -- The number of continuous inputs.
                continuous_input_names -- A list of the continuous input string names.
                continuous_init -- A list of initial values for each continuous input symbol, in the
                same order as the symbols are listed in continuous_symbol_list

        """
        #get names, number and indices of continuous inputs
        continuous_input_names = continuous_inputs['Inputs']
        continuous_input_num = len(continuous_input_names)
        continuous_input_bounds = continuous_inputs['Bounds']
        continuous_input_structure = continuous_inputs['Structure']
        if 'Initial' in continuous_inputs:
            continuous_input_init = continuous_inputs['Initial']
        # if 'Initial' in continuous_inputs:
        #     for 
        #     continuous_input_init = continuous_inputs['Initial']
        keyword_symbol_list_dict = []
        #add a dictionary to the list for each continuous input
        for j in range(continuous_input_num):
            keyword_symbol_list_dict.append({})
        #create a list to store casadi optimization symbols for continuous input archetypes
        continuous_symbol_archetypes = []
        continuous_symbol_list = []
        continuous_lowerbounds = []
        continuous_upperbounds = []
        continuous_init = []
        symbol_index = 0
        #loop over continuous input structure rows 
        for i in range(len(continuous_input_structure)):
            keyword_row = continuous_input_structure[i]
            continuous_archetype_row_dict = {}
            if 'Initial' in continuous_inputs:
                init_row = continuous_input_init[i]
            #archetype_map_row_dict = {}
            #current_arch_index_list = []
            #loop over keywords in row
            for j in range(len(keyword_row)):
                input_name = continuous_input_names[j]
                keyword = keyword_row[j]
                if 'Initial' in continuous_inputs:
                    init_val = init_row[j]
                else:
                    init_val = np.random.uniform(continuous_input_bounds[j][0],
                                                 continuous_input_bounds[j][1],1)[0]
                #check if this keyword has been seen before 
                # NOTE: should really only restrict the use to unique keyworks per column of ExactInputStructure, otherwise we can get bad behaviour
                if keyword in keyword_symbol_list_dict[j].keys():
                    #if the keyword has been seen then a symbol already exists, add the casadi symbol to the current archetype list
                    continuous_archetype_row_dict[input_name] = keyword_symbol_list_dict[j][keyword]
                else:
                    #if the keyword is new, a casadi symbol does not exist, create optimizaton symbols for the corresponding continuous input
                    new_continuous_symbol = cs.MX.sym('ExactSym_'+input_name+'_entry'+str(i)+'_elmnt'+ str(j))
                    symbol_index_pair = {'Symbol':new_continuous_symbol,'Index':symbol_index}
                    #now add the new casadi symbol to the current archetype list
                    continuous_archetype_row_dict[input_name] = symbol_index_pair
                    keyword_symbol_list_dict[j][keyword]=symbol_index_pair
                    continuous_symbol_list.append(new_continuous_symbol)
                    continuous_lowerbounds.append(continuous_input_bounds[j][0])
                    continuous_upperbounds.append(continuous_input_bounds[j][1])
                    continuous_init.append(init_val)
                    symbol_index += 1
            #add the current archetype list to the list of archetype lists
            continuous_symbol_archetypes.append(continuous_archetype_row_dict)
            continuous_archetype_num = len(continuous_symbol_archetypes)
            continuous_symbol_num = len(continuous_symbol_list)

        return [continuous_symbol_archetypes,
                continuous_symbol_list,
                continuous_lowerbounds,
                continuous_upperbounds,
                continuous_archetype_num,
                continuous_symbol_num,
                continuous_input_num,
                continuous_input_names,
                continuous_init]

    def __weighting_settup(self, discrete_input_names, discrete_input_grid, continuous_input_names, 
                                continuous_symbol_archetypes, fixed_design, options):
        #PACKAGE THIS AS A FUNCTION; returns obs vars, samp sum and fim symbols?
        """ A private helper function used to create Casadi optimization symbols for the relaxed
        sampling weights of the design optimization problem.

        This function is private helper function called during instantiation of a Design object.

        This function through the continuous input symbol archetypes and the discrete input
        grid and assembles complete input vectors for the model. These input vectors, in input_list,
        contain both real values from the discrete inputs and Casadi symbols from the continuous 
        inputs. This function then adds the appropriate weighting symbols to each of these cadindidate
        points and assembles the total fisher information matrix for the optimized design using
        said weighting. This function also adds any fixed design aspects to the fim computation
        and constructs the weight sum constraints.

        Args:
            discrete_input_names (list of strings): A list of the names of the discrete inputs.
            discrete_input_grid (list of dictionaries): Each element is a dictionary specifying a 
                candidate point in the discrete input grid. The dictionary keys are discrete input 
                names and their values are the discrete input values at the given point.
            continuous_input_names (list of strings): A list of the names of the continuous inputs.
            continuous_symbol_archetypes (list of dictionaries): Each element is a dictionary specifying a 
                archetypal input subset of the continuous inputs. The dictionary keys are continuous input 
                names and their values are the Casadi symbols for the continuous input levels, which
                will be optimized.
            fixed_design (dictionary): A dictionary with a 'Weight' key, pointing to a real number
                between 0 and 1 representing the fraction of overall samples proportioned to the fixed
                design, and a 'Design' key pointing to dataframe containing the fixed experimental design.
            options (dictionary, optional): A dictionary of user-defined options, possible key-value
                pairs include:

        Return:
            list: This function returns a list of data structures, as follows: fim_list -- a list
            of Casadi FIM symbols with one for each model, weight_symbol_list -- a list of Casadi
            symbols for the sampling weights of each candidate input point that will be optimized, 
            weight_sum -- a Casadi symbol for the sum of the sampling weights, weight_num -- an integer
            indicating the number of weight symbols, weight_init -- a list of values corresponding to
            the sampling weight symbols at which the IPOP solver will initialized the optimization, 
            weight_design_map -- list of dictionaries, one for each weight symbol, mapping the sampling
            weight symbol to its corresponding continuous input archetype and discrete input grid point
            as well as its observation group; used to reconstruct the final optimized design from
            the solver output.
        """

        fim_list = []
        for model in self.model_list:
            fim_list.append(np.zeros((model.num_param,model.num_param)))

        if fixed_design:
            #fixed_design
            #NOTE: perhaps change weight to be of new design?
            pre_weight = fixed_design['Weight']
            post_weight = 1 - pre_weight
            #if 'Design' in fixed_design:
            pre_design = cp.deepcopy(fixed_design['Design'])
            pre_design['Weights'] = pre_design['Replicates']/sum(pre_design['Replicates'])
            pre_design.drop('Replicates',axis=1,inplace=True)
            #loop over the number of replicates
            for index,row in pre_design.iterrows():
                for mod in range(self.num_models):
                    #get the model 
                    model = self.model_list[mod]
                    input_row = row[self.input_name_list].to_numpy()
                    observ_name = row['Variable']
                    weight = row['Weights']
                    fim_list[mod] += pre_weight * weight *\
                                        model.fisher_info_matrix[observ_name](input_row, 
                                                                    self.param_list[mod]).full()
            # elif 'FIM' in fixed_design:
            #     pre_fim = cp.deepcopy(fixed_design['FIM'])
            #     for mod in range(self.num_models):
            #         fim_list[mod] += pre_weight * pre_fim[mod]
            # else:
            #     raise Exception('Invalid key in fixed design data!')
        else:
            post_weight = 1

        # declare sum for discrete weights
        weight_sum = 0
        weight_symbol_list = []
        weight_design_map = []
        weight_init = []
        # loop over continuous symbol archetypes, or if continuous wasn't passed then enter the loop only once
        for i  in range(len(continuous_symbol_archetypes)):
            #current_continuous_weights = []
            # loop over discrete grid, or if discrete wasn't passed then enter the loop only once
            for j in range(len(discrete_input_grid)):
                #create a list to hold the current input to the model
                input_list = []
                input_map_dict = {}
                #loop over model inputs
                for input_name in self.input_name_list:
                    #check if input index is in discrete or continuous inputs
                    if input_name in discrete_input_names:
                        #if in index corresponds to an discrete input add the appropriate numerical grid values to the input vector
                        input_list.append(cs.MX(discrete_input_grid[j][input_name]))
                        input_map_dict[input_name] = {'Index':j,'Type':'A'}
                    elif input_name in continuous_input_names:
                        #if in index corresponds to an continuous input add the appropriate symbol to the input vector
                        input_list.append(continuous_symbol_archetypes[i][input_name]['Symbol'])
                        input_map_dict[input_name] = {'Index':continuous_symbol_archetypes[i][input_name]['Index'],'Type':'E'}
                    else:
                        #if we can't find the index k in either the discrete or continuous indices, throw an error
                        raise Exception('Model input with name '+input_name+' does not match any inputs passed as discrete or continuous!')
                #concatinate input list into a single MX
                input_vector = cs.horzcat(*input_list)
                #current_discrete_weights = []
                #loop over the obeservation groups
                for k in range(self.num_observ_group):
                    obs_group = self.observ_group_list[k]
                    if not options['LockWeights']:
                        #create a sampling weight symbol for the current input and add it to the optimization variable list
                        new_weight = cs.MX.sym('sample_weight_'+ str(i)+'_'+ str(j)+'_'+ str(k))
                        weight_symbol_list.append(new_weight)
                        #add sampling weight symbol to the running total, used to constrain sum of sampleing weights to 1 latee
                        weight_sum += new_weight
                    else:
                        new_weight = 1/len(continuous_symbol_archetypes)
                    map_info = {'InputMap':input_map_dict,
                                'ObsMap':obs_group}
                    weight_design_map.append(map_info)
                    # get the length of the observation group
                    # this is used to scale sampling weight so FIM stays normalized w.r.t. sample number
                    group_size = len(obs_group)
                    #loop over observatino variables in sample group
                    for observ_name in obs_group:
                        #loop over each model
                        for mod in range(self.num_models):
                            #get the model 
                            model = self.model_list[mod]
                            #NOTE: Bayesian sigma point loop goes here
                            #get the model's parameter symbols
                            param = self.param_list[mod]
                            #add the weighted FIM to the running total for the experiment (for each model)
                            fim_list[mod] += post_weight*(new_weight / group_size)\
                                * model.fisher_info_matrix[observ_name](input_vector,param)
                
                #current_continuous_weights.append(current_discrete_weights)
            #.append(current_continuous_weights)
        weight_num = len(weight_symbol_list)

        if not options['LockWeights']:
            weight_init = [1/weight_num] * weight_num
        else:
            weight_init = [1/len(continuous_symbol_archetypes)] * len(continuous_symbol_archetypes)

        return [fim_list, weight_symbol_list, weight_sum, weight_num, weight_init, weight_design_map]

    def __optim_settup(self, fim_list, continuous_symbol_list, continuous_lowerbounds, continuous_upperbounds, continuous_init, weight_symbol_list, weight_sum, weight_init, options):
        """ A private helper function  

        This function 

        Args:
            fim_list
            continuous_symbol_list ():
            continuous_lowerbounds ():
            continuous_upperbounds ():
            continuous_init ():
            weight_symbol_list ():
            weight_sum ():
            weight_init ():
            options :

        Return:

        """
        
        #SETTUP OPTIM VARS, BOUNDS and MAP here

        if not options['LockWeights']:
            optim_symbol_list = continuous_symbol_list + weight_symbol_list
            optim_init = continuous_init + weight_init

            var_lowerbounds = continuous_lowerbounds + [0]*len(weight_symbol_list)
            var_upperbounds = continuous_upperbounds + [1]*len(weight_symbol_list)

            #add a constraint function to ensure sample weights sum to 1
            constraint_funcs = [weight_sum - 1]
            #bound the constrain function output to 0
            constraint_lowerbounds = [0]
            constraint_upperbounds = [0]
        else:
            optim_symbol_list = continuous_symbol_list
            optim_init = continuous_init

            var_lowerbounds = continuous_lowerbounds
            var_upperbounds = continuous_upperbounds

            #add a constraint function to ensure sample weights sum to 1
            constraint_funcs = []
            #bound the constrain function output to 0
            constraint_lowerbounds = []
            constraint_upperbounds = []
        #MOST RECENT
        #optim_init = [np.random.uniform(continuous_lowerbounds[i],continuous_upperbounds[i],1)[0]\
        #                        for i in range(len(continuous_symbol_list))]\
        #             + [1/len(weight_symbol_list)] * len(weight_symbol_list)
        #OLD COMMENTED OUT
        # optim_init = [0.5*(continuous_upperbounds[i] + continuous_lowerbounds[i])\
        #                         for i in range(len(continuous_symbol_list))]\
        #              + [1/len(weight_symbol_list)] * len(weight_symbol_list)
        # optim_init = 0.5*(continuous_upperbounds + continuous_lowerbounds)\
        #                     +[1/(self.discrete_grid_length \
        #                             * self.continuous_archetype_num \
        #                             * self.num_observ_group)] * len(weight_symbol_list)

        cumulative_objective_symbol = 0
        for m in range(self.num_models): 
            cumulative_objective_symbol += self.objective_list[m](fim_list[m])/self.num_models

        ipopt_problem_struct = {'f': cumulative_objective_symbol,
                                'x': cs.vertcat(*optim_symbol_list),
                                'g': cs.vertcat(*constraint_funcs)}

        return [ipopt_problem_struct,
                optim_init,
                var_lowerbounds,
                var_upperbounds,
                constraint_funcs,
                constraint_lowerbounds,
                constraint_upperbounds]
    
    #Function that recursively builds a grid point list from a set of candidate levels of the provided inputs
    def __create_grid(self,input_names,input_candidates,constraints):
        """ A function to 

        This function 

        Args:
            input_names ():
            input_candidates ():
            constraints ():

        Return:

        """
        new_grid=[]
        current_dim = input_candidates.pop()
        current_name = input_names.pop()

        if len(input_candidates)>0:
            current_grid = self.__create_grid(input_names, input_candidates, constraints)
            for grid_point in current_grid:
                for dim_value in current_dim:
                    temp_grid_point = grid_point.copy()
                    temp_grid_point[current_name] = dim_value
                    new_grid.append(temp_grid_point)
        else:
            new_grid = [{current_name:d} for d in current_dim]

        return new_grid

    #Function uses recursion to find the insertion index to ensure sorted designs
    def __sort_inputs(self,newrow,rows,rowpntr=0,colpntr=0):
        """ A function to 

        This function 

        Args:

        Return:

        """
        if not(len(rows)==0) and colpntr<len(rows[0]):
            i=rowpntr
            while i<len(rows):
                if newrow[colpntr]>rows[i][colpntr]:
                    rowpntr+=1
                elif newrow[colpntr]==rows[i][colpntr]:
                    rowpntr=max(rowpntr,self.__sort_inputs(newrow,rows,rowpntr,colpntr+1))
                i+=1
        return rowpntr

#NOTE: Note sure if these can be included in the class somehow
@contextmanager
def silence_stdout(flag):
    if flag:
        old_target = sys.stdout
        try:
            with open(os.devnull, "w") as new_target:
                sys.stdout = new_target
                yield new_target
        finally:
            sys.stdout = old_target
    else:
        yield None

# def design(models, discreteinputs=None, continuousinputs=None, observgroups=None, fixeddesign=None):
#     """ 
#     Add a docstring
#     """
#NOTE: [IMPORTANT] should design be a class that power is called on, makes set up of power/samplesize/rounding much less redundant (passing models, objectives, past designs, constraints etc. -- all used for jitter objective)
#       downside is approximate design becomes the latent data within design class, less modular but perhas in a good way, prevents using power to round designs produce using very different settings

#Fixes/asthetics:   keywords are shared across all inputs, can cause unexpected collisions

#Easy Changes:      model priors weights
#                   observation weight caps
#                   replicate arg for model structure
#                   resolution option passing to IPOPT/custom options
#                   custom grid/candidate list
#                   turn off observation selection for continuous (forces you to observe all ouputs at each unique input level, no discrete??, is this even possible in a simple way?)
#Medium Changes:    passing in fixed design (or data, useful for MC simulations with past data or just previous fixed design)
#                   start values for continuous
#                   Ds optimality
#Hard Changes:      partition into subfunctions
#                   continuous/discrete constraints
#                   bayesian parameter prior with sigma points
#                   Custom optimality
#                   [NO] T optimality
#                   Bias optimality, consistent with order of FIM (is bias 2nd order, can we do 2nd order cov)
#                   [No] Constrained parameter optimality
#Speculative:       grid points on constraint boundary
#                   grid refinment
#                   L2 regularization on weights for unique design

#NOTE: have matrix or past data as option, cov matrix allows for designing with bayes/regularized models, singular fims
#NOTE: can we allow for paramsa and inputs to be high-dim vectors to allow for glm bayesian design for P>N setting, i.e. genomic ridge (GLM) regression with interactions?? matrix products makes model easie to write, can ipopt handle efficiently?
#NOTE: bias estimation won't work for ridge stuff though

#NOTE:should we drop T-opt form v1.0?? Bit of a stretch structure wise, need delta-method etc. to evaluate efficacy of model, need 

#NOTE: for fixed/pre-existing design, do we add it to the output design (probably not)

#NOTE: should add simple if statment to check if models is a model or a list of models, allows user to pass more conveniently 

#NOTE: maybe add some L2 regularization (or Linf??, L1 does nothing given sum-to-one constraint) or some fancy exchange of coordinates at the end using derivatives to produce equivlant designs using fewer support points
# (i.e. pathological model with two responses and few interactions between inputs and parameters across responses tends to give huge number of suppor points)

#NOTE: ***[important]*** Need to handle multiple observations (groups too), w.r.t. to sample weights and observation selection, need good default
#NOTE: should add replicates code, will make good defaults easier maaybe (double check)....
#NOTE: design ouput and related code needs some clean up (i.e. map)
#NOTE: start thinknig about more modular code (i.e. other functions for design.py), user may want to access something like
#      creategrid for custome grid generation (i.e. for log distributed grid), but not internal functions, private funcs somehow??, leading '__'

#NOTE: need to decide on design/experiment/data structure, thinking dictionary with input and output names as keys
#      list of settings as value, list of list for observations, need to implement a sort alg. for inputs to standardize inoput order
#      need to combine non-unique elements, this will make implementing fixeddesign, power, fitting easier
#      need to make sure weighting works out on grouped observations (half the optimal weight to each??)
#NOTE: with current code, struct matrix provided to continuous doesn't allow for you to assign specific  observations to each input
#       inputs arechetype in a fully continuous design, this is limiting, but forces user to select optimal observation vars (which maybe okay)
#       or they have to observe all (specific subset of) vars at each continuous input archetype (probabyl okay, otherwise gets too complex I think)
#       user can lock in weight fractions (with observation weights field) to compensate for diff in obs. cost, but can't have a hard max on an assay type for example
#NOTE: need to change design ouput so it can build dictionary (with proper obervartion vect handling), according to above
#       maybe observations with no weight get 0 weight in design output, or blanks??, or we just list obervations that are observed
#       but then we can't handle weights...

#NOTE: Add to model later: Weight, Prior (cov, norm only, user can transform var), POI (parameters-of-interest)
#NOTE: Add to discrete later: Constraints, Resolution (stepsize), Grid (user defined set of possible input levels, or full grid?)
#NOTE: Add to continuous later: Replicates (code to default with full reps of all unique vec), Constraints, Start values (pass in struct matrix??)
#NOTE: Add to obs later: Weights, if not passed, treat each observation as individual option

#NOTE: should we avoid model 'Input' as it makes error strings awkward when talking about function inputs?!

#NOTE: should bayesian priors be treated as symbolics in group loop, and loop over sigma points done just before ipopt pass
#NOTE: OR should sigma points be generatedin inital model loop as numbers, and looped over within group loop, with FIMList being sigmaXmodels in size
#NOTE: leaning towards latter at least initially

#NOTE: current structure makes grid refinments difficult
#NOTE: fixeddesign (observations), data or past fim (as function of beta???) Probably just pass in design/data, fim comp for data will ignore obseved y info anyways for asympotitic fim
#NOTE: models must have the same x dim and input names, not same parameters though
#NOTE: should fixed be an discrete design (by weight) or an continuous design (count), discrete does everything more flexible, continuous enforces 'real' data
#NOTE: leaning towards discrete, or auto configure for both
#NOTE: should sort design output so it has a common ordering (ie by x1, then x2 etc.), should group non-unique elements and merge their weights

#OLD DISCUSSION
#Should this be a class?? would make rounding easier

# rough specs
# inputs: model list ,grouping matrix, beta info, past data
# return: ys, xs, weights (not ready for actual experiment because you need to do power/rounding analysis)

# Goals
# can enter either past data or maybe fisher info from past data as starting point to improve an experiment
# grouping response variables that need to be sampled together, paritioning overall sample number across groups
#       (ie 80% and 20%, or if 0% will find optimal weights, perhaps fast weight to specifiy equal weights)
# [DONE] having x split into two types, grided values with weights xi vs continuous ranges which are constant within a group
# OptimConstraintsaints on x, linear and nonlinear
# grid refinment for grided x, starting values for non-grided
# optional optimality criteria, D, Ds (need to group parameters), A, custom (but simple casadi function of fisher info)
# average designs for multiple models, with priors over the models
# bayesian (normal prior) using sigma points for all or subset of the parameters
# possible support for rounding design into an experiment

# Questions
# Do we scale the FIM?, only D maybe Ds is invariant under linear re-scaling (maybe all rescaling), but not others or custom, however is more numerically stable
# Also do we log the opt(FIM)? for numerical stability, this maybe generally useful
# how to select sigma points
# how to do constrained and curvature/bias optimality measures


# #check that either continuous xor discrete has been passed 
# if not( discreteinputs) and not(continuousinputs):
#     raise Exception('The design function requires at least one of the discrete or continuous values to be passed!')

# #get number of models
# NumModels=len(models)
# # fim list for each model, keeps a running sum of symbols for each model (and prior parameter evaluation)
# FIMList = []
# # List of beta symbols for each model 
# # or create sigma points first and nest loop within model loop for fim, with fim list sigmaXmodels big
# ParamList = []
# #list for casadi objective functions for each model, as matrix dim.s change, we need to define at run-time
# ObjectiveFuncs = []
# if not('Model' in models[0]):
#         raise Exception('Missing objective for model at index '+str(0)+' in models list!')
# #set common dimensions for all model
# InputDim = models[0]['Model'].NumInputs
# ObservDim = models[0]['Model'].NumObserv
# #set common dicts for inputs and observations, NOTE: could later allow for different orderings
# InputDict = models[0]['Model'].InputNameDict
# ObservDict = models[0]['Model'].ObservNameDict
# #create lists for reverse lookup, index to name mapping NOTE: can get these from model[0] now
# InputNameList=list(InputDict.keys())
# ObservNameList=list(ObservDict.keys())
# #loop over models check dimensions and dicts, build objective functions, create fim and beta lists 
# for m in range(NumModels):
#     if not('Model' in models[m]):
#         raise Exception('Missing model field for model at index '+str(m)+' in models list!')
#     if not('Parameters' in models[m]):
#         raise Exception('Missing objective field for model at index '+str(m)+' in models list!')
#     if not('Objective' in models[m]):
#         raise Exception('Missing parameters field for model at index '+str(m)+' in models list!')
#     Model = models[m]['Model']
#     Params = models[m]['Parameters']
#     ObjectiveType = models[m]['Objective']
#     #check if each model has the continuous same dimensions and input/ouput naming, if not throw error
#     if not(ObservDim == Model.NumObserv):
#         raise Exception('All model output dimensions must match!')
#     if not(InputDim == Model.NumInputs ):
#         raise Exception('All model input dimensions must match!')
#     if not(InputDict == Model.InputNameDict):
#         raise Exception('Model input name and ordering must match!')
#     if not(ObservDict == Model.ObservNameDict):
#         raise Exception('Model output name and ordering must match!')
#     #NOTE:model D score must be weighted/rooted log-divided according to number of params, need to check this
#     if ObjectiveType =='D':
#         Matrx = cs.SX.sym('Matrx',Model.NumParams, Model.NumParams)
#         RFact = cs.qr(Matrx)[1]
#         NormalizedLogDet = cs.trace(cs.log(RFact))/Model.NumParams
#         ObjectiveFuncs.append( cs.Function('ObjFunc'+str(m),[Matrx],[-NormalizedLogDet]) )
#     elif ObjectiveType == 'Ds':
#         if not('POI' in Model[m]):
#             raise Exception('No parameters of interest provided for Ds design objective! Need list of parameter-of-interest names!')
#         poi=models[m]['POI']
#         #need to write this
#         #ObjectiveFuncs.append( cs.Function('ObjFunc'+str(m),[M],[-logdet]) )
#     elif ObjectiveType == 'T':
#         i=0
#         #add for model difference, need to flag this and switch to computing output rather than fim
#     elif ObjectiveType == 'Custom':
#         i=0
#         #add for custom function of FIM
#     else:
#         raise Exception('Unknown objective: '+str(ObjectiveType)+'!')

#     #create the fim list for summing fim symbolics in group loop and parameter symbols for each model 
#     #ParamList used for bayesian priors
#     #NOTE:should maybe be an 'output' list for model selection objective; T-optimality etc.
#     FIMList.append(np.zeros((Model.NumParams,Model.NumParams) ))
#     #ParamList.append(cs.MX.sym('beta_model'+str(m),model.Nb))
#     ParamList.append(Params)

# #counter to track the total number inputs across continuous and discrete, must sum to total for the model(s)
# InputNumCheck=0
# #if user has passed discrete inputs
# if  discreteinputs:
#     #get names, number and indices of discrete inputs
#     ApproxInputNames =  discreteinputs['Inputs']
#     ApproxInputNum = len(ApproxInputNames)
#     ApproxInputIndices = [InputDict[a] for a in ApproxInputNames] 
#     #add discrete inputs to the total input number (used to check all inputs are accounted for after loading continuous)
#     InputNumCheck = InputNumCheck + ApproxInputNum
#     #check if discrete bounds have been passed, if not throw error, if so get them
#     if not('Bounds' in  discreteinputs):
#         raise Exception('Approximate inputs have no bounds!')
#     ApproxInputBounds =  discreteinputs['Bounds']
#     #check if we have bounds for each discrete input
#     if not(ApproxInputNum == len(ApproxInputBounds)):
#         raise Exception('There are '+str(len(ApproxInputNames))+' discrete inputs listed, but there are '+str(len(ApproxInputBounds))+' bounds, these must match!')
#     #check if inquality OptimConstraintsains have been passed, if so store them 
#     ApproxInputConstr = []
#     if 'Constraints' in  discreteinputs:
#         ApproxInputConstr  =  discreteinputs['Constraints']          
#     #set resolution of grid NOTE: this should be able to be specified by the user, will change
#     N = 5
#     #create a list for storing possible levels of each discretemate input
#     ApproxInputCandidates = []
#     #loop over bounds passed, and use resolution and bounds to populate xlist
#     for b in ApproxInputBounds:
#         ApproxInputCandidates.extend([np.linspace(b[0],b[1],N).tolist()])
#     #call recursive createGrid function to generate ApproxInputGrid, a list of all possible permuations of xlist's that also satisfy inequality OptimConstraintsaints
#     #NOTE: currently, createGrid doesn't actually check the inequality OptimConstraintsaints, need to add, and perhaps add points on inequality boundary??!
#     ApproxInputGrid = creategrid(ApproxInputCandidates,ApproxInputConstr)
#     NumApproxGrid=len(ApproxInputGrid)
# else:
#     NumApproxGrid=1 #NOTE: this is ugly, but needed for now so that while loops and weight initialization works out if discrete isn't passed
#     ApproxInputIndices=[]

# # these data structures are used to track where continuous input symbols, discrete input-observation weights end up in the final optimization solution
# # OptimSolutionMap is a dictionary mapping optimization vector input indices (only ones that correspond to a sample weights)
# # to a dictionary with information on reconstructing the corresponding intput vector and observation group
# OptimSolutionMap={}
# # List_Of_SampleWeightOptimIndices is a list of optimization vector indices that correspond to a sample weight (as opposed to an continuous input value)
# List_Of_SampleWeightOptimIndices=[]
# # ArchetypeIndex_To_OptimIndices ia a list that maps and index in the continuous archetype list to the set of optimizaion vector indices that contain the archetype continuous input value after optimization
# # one-to-many, ordering here is identical to user-passed ordering
# ArchetypeIndex_To_OptimIndices=[]
# # Keyword_To_OptimIndex is a dictionary that maps a keyword passed via the continuousinput 'Structure' field to an index in the optimization vector (one-to-one)
# Keyword_To_OptimIndex={}

# # list of optimization variables (continuous input settings and discrete weights), and a list of starting values
# OptimSymbolList = []
# OptimVariableStart = []
# # list of Casadi expressions for (non-)linear ineqaulity constraints on continuous settings (e.g. g(X)>0), and linear constraints on discrete problem (i.e. sum(xi)=1)
# OptimConstraints = []
# # lower and upper bounds for optimization variables and for optimization constraints in OptimConstraints
# LowerBoundVariables = []
# UpperBoundVariables = []
# LowerBoundConstraints = []
# UppperBoundConstraints = []
# #if the user has passed continuous inputs
# if continuousinputs:
#     #get names, number and indices of continuous inputs
#     ExactInputNames = continuousinputs['Inputs']
#     ExactInputNum = len(ExactInputNames)
#     ExactInputIndices = [InputDict[e] for e in ExactInputNames]
#     #add these to the total input number check for this group
#     InputNumCheck = InputNumCheck + ExactInputNum
#     #if no bounds passed for continuous inputs, throw error, if not get the continuous input bounds
#     if not('Bounds' in continuousinputs):
#         raise Exception('Exact inputs have no bounds!')
#     ExactInputBounds=continuousinputs['Bounds']
#     #if the number of bounds don't match the continuous names, throw error
#     if not(ExactInputNum == len(ExactInputBounds)):
#         raise Exception('There are '+str(len(ExactInputNames))+' continuous inputs listed, but there are '+str(len(ExactInputBounds))+' bounds, these must match!')
#     #if structure for continuous inputs is not provided throw error, else get
#     if not('Structure' in continuousinputs):
#         raise Exception('No continuous input structure was provided!')
#     ExactInputStructure = continuousinputs['Structure']
#     #create a list of dicts for tracking existing symbols 
#     ExistingExactSymbols = []
#     #add a dictionary to the list for each continuous input
#     for i in range(ExactInputNum):
#         ExistingExactSymbols.append({})
#     #create a list to store casadi optimization symbols for continuous input archetypes
#     ExactSymbolArchetypes=[]
#     #loop over continuous input structure and create archetype symbol list for continuous inputs provided
#     for i in range(len(ExactInputStructure)):
#         ExactStructRow=ExactInputStructure[i]
#         CurrentArchetype=[]
#         CurrentArchOptimIndices=[]
#         if not(len(ExactStructRow) == ExactInputNum):
#             raise Exception('Row number '+str(i)+' in the continuous structure passed has a length of '+str(len(ExactStructRow))+' but should be '+str(ExactInputNum)+' long, with an element for each continuous input!')
#         for j in range(len(ExactStructRow)):
#             Keyword=ExactStructRow[j]
#             #check if this keyword has been seen before NOTE: should really only restrict the use to unique keyworks per column of ExactInputStructure, otherwise we can get bad behaviour
#             if Keyword in ExistingExactSymbols[j]:
#                 #if the keyword has been seen then a symbol already exists, add the casadi symbol to the current archetype list
#                 CurrentArchetype.append(ExistingExactSymbols[j][Keyword])
#                 #add the index within OptimSymbolList corresponding to the existing symbol to the current archetype's index list
#                 CurrentArchOptimIndices.append(Keyword_To_OptimIndex[Keyword])
#             else:
#                 #if the keyword is new, a casadi symbol does not exist, create optimizaton symbols for the corresponding continuous input
#                 NewExactSymbol = cs.MX.sym('ExactSym_'+ExactInputNames[j]+'_entry'+str(i)+'_elmnt'+ str(j))
#                 #now add the new casadi symbol to the current archetype list
#                 CurrentArchetype.append(NewExactSymbol)
#                 #add new keyword-symbol pair to the existing symbol dictionary
#                 ExistingExactSymbols[j][Keyword] = NewExactSymbol
#                 #add new keyword to the map so we can find the optimization index that matches that keyword
#                 Keyword_To_OptimIndex[Keyword]=len(OptimSymbolList)
#                 #add the index within OptimSymbolList corresponding to the new symbol to the current archetype's index list
#                 CurrentArchOptimIndices.append(len(OptimSymbolList))
#                 #add continuous symbols for this group to optimzation symbol list
#                 OptimSymbolList += [NewExactSymbol]
#                 #get the current bounds
#                 lb = ExactInputBounds[j][0]
#                 ub = ExactInputBounds[j][1]
#                 #set the continuous input start value randomly within bounds and add it to start value list
#                 OptimVariableStart += [rn.uniform(lb,ub)]
#                 #get the upper and lower bound and add them to the opt var bound lists
#                 LowerBoundVariables += [lb]
#                 UpperBoundVariables += [ub]
#         #add the current archetype list to the list of archetype lists
#         ExactSymbolArchetypes.append(CurrentArchetype)
#         #append the current archetype's index list to the ArchetypeIndex_To_OptimIndices list within the optimal solution map
#         ArchetypeIndex_To_OptimIndices.append(CurrentArchOptimIndices) 
#         NumExactArchetypes=len(ExactSymbolArchetypes)
# else:
#     NumExactArchetypes=1 #NOTE: this is ugly, but needed for now so that while loops and weight initialization works out if continuous isn't passed
#     ExactInputIndices=[]

# #check if total inputs passed, continuous + discrete, is equal to total model inputs, if not throw error
# if not(InputNumCheck == InputDim):
#     raise Exception('All input variables must be passed as either discrete or continuous, and the total number of inputs passed must be the same as recieved by the model(s)!\n'
#                     'There are '+str(ApproxInputNum)+' discrete inputs and '+str(ExactInputNum)+' continuous inputs passed but model(s) expect '+str(InputDim)+'!')

# #check if observ passed 
# if not( observgroups):
#     observgroups={}
#     observgroups['Observations']=[[o] for o in list(ObservDict.keys())]
# #    raise Exception('No observation dictionary has been passed!')
# #check if observation groups have been passed, if not throw error, if so get
# if not('Observations' in  observgroups):
#     raise Exception('No observation field was passed!')
# ObservGroups =  observgroups['Observations']
# NumObservGroups=len(ObservGroups)
# #list for observation group indices in the models
# ObservGroupIndices = []
# for i in range(NumObservGroups):
#     CurrentGroupNames = ObservGroups[i]
#     #NOTE: need to add check that names exist here
#     #lookup the indices for the y variable names
#     CurrentIndices = [ObservDict[n] for n in CurrentGroupNames] 
#     ObservGroupIndices.append(CurrentIndices)


# # declare sum for discrete weights
# ApproxWeightSum = 0
# #set up loop counters
# i=0
# # loop over continuous symbol archetypes, or if continuous wasn't passed then enter the loop only once
# while i < NumExactArchetypes or (not(continuousinputs) and i==0):
#     j=0
#     # loop over discrete grid, or if discrete wasn't passed then enter the loop only once
#     while j < NumApproxGrid or (not( discreteinputs) and j==0):
#         #create a list to hold the current input to the model
#         InputList = []
#         #labels each input as continuous 'E' or discrete 'A', for use in OptimSolutionMap
#         CurrentInputTypeLabels=[]
#         #stores either the grid index (for discrete inputs) or the opimal vector index (for continuous inputs)
#         CurrentInputLookupIndices=[]
#         #loop over model inputs
#         for k in range(InputDim):
#             #check if input index is in discrete or continuous inputs
#             if k in ApproxInputIndices:
#                 #if in index corresponds to an discrete input add the appropriate numerical grid values to the input vector
#                 InputList.append(cs.MX(ApproxInputGrid[j][ApproxInputIndices.index(k)]))
#                 CurrentInputTypeLabels.append('A')
#                 CurrentInputLookupIndices.append(ApproxInputIndices.index(k))
#             elif k in ExactInputIndices:
#                 #if in index corresponds to an continuous input add the appropriate symbol to the input vector
#                 InputList.append(ExactSymbolArchetypes[i][ExactInputIndices.index(k)])
#                 CurrentInputTypeLabels.append('E')
#                 CurrentInputLookupIndices.append(ArchetypeIndex_To_OptimIndices[i][k])
#             else:
#                 #if we can't find the index k in either the discrete or continuous indices, throw an error
#                 raise Exception('Model input with index'+str(k)+' does not match any inputs passed as discrete or continuous!')
#         #concatinate input list into a single MX
#         InputVector=cs.horzcat(*InputList)
#         #loop over the obeservation groups
#         for k in range(NumObservGroups):
#             obs=ObservGroupIndices[k]
#             #create a sampling weight symbol for the current input and add it to the optimization variable list
#             NewSampleWeight = cs.MX.sym('SampWeight_'+ str(i)+'_'+ str(j)+'_'+ str(k))
#             #label the current index in OptimSymbolList as a sample weight in the optimization variable map
#             List_Of_SampleWeightOptimIndices.append(len(OptimSymbolList))
#             #add an entry mapping the current index in OptimSymbolList to the index of the current continuous symbol archetype and discrete grid entry
#             OptimSolutionMap[len(OptimSymbolList)]={'InputType':CurrentInputTypeLabels,'InputLookUpIndex':CurrentInputLookupIndices,'GridIndex':j,'ObsGroupIndices':obs} 
#             #add sample weight symbol to the optimization symbol list
#             OptimSymbolList += [NewSampleWeight]
#             #set the starting weights so that all grid points at all archetypes have equal weights NOTE: should probably clean this up, conditional on discrete and continuous being passed
#             OptimVariableStart += [1/(NumApproxGrid*NumExactArchetypes*NumObservGroups)]
#             #apply appropriate bounds for the sampling weight NOTE: upper bound 1 should be scaled by passed observation weights when this functionality is added
#             LowerBoundVariables += [0]
#             UpperBoundVariables += [1]
#             #add sampling weight symbol to the running total, used to constrain sum of sampleing weights to 1 latee
#             ApproxWeightSum = ApproxWeightSum+NewSampleWeight
#             #get the length of the observation group
#             # this is used to scale sampling weight so FIM stays normalized w.r.t. sample number
#             N=len(obs)
#             #loop over observatino variables in sample group
#             for var in obs:
#                 #loop over each model
#                 for mod in range(NumModels):
#                     #get the model 
#                     Model = models[mod]['Model']
#                     #NOTE: Bayesian sigma point loop goes here
#                     #get the model's parameter symbols
#                     Params = ParamList[mod]
#                     #add the weighted FIM to the running total for the experiment (for each model)
#                     FIMList[mod]= FIMList[mod] + (NewSampleWeight / N) * Model.FIM[var](Params,InputVector)

#         j+=1
#     i+=1

# #add a constraint function to ensure sample weights sum to 1
# OptimConstraints += [ApproxWeightSum - 1]
# #bound the constrain function output to 0
# LowerBoundConstraints += [0]
# UppperBoundConstraints += [0]

# OverallObjectiveSymbol=0
# for m in range(NumModels): 
#      OverallObjectiveSymbol += ObjectiveFuncs[m](FIMList[m])/NumModels

# #NOTE: should be checking solution for convergence, should allow use to pass options to ipopt
# # Create an IPOPT solver
# IPOPTProblemStructure = {'f': OverallObjectiveSymbol, 'x': cs.vertcat(*OptimSymbolList), 'g': cs.vertcat(*OptimConstraints)}
# print('Setting up optimization problem, this can take some time...')
# #"verbose":True,
# IPOPTSolver = cs.nlpsol('solver', 'ipopt', IPOPTProblemStructure,{'ipopt.hessian_discreteimation':'limited-memory'}) #NOTE: need to give option to turn off full hessian (or coloring), may need to restucture problem mx/sx, maybe use quadratic programming in full discrete mode?
# print('Problem set up complete.')
# # Solve the NLP with IPOPT call
# print('Begining optimization...')
# IPOPTSolutionStruct = IPOPTSolver(x0=OptimVariableStart, lbx=LowerBoundVariables, ubx=UpperBoundVariables, lbg=LowerBoundConstraints, ubg=UppperBoundConstraints)
# OptimSolution = IPOPTSolutionStruct['x'].full().flatten()

# Tol=1e-4 #NOTE: this should probably be a function of the numeber of parameters and maybe the bayesian/model mixture inputs
# #get the samle weights indices that have a non-trivial non-zero value
# NonZeroWeightOptimIndices=[i for i in List_Of_SampleWeightOptimIndices if OptimSolution[i]>Tol]

# #create the design dictionary
# Design={'InputNames':InputNameList,'ObservationNames':ObservNameList,'Inputs':[],'Weight':[]}
# #loop over the non-zero sample weights
# for i in NonZeroWeightOptimIndices:
#     #get relevant information for looking up input values and observation indices from OptimSolutionMap
#     CurrentInputTypeList = OptimSolutionMap[i]['InputType']
#     CurrentInputLookupIndices = OptimSolutionMap[i]['InputLookUpIndex']
#     CurrentApproxGridIndex = OptimSolutionMap[i]['GridIndex']
#     CurrentObservGroupIndices = OptimSolutionMap[i]['ObsGroupIndices']
#     #create a list to store the potential new input vector
#     NewInputRow = []
#     #loop over input indices to the full model
#     for k in range(InputDim):
#         #get the current type 'A' discrete, 'E' continuous
#         CurrentType=CurrentInputTypeList[k]
#         #check if input is discrete or continuous
#         if CurrentType =='A':
#             #if discrete, look up the discretegrid location and the input index within the grid and add to new row
#             NewInputRow.append(round(ApproxInputGrid[CurrentApproxGridIndex][CurrentInputLookupIndices[k]],4))
#         elif CurrentType =='E':
#             #if continuous, look up the optimal solution of the given continuous input and add to the new row
#             NewInputRow.append(round(OptimSolution[CurrentInputLookupIndices[k]],4))
#         else:
#             #this should never be reached
#             raise Exception('Error formating the experimental design! Contact the developers for further support')

#     #check if the input row is unique (i.e. if not alread inserted into the design)
#     InputUnique=not(NewInputRow in Design['Inputs'])

#     #get the number of observations in the observation group associated with this non-zero sample weight
#     Nobs=len(CurrentObservGroupIndices)
#     #create a new observation row
#     NewObservRow=[]
#     #loop over the observation variables #NOTE: this loop could use some clean up, if statments and loop interaction with unique flag are sloppy
#     for k in range(ObservDim):
#         #check the current index of the observation variables is in the current observation group
#         if k in CurrentObservGroupIndices:
#             #if the index is in the group, add the optimal sampling weight, paritioned according to the number of observation variables in the group
#             NewWeight = round(OptimSolution[i]/Nobs,4)
#         else: 
#             #if the observation index is not in the group than that variable gets a zero weight
#             NewWeight = 0.0
#         #check if new input row was unique
#         if InputUnique:
#             #if unique add the new weight to 
#             NewObservRow.append(NewWeight)
#         else:
#             #if not unique, add the weights to the corresponding existing input
#             ExistingIndex = Design['Inputs'].index(NewInputRow)
#             Design['Weight'][ExistingIndex][k]=Design['Weight'][ExistingIndex][k] + NewWeight

#     #if the input was unique add the new input row and new input weights, sort while adding to provide a standardized order to the design
#     if InputUnique:
#         InsertIndex=sortinputs(NewInputRow,Design['Inputs'])
#         Design['Inputs'].insert(InsertIndex,NewInputRow)
#         Design['Weight'].insert(InsertIndex,NewObservRow)

# return Design


