import casadi as cs
import numpy as np
import random as rn

def design(models, approxinputs=None, exactinputs=None, observgroups=None, fixeddesign=None):
    """ 
    Add a docstring
    """
    #Fixes/asthetics:   keywords are shared across all inputs, can cause unexpected collisions

    #Easy Changes:      model priors weights
    #                   observation weight caps
    #                   replicate arg for model structure
    #                   resolution option passing to IPOPT/custom options
    #                   custom grid/candidate list
    #Medium Changes:    passing in fixed design
    #                   start values for exact
    #                   Ds optimality
    #Hard Changes:      partition into subfunctions
    #                   exact/approx constraints
    #                   bayesian parameter prior with sigma points
    #                   Custom optimality
    #                   T optimality
    #                   Beal/Hunter curvature/bias optimality
    #                   Constrained parameter optimality
    #Speculative:       grid points on constraint boundary
    #                   grid refinment
    #                   L2 regularization on weights for unique design

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
    #NOTE: with current code, struct matrix provided to exact doesn't allow for you to assign specific  observations to each input
    #       inputs arechetype in a fully exact design, this is limiting, but forces user to select optimal observation vars (which maybe okay)
    #       or they have to observe all (specific subset of) vars at each exact input archetype (probabyl okay, otherwise gets too complex I think)
    #       user can lock in weight fractions (with observation weights field) to compensate for diff in obs. cost, but can't have a hard max on an assay type for example
    #NOTE: need to change design ouput so it can build dictionary (with proper obervartion vect handling), according to above
    #       maybe observations with no weight get 0 weight in design output, or blanks??, or we just list obervations that are observed
    #       but then we can't handle weights...

    #NOTE: Add to model later: Weight, Prior (cov, norm only, user can transform var), POI (parameters-of-interest)
    #NOTE: Add to approx later: Constraints, Resolution (stepsize), Grid (user defined set of possible input levels, or full grid?)
    #NOTE: Add to exact later: Replicates (code to default with full reps of all unique vec), Constraints, Start values (pass in struct matrix??)
    #NOTE: Add to obs later: Weights, if not passed, treat each observation as individual option

    #NOTE: should we avoid model 'Input' as it makes error strings awkward when talking about function inputs?!
    
    #NOTE: should bayesian priors be treated as symbolics in group loop, and loop over sigma points done just before ipopt pass
    #NOTE: OR should sigma points be generatedin inital model loop as numbers, and looped over within group loop, with FIMList being sigmaXmodels in size
    #NOTE: leaning towards latter at least initially
    
    #NOTE: current structure makes grid refinments difficult
    #NOTE: fixeddesign (observations), data or past fim (as function of beta???) Probably just pass in design/data, fim comp for data will ignore obseved y info anyways for asympotitic fim
    #NOTE: models must have the same x dim and input names, not same parameters though
    #NOTE: should fixed be an approx design (by weight) or an exact design (count), approx does everything more flexible, exact enforces 'real' data
    #NOTE: leaning towards approx, or auto configure for both
    #NOTE: should sort design output so it has a common ordering (ie by x1, then x2 etc.), should group non-unique elements and merge their weights

    #check that either exact xor approx has been passed 
    if not( approxinputs) and not(exactinputs):
        raise Exception('The design function requires at least one of the approximate or exact values to be passed!')

    #get number of models
    NumModels=len(models)
    # fim list for each model, keeps a running sum of symbols for each model (and prior parameter evaluation)
    FIMList = []
    # List of beta symbols for each model 
    # or create sigma points first and nest loop within model loop for fim, with fim list sigmaXmodels big
    ParamList = []
    #list for casadi objective functions for each model, as matrix dim.s change, we need to define at run-time
    ObjectiveFuncs = []
    if not('Model' in models[0]):
            raise Exception('Missing objective for model at index '+str(0)+' in models list!')
    #set common dimensions for all model
    InputDim = models[0]['Model'].NumInputs
    ObservDim = models[0]['Model'].NumObserv
    #set common dicts for inputs and observations, NOTE: could later allow for different orderings
    InputDict = models[0]['Model'].InputNameDict
    ObservDict = models[0]['Model'].ObservNameDict
    #create lists for reverse lookup, index to name mapping 
    InputNameList=list(InputDict.keys())
    ObservNameList=list(ObservDict.keys())
    #loop over models check dimensions and dicts, build objective functions, create fim and beta lists 
    for m in range(NumModels):
        if not('Model' in models[m]):
            raise Exception('Missing model field for model at index '+str(m)+' in models list!')
        if not('Parameters' in models[m]):
            raise Exception('Missing objective field for model at index '+str(m)+' in models list!')
        if not('Objective' in models[m]):
            raise Exception('Missing parameters field for model at index '+str(m)+' in models list!')
        Model = models[m]['Model']
        Params = models[m]['Parameters']
        ObjectiveType = models[m]['Objective']
        #check if each model has the exact same dimensions and input/ouput naming, if not throw error
        if not(ObservDim == Model.NumObserv):
            raise Exception('All model output dimensions must match!')
        if not(InputDim == Model.NumInputs ):
            raise Exception('All model input dimensions must match!')
        if not(InputDict == Model.InputNameDict):
            raise Exception('Model input name and ordering must match!')
        if not(ObservDict == Model.ObservNameDict):
            raise Exception('Model output name and ordering must match!')
        #NOTE:model D score must be weighted/rooted log-divided according to number of params, need to check this
        if ObjectiveType =='D':
            Matrx = cs.SX.sym('Matrx',Model.NumParams, Model.NumParams)
            RFact = cs.qr(Matrx)[1]
            NormalizedLogDet = cs.trace(cs.log(RFact))/Model.NumParams
            ObjectiveFuncs.append( cs.Function('ObjFunc'+str(m),[Matrx],[-NormalizedLogDet]) )
        elif ObjectiveType == 'Ds':
            if not('POI' in Model[m]):
                raise Exception('No parameters of interest provided for Ds design objective! Need list of parameter-of-interest names!')
            poi=models[m]['POI']
            #need to write this
            #ObjectiveFuncs.append( cs.Function('ObjFunc'+str(m),[M],[-logdet]) )
        elif ObjectiveType == 'T':
            i=0
            #add for model difference, need to flag this and switch to computing output rather than fim
        elif ObjectiveType == 'Custom':
            i=0
            #add for custom function of FIM
        else:
            raise Exception('Unknown objective: '+str(ObjectiveType)+'!')

        #create the fim list for summing fim symbolics in group loop and parameter symbols for each model 
        #ParamList used for bayesian priors
        #NOTE:should maybe be an 'output' list for model selection objective; T-optimality etc.
        FIMList.append(np.zeros((Model.NumParams,Model.NumParams) ))
        #ParamList.append(cs.MX.sym('beta_model'+str(m),model.Nb))
        ParamList.append(Params)

    #counter to track the total number inputs across exact and approx, must sum to total for the model(s)
    InputNumCheck=0
    #if user has passed approximate inputs
    if  approxinputs:
        #get names, number and indices of approximate inputs
        ApproxInputNames =  approxinputs['Inputs']
        ApproxInputNum = len(ApproxInputNames)
        ApproxInputIndices = [InputDict[a] for a in ApproxInputNames] 
        #add approx inputs to the total input number (used to check all inputs are accounted for after loading exact)
        InputNumCheck = InputNumCheck + ApproxInputNum
        #check if approximate bounds have been passed, if not throw error, if so get them
        if not('Bounds' in  approxinputs):
            raise Exception('Approximate inputs have no bounds!')
        ApproxInputBounds =  approxinputs['Bounds']
        #check if we have bounds for each approx input
        if not(ApproxInputNum == len(ApproxInputBounds)):
            raise Exception('There are '+str(len(ApproxInputNames))+' approximate inputs listed, but there are '+str(len(ApproxInputBounds))+' bounds, these must match!')
        #check if inquality OptimConstraintsains have been passed, if so store them 
        ApproxInputConstr = []
        if 'Constraints' in  approxinputs:
            ApproxInputConstr  =  approxinputs['Constraints']          
        #set resolution of grid NOTE: this should be able to be specified by the user, will change
        N = 5
        #create a list for storing possible levels of each approxmate input
        ApproxInputCandidates = []
        #loop over bounds passed, and use resolution and bounds to populate xlist
        for b in ApproxInputBounds:
            ApproxInputCandidates.extend([np.linspace(b[0],b[1],N).tolist()])
        #call recursive createGrid function to generate ApproxInputGrid, a list of all possible permuations of xlist's that also satisfy inequality OptimConstraintsaints
        #NOTE: currently, createGrid doesn't actually check the inequality OptimConstraintsaints, need to add, and perhaps add points on inequality boundary??!
        ApproxInputGrid = creategrid(ApproxInputCandidates,ApproxInputConstr)
        NumApproxGrid=len(ApproxInputGrid)
    else:
        NumApproxGrid=1 #NOTE: this is ugly, but needed for now so that while loops and weight initialization works out if approx isn't passed
        ApproxInputIndices=[]

    # these data structures are used to track where exact input symbols, approx input-observation weights end up in the final optimization solution
    # OptimSolutionMap is a dictionary mapping optimization vector input indices (only ones that correspond to a sample weights)
    # to a dictionary with information on reconstructing the corresponding intput vector and observation group
    OptimSolutionMap={}
    # List_Of_SampleWeightOptimIndices is a list of optimization vector indices that correspond to a sample weight (as opposed to an exact input value)
    List_Of_SampleWeightOptimIndices=[]
    # ArchetypeIndex_To_OptimIndices ia a list that maps and index in the exact archetype list to the set of optimizaion vector indices that contain the archetype exact input value after optimization
    # one-to-many, ordering here is identical to user-passed ordering
    ArchetypeIndex_To_OptimIndices=[]
    # Keyword_To_OptimIndex is a dictionary that maps a keyword passed via the exactinput 'Structure' field to an index in the optimization vector (one-to-one)
    Keyword_To_OptimIndex={}

    # list of optimization variables (exact input settings and approximate weights), and a list of starting values
    OptimSymbolList = []
    OptimVariableStart = []
    # list of Casadi expressions for (non-)linear ineqaulity constraints on exact settings (e.g. g(X)>0), and linear constraints on approx problem (i.e. sum(xi)=1)
    OptimConstraints = []
    # lower and upper bounds for optimization variables and for optimization constraints in OptimConstraints
    LowerBoundVariables = []
    UpperBoundVariables = []
    LowerBoundConstraints = []
    UppperBoundConstraints = []
    #if the user has passed exact inputs
    if exactinputs:
        #get names, number and indices of exact inputs
        ExactInputNames = exactinputs['Inputs']
        ExactInputNum = len(ExactInputNames)
        ExactInputIndices = [InputDict[e] for e in ExactInputNames]
        #add these to the total input number check for this group
        InputNumCheck = InputNumCheck + ExactInputNum
        #if no bounds passed for exact inputs, throw error, if not get the exact input bounds
        if not('Bounds' in exactinputs):
            raise Exception('Exact inputs have no bounds!')
        ExactInputBounds=exactinputs['Bounds']
        #if the number of bounds don't match the exact names, throw error
        if not(ExactInputNum == len(ExactInputBounds)):
            raise Exception('There are '+str(len(ExactInputNames))+' exact inputs listed, but there are '+str(len(ExactInputBounds))+' bounds, these must match!')
        #if structure for exact inputs is not provided throw error, else get
        if not('Structure' in exactinputs):
            raise Exception('No exact input structure was provided!')
        ExactInputStructure = exactinputs['Structure']
        #create a list of dicts for tracking existing symbols 
        ExistingExactSymbols = []
        #add a dictionary to the list for each exact input
        for i in range(ExactInputNum):
            ExistingExactSymbols.append({})
        #create a list to store casadi optimization symbols for exact input archetypes
        ExactSymbolArchetypes=[]
        #loop over exact input structure and create archetype symbol list for exact inputs provided
        for i in range(len(ExactInputStructure)):
            ExactStructRow=ExactInputStructure[i]
            CurrentArchetype=[]
            CurrentArchOptimIndices=[]
            if not(len(ExactStructRow) == ExactInputNum):
                raise Exception('Row number '+str(i)+' in the exact structure passed has a length of '+str(len(ExactStructRow))+' but should be '+str(ExactInputNum)+' long, with an element for each exact input!')
            for j in range(len(ExactStructRow)):
                Keyword=ExactStructRow[j]
                #check if this keyword has been seen before NOTE: should really only restrict the use to unique keyworks per column of ExactInputStructure, otherwise we can get bad behaviour
                if Keyword in ExistingExactSymbols[j]:
                    #if the keyword has been seen then a symbol already exists, add the casadi symbol to the current archetype list
                    CurrentArchetype.append(ExistingExactSymbols[j][Keyword])
                    #add the index within OptimSymbolList corresponding to the existing symbol to the current archetype's index list
                    CurrentArchOptimIndices.append(Keyword_To_OptimIndex[Keyword])
                else:
                    #if the keyword is new, a casadi symbol does not exist, create optimizaton symbols for the corresponding exact input
                    NewExactSymbol = cs.MX.sym('ExactSym_'+ExactInputNames[j]+'_entry'+str(i)+'_elmnt'+ str(j))
                    #now add the new casadi symbol to the current archetype list
                    CurrentArchetype.append(NewExactSymbol)
                    #add new keyword-symbol pair to the existing symbol dictionary
                    ExistingExactSymbols[j][Keyword] = NewExactSymbol
                    #add new keyword to the map so we can find the optimization index that matches that keyword
                    Keyword_To_OptimIndex[Keyword]=len(OptimSymbolList)
                    #add the index within OptimSymbolList corresponding to the new symbol to the current archetype's index list
                    CurrentArchOptimIndices.append(len(OptimSymbolList))
                    #add exact symbols for this group to optimzation symbol list
                    OptimSymbolList += [NewExactSymbol]
                    #get the current bounds
                    lb = ExactInputBounds[j][0]
                    ub = ExactInputBounds[j][1]
                    #set the exact input start value randomly within bounds and add it to start value list
                    OptimVariableStart += [rn.uniform(lb,ub)]
                    #get the upper and lower bound and add them to the opt var bound lists
                    LowerBoundVariables += [lb]
                    UpperBoundVariables += [ub]
            #add the current archetype list to the list of archetype lists
            ExactSymbolArchetypes.append(CurrentArchetype)
            #append the current archetype's index list to the ArchetypeIndex_To_OptimIndices list within the optimal solution map
            ArchetypeIndex_To_OptimIndices.append(CurrentArchOptimIndices) 
            NumExactArchetypes=len(ExactSymbolArchetypes)
    else:
        NumExactArchetypes=1 #NOTE: this is ugly, but needed for now so that while loops and weight initialization works out if exact isn't passed
        ExactInputIndices=[]

    #check if total inputs passed, exact + approx, is equal to total model inputs, if not throw error
    if not(InputNumCheck == InputDim):
        raise Exception('All input variables must be passed as either approximate or exact, and the total number of inputs passed must be the same as recieved by the model(s)!\n'
                        'There are '+str(ApproxInputNum)+' approximate inputs and '+str(ExactInputNum)+' exact inputs passed but model(s) expect '+str(InputDim)+'!')
    
    #check if observ passed 
    if not( observgroups):
        observgroups={}
        observgroups['Observations']=[[o] for o in list(ObservDict.keys())]
    #    raise Exception('No observation dictionary has been passed!')
    #check if observation groups have been passed, if not throw error, if so get
    if not('Observations' in  observgroups):
        raise Exception('No observation field was passed!')
    ObservGroups =  observgroups['Observations']
    NumObservGroups=len(ObservGroups)
    #list for observation group indices in the models
    ObservGroupIndices = []
    for i in range(NumObservGroups):
        CurrentGroupNames = ObservGroups[i]
        #NOTE: need to add check that names exist here
        #lookup the indices for the y variable names
        CurrentIndices = [ObservDict[n] for n in CurrentGroupNames] 
        ObservGroupIndices.append(CurrentIndices)
    

    # declare sum for approximate weights
    ApproxWeightSum = 0
    #set up loop counters
    i=0
    # loop over exact symbol archetypes, or if exact wasn't passed then enter the loop only once
    while i < NumExactArchetypes or (not(exactinputs) and i==0):
        j=0
        # loop over approximate grid, or if approx wasn't passed then enter the loop only once
        while j < NumApproxGrid or (not( approxinputs) and j==0):
            #create a list to hold the current input to the model
            InputList = []
            #labels each input as exact 'E' or approximate 'A', for use in OptimSolutionMap
            CurrentInputTypeLabels=[]
            #stores either the grid index (for approx inputs) or the opimal vector index (for exact inputs)
            CurrentInputLookupIndices=[]
            #loop over model inputs
            for k in range(InputDim):
                #check if input index is in approximate or exact inputs
                if k in ApproxInputIndices:
                    #if in index corresponds to an approximate input add the appropriate numerical grid values to the input vector
                    InputList.append(cs.MX(ApproxInputGrid[j][ApproxInputIndices.index(k)]))
                    CurrentInputTypeLabels.append('A')
                    CurrentInputLookupIndices.append(ApproxInputIndices.index(k))
                elif k in ExactInputIndices:
                    #if in index corresponds to an exact input add the appropriate symbol to the input vector
                    InputList.append(ExactSymbolArchetypes[i][ExactInputIndices.index(k)])
                    CurrentInputTypeLabels.append('E')
                    CurrentInputLookupIndices.append(ArchetypeIndex_To_OptimIndices[i][k])
                else:
                    #if we can't find the index k in either the approximate or exact indices, throw an error
                    raise Exception('Model input with index'+str(k)+' does not match any inputs passed as approximate or exact!')
            #concatinate input list into a single MX
            InputVector=cs.horzcat(*InputList)
            #loop over the obeservation groups
            for k in range(NumObservGroups):
                obs=ObservGroupIndices[k]
                #create a sampling weight symbol for the current input and add it to the optimization variable list
                NewSampleWeight = cs.MX.sym('SampWeight_'+ str(i)+'_'+ str(j)+'_'+ str(k))
                #label the current index in OptimSymbolList as a sample weight in the optimization variable map
                List_Of_SampleWeightOptimIndices.append(len(OptimSymbolList))
                #add an entry mapping the current index in OptimSymbolList to the index of the current exact symbol archetype and approximate grid entry
                OptimSolutionMap[len(OptimSymbolList)]={'InputType':CurrentInputTypeLabels,'InputLookUpIndex':CurrentInputLookupIndices,'GridIndex':j,'ObsGroupIndices':obs} 
                #add sample weight symbol to the optimization symbol list
                OptimSymbolList += [NewSampleWeight]
                #set the starting weights so that all grid points at all archetypes have equal weights NOTE: should probably clean this up, conditional on approx and exact being passed
                OptimVariableStart += [1/(NumApproxGrid*NumExactArchetypes*NumObservGroups)]
                #apply appropriate bounds for the sampling weight NOTE: upper bound 1 should be scaled by passed observation weights when this functionality is added
                LowerBoundVariables += [0]
                UpperBoundVariables += [1]
                #add sampling weight symbol to the running total, used to constrain sum of sampleing weights to 1 latee
                ApproxWeightSum = ApproxWeightSum+NewSampleWeight
                #get the length of the observation group
                # this is used to scale sampling weight so FIM stays normalized w.r.t. sample number
                N=len(obs)
                #loop over observatino variables in sample group
                for var in obs:
                    #loop over each model
                    for mod in range(NumModels):
                        #get the model 
                        Model = models[mod]['Model']
                        #NOTE: Bayesian sigma point loop goes here
                        #get the model's parameter symbols
                        Params = ParamList[mod]
                        #add the weighted FIM to the running total for the experiment (for each model)
                        FIMList[mod]= FIMList[mod] + (NewSampleWeight / N) * Model.FIM[var](Params,InputVector)

            j+=1
        i+=1

    #add a constraint function to ensure sample weights sum to 1
    OptimConstraints += [ApproxWeightSum - 1]
    #bound the constrain function output to 0
    LowerBoundConstraints += [0]
    UppperBoundConstraints += [0]

    OverallObjectiveSymbol=0
    for m in range(NumModels): 
         OverallObjectiveSymbol += ObjectiveFuncs[m](FIMList[m])/NumModels

    # Create an IPOPT solver
    IPOPTProblemStructure = {'f': OverallObjectiveSymbol, 'x': cs.vertcat(*OptimSymbolList), 'g': cs.vertcat(*OptimConstraints)}
    print('Setting up optimization problem, this can take some time...')
    #"verbose":True,
    IPOPTSolver = cs.nlpsol('solver', 'ipopt', IPOPTProblemStructure,{'ipopt.hessian_approximation':'limited-memory'}) #NOTE: need to give option to turn off full hessian (or coloring), may need to restucture problem mx/sx, maybe use quadratic programming in full approx mode?
    print('Problem set up complete.')
    # Solve the NLP with IPOPT call
    print('Begining optimization...')
    IPOPTSolutionStruct = IPOPTSolver(x0=OptimVariableStart, lbx=LowerBoundVariables, ubx=UpperBoundVariables, lbg=LowerBoundConstraints, ubg=UppperBoundConstraints)
    OptimSolution = IPOPTSolutionStruct['x'].full().flatten()

    Tol=1e-4 #NOTE: this should probably be a function of the numeber of parameters and maybe the bayesian/model mixture inputs
    #get the samle weights indices that have a non-trivial non-zero value
    NonZeroWeightOptimIndices=[i for i in List_Of_SampleWeightOptimIndices if OptimSolution[i]>Tol]

    #create the design dictionary
    Design={'InputNames':InputNameList,'ObservationNames':ObservNameList,'Inputs':[],'Weight':[]}
    #loop over the non-zero sample weights
    for i in NonZeroWeightOptimIndices:
        #get relevant information for looking up input values and observation indices from OptimSolutionMap
        CurrentInputTypeList = OptimSolutionMap[i]['InputType']
        CurrentInputLookupIndices = OptimSolutionMap[i]['InputLookUpIndex']
        CurrentApproxGridIndex = OptimSolutionMap[i]['GridIndex']
        CurrentObservGroupIndices = OptimSolutionMap[i]['ObsGroupIndices']
        #create a list to store the potential new input vector
        NewInputRow = []
        #loop over input indices to the full model
        for k in range(InputDim):
            #get the current type 'A' approx, 'E' exact
            CurrentType=CurrentInputTypeList[k]
            #check if input is approximate or exact
            if CurrentType =='A':
                #if approx, look up the approxgrid location and the input index within the grid and add to new row
                NewInputRow.append(round(ApproxInputGrid[CurrentApproxGridIndex][CurrentInputLookupIndices[k]],4))
            elif CurrentType =='E':
                #if exact, look up the optimal solution of the given exact input and add to the new row
                NewInputRow.append(round(OptimSolution[CurrentInputLookupIndices[k]],4))
            else:
                #this should never be reached
                raise Exception('Error formating the experimental design! Contact the developers for further support')

        #check if the input row is unique (i.e. if not alread inserted into the design)
        InputUnique=not(NewInputRow in Design['Inputs'])

        #get the number of observations in the observation group associated with this non-zero sample weight
        Nobs=len(CurrentObservGroupIndices)
        #create a new observation row
        NewObservRow=[]
        #loop over the observation variables #NOTE: this loop could use some clean up, if statments and loop interaction with unique flag are sloppy
        for k in range(ObservDim):
            #check the current index of the observation variables is in the current observation group
            if k in CurrentObservGroupIndices:
                #if the index is in the group, add the optimal sampling weight, paritioned according to the number of observation variables in the group
                NewWeight = round(OptimSolution[i]/Nobs,4)
            else: 
                #if the observation index is not in the group than that variable gets a zero weight
                NewWeight = 0.0
            #check if new input row was unique
            if InputUnique:
                #if unique add the new weight to 
                NewObservRow.append(NewWeight)
            else:
                #if not unique, add the weights to the corresponding existing input
                ExistingIndex = Design['Inputs'].index(NewInputRow)
                Design['Weight'][ExistingIndex][k]=Design['Weight'][ExistingIndex][k] + NewWeight
        
        #if the input was unique add the new input row and new input weights, sort while adding to provide a standardized order to the design
        if InputUnique:
            InsertIndex=sortinputs(NewInputRow,Design['Inputs'])
            Design['Inputs'].insert(InsertIndex,NewInputRow)
            Design['Weight'].insert(InsertIndex,NewObservRow)

    return Design

#Function that recursively builds a grid point list from a set of candidate levels of the provided inputs
def creategrid(inputcandidates,constraints):
    NewGrid=[]
    CurrentDim=inputcandidates.pop()

    if len(inputcandidates)>0:
        CurrentGrid=creategrid(inputcandidates,constraints)
        for g in CurrentGrid:
            for d in CurrentDim:
                TempGridPoint=g.copy()
                TempGridPoint.append(d)
                NewGrid.extend([TempGridPoint])
    else:
        NewGrid=[[d] for d in CurrentDim]

    return NewGrid

#Function uses recursion to find the insertion index to ensure sorted designs
def sortinputs(newrow,rows,rowpntr=0,colpntr=0):
    if not(len(rows)==0) and colpntr<len(rows[0]):
        i=rowpntr
        while i<len(rows):
            if newrow[colpntr]>rows[i][colpntr]:
                rowpntr+=1
            elif newrow[colpntr]==rows[i][colpntr]:
                rowpntr=max(rowpntr,sortinputs(newrow,rows,rowpntr,colpntr+1))
            i+=1
    return rowpntr



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
