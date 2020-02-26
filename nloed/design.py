import casadi as cs
import numpy as np
import random as rn

def design(models, exact=None, approx=None, observ=None, fixed=None):

    #NOTE: should we avoid 'Input' as it makes error strings awkward when talking about function inputs?!
    #NOTE: should bayesian priors be treated as symbolics in group loop, and loop over sigma points done just before ipopt pass
    #NOTE: OR should sigma points be generatedin inital model loop as numbers, and looped over within group loop, with FIMList being sigmaXmodels in size
    #NOTE: leaning towards latter at least initially
    #NOTE: current structure makes grid refinments difficult
    #NOTE: is exact vs approx in each group too flexible??
    #NOTE: can't have a common exact for all groups, i.e. initial conditions??
    #NOTE: maybe we should have exact constants for all groups, and then exact vs approximate but common to all groups
    #NOTE: fixed (observations), data or past fim (as function of beta???) Probably just pass in design/data, fim comp for data will ignore obseved y info anyways for asympotitic fim
    #NOTE:models must have the same x dim and input names, not same parameters though

    if not(approx) and not(exact):
        raise Exception('The design function requires at least one of the approximate or exact values to be passed!')

    # fim list for each model, keeps a running sum of symbols for each model (and prior parameter evaluation)
    FIMList = []
    # List of beta symbols for each model NOTE: need either symbols and build funcs at end for sigma point
    # or create sigma points first and nest loop within model loop for fim, with fim list sigmaXmodels big
    ParamList = []
    #list for casadi objective functions for each model, as matrix dim.s change, we need to define at run-time
    ObjectiveFuncs = []
    #set common dimensions for all model
    InputDim = models[0]['Model'].Nx
    ObservDim = models[0]['Model'].Ny
    #set common dicts for inputs and observations, NOTE: could later allow for different orderings
    InputDict = models[0]['Model'].xdict
    ObservDict = models[0]['Model'].ydict
    #loop over models check dimensions and dicts, build objective functions, create fim and beta lists
    for m in range(len(models)):
        model = models[m]['Model']
        if not(ObservDim == model.Ny):
            raise Exception('All model output dimensions must match!')
        if not(InputDim == model.Nx ):
            raise Exception('All model input dimensions must match!')
        if not(InputDict == model.xdict):
            raise Exception('Model input name and ordering must match!')
        if not(ObservDict == model.ydict):
            raise Exception('Model output name and ordering must match!')
        if not('Objective' in models[m]):
            raise Exception('Missing objective for model'+str(m)+'!')

        #NOTE:model D score must be weighted/rooted log-divided according to number of params
        if models[m]['Objective']=='D':
            Matrx = cs.SX.sym('Matrx',model.Nb, model.Nb)
            RFact = cs.qr(Matrx)[1]
            NormalizedLogDet = cs.trace(cs.log(RFact))/model.Nb
            ObjectiveFuncs.append( cs.Function('ObjFunc'+str(m),[Matrx],[-NormalizedLogDet]) )
        elif models[m]['Objective'] == 'Ds':
            if not('POI' in models[m]):
                raise Exception('No parameters of interest provided for Ds design objective! Need list of parameter-of-interest names!')
            poi=models[m]['POI']
            #need to write this
            #ObjectiveFuncs.append( cs.Function('ObjFunc'+str(m),[M],[-logdet]) )
        elif models[m]['Objective'] == 'T':
            i=0
            #add for model difference, need to flag this and switch to computing output rather than fim
        elif models[m]['Objective'] == 'Custom':
            i=0
            #add for custom function of FIM
        else:
            raise Exception('Unknown objective: '+str(models[m]['Objective'])+'!')

        #create the fim list for summing fim symbolics in group loop and parameter symbols for each model 
        #ParamList used for bayesian priors
        #NOTE:should maybe be an 'output' list for model selection objective; T-optimality etc.
        FIMList.append(np.zeros((model.Nb,model.Nb) ))
        ParamList.append(cs.MX.sym('beta_model'+str(m),model.Nb))

    #counter to track the total number inputs across exact and approx, must sum to total for the model(s)
    InputNumCheck=0
    #if user has passed approximate inputs
    if approx:
        #get names, number and indices of approximate inputs
        ApproxInputNames = approx['Inputs']
        ApproxInputNum = len(ApproxInputNames)
        ApproxInputIndices = [InputDict[a] for a in ApproxInputNames] 
        #add approx inputs to the total input number (used to check all inputs are accounted for after loading exact)
        InputNumCheck = InputNumCheck + ApproxInputNum
        #check if approximate bounds have been passed, if not throw error, if so get them
        if not('Bounds' in approx):
            raise Exception('Approximate inputs have no bounds!')
        ApproxInputBounds = approx['Bounds']
        #check if we have bounds for each approx input
        if not(ApproxInputNum == len(ApproxInputBounds)):
            raise Exception('There are '+str(len(ApproxInputNames))+' approximate inputs listed, but there are '+str(len(ApproxInputBounds))+' bounds, these must match!')
        #check if inquality OptimConstraintsains have been passed, if so store them 
        ApproxInputConstr = []
        if 'Constraints' in approx:
            ApproxInputConstr  = approx['Constraints']          
        #set resolution of grid NOTE: this should be able to be specified by the user, will change
        N = 5
        #create a list for storing possible levels of each approxmate input
        ApproxInputLists = []
        #loop over bounds passed, and use resolution and bounds to populate xlist
        for b in ApproxInputBounds:
            ApproxInputLists.extend([np.linspace(b[0],b[1],N).tolist()])
        #call recursive createGrid function to generate ApproxInputGrid, a list of all possible permuations of xlist's that also satisfy inequality OptimConstraintsaints
        #NOTE: currently, createGrid doesn't actually check the inequality OptimConstraintsaints, need to add, and perhaps add points on inequality boundary??!
        ApproxInputGrid = creategrid(ApproxInputLists,ApproxInputConstr)

    #dictionary tracking where xi's and exact vars are in opt list, for repackaging output for return
    #NOTE:also useful for setting intial xi's (b/c we don't have total grid size until after group loop)
    OptimVariableMap={}

    # list of optimization variables (exact input settings and approximate weights), and a list of starting values
    OptimVariableList = []
    OptVar_start = []
    # list of Casadi expressions for (non-)linear ineqaulity constraints on exact settings (e.g. g(X)>0), and linear constraints on approx problem (i.e. sum(xi)=1)
    OptimConstraints = []
    # lower and upper bounds for optimization variables and for optimization constraints in OptimConstraints
    LowerBoundVariables = []
    UpperBoundVariables = []
    LowerBoundConstraints = []
    UppperBoundConstraints = []
    #if the user has passed exact inputs
    if exact:
        #get names, number and indices of exact inputs
        ExactInputNames = exact['Inputs']
        ExactInputNum = len(ExactInputNames)
        ExactInputIndices = [InputDict[e] for e in ExactInputNames]
        #add these to the total input number check for this group
        InputNumCheck = InputNumCheck + ExactInputNum
        #if no bounds passed for exact inputs, throw error, if not get the exact input bounds
        if not('Bounds' in exact):
            raise Exception('Exact inputs have no bounds!')
        ExactInputBounds=obs['Bounds']
        #if the number of bounds don't match the exact names, throw error
        if not(ExactInputNum == len(eXbounds)):
            raise Exception('There are '+str(len(ExactInputNames))+' exact inputs listed, but there are '+str(len(eXbounds))+' bounds, these must match!')
        #if structure for exact inputs is not provided throw error, else get
        if not('Structure' in exact):
            raise Exception('No exact input structure was provided!')
        ExactInputStructure = exact['Structure']
        #create a list of dicts for tracking existing symbols NOTE: do this with list comprehension???
        ExistingExactSymbols = []
        #add a dictionary to the list for each exact input
        for i in range(ExactInputNum):
            ExistingExactSymbols.append({})
        #create a list to store casadi optimization symbols for exact input archetypes
        ExactSymbolArchetypes=[]
        #loop over exact input structure and create archetype symbol list for exact inputs provided
        for i in range(len(ExactInputStructure)):
            Entry=ExactInputStructure[i]
            Archetype=[]
            if not(len(Entry) == ExactInputNum):
                raise Exception('Entry number '+str(i)+' in the exact structure passed has a length of '+str(len(Entry))+' but should be '+str(ExactInputNum)+' long, with an element for each exact input!')
            for j in range(len(entry)):
                Element=entry[j]
                if Element in ExistingExactSymbols[j]:
                    Archetype.append(ExistingExactSymbols[j][Element])
                else:
                    #create optimizaton symbols for the exact variables in this 
                    NewExactSymbol = cs.MX.sym('ExactSym_entry'+str(i)+'_elmnt'+ str(j))
                    #add new symbol to the existing symbol dictionary
                    ExistingExactSymbols[j][Element] = NewExactSymbol
                    #add exact symbols for this group to optimzation var list
                    OptimVariableList+=NewExactSymbol
                    #get the current bounds
                    lb = ExactInputBounds[j][0]
                    ub = ExactInputBounds[j][1]
                    #set the exact input start value randomly within bounds and add it to start value list
                    OptimVariableStart += rn.uniform(lb,ub)
                    #get the upper and lower bound and add them to the opt var bound lists
                    LowerBoundVariables += [lb]
                    UpperBoundVariables += [ub]
            ExistingExactSymbols.append(Archetype)
                    

    #check if total inputs passed, exact + approx, is equal to total model inputs, if not throw error
    if not(InputNumCheck == InputDim):
        raise Exception('All input variables must be passed as either approximate or exact, and the total number of inputs passed must be the same as recieved by the model(s)!\n'
                        'There are '+str(ApproxInputNum)+' approximate inputs and '+str(ExactInputNum)+' exact inputs passed but model(s) expect '+str(InputDim)'!')
    
    #NOTE: add checks for observ, with weights loop
    # if not("Weight" in observ):
    #        raise Exception('A observation weighting must be provided for every observation variable or for none!')

    # declare sum for approximate weights
    ApproxWeightSum=0
    # group loop, build optimization symbolics for each group (all models)
    for i in range(Ngroups):
        obs=ostruct[o]

        # get oobervation variable names for this group
        Ynames=obs['Group']
        #lookup the indices for the y variable names
        Yindex=[ObservDict[y] for y in Ynames] 

            if not(weightFlag):
                xi_sum=0
            for k in range(len(ApproxInputGrid)):
                xi_k=cs.MX.sym('xi_'+ str(o)+'_'+ str(k))
                LowerBoundVariables += [0]
                UpperBoundVariables += [1]
                OptimVariableList += [xi_k]
                if weightFlag:
                    wght=obs['Weight']
                    OptimVariableStart +=  [wght/k]
                else:
                    OptimVariableStart +=  [1]

                xvec=[]
                for i in range(InputDim):
                    if i in ApproxInputIndices:
                        xvec.append(ApproxInputGrid[k][ApproxInputIndices.index(i)])
                    elif i in ExactInputIndices:
                        xvec.append(eXsym[ExactInputIndices.index(i)])
                    else:
                        raise Exception('Cannot find !'+str(i)+' model input in approximate or exact inputs provided!')

                for m in range(len(models)):
                    model=models[m]['Model']
                    betasym=ParamList[m]
                    for y in Yindex:
                        FIMList[m]= FIMList[m] + xi_k*model.fim[y](betasym[m],xvec)
                xi_sum=xi_sum+xi_k

            if weightFlag:
                wght=obs['Weight']
                OptimConstraints += [xi_sum - wght]
                LowerBoundConstraints += [0]
                UppperBoundConstraints += [0]

        else:

            if not(ExactInputNum==InputDim):
                raise Exception('All input variables must be passed as either approximate or exact and total to match the number of inputs for the models! There are '+str(ApproxInputNum+ExactInputNum)+' inputs passed but '+str(InputDim)+' expected!')
            
            if not(weightFlag):
                xi_sum=0
                xi_k=cs.MX.sym('xi_'+ str(o))
                LowerBoundVariables += [0]
                UpperBoundVariables += [1]
                OptimVariableList += [xi_k]
                OptimVariableStart += [1/Ngroups]
            else:
                wght=obs['Weight']
                xi_k=wght

            xvec=[]
            for i in range(InputDim):
                if i in ExactInputIndices:
                    xvec.append(eXsym[ExactInputIndices.index(i)])
                else:
                    raise Exception('Cannot find !'+str(i)+' model input in approximate or exact inputs provided!')

            for m in range(len(models)):
                model=models[m]['Model']
                betasym=ParamList[m]
                for y in Yindex:
                    FIMList[m]= FIMList[m] + xi_k*model.fim[y](betasym[m],xvec)
            xi_sum=xi_sum+xi_k

    if not(weightFlag):
        OptimConstraints+=[xi_sum-wght]
        LowerBoundConstraints += [0]
        UppperBoundConstraints += [0]

    #NOTE: bayesiam stuff, with casadi functions, tricky with optvar_input list, might remove
    ObjParameterFuncs=[]
    ObjTot=0
    for m in range(len(models)): 
        model=models[m]['Model']

        ObjSymbol=ObjectiveFuncs[m](FIMList[m])
        ObjParameterFuncs.append( cs.Function('ObjParFunc'+str(m),[ParamList[m]+OptimVariableList],[ObjSymbol]) )
        #loop over bayesian prior here
        # for p in parameters
        beta0=models[m]['Beta']
        wght=models[m]['Weight']
        ObjTot=ObjTot+wght*ObjParameterFuncs[m](beta0,OptimVariableList)

    # Create an NLP solver
    prob = {'f': ObjTot, 'x': cs.vertcat(*OptimVariableList), 'g': cs.vertcat(*OptimConstraints)}
    solver = cs.nlpsol('solver', 'ipopt', prob)

    # Solve the NLP
    sol = solver(x0=OptimVariableStart, lbx=LowerBoundVariables, ubx=UpperBoundVariables, lbg=LowerBoundConstraints, ubg=UppperBoundConstraints)
    opt_output = sol['x'].full().flatten()

    #HERE: code to unpack opt_output into a design matrix using labels dictionar
    #will be a loop over groups and support points with nontivial weight (xi>n*eps??, xi>tol)

    return opt_output


def creategrid(xlists,aXOptimConstraintsaints):

    #print('prepop len xlist: '+str(len(xlists)))

    newgrid=[]
    currentdim=xlists.pop()

    if len(xlists)>0:
        currentgrid=createGrid(xlists,aXOptimConstraintsaints)
        #print('current grid: '+str(currentgrid))
        #print('currentdim'+str(currentdim))
        for g in currentgrid:
            #print('g: '+str(g))
            for d in currentdim:
                #print('d: '+str(d))
                temp=g.copy()
                temp.append(d)
                #print('g'+str(temp))
                newgrid.extend([temp])
    else:
        newgrid=[[d] for d in currentdim]

    return newgrid



#Should this be a class?? would make rounding easier

# rough specs
# inputs: model list ,grouping matrix, beta info, past data
# return: ys, xs, weights (not ready for actual experiment because you need to do power/rounding analysis)


# Goals
# can enter either past data or maybe fisher info from past data as starting point to improve an experiment
# grouping response variables that need to be sampled together, paritioning overall sample number across groups
#       (ie 80% and 20%, or if 0% will find optimal weights, perhaps fast weight to specifiy equal weights)
# having x split into two types, grided values with weights xi vs continuous ranges which are constant within a group
# OptimConstraintsaints on x, linear and nonlinear
# grid refinment for grided x, starting values for non-grided
# optional optimality criteria, D, Ds (need to group parameters), A, custom (but simple casadi function of fisher info)
# otimality measures for curvature/bias measure, also OptimConstraintsained measure (maybe not possible with bayesian???)
# average designs for multiple models, with priors over the models
# bayesian (normal prior) using sigma points for all or subset of the parameters
# possible support for rounding design into an experiment


# Questions
# Do we scale the FIM?, only D maybe Ds is invariant under linear re-scaling (maybe all rescaling), but not others or custom, however is more numerically stable
# Also do we log the opt(FIM)? for numerical stability, this maybe generally useful
# how to select sigma points
# how to do OptimConstraintsained and curvature/bias optimality measures
