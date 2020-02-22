import casadi as cs
import numpy as np

def design(models, ostruct, data):

    #NOTE: should bayesian priors be treated as symbolics in group loop, and loop over sigma points done just before ipopt pass
    #NOTE: OR should sigma points be generatedin inital model loop as numbers, and looped over within group loop, with fimlist being sigmaXmodels in size
    #NOTE: leaning towards latter at least initially
    #NOTE: current structure makes grid refinments difficult
    #NOTE: is exact vs approx in each group too flexible??
    #NOTE: can't have a common exact for all groups, i.e. initial conditions??
    #NOTE: maybe we should have exact constants for all groups, and then exact vs approximate but common to all groups
    #NOTE:data or past fim (as function of beta???) Probably just pass in design/data, fim comp for data will ignore obseved y info anyways for asympotitic fim
    #NOTE:models must have the same x dim and input names, not same parameters though

    # list of optimization varibales, containing exact input settings and approximate input weights for each obs. group
    optvar_list=[]
    # starting values for optimization variables, exact settings are passed, weights are automated
    optvar_start=[]
    # (non-)linear ineqaulity constraints g(X)>0 on exact settings and approx problem (i.e. sum(xi)=1)
    constr=[]
    # lower and upper bounds for optimization variables
    lowerbound = []
    upperbound = []
    # lower and upper bounds for constraints in constr, user passed constraints must be g(X)>0
    lowerboundconstr = []
    upperboundconstr = []
    # fim list for each model, keeps a running sum of symbols for each model across all groups and y's
    fimlist=[]
    # List of beta symbols for each model NOTE: need either symbols and build funcs at end for sigma point
    # or create sigma points first and nest loop within model loop for fim, with fim list sigmaXmodels big
    betalist=[]
    #dictionary tracking where xi's and exact vars are in opt list, for repackaging output for return
    #NOTE:also useful for setting intial xi's (b/c we don't have total grid size until after group loop)
    labels={}

    #list for casadi objective functions for each model, user could provide, matrix dim changes so need to define
    #at run time
    ObjectiveFuncs=[]
    #Common dimensions for all model, also used to check other passed models
    NxAll=models[0]['Model'].Nx
    NyAll=models[0]['Model'].Ny
    #Common dicts for inputs and response, (could later allow for different orderings)
    xDictAll=models[0]['Model'].xdict
    yDictAll=models[0]['Model'].ydict
    #loop over models check dimensions and dicts, build objective functions, create fim and beta lists
    for m in range(len(models)):
        model=models[m]['Model']
        if not(NyAll==model.Ny):
            raise Exception('All model output dimensions must match!')
        if not(NxAll==model.Nx ):
            raise Exception('All model input dimensions must match!')
        if not(xDictAll==model.xdict):
            raise Exception('Model input name and ordering must match!')
        if not(yDictAll==model.ydict):
            raise Exception('Model output name and ordering must match!')

        if not('Objective' in models[m]):
            raise Exception('Missing objective for model'+str(m)+'!')

        #NOTE:model D score must be weighted/rooted log-divided according to number of params
        if models[m]['Objective']=='D':
            M = cs.SX.sym('M',model.Nb, model.Nb)
            R = cs.qr(M)[1]
            normlogdet = cs.trace(cs.log(R))/model.Nb
            ObjectiveFuncs.append( cs.Function('ObjFunc'+str(m),[M],[-normlogdet]) )
        elif models[m]['Objective']=='Ds':
            if not('POI' in models[m]):
                raise Exception('No parameters of interest provided for Ds design objective! Need list of parameter-of-interest names!')
            poi=models[m]['POI']
            #need to write this
            #ObjectiveFuncs.append( cs.Function('ObjFunc'+str(m),[M],[-logdet]) )
        elif models[m]['Objective']=='T':
            i=0
            #add for model difference, need to flag this and switch to computing output rather than fim
        elif models[m]['Objective']=='Custom':
            i=0
            #add for custom function of FIM
        else:
            raise Exception('Unknown objective: '+str(models[m]['Objective'])+'!')

        #create the fim list for summing fim symbolics in group loop and parameter symbols for each model 
        #betalist used for bayesian priors
        #NOTE:should maybe be an 'output' list for model selection objective
        fimlist.append(np.zeros((model.Nb,model.Nb) ))
        betalist.append(cs.MX.sym('beta_model'+str(m),model.Nb))

    #store number of groups passed
    Ngroups=len(ostruct)
    #declare sum for xi weights, will reset if weight's passed, if not is for all groups
    xi_sum=0
    #boolean for if weighting is passed, for logic within group loop
    if 'Weight' in ostruct[0]:
        weightFlag=True
    else:
        weightFlag=False
    #group loop, build optimization symbolics for each group (all models)
    for o in range(Ngroups):
        obs=ostruct[o]
        #counter to track the total number inputs across exact and approx, must sum to total for the model(s)
        Nxcheck=0
        #check that group has basic requirments
        #must have a 'Group' field (for obs vars) NOTE: should add more checks here, i.e. string list ??
        if not('Group' in obs):
            raise Exception('No observation variable group passed!')
        #must have approximate or exact inputs NOTE: add more checks, i.e. string list ??
        if not('eX' in obs or 'aX' in obs):
            raise Exception('A design requires inputs to be identified as exact (\'ex\') or approximate (\'aX\')!, neither were provided!')
        #check that all groups have weights if one does, or vise versa NOTE
        if not("Weight" in obs == weightFlag):
            raise Exception('A sample weight must be provided for every observation group or for none!')

        # get oobervation variable names for this group
        Ynames=obs['Group']
        #lookup the indices for the y variable names
        Yindex=[yDictAll[y] for y in Ynames] 

        #if the user has specified some inputs to be treated exactly in this group
        if 'eX' in obs: 
            #get names of inputs to be treated exactly
            eXnames=obs['eX']
            #get total number of exact inputs in this group
            NeX=len(eXnames)
            #add these to the total input number check for this group
            Nxcheck=Nxcheck+NeX
            #get the model input indices for the given exact input names
            eXindex=[xDictAll[e] for e in eXnames]
            #create optimizaton symbols for the exact variables in this 
            eXsym=cs.MX.sym('eXsym'+ str(o), NeX)
            #add exact symbols for this group to optimzation var list
            optvar_list+=eXsym
            #check if use passed exact start values, if not throw error
            if not(eXstart in obs):
                raise Exception('No starting values for exact inputs was passed!')
            #get the exact start value and add them to start value list
            eXstart=obs['eXstart']
            optvar_start+=eXstart
            #if no bounds passed for exact variables, throw error
            if not('eXbounds' in obs):
                raise Exception('Exact inputs have no bounds!')
            #get the exact bounds list for this group
            eXbounds=obs['eXbounds']
            #if num. bounds don't match the exact names, throw error
            if not(NeX==len(eXbounds)):
                raise Exception('There are '+str(len(eXnames))+' exact inputs listed, but there are '+str(len(eXbounds))+' bounds, these must match!')
            #get the upper and lower bound and add them to the opt var bound lists
            for b in eXbounds:
                lowerbound += [b[0]]
                upperbound += [b[1]]
            #if exact inequality constrains have been passed, get them
            if 'eXconstraints' in obs:
                eXconstraints=obs['eXconstraints']
                #loop over them and add them to constraint list for optimization sovler
                #NOTE: should add a check to see if contraints only depend on X and inputs match NeX
                for eXcon in eXconstraints:
                    constr+=eXcon(eXsym)
                    #restrict inequality constraints so that g(X)>0
                    lowerboundconstr += [0]
                    upperboundconstr += [cs.inf]

        #if user has passed approximate input field
        if 'aX' in obs:
            #get names of approximate inputs
            aXnames=obs['aX']
            #get the number of approximate inputs
            NaX=len(aXnames)
            #add the number to the total number of inputs (exact and approx) for current group
            Nxcheck=Nxcheck+NaX
            #get the model input indices for the specified approximate input names
            aXindex=[xDictAll[a] for a in aXnames] 
            #check if approximate bounds have been passed, if not throw error
            if not('aXbounds' in obs):
                raise Exception('Approximate inputs have no bounds!')
            #get approximate bounds list for this group
            aXbounds=obs['aXbounds']
            if not(NaX==len(aXbounds)):
                raise Exception('There are '+str(len(aXnames))+' approximate inputs listed, but there are '+str(len(aXbounds))+' bounds, these must match!')
            #check if total inputs passed exact+approx is equal to total model inputs, if not throw error
            if not(Nxcheck==NxAll):
                raise Exception('All input variables must be passed as either approximate or exact and total to match the number of inputs for the models! There are '+str(NaX+NeX)+' inputs passed but '+str(NxAll)+' expected!')
            #check if inquality constrains have been passed, if so store them NOTE: maybe don't need this if block
            if 'aXconstraints' in obs:
                aXconstraints=obs['aXconstraints']
            else:
                aXconstraints=()
            #set resolution of grid NOTE: this should be able to be specified by the user, will change
            N=10
            #create a list for storing lists of each approxmate input vars, potential values, perimuations between which creates the xgrid
            xlists=[]
            #loop over bounds passed, and use resolution and bounds to populate xlist
            for b in aXbounds:
                xlists.extend([np.linspace(b[0],b[1],N).tolist()])
            #call recursive createGrid function to generate xgrid, a list of all possible permuations of xlist's that also satisfy inequality constraints
            #NOTE: currently, createGrid doesn't actually check the inequality constrains, need to add, and perhaps add points on inequality boundary??!
            xgrid=createGrid(xlists,aXconstraints)

            if not(weightFlag):
                xi_sum=0
            for k in range(len(xgrid)):
                xi_k=cs.MX.sym('xi_'+ str(o)+'_'+ str(k))
                lowerbound += [0]
                upperbound += [1]
                optvar_list += [xi_k]
                if weightFlag:
                    wght=obs['Weight']
                    optvar_start +=  [wght/k]
                else:
                    optvar_start +=  [1]

                xvec=[]
                for i in range(NxAll):
                    if i in aXindex:
                        xvec.append(xgrid[k][aXindex.index(i)])
                    elif i in eXindex:
                        xvec.append(eXsym[eXindex.index(i)])
                    else:
                        raise Exception('Cannot find !'+str(i)+' model input in approximate or exact inputs provided!')

                for m in range(len(models)):
                    model=models[m]['Model']
                    betasym=betalist[m]
                    for y in Yindex:
                        fimlist[m]= fimlist[m] + xi_k*model.fim[y](betasym[m],xvec)
                xi_sum=xi_sum+xi_k

            if weightFlag:
                wght=obs['Weight']
                constr += [xi_sum - wght]
                lowerboundconstr += [0]
                upperboundconstr += [0]

        else:

            if not(NeX==NxAll):
                raise Exception('All input variables must be passed as either approximate or exact and total to match the number of inputs for the models! There are '+str(NaX+NeX)+' inputs passed but '+str(NxAll)+' expected!')
            
            if not(weightFlag):
                xi_sum=0
                xi_k=cs.MX.sym('xi_'+ str(o))
                lowerbound += [0]
                upperbound += [1]
                optvar_list += [xi_k]
                optvar_start += [1/Ngroups]
            else:
                wght=obs['Weight']
                xi_k=wght

            xvec=[]
            for i in range(NxAll):
                if i in eXindex:
                    xvec.append(eXsym[eXindex.index(i)])
                else:
                    raise Exception('Cannot find !'+str(i)+' model input in approximate or exact inputs provided!')

            for m in range(len(models)):
                model=models[m]['Model']
                betasym=betalist[m]
                for y in Yindex:
                    fimlist[m]= fimlist[m] + xi_k*model.fim[y](betasym[m],xvec)
            xi_sum=xi_sum+xi_k

    if not(weightFlag):
        constr+=[xi_sum-wght]
        lowerboundconstr += [0]
        upperboundconstr += [0]

    #NOTE: bayesiam stiff, with casadi functions, tricky with optvar_input list, might remove
    ObjParameterFuncs=[]
    ObjTot=0
    for m in range(len(models)): 
        model=models[m]['Model']

        ObjSymbol=ObjectiveFuncs[m](fimlist[m])
        ObjParameterFuncs.append( cs.Function('ObjParFunc'+str(m),[betalist[m]+optvar_list],[ObjSymbol]) )
        #loop over bayesian prior here
        # for p in parameters
        beta0=models[m]['Beta']
        wght=models[m]['Weight']
        ObjTot=ObjTot+wght*ObjParameterFuncs[m](beta0,optvar_list)

    # Create an NLP solver
    prob = {'f': ObjTot, 'x': cs.vertcat(*optvar_list), 'g': cs.vertcat(*constr)}
    solver = cs.nlpsol('solver', 'ipopt', prob)

    # Solve the NLP
    sol = solver(x0=optvar_start, lbx=lowerbound, ubx=upperbound, lbg=lowerboundconstr, ubg=upperboundconstr)
    opt_output = sol['x'].full().flatten()

    #HERE: code to unpack opt_output into a design matrix using labels dictionar
    #will be a loop over groups and support points with nontivial weight (xi>n*eps??, xi>tol)

    return opt_output


def createGrid(xlists,aXconstraints):

    #print('prepop len xlist: '+str(len(xlists)))

    newgrid=[]
    currentdim=xlists.pop()

    if len(xlists)>0:
        currentgrid=createGrid(xlists,aXconstraints)
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
# constraints on x, linear and nonlinear
# grid refinment for grided x, starting values for non-grided
# optional optimality criteria, D, Ds (need to group parameters), A, custom (but simple casadi function of fisher info)
# otimality measures for curvature/bias measure, also constrained measure (maybe not possible with bayesian???)
# average designs for multiple models, with priors over the models
# bayesian (normal prior) using sigma points for all or subset of the parameters
# possible support for rounding design into an experiment


# Questions
# Do we scale the FIM?, only D maybe Ds is invariant under linear re-scaling (maybe all rescaling), but not others or custom, however is more numerically stable
# Also do we log the opt(FIM)? for numerical stability, this maybe generally useful
# how to select sigma points
# how to do constrained and curvature/bias optimality measures
