import casadi as cs
import numpy as np

def design(models, ostruct, data):

    #data or past fim (as function of beta???) Probably just pass in design/data, fim comp for data will ignore obseved y info anyways for asympotitic fim

    #models must have the same x domain and exact names, not same parameters though
    #model D score must be weighted/rooted log-divided according to number of params

    optvar_list=[]
    optvar_start=[]
    constr=[]
    lowerbound = []
    upperbound = []
    lowerboundconstr = []
    upperboundconstr = []

    fimlist=[]
    betalist=[]

    label=[]

    Nxcheck=0
    ObjectiveFuncs=[]
    NxAll=models[0]['Model'].Nx
    NyAll=models[0]['Model'].Ny
    xDictAll=models[0]['Model'].xdict
    yDictAll=models[0]['Model'].ydict
    for m in range(len(models)):
        model=models[m]['Model']
        if not(NyAll==model.Ny):
            raise Exception('Model output dimensions must match!')
        if not(NxAll==model.Nx ):
            raise Exception('All model input dimensions must match!')
        if not(xDictAll==model.xdict):
            raise Exception('Model input name and ordering must match!')
        if not(yDictAll==model.ydict):
            raise Exception('Model output name and ordering must match!')

        if not('Objective' in models[m]):
            raise Exception('Missing objective for model'+str(m)+'!')

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

        #should maybe output list for model selection objective
        fimlist.append(np.zeros((model.Nb,model.Nb) ))
        betalist.append(cs.MX.sym('beta_model'+str(m),model.Nb))

    xi_sum=0
    if 'Weight' in ostruct[0]:
        weightFlag=True
    else:
        weightFlag=False

    Ngroups=len(ostruct)

    for o in range(Ngroups):
        obs=ostruct[o]

        if not('Group' in obs):
            raise Exception('No observation variable group passed!')
        if not('eX' in obs or 'aX' in obs):
            raise Exception('A design requires inputs to be identified as exact (\'ex\') or approximate (\'aX\')!, neither were provided!')
        if not("Weight" in obs == weightFlag):
            raise Exception('A sample weight must be provided for every observation group or none!')

        Ynames=obs['Group']
        Yindex=[models[0]['Model'].ydict[y] for y in Ynames] #hack, should maybe creat shared name list for all models after check
        
        if 'eX' in obs: 
            if not('eXbounds' in obs):
                raise Exception('Exact inputs have no bounds!')
            eXnames=obs['eX']
            NeX=len(eXnames)
            Nxcheck=Nxcheck+NeX
            eXindex=[m.xdict[e] for e in eXnames]
            eXsym=cs.MX.sym('eXsym'+ str(o), NeX)
            optvar_list+=eXsym

            if not(eXstart in obs):
                raise Exception('No starting values for exact inputs was passed!')
            eXstart=obs['eXstart']
            optvar_start+=eXstart

            eXbounds=obs['eXbounds']
            if not(NeX==len(eXbounds)):
                raise Exception('There are '+str(len(eXnames))+' exact inputs listed, but there are '+str(len(eXbounds))+' bounds, these must match!')
            
            
            for b in eXbounds:
                lowerbound += [b[0]]
                upperbound += [b[1]]

            if 'eXconstraints' in obs:
                eXconstraints=obs['eXconstraints']
                for eXcon in eXconstraints:
                    constr+=eXcon(eXsym)
                    lowerboundconstr += [0]
                    upperboundconstr += [cs.inf]

        if 'aX' in obs:
            if not('aXbounds' in obs):
                raise Exception('Approximate inputs have no bounds!')
            aXnames=obs['aX']
            NaX=len(aXnames)
            Nxcheck=Nxcheck+NaX
            aXindex=[models[0]['Model'].xdict[a] for a in aXnames] #hack, should have shared model list of inputs and outputs
            aXbounds=obs['aXbounds']
            if not(NaX==len(aXbounds)):
                raise Exception('There are '+str(len(aXnames))+' approximate inputs listed, but there are '+str(len(aXbounds))+' bounds, these must match!')
            
            if not(Nxcheck==NxAll):
                raise Exception('All input variables must be passed as either approximate or exact and total to match the number of inputs for the models! There are '+str(NaX+NeX)+' inputs passed but '+str(NxAll)+' expected!')
            if 'aXconstraints' in obs:
                aXconstraints=obs['aXconstraints']
            else:
                aXconstraints=()

            N=10
            xlists=[]
            for b in aXbounds:
                xlists.extend([np.linspace(b[0],b[1],N).tolist()])
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
    sol = solver(x0=xi0, lbx=lowerbound, ubx=upperbound, lbg=lowerboundconstr, ubg=upperboundconstr)
    opt_output = sol['x'].full().flatten()

    return opt_output
            #for x in xgrid:

# build dict that maps grid point to x values, extend on grid refinment


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







#### Old stuff
    # nparameters=model.beta.size1()

    # xgrid=np.linspace(xbounds[0],xbounds[1],101)

    # xi0=np.ones(np.shape(xgrid))/xgrid.size
    # xi_list=[]
    # xi_sum=0

    # fim_sum=np.zeros((nparameters,nparameters) )

    # constr=[]
    # lowerboundxi = []
    # upperboundxi = []
    # lowerboundconstr = []
    # upperboundconstr = []

    # for k in range(xgrid.size):

    #     xi_k=cs.MX.sym('xi_'+ str(k))
    #     lowerboundxi += [0]
    #     upperboundxi += [1]
    #     xi_list += [xi_k]
            
    #     fim_sum= fim_sum + xi_k*model.fim[0](beta0,xgrid[k])
    #     xi_sum=xi_sum+xi_k

    # constr+=[xi_sum-1]
    # lowerboundconstr += [0]
    # upperboundconstr += [0]
    
    # M = cs.SX.sym('M',nparameters, nparameters)
    # R = cs.qr(M)[1]
    # det = cs.exp(cs.trace(cs.log(R)))
    # qrdeterminant = cs.Function('qrdeterminant',[M],[-det])

    # #objective deff (include ds as well)
    # #objective=-cs.log(cs.det(fim_sum))
    # objective=qrdeterminant(fim_sum)

    # # Create an NLP solver
    # prob = {'f': objective, 'x': cs.vertcat(*xi_list), 'g': cs.vertcat(*constr)}
    # solver = cs.nlpsol('solver', 'ipopt', prob)

    # # Solve the NLP
    # sol = solver(x0=xi0, lbx=lowerboundxi, ubx=upperboundxi, lbg=lowerboundconstr, ubg=upperboundconstr)
    # xi_opt = sol['x'].full().flatten()

    # return xi_opt


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
