import casadi as cs
import numpy as np

def design(models, ostruct, data):

    #data or past fim (as function of beta???)

    #models must have the same x domain and exact names, not same parameters though
    #model D score must be weighted/rooted log-divided according to number of params

    optvar_list=[]
    constr=[]
    lowerbound = []
    upperbound = []
    lowerboundconstr = []
    upperboundconstr = []

    fimlist=[]
    betalist=[]
    Nxcheck=models[0].Nx
    Nycheck=models[0].Ny
    for m in range(len(models)):
        model=models[m]['Model']
        if not(Nxcheck==model.Nx and Nycheck==model.Ny):
            raise Exception('All model input and output dimensions must match!')
        fimlist.append(np.zeros((model.Nb,model.Nb) ))
        betalist.append(cs.MX.sym('beta_'+str(m)))

    for o in range(len(ostruct)):
        obs=ostruct[o]

        if not('Groups' in obs):
            raise Exception('No observations variable group passed!')
        Ynames=obs['Groups']
        Yindex=[m.ydict[y] for y in Ynames]

        if not('eX' in obs or 'aX' in obs):
            raise Exception('A design requires inputs to be assigned as exact (\'ex\') or approximate (\'aX\')!, neither were provided!')
        #need check for sum of ax and ex len equal to total x

        if 'eX' in obs: 
            if not('eXbounds' in obs):
                raise Exception('Exact inputs have no bounds!')
            eXnames=obs['eX']
            eXindex=[m.xdict[e] for e in eXnames]
            eXbounds=obs['eXbounds']
            if not(len(eXnames)==len(eXbounds)):
                raise Exception('There are '+str(len(eXnames))+' exact inputs listed, but there are '+str(len(eXbounds))+' bounds, these must match!')
            NeX=len(eXnames)
            eXsym=cs.MX.sym('eXsym'+ str(o), NeX)
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
            aXindex=[m.xdict[a] for a in aXnames]
            aXbounds=obs['aXbounds']
            if not(len(aXnames)==len(aXbounds)):
                raise Exception('There are '+str(len(aXnames))+' approximate inputs listed, but there are '+str(len(aXbounds))+' bounds, these must match!')
            NeX=len(eXnames)
            if 'aXconstraints' in obs:
                aXconstraints=obs['aXconstraints']
            else:
                aXconstraints=()

            N=5
            xlists=[]
            for b in aXbounds:
                xlists.extend([np.linspace(b[0],b[1],N).tolist()])
                print(xlists)
            #print('enter grid func')
            xgrid=createGrid(xlists,aXconstraints)
    
            for k in range(xgrid.size):
                xi_k=cs.MX.sym('xi_'+ str(m)+'_'+ str(k))
                lowerbound += [0]
                upperbound += [1]
                optvar_list += [xi_k]
                for m in range(len(models)):
                    model=m[m]['Model']
                    betasym=betalist[m]
                    for y in Yindex:
                        fimlist[m]= fimlist[m] + xi_k*model.fim[y](betasym,xgrid[k])
                xi_sum=xi_sum+xi_k

            #only if we are capping weight within group, else its all together and constrains added at the end
            constr+=[xi_sum-1]
            lowerboundconstr += [0]
            upperboundconstr += [0]

        else:
            #account for FIM when no approximate aX and grid
            i=0


    # NEED one of these for each parameter set, bit awkward
    # M = cs.SX.sym('M',nparameters, nparameters)
    # R = cs.qr(M)[1]
    # det = cs.exp(cs.trace(cs.log(R)))
    # qrdeterminant = cs.Function('qrdeterminant',[M],[-det])

    # #Switch case for objective type
    # objective=qrdeterminant(fim_sum)

    # # Create an NLP solver
    # prob = {'f': objective, 'x': cs.vertcat(*xi_list), 'g': cs.vertcat(*constr)}
    # solver = cs.nlpsol('solver', 'ipopt', prob)

    # # Solve the NLP
    # sol = solver(x0=xi0, lbx=lowerboundxi, ubx=upperboundxi, lbg=lowerboundconstr, ubg=upperboundconstr)
    # xi_opt = sol['x'].full().flatten()

    xi=[]
    return xi
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
