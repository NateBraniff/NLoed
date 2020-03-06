""" 
Add a docstring
"""
import numpy as np
import casadi as cs
from nloed import model
from nloed import design

#Declare model inputs
x=cs.SX.sym('Xs',1)

#Declare the number of parameters-of-interest
#These are the parameters you are interested in estimating as accurately as possible (or nuisance parameters, you can decide which is which later)
beta=cs.SX.sym('Betas',2)

#Now we define how x and beta are linked to the distribution parameters, theta, of the normal response distribution
#Define a 1D linear model with slope and intercept for the mean, constant variance, 
#combine these into a function computing the response distribution parameters, theta (in this case mean and var)
theta=cs.Function('theta',[beta,x],[beta[0] + x*beta[1], 0.01])

#Enter the above model into the list of reponse variables
response= [('y1','Normal',theta)]
xnames=['x1']
betanames=['beta0','beta1']

#Instantiate class
linear1d=model(response,xnames,betanames)

Experiment={}
Experiment['InputNames']=xnames
Experiment['ObservationNames']=['y1']
Experiment['Inputs']=[[0],[1],[2]]
Experiment['Count']=[[5],[1],[5]]
dataset1=linear1d.sample(Experiment,[0,1],1)

pars=linear1d.fit(dataset1,[0,1],opts={'Confidence':'Profiles'})

paramCov1=np.cov(pars,rowvar=False)
paramMean1=np.mean(pars,axis=0)

dataset2=linear1d.sample([Experiment,Experiment],[0,1])

model={'Model':linear1d, 'Parameters': [0.5, 1],'Objective':'D'}
approx={'Inputs':['x1'],'Bounds':[(-1,1)]}
obs={'Observations':[['y1']]}

opt_approx=design([model],approxinputs=approx,observgroups=obs)
print(opt_approx)

opt_approx=design([model],approx)
print(opt_approx)

model={'Model':linear1d, 'Parameters': [0.5, 1],'Objective':'D'}
struct=[['x1_lvl1'],['x1_lvl2'],['x1_lvl3']]
exact={'Inputs':['x1'],'Bounds':[(-1,1)],'Structure':struct}
obs={'Observations':[['y1']]}

opt_exact=design([model],exactinputs=exact,observgroups=obs)
print(opt_exact)

opt_exact=design([model],exactinputs=exact)
print(opt_exact)

