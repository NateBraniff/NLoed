""" 
Add a docstring
"""
import numpy as np
import casadi as cs
from nloed.model import Model
from nloed import design
import time



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
linear1d=Model(response,xnames,betanames)

pred_struct1=linear1d.eval_model([1,1],1)
pred_struct2=linear1d.eval_model([1,1],1,param_covariance=[[1,0],[0,1]])
pred_struct3=linear1d.eval_model([1,1],1,sensitivity=True)
pred_struct4=linear1d.eval_model([1,1],1,[[1,0],[0,1]],True,options={'ErrorMethod':'Delta'})
pred_struct5=linear1d.eval_model([1,1],1,[[1,0],[0,1]],True,options={'ErrorMethod':'MonetCarlo'})

Experiment={}
Experiment['InputNames']=xnames
Experiment['ObservationNames']=['y1']
Experiment['Inputs']=[[0],[1],[2]]
Experiment['Count']=[[5],[1],[5]]
dataset1=linear1d.sample(Experiment,[0,1],5)



start = time.time()
parsfits=linear1d.fit(dataset1,[0,1],options={'Confidence':'Contours'})
end = time.time()
print(end - start)

pars=parsfits[0]

paramCov1=np.cov(pars,rowvar=False)
paramMean1=np.mean(pars,axis=0)

dataset2=linear1d.sample([Experiment,Experiment],[0,1])

modelinfo={'Model':linear1d, 'Parameters': [0.5, 1],'Objective':'D'}
approx={'Inputs':['x1'],'Bounds':[(-1,1)]}
obs={'Observations':[['y1']]}

opt_approx=design([modelinfo],approxinputs=approx,observgroups=obs)
print(opt_approx)

opt_approx=design([modelinfo],approx)
print(opt_approx)

modelinfo={'Model':linear1d, 'Parameters': [0.5, 1],'Objective':'D'}
struct=[['x1_lvl1'],['x1_lvl2'],['x1_lvl3']]
exact={'Inputs':['x1'],'Bounds':[(-1,1)],'Structure':struct}
obs={'Observations':[['y1']]}

opt_exact=design([modelinfo],exactinputs=exact,observgroups=obs)
print(opt_exact)

opt_exact=design([modelinfo],exactinputs=exact)
print(opt_exact)

print('Done')

