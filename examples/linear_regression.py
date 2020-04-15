""" 
Add a docstring
"""
import numpy as np
import casadi as cs
import pandas as pd
from nloed.model import Model
from nloed import design
import time

#Declare model inputs
x=cs.SX.sym('Xs',1)

#Declare the number of parameters-of-interest
#These are the parameters you are interested in estimating as accurately as possible (or nuisance parameters, you can decide which is which later)
p=cs.SX.sym('Betas',2)

#Now we define how x and beta are linked to the distribution parameters, theta, of the normal response distribution
#Define a 1D linear model with slope and intercept for the mean, constant variance, 
#combine these into a function computing the response distribution parameters, theta (in this case mean and var)
predictor = cs.Function('theta',[p,x],[p[0] + x*p[1], 0.01])

#Enter the above model into the list of reponse variables
response= [('y1','Normal',predictor)]
input_names = ['x1']
param_names = ['p1','p2']

#Instantiate class
linear1d=Model(response,input_names,param_names)

design1 = pd.DataFrame({'x1':[0,1,2],'y1':[5,1,5]})
design2 = pd.DataFrame({'x1':[-1,0,1],'y1':[4,3,4]})

dataset_list = linear1d.sample([design1,design2],[0,1],5)

pars_info = linear1d.fit(dataset_list,[0,1],options={'Confidence':'Contours'})

inputs = design1[input_names]
par_est = pars_info['Estimate'][param_names].to_numpy()

#pred_structA=linear1d.eval_model(inputs, par_est, [[1,0],[0,1]], True,options={'ErrorMethod':'Delta'})
#pred_structB=linear1d.eval_model(inputs, par_est, [[1,0],[0,1]], True,options={'ErrorMethod':'MonteCarlo','SampleSize':10000})
# print(pred_structA['y1']['Mean']['Bounds'])
# print(pred_structB['y1']['Mean']['Bounds'])

# pred_struct4=linear1d.eval_model([1,1],[[i] for i in range(10)],[[1,0],[0,1]],True,options={'ErrorMethod':'Delta'})
# pred_struct5=linear1d.eval_model([1,1],[[i] for i in range(10)],[[1,0],[0,1]],True,options={'ErrorMethod':'MonteCarlo'})
# pred_struct1=linear1d.eval_model([1,1],1)
# pred_struct2=linear1d.eval_model([1,1],1,param_covariance=[[1,0],[0,1]])
# pred_struct3=linear1d.eval_model([1,1],1,sensitivity=True)






# paramCov1=np.cov(pars,rowvar=False)
# paramMean1=np.mean(pars,axis=0)

# dataset2=linear1d.sample([Experiment,Experiment],[0,1])

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

