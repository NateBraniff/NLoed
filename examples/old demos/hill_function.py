""" 
Add a docstring
"""
import numpy as np
import pandas as pd
import casadi as cs
from nloed import Model
from nloed import Design

####################################################################################################
# SET UP MODEL
####################################################################################################

#define input and parameters
x = cs.SX.sym('x',1)
xnames = ['Light']
p = cs.SX.sym('p',4)
pnames = ['Basal','Rate','Hill','HalfConst']

#define the deterministic model
hull_func = cs.exp(p[0]) + cs.exp(p[1])*x[0]**cs.exp(p[2])/(cs.exp(p[3])**cs.exp(p[2])+x[0]**cs.exp(p[2]))

#link the deterministic model to the sampling statistics (here normal mean and variance)
mean, var = hull_func, 0.1
normal_stats = cs.vertcat(mean, var)

#create a casadi function mapping input and parameters to sampling statistics (mean and var)
y = cs.Function('y',[x,p],[normal_stats])

# enter the function as a tuple with label indicating normal error, into observation list
observ_list = [(y,'Normal')]

#instantiate nloed model class
nloed_model = Model(observ_list,xnames,pnames)

####################################################################################################
# GENERATE OPTIMAL DESIGN
####################################################################################################

#set up the design algorithm to use continuous (continuous) optimization with two unique inputs points
continuous_inputs={'Inputs':['Light'],'Bounds':[(.1,5)],'Structure':[['x1'],['x2'],['x3'],['x4']]}

true_param = [1,5,2,1]
# generate the optimal discreteimate (relaxed) design
relaxed_design = Design(nloed_model,np.log(true_param),'D',continuous_inputs=continuous_inputs)

sample_size = 30
#generate a rounded exact design 
exact_design = relaxed_design.round(sample_size)

print(exact_design)

####################################################################################################
# GENERATE SAMPLE DATA & FIT MODEL
####################################################################################################

#generate some data, to stand in for an initial experiment
data = nloed_model.sample(exact_design,np.log(true_param))

print(data)

#pass some additional options to fitting alglorithm, including Profile likelihood
fit_options={'Confidence':'Intervals',  #NOTE: ISSUE WITH PROFILES HERE
             'InitParamBounds':[(-1,4),(-1,4),(-1,4),(-1,4)],
             'InitSearchNumber':7,
             'SearchBound':5.}
#fit the model to the init data
fit_info = nloed_model.fit(data, options=fit_options)

print(fit_info)

fit_params = fit_info['Estimate'].to_numpy().flatten()

####################################################################################################
# EVALUATE DESIGN & PREDICT OUTPUT
####################################################################################################

#get estimated covariance, bias and MSE of parameter fit (use asymptotic method here) 
opts={'Covariance':True,'Bias':True,'MSE':True,'SampleNumber':100}
diagnostic_info = nloed_model.evaluate(exact_design,fit_params,opts)
print(diagnostic_info)

#generate predictions with error bars fdor a random selection of inputs)
prediction_inputs = pd.DataFrame({ 'Light':np.random.uniform(0,5,10),
                                'Variable':['y']*10})
cov_mat = diagnostic_info['Covariance'].to_numpy()
pred_options = {'PredictionInterval':True,
                'ObservationInterval':True,
                'Sensitivity':True}
predictions = nloed_model.predict(prediction_inputs,
                                  fit_params,
                                  covariance_matrix = cov_mat,
                                  options=pred_options)
print(predictions)

t=0
