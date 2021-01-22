""" 
Add a docstring
"""
import numpy as np
import pandas as pd
import casadi as cs
from nloed import Model
from nloed import Design

import re as re
import copy as cp
import matplotlib.pyplot as plt

####################################################################################################
# SET UP RHS OF ODE
####################################################################################################

# state variales
y = cs.SX.sym('y',2)
# control inputs
u = cs.SX.sym('u')
# parameters
p = cs.SX.sym('p',4)

#symbolic RHS
rhs = cs.vertcat(cs.exp(p[0])*u -cs.exp(p[1])*y[0], cs.exp(p[2])*y[0]-cs.exp(p[3])*y[1])
#casadi RHS function
ode = cs.Function('ode',[y,u,p],[rhs])

####################################################################################################
# DEFINE RK-4 STEP ALGEBRA
####################################################################################################

#time step size
dt = 1
# Create symbolics for RK4 integration, as shown in Casadi examples
k1 = ode(y, u, p)
k2 = ode(y + dt/2.0*k1, u, p)
k3 = ode(y + dt/2.0*k2, u, p)
k4 = ode(y + dt*k3, u, p)
y_step = y + dt/6.0 * (k1 + 2*k2 + 2*k3 + k4)
# Create a function to perform one step of the RK integration
step = cs.Function('step',[y, u, p],[y_step])

####################################################################################################
# DEFINE TIME DISCRETIZATION, AND CONSTRUCT DETERMINISTIC MODEL VIA INTEGRATION
####################################################################################################

#set number of control intervals
num_cntrl_intervals = 3
#define a sample pattern to apply in each control interval
sample_pattern= [1,2,3] 

#  cntrl_int1    cntrl_int2    cntrl_int3
#|------------||------------||------------|
#|-1---2-----3||-1---2-----3||-1---2-----3|
#repeated sampling pattern (1+2+3=6 steps long)
#total of 3*6

# IC symbols (considered as controls here)
y0 = cs.SX.sym('y0',2)
# control values (one per control interval)
uvec = cs.SX.sym('uvec',3)
# control values (one per control interval)
x = cs.vertcat(y0,uvec)

# create list to store symbols for each sample point
sample_list=[]
# create list to store times of each sample (for naming observ. vars)
times=[]

#Loop over the ste function to create symbolics for integrating across a sample interval
y_sample = y0
cntr=0
#loop over control invervals
for i in range(num_cntrl_intervals):
  # loop over sample pattern
  for num_stps in sample_pattern:
    #iterate steps indicated by pattern
    for k in range(num_stps):
      #propogate the state variables
      y_sample = step(y_sample, uvec[i], p)
      cntr+=1
    #save the state symbols and times of each sample
    sample_list.append(y_sample)
    times.append(cntr*dt)

####################################################################################################
# CREATE NLOED MODEL
####################################################################################################

# create list for observation structure 
ode_response = []
#create list to store response names
response_names=[]
# loop over samples (time points)
for i in range(len(sample_list)):
  #create a unique name for mrna and prot samples
  mrna_name = 'mrna_'+'t'+"{0:0=2d}".format(times[i])
  prot_name = 'prot_'+'t'+"{0:0=2d}".format(times[i])
  #create mean and var tuple for mrna and prot observ.
  mrna_stats = cs.vertcat(sample_list[i][0], 0.001)
  prot_stats = cs.vertcat(sample_list[i][1], 0.001)
  #create casadi function for mrna and prot stats
  mrna_func = cs.Function(mrna_name,[x,p],[mrna_stats])
  prot_func = cs.Function(prot_name,[x,p],[prot_stats])
  #append the casadi function and distribution type to obs struct
  ode_response.append((mrna_func,'Normal'))
  ode_response.append((prot_func,'Normal'))
  #store response names for plotting
  response_names.append(mrna_name)
  response_names.append(prot_name)

xnames = ['mrna_ic','prot_ic','cntrl_1','cntrl_2','cntrl_3']
pnames = ['alpha','delta','beta','gamma']

ode_model = Model(ode_response,xnames,pnames)

####################################################################################################
# GENERATE DESIGN
####################################################################################################

true_pars = np.log([0.5,1.1,2.1,0.3])

# discrete_inputs = {'Inputs':['mrna_ic','prot_ic','cntrl_1','cntrl_2','cntrl_3'],
#                  'Bounds':[(0,1),(0,1),(0,1),(0,1),(0,1)]}
# opt_design = Design(ode_model,true_pars,'D',discrete_inputs)
continuous_inputs = {'Inputs':['mrna_ic','prot_ic','cntrl_1','cntrl_2','cntrl_3'],
                     'Bounds':[(0,1),(0,1),(0,1),(0,1),(0,1),(0,1)],
                     'Structure':[['mrna_ic1','prot_ic1','c1_lvl1','c2_lvl1','c3_lvl1'],
                                  ['mrna_ic2','prot_ic2','c1_lvl2','c2_lvl2','c3_lvl2'],
                                  ['mrna_ic3','prot_ic3','c1_lvl3','c2_lvl3','c3_lvl3']]}
opt_design = Design(ode_model, true_pars, 'D', continuous_inputs=continuous_inputs)

sample_size = 30
exact_design = opt_design.round(sample_size)

print(exact_design)

####################################################################################################
# SAMPLE DESIGN AND FIT
####################################################################################################

#evaluate the proposed deign in terms of expected Covariance, Bias and MSE
opts = {'Covariance':True,'Bias':True,'MSE':True}
eval_dat = ode_model.evaluate(exact_design,true_pars,opts)
print(eval_dat)

#generate some data, to stand in for an initial experiment
data = ode_model.sample(exact_design,true_pars)
print(data)

#pass some additional options to fitting alglorithm, including Profile likelihood
fit_options={'Confidence':'Profiles',  #NOTE: ISSUE WITH PROFILES HERE
             'InitParamBounds':[(-2,2)]*4,
             'InitSearchNumber':3,
             'SearchBound':5.,
             'MaxSteps':100000}
#fit the model to the init data
fit_info = ode_model.fit(data, options=fit_options)
print(fit_info)

#extract fitted parameters
fit_params = fit_info['Estimate'].to_numpy().flatten()

####################################################################################################
# TESTING PREDICTIONS AND EVAL
####################################################################################################

# create data frame of inputs that need predictions
predict_inputs = pd.DataFrame({ 'mrna_ic':[1]*len(response_names),
                                'prot_ic':[1]*len(response_names),
                                'cntrl_1':[0.1]*len(response_names),
                                'cntrl_2':[1.0]*len(response_names),
                                'cntrl_3':[0.1]*len(response_names),
                                'Variable':response_names})

predict_design = cp.deepcopy(predict_inputs)
predict_design['Replicats'] = [5]*len(response_names)

#generate the predictions at fit parameters
predictions = ode_model.predict(predict_inputs,fit_params)
print(predictions)

#generate data at true parameters
plot_data = ode_model.sample(predict_design,true_pars)
print(plot_data)

# create regular rexpressions (re package) search strings
digit_re = re.compile('[a-z]+_t(\d+)')
type_re = re.compile('([a-z]+)_t\d+')

# use regular rexpressions (re package) to parse observations into times and rna/prot
plot_data['Time'] = plot_data['Variable'].apply(lambda x: int(digit_re.search(x).group(1)))
plot_data['Type'] = plot_data['Variable'].apply(lambda x: type_re.search(x).group(1))
#organize the data for plotting
mrna_obs = plot_data.loc[plot_data['Type'] == 'mrna']
prot_obs = plot_data.loc[plot_data['Type'] == 'prot']
#plot true data
ax1 = mrna_obs.plot.scatter(x=('Time'),y=('Observation'),c='Blue')
ax2 = prot_obs.plot.scatter(x=('Time'),y=('Observation'),c='Blue')

# use regular rexpressions (re package) to parse predictions into times and rna/prot
predictions['Prediction','Time'] = predictions['Inputs','Variable'].apply(lambda x: int(digit_re.search(x).group(1)))
predictions['Prediction','Type'] = predictions['Inputs','Variable'].apply(lambda x: type_re.search(x).group(1))
#organize the data for plotting
mrna_pred = predictions.loc[predictions['Prediction','Type'] == 'mrna']
prot_pred = predictions.loc[predictions['Prediction','Type'] == 'prot']
#plot the predicted mean values for the model at fit params
mrna_pred.plot.scatter(x=('Prediction','Time'),y=('Prediction','Mean'),c='Red',ax=ax1)
prot_pred.plot.scatter(x=('Prediction','Time'),y=('Prediction','Mean'),c='Red',ax=ax2)

plt.show()

t=0

####################################################################################################
# MANUALLY CREATE A DESIGN FOR TESTING
####################################################################################################

# design = pd.DataFrame({ 'mrna_ic':[1],
#                         'prot_ic':[1],
#                         'cntrl_1':[0.1],
#                         'cntrl_2':[1],
#                         'cntrl_3':[0.1]})

# design=design.reindex(design.index.repeat(len(response_names)))
# design['Variable'] = response_names
# design['Replicats'] = replicates
# design = design.sort_values(by='Variable').reset_index()

# fit_pars = mixed_fit['Estimate'].to_numpy().flatten()
# print(fit_pars)

# #evaluate goes here

# predict_inputs = pd.DataFrame({ 'x1':[-1,-1,-1,0,0,0,1,1,1]*3,
#                                 'x2':[-1,0,1,-1,0,1,-1,0,1]*3,
#                                 'Variable':['y_norm']*9+['y_bern']*9+['y_pois']*9})

# cov_mat = np.diag([0.1,0.1,0.1,0.1])
# pred_options = {'Method':'MonteCarlo',
#                 'PredictionInterval':True,
#                 'ObservationInterval':True,
#                 'Sensitivity':True}
# predictions_dlta = mixed_model.predict(predict_inputs,fit_pars,covariance_matrix = cov_mat,options=pred_options)


