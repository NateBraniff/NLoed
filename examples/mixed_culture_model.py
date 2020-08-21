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
y = cs.SX.sym('y',4)
# parameters
p = cs.SX.sym('p',8)

#symbolic RHS
rhs = cs.vertcat(y[0]*( cs.exp(p[0])*(y[2]/(cs.exp(p[1])+y[2])) - cs.exp(p[2])*y[1] ),
                 cs.exp(p[3])*(y[3]/(cs.exp(p[4])+y[3]))*y[1],
                 -cs.exp(p[5])*(y[2]/(cs.exp(p[1])+y[2]))*y[0],
                 y[1]*( cs.exp(p[6])*y[0] - cs.exp(p[7])*(y[3]/(cs.exp(p[4])+y[3])) ) )
#casadi RHS function
ode = cs.Function('ode',[y,p],[rhs])

####################################################################################################
# DEFINE RK-4 STEP ALGEBRA
####################################################################################################

#time step size
dt = 0.01
# Create symbolics for RK4 integration, as shown in Casadi examples
k1 = ode(y, p)
k2 = ode(y + dt/2.0*k1, p)
k3 = ode(y + dt/2.0*k2, p)
k4 = ode(y + dt*k3, p)
y_step = y + dt/6.0 * (k1 + 2*k2 + 2*k3 + k4)
# Create a function to perform one step of the RK integration
step = cs.Function('step',[y, p],[y_step])

####################################################################################################
# DEFINE TIME DISCRETIZATION, AND CONSTRUCT DETERMINISTIC MODEL VIA INTEGRATION
####################################################################################################

#define a sample pattern to apply in each control interval
sample_pattern= [500,500,500,500,500] 

# IC symbols (considered as controls here)
u = cs.SX.sym('u',2)
y0 = cs.vertcat(u,cs.SX([5,0]))

# create list to store symbols for each sample point
sample_list=[]
# create list to store times of each sample (for naming observ. vars)
times=[]

#Loop over the ste function to create symbolics for integrating across a sample interval
y_sample = y0
cntr=0
# loop over sample pattern
for num_stps in sample_pattern:
    #iterate steps indicated by pattern
    for k in range(num_stps):
        #propogate the state variables
        y_sample = step(y_sample, p)
        cntr+=1
    #save the state symbols and times of each sample
    sample_list.append(y_sample)
    times.append(cntr)

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
    ecoli_name = 'EColi_'+'t'+"{0:0=2d}".format(times[i])
    salm_name = 'Salmonella_'+'t'+"{0:0=2d}".format(times[i])
    primary_name = 'PrimarySubstrate_'+'t'+"{0:0=2d}".format(times[i])
    secondary_name = 'SecondarySubstrate_'+'t'+"{0:0=2d}".format(times[i])
    #create mean and var tuple for mrna and prot observ.
    ecoli_stats = cs.vertcat(sample_list[i][0], 0.001)
    salm_stats = cs.vertcat(sample_list[i][1], 0.001)
    primary_stats = cs.vertcat(sample_list[i][2], 0.001)
    secondary_stats = cs.vertcat(sample_list[i][3], 0.001)
    #create casadi function for mrna and prot stats
    ecoli_func = cs.Function(ecoli_name,[u,p],[ecoli_stats])
    salm_func = cs.Function(salm_name,[u,p],[salm_stats])
    primary_func = cs.Function(primary_name,[u,p],[primary_stats])
    secondary_func = cs.Function(secondary_name,[u,p],[secondary_stats])
    #append the casadi function and distribution type to obs struct
    ode_response.append((ecoli_func,'Normal'))
    ode_response.append((salm_func,'Normal'))
    ode_response.append((primary_func,'Normal'))
    ode_response.append((secondary_func,'Normal'))
    #store response names for plotting
    response_names.append(ecoli_name)
    response_names.append(salm_name)
    response_names.append(primary_name)
    response_names.append(secondary_name)

#xnames = ['ecoli_ic','salm_ic','prim_ic','sec_ic']
xnames = ['ecoli_ic','salm_ic']
pnames = ['lambda_E','K_L','delta_E','lambda_S','K_G','alpha_E','alpha_G','alpha_S']

ode_model = Model(ode_response,xnames,pnames)

####################################################################################################
# GENERATE DESIGN
####################################################################################################

true_pars = np.log([0.5,0.01,0.1,0.4,0.01,0.2,0.1,0.1])

# continuous_inputs = {'Inputs':['ecoli_ic','salm_ic','prim_ic','sec_ic'],
#                      'Bounds':[(0.1,1),(0.1,0.2),(1,5),(0.01,0.1)],
#                      'Structure':[['ecoli_ic1','salm_ic1','prim_ic1','sec_ic1'],
#                                   ['ecoli_ic2','salm_ic2','prim_ic2','sec_ic2'],
#                                   ['ecoli_ic3','salm_ic3','prim_ic3','sec_ic3']]}
# continuous_inputs = {'Inputs':['ecoli_ic','salm_ic'],
#                      'Bounds':[(0.01,0.1),(0.1,0.1)],
#                      'Structure':[['ecoli_ic1','salm_ic1'],
#                                   ['ecoli_ic2','salm_ic2'],
#                                   ['ecoli_ic3','salm_ic3']]}
discrete_inputs = {'Inputs':['ecoli_ic','salm_ic'],
                     'Bounds':[(0.01,0.1),(0.01,0.1)]}
opt_design = Design(ode_model, true_pars, 'D', discrete_inputs=discrete_inputs)

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
             'InitParamBounds':[(-5,1)]*8,
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
predict_inputs = pd.DataFrame({ 'ecoli_ic':[0.1]*len(response_names),
                                'salm_ic':[0.01]*len(response_names),
                                'prim_ic':[5]*len(response_names),
                                'sec_ic':[0.001]*len(response_names),
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
digit_re=re.compile('[a-z]+_t(\d+)')
type_re=re.compile('([a-z]+)_t\d+')

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


