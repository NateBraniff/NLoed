""" 
Add a docstring
"""
import numpy as np
import pandas as pd
import casadi as cs
import matplotlib.pyplot as plt
from nloed import Model
from nloed import Design
####################################################################################################
# Block 1: Casadi RHS setup
####################################################################################################
# state variales
states = cs.SX.sym('states',2)
# control inputs
inducer = cs.SX.sym('inducer')
# parameters
parameters = cs.SX.sym('parameters',6)
#log-transform the parameters
alpha = cs.exp(parameters[0]) 
K = cs.exp(parameters[1]) 
delta = cs.exp(parameters[2])
beta = cs.exp(parameters[3]) 
L = cs.exp(parameters[4]) 
gamma = cs.exp(parameters[5])
#symbolic RHS
rhs = cs.vertcat(alpha*inducer/(K + inducer) - delta*states[0],
                 beta*states[0]/(L + states[0]) - gamma*states[1])
#casadi RHS function
rhs_func = cs.Function('rhs_func',[states,inducer,parameters],[rhs])

####################################################################################################
# Block 2: Integrator setup
####################################################################################################
#time step size
dt = 1
# Create symbolics for RK4 integration, as shown in Casadi examples
k1 = rhs_func(states, inducer, parameters)
k2 = rhs_func(states + dt/2.0*k1, inducer, parameters)
k3 = rhs_func(states + dt/2.0*k2, inducer, parameters)
k4 = rhs_func(states + dt*k3, inducer, parameters)
state_step = states + dt/6.0 * (k1 + 2*k2 + 2*k3 + k4)
# Create a function to perform one step of the RK integration
step_func = cs.Function('step_func',[states, inducer, parameters],[state_step])

####################################################################################################
# Block 3: define sampling and control intervals
####################################################################################################
#  cntrl_int1    cntrl_int2    cntrl_int3
#|------------||------------||------------|
#|-1---2-----3||-1---2-----3||-1---2-----3|
#repeated sampling pattern (1+2+3=6 steps long)
#total of 3*6
#set number of control intervals
num_cntrl_intervals = 3
#define a sample pattern to apply in each control interval
sample_pattern= [1,2,3] 
# define initial steady state input for ICs
initial_inducer = cs.SX.sym('init_inducer')
#define the steady-state initial states in terms of the initial inducer
init_mrna = (alpha/delta)*initial_inducer/(K+initial_inducer)
ini_prot = (beta/gamma)*init_mrna/(L+init_mrna)
# zip the initial states into a vector
initial_states = cs.vertcat(init_mrna,ini_prot)
# control values (one per control interval)
inducer_vector = cs.SX.sym('inducer_vec',3)
# experimental inputs
inputs = cs.vertcat(initial_inducer,inducer_vector)
# create list to store symbols for each sample point
sample_list=[]
# create list to store times of each sample (for naming observ. vars)
times=[]
#Loop over the ste function to create symbolics for integrating across a sample interval
state_sample = initial_states
step_counter=0
#loop over control invervals
for interval in range(num_cntrl_intervals):
  # loop over sample pattern
  for num_stps in sample_pattern:
    #iterate steps indicated by pattern
    for k in range(num_stps):
      #propogate the state variables
      state_sample = step_func(state_sample, inducer_vector[interval], parameters)
      step_counter+=1
    #save the state symbols and times of each sample
    sample_list.append(state_sample)
    times.append(step_counter*dt)

####################################################################################################
# Block 2: Model creation
####################################################################################################
# create list for observation structure 
observation_structure = []
#create list to store response names
observation_names, observation_type, observation_times = [], [], []
# loop over samples (time points)
for i in range(len(sample_list)):
  #create a unique name for mrna and prot samples
  mrna_name = 'mrna_'+'t'+"{0:0=2d}".format(times[i])
  prot_name = 'prot_'+'t'+"{0:0=2d}".format(times[i])
  #create mean and var tuple for mrna and prot observ.
  mrna_stats = cs.vertcat(sample_list[i][0], (0.1*sample_list[i][0])**2)
  prot_stats = cs.vertcat(sample_list[i][1], (0.1*sample_list[i][1])**2)
  #create casadi function for mrna and prot stats
  mrna_func = cs.Function(mrna_name,[inputs,parameters],[mrna_stats])
  prot_func = cs.Function(prot_name,[inputs,parameters],[prot_stats])
  #append the casadi function and distribution type to obs struct
  observation_structure.extend([(mrna_func,'Normal'),(prot_func,'Normal')])
  #store observation names, useful for plotting
  observation_names.extend([mrna_name,prot_name])
  #store observation type
  observation_type.extend(['RNA','Prot'])
  #store observation time
  observation_times.extend([times[i]]*2)

input_names = ['Init_Inducer','Inducer_1','Inducer_2','Inducer_3']
parameter_names = ['log_Alpha','log_K','log_Delta','log_Beta','log_L','log_Gamma']

model_object = Model(observation_structure, input_names, parameter_names)

#***hidden_start****
#generate initial dataset 
# create data frame of inputs that need predictions
init_design = pd.DataFrame({ 'Init_Inducer':[0.]*len(observation_names),
                                'Inducer_1':[1.]*len(observation_names),
                                'Inducer_2':[0.]*len(observation_names),
                                'Inducer_3':[3.]*len(observation_names),
                                'Variable':observation_names,
                                'Replicats':[1]*len(observation_names)})
true_param = np.log([2,1.5,1,3,0.75,0.5])
init_data = model_object.sample(init_design,true_param)
print('')
print(init_data)
print('')
#***hidden_end****

####################################################################################################
# Block 3: Inital data and fit
####################################################################################################
#request contours and use a simple initial search
fit_options={'Confidence':'None',
             'InitParamBounds':[(-1,2),(-1,2),(-1,2),(-1,2),(-1,2),(-1,2)],
             'InitSearchNumber':7,
             'SearchBound':5.}
#fit the model to the initial data
fit_info = model_object.fit(init_data, options=fit_options)
#extract the parameter values
fit_params = fit_info['Estimate'].to_numpy().flatten()
#***hidden_start****
print('')
print(np.exp(fit_params))
print('')
#***hidden_end****

####################################################################################################
# Block 4: CI comparison for initial data
####################################################################################################
# #***hidden_start****
# lik_bnds_array = np.hstack((fit_info['Lower'].to_numpy().T,
#                         fit_info['Estimate'].to_numpy().T,
#                         fit_info['Upper'].to_numpy().T))
# lik_bnds = pd.DataFrame(np.exp(lik_bnds_array),
#                         index=['Alpha0','Alpha','n','K'],
#                         columns=['Lower','Estimate','Upper'])
# print('')
# print(lik_bnds)
# print('')
# #***hidden_end****
#request the asymptotic covariance matrix
eval_options={'Method':'Asymptotic',
              'Covariance':True}
# call evaluate to compute the asymptotic covariance
asymptotic_covariance = model_object.evaluate(init_design,fit_params,eval_options)
#compute the asymptotic upper and lower 95% bounds
asymptotic_lower_bound = fit_params - 2*np.sqrt(np.diag(asymptotic_covariance))
asymptotic_upper_bound = fit_params + 2*np.sqrt(np.diag(asymptotic_covariance))
#***hidden_start****
fim_bnds_array = np.vstack((asymptotic_lower_bound,
                            fit_params,
                            asymptotic_upper_bound)).T
fim_bnds0 = pd.DataFrame(np.exp(fim_bnds_array),
                        index=['Alpha','K','Delta','Beta','L','Gamma'],
                        columns=['Lower','Estimate','Upper'])
print('')
print(fim_bnds0)
print('')
# #***hidden_end****
# #generate 500 simulated datasets
# mc_data = model_object.sample(init_design,fit_params,design_replicats=500)
# #use the same pre-search settings as for the initial fit
# # fit_options={'InitParamBounds':[(-1,2),(1,3),(-1,2),(-1,2)],
# #              'InitSearchNumber':7}
# #fit the 500 datasets
# mc_fits = model_object.fit(mc_data, start_param = fit_params)
# #compute the Monte Carlo covariance estimate
# mc_covariance = np.cov(np.exp(mc_fits['Estimate'].to_numpy().T))
# #compute the asymptotic upper and lower 95% bounds
# mc_lower_bound = fit_params - 2*np.sqrt(np.diag(mc_covariance))
# mc_upper_bound = fit_params + 2*np.sqrt(np.diag(mc_covariance))
# #***hidden_start****
# mc_bnds_array = np.vstack((mc_lower_bound,
#                            fit_params,
#                            mc_upper_bound)).T
# mc_bnds = pd.DataFrame(np.exp(mc_bnds_array),
#                         index=['Alpha0','Alpha','n','K'],
#                         columns=['Lower','Estimate','Upper'])
# print('')
# print(mc_bnds)
# print('')
# #***hidden_end****

####################################################################################################
# Block 5: plot model predictions and initial data
####################################################################################################
#convert the covariance matrix to numpy
covariance_matrix = asymptotic_covariance.to_numpy()
#generate predictions with error bars fdor a random selection of inputs)
prediction_inputs = pd.DataFrame({'Init_Inducer':[0.]*len(observation_names),
                                  'Inducer_1':[1.]*len(observation_names),
                                  'Inducer_2':[0.]*len(observation_names),
                                  'Inducer_3':[3.]*len(observation_names),
                                  'Variable':observation_names})
#request prediction and observation intervals
prediction_options = {'PredictionInterval':True,
                      'ObservationInterval':True}
#generate predictions and intervals
predictions = model_object.predict(prediction_inputs,
                                   fit_params,
                                   covariance_matrix = covariance_matrix,
                                   options=prediction_options)
#***hidden_start****
#partition the predictions by state type
mrna_pred = predictions.loc[[obs == 'RNA' for obs in observation_type]]
prot_pred = predictions.loc[[obs == 'Prot' for obs in observation_type]]
#partition the initial data predictions by state type
mrna_dat = init_data.loc[[obs == 'RNA' for obs in observation_type]]
prot_dat = init_data.loc[[obs == 'Prot' for obs in observation_type]]
#plot the predictions, intervals and data, two subplots; mrna and protein
fig, ax = plt.subplots(2)
obs_bars = [(mrna_pred['Prediction','Mean']  - mrna_pred['Observation','Lower']).to_list(),
            (mrna_pred['Observation','Upper'] - mrna_pred['Prediction','Mean']).to_list()]
ax[0].errorbar(times, mrna_pred['Prediction','Mean'], yerr=obs_bars, fmt='none', color='C1')
pred_bars = [(mrna_pred['Prediction','Mean']  - mrna_pred['Prediction','Lower']).to_list(),
             (mrna_pred['Prediction','Upper'] - mrna_pred['Prediction','Mean']).to_list()]
ax[0].errorbar(times, mrna_pred['Prediction','Mean'], yerr=pred_bars, fmt='.', color='C0')
ax[0].plot(times, mrna_dat['Observation'], 'x', color='C1')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('mRNA')
obs_bars = [(prot_pred['Prediction','Mean']  - prot_pred['Observation','Lower']).to_list(),
            (prot_pred['Observation','Upper'] - prot_pred['Prediction','Mean']).to_list()]
ax[1].errorbar(times, prot_pred['Prediction','Mean'], yerr=obs_bars, fmt='none', color='C1')
pred_bars = [(prot_pred['Prediction','Mean']  - prot_pred['Prediction','Lower']).to_list(),
             (prot_pred['Prediction','Upper'] - prot_pred['Prediction','Mean']).to_list()]
ax[1].errorbar(times, prot_pred['Prediction','Mean'], yerr=pred_bars, fmt='.', color='C0')
ax[1].plot(times, prot_dat['Observation'], 'x', color='C1')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Prot')
plt.show()
# #***hidden_end****

####################################################################################################
# Block 6: optimal design
####################################################################################################
#set up the design algorithm to use continuous (continuous) optimization with two unique inputs points
continuous_inputs={'Inputs':['Init_Inducer','Inducer_1','Inducer_2','Inducer_3'],
                   'Bounds':[(.01,5),(.01,5),(.01,5),(.01,5)],
                   'Structure':[['I0','I1','I2','I3']]}
#set up fixed design dictionary
fixed_dict ={'Weight':0.5, 'Design':init_design}
# generate the optimal discreteimate (relaxed) design
design_object = Design(model_object,fit_params,'D',
                       fixed_design = fixed_dict,
                       continuous_inputs = continuous_inputs)
#extract the relaxed design structure
relaxed_design = design_object.relaxed()
#***hidden_start****
print('')
print(relaxed_design)
print('')
#***hidden_end****
#set the sample size to 30
sample_size = 18
#generate a rounded exact design 
exact_design = design_object.round(sample_size)
#***hidden_start****
print('')
print(exact_design)
print('')
#***hidden_end****

####################################################################################################
# Block 7: fit combined data
####################################################################################################
#***hidden_start****
#generate some data for optimal design and combine 
optimal_data = model_object.sample(exact_design,true_param)
combined_data = pd.concat([init_data, optimal_data], ignore_index=True)
#***hidden_end****
#combine the initial and optimal design
combined_design = pd.concat([init_design, exact_design], ignore_index=True)
#request contours and use a simple initial search
fit_options={'Confidence':'None'}
#fit the model to the initial data
fit_info = model_object.fit(combined_data,start_param = fit_params, options=fit_options)
#extract the parameter values
fit_params = fit_info['Estimate'].to_numpy().flatten()
#***hidden_start****
print('')
print(np.exp(fit_params))
print('')
#***hidden_end****

####################################################################################################
# Block 8: CI comparison for initial data
####################################################################################################
#***hidden_start****
# lik_bnds_array = np.hstack((fit_info['Lower'].to_numpy().T,
#                         fit_info['Estimate'].to_numpy().T,
#                         fit_info['Upper'].to_numpy().T))
# lik_bnds = pd.DataFrame(np.exp(lik_bnds_array),
#                         index=['Alpha0','Alpha','n','K'],
#                         columns=['Lower','Estimate','Upper'])
# print('')
# print(lik_bnds)
# print('')
#***hidden_end****
#request the asymptotic covariance matrix
eval_options={'Method':'Asymptotic',
              'Covariance':True}
# call evaluate to compute the asymptotic covariance
asymptotic_covariance = model_object.evaluate(combined_design,fit_params,eval_options)
#compute the asymptotic upper and lower 95% bounds
asymptotic_lower_bound = fit_params - 2*np.sqrt(np.diag(asymptotic_covariance))
asymptotic_upper_bound = fit_params + 2*np.sqrt(np.diag(asymptotic_covariance))
#***hidden_start****
fim_bnds_array = np.vstack((asymptotic_lower_bound,
                            fit_params,
                            asymptotic_upper_bound)).T
fim_bnds = pd.DataFrame(np.exp(fim_bnds_array),
                        index=['Alpha','K','Delta','Beta','L','Gamma'],
                        columns=['Lower','Estimate','Upper'])
print('')
print(fim_bnds)
print('')
# #***hidden_end****
# #generate 500 simulated datasets
# mc_data = model_object.sample(init_design,fit_params,design_replicats=500)
# #fit the 500 datasets
# mc_fits = model_object.fit(mc_data, start_param = fit_params)
# #compute the Monte Carlo covariance estimate
# mc_covariance = np.cov(np.exp(mc_fits['Estimate'].to_numpy().T))
# #compute the asymptotic upper and lower 95% bounds
# mc_lower_bound = fit_params - 2*np.sqrt(np.diag(mc_covariance))
# mc_upper_bound = fit_params + 2*np.sqrt(np.diag(mc_covariance))
# #***hidden_start****
# mc_bnds_array = np.vstack((mc_lower_bound,
#                            fit_params,
#                            mc_upper_bound)).T
# mc_bnds = pd.DataFrame(np.exp(mc_bnds_array),
#                         index=['Alpha0','Alpha','n','K'],
#                         columns=['Lower','Estimate','Upper'])
# print('')
# print(mc_bnds)
# print('')
# #***hidden_end****

####################################################################################################
# Block 5: plot model predictions and initial data
####################################################################################################
#convert the covariance matrix to numpy
covariance_matrix = asymptotic_covariance.to_numpy()
#generate predictions with error bars fdor a random selection of inputs)
prediction_inputs = pd.DataFrame({'Init_Inducer':[0.]*len(observation_names),
                                  'Inducer_1':[1.]*len(observation_names),
                                  'Inducer_2':[0.]*len(observation_names),
                                  'Inducer_3':[3.]*len(observation_names),
                                  'Variable':observation_names})
#request prediction and observation intervals
prediction_options = {'PredictionInterval':True,
                      'ObservationInterval':True}
#generate predictions and intervals
predictions = model_object.predict(prediction_inputs,
                                   fit_params,
                                   covariance_matrix = covariance_matrix,
                                   options=prediction_options)
#***hidden_start****
#partition the predictions by state type
mrna_pred = predictions.loc[[obs == 'RNA' for obs in observation_type]]
prot_pred = predictions.loc[[obs == 'Prot' for obs in observation_type]]
#partition the initial data predictions by state type
mrna_dat = init_data.loc[[obs == 'RNA' for obs in observation_type]]
prot_dat = init_data.loc[[obs == 'Prot' for obs in observation_type]]
#plot the predictions, intervals and data, two subplots; mrna and protein
fig, ax = plt.subplots(2)
obs_bars = [(mrna_pred['Prediction','Mean']  - mrna_pred['Observation','Lower']).to_list(),
            (mrna_pred['Observation','Upper'] - mrna_pred['Prediction','Mean']).to_list()]
ax[0].errorbar(times, mrna_pred['Prediction','Mean'], yerr=obs_bars, fmt='none', color='C1')
pred_bars = [(mrna_pred['Prediction','Mean']  - mrna_pred['Prediction','Lower']).to_list(),
             (mrna_pred['Prediction','Upper'] - mrna_pred['Prediction','Mean']).to_list()]
ax[0].errorbar(times, mrna_pred['Prediction','Mean'], yerr=pred_bars, fmt='.', color='C0')
ax[0].plot(times, mrna_dat['Observation'], 'x', color='C1')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('mRNA')
obs_bars = [(prot_pred['Prediction','Mean']  - prot_pred['Observation','Lower']).to_list(),
            (prot_pred['Observation','Upper'] - prot_pred['Prediction','Mean']).to_list()]
ax[1].errorbar(times, prot_pred['Prediction','Mean'], yerr=obs_bars, fmt='none', color='C1')
pred_bars = [(prot_pred['Prediction','Mean']  - prot_pred['Prediction','Lower']).to_list(),
             (prot_pred['Prediction','Upper'] - prot_pred['Prediction','Mean']).to_list()]
ax[1].errorbar(times, prot_pred['Prediction','Mean'], yerr=pred_bars, fmt='.', color='C0')
ax[1].plot(times, prot_dat['Observation'], 'x', color='C1')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Prot')
plt.show()
# #***hidden_end****