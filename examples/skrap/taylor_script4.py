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
# Block 1: Casadi Setup
####################################################################################################
#define input and parameter symbols
inputs = cs.SX.sym('inputs',1)
parameters = cs.SX.sym('parameters',4)
#log-transormation of the parameters
alpha0 = cs.exp(parameters[0]) 
alpha = cs.exp(parameters[1]) 
n = cs.exp(parameters[2]) 
K = cs.exp(parameters[3]) 
#define the deterministic model for the GFP mean
gfp_mean = alpha0 + alpha*inputs[0]**n/(K**n+inputs[0]**n)
#assume some hetroskedasticity, std_dev 5% of mean expression level
gfp_var = (0.05**2)*gfp_mean**2
#link the deterministic model to the sampling statistics (here normal mean and variance)
gfp_stats = cs.vertcat(gfp_mean, gfp_var)
#create a casadi function mapping input and parameters to sampling statistics (mean and var)
gfp_model = cs.Function('GFP',[inputs,parameters],[gfp_stats])

####################################################################################################
# Block 2: Model creation
####################################################################################################
# enter the function as a tuple with label indicating normal error, into observation list
observ_list = [(gfp_model,'Normal')]
#create names for inputs
input_names = ['Light']
#create names for parameters
parameter_names = ['log_Alpha0','log_Alpha','log_n','log_K']
#instantiate nloed model class
model_object = Model(observ_list,input_names,parameter_names)
#***hidden_start****
#generate initial dataset 
init_design = pd.DataFrame({'Light':np.linspace(0.1,4095,5),
                            'Variable':['GFP']*5 ,
                            'Replicates':[3]*5})
true_param = np.log([5.53127845e+02, 9.52661655e+03, 2.41382438e+00, 3.62505725e+02])
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
             'InitParamBounds':[(5,8),(8,10),(-1,2),(3,5)],
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
                        index=['Alpha0','Alpha','n','K'],
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
# #convert the covariance matrix to numpy
# covariance_matrix = asymptotic_covariance.to_numpy()
# #generate predictions with error bars fdor a random selection of inputs)
# prediction_inputs = pd.DataFrame({'Light':np.linspace(0.1,4095,100),
#                                   'Variable':['GFP']*100})
# #request prediction and observation intervals
# prediction_options = {'PredictionInterval':True,
#                       'ObservationInterval':True}
# #generate predictions and intervals
# predictions = model_object.predict(prediction_inputs,
#                                    fit_params,
#                                    covariance_matrix = covariance_matrix,
#                                    options=prediction_options)

# tru_pred = model_object.predict(prediction_inputs,true_param)
# #create plot
# fig, ax = plt.subplots()
# #plot observation interval
# ax.fill_between(predictions['Inputs','Light'],
#                 predictions['Observation','Lower'],
#                 predictions['Observation','Upper'],
#                 alpha=0.3,
#                 color='C1')
# #plot prediction interval
# ax.fill_between(predictions['Inputs','Light'],
#                 predictions['Prediction','Lower'],
#                 predictions['Prediction','Upper'],
#                 alpha=0.4,
#                 color='C0')
# #plot mean model prediction
# ax.plot(predictions['Inputs','Light'], predictions['Prediction','Mean'], '-')
# ax.plot(tru_pred['Inputs','Light'], tru_pred['Prediction','Mean'], '-',color='red')
# #plot initial dataset
# ax.plot(init_data['Light'], init_data['Observation'], 'o', color='C1')
# ax.set_xlabel('Light')
# ax.set_ylabel('GFP')
# plt.show()

####################################################################################################
# Block 6: optimal design
####################################################################################################
#set up the design algorithm to use continuous (continuous) optimization with two unique inputs points
continuous_inputs={'Inputs':['Light'],
                   'Bounds':[(.01,4095)],
                   'Structure':[['x1'],
                                ['x2'],
                                ['x3'],
                                ['x4'],
                                ['x5']]}
#set up fixed design dictionary
fixed_dict ={'Weight':0.5,'Design':init_design}
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
sample_size = 15
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
intu_design = pd.DataFrame({'Light':[.1]+[63]+[254]+[1022]+[4095],
                          'Variable':['GFP']*5,
                          'Replicats':[3]*5}) 
intu_data = model_object.sample(intu_design,true_param)
optimal_data = model_object.sample(exact_design,true_param)
combined_data = pd.concat([init_data, optimal_data], ignore_index=True)
total_null_data = pd.concat([init_data, intu_data], ignore_index=True)
#***hidden_end****
total_null_design = pd.concat([init_design, intu_design], ignore_index=True)
#combine the initial and optimal design
combined_design = pd.concat([init_design, exact_design], ignore_index=True)

eval_options={'Method':'Asymptotic',
              'Covariance':True}
# call evaluate to compute the asymptotic covariance
asymptotic_covariance = model_object.evaluate(combined_design,true_param,eval_options)
#compute the asymptotic upper and lower 95% bounds
asymptotic_lower_bound = fit_params - 2*np.sqrt(np.diag(asymptotic_covariance))
asymptotic_upper_bound = fit_params + 2*np.sqrt(np.diag(asymptotic_covariance))
#***hidden_start****
fim_bnds_array = np.vstack((asymptotic_lower_bound,
                            fit_params,
                            asymptotic_upper_bound)).T
fim_bnds = pd.DataFrame(np.exp(fim_bnds_array),
                        index=['Alpha0','Alpha','n','K'],
                        columns=['Lower','Estimate','Upper'])
print('')
print(fim_bnds)
print('')

eval_options={'Method':'Asymptotic',
              'Covariance':True}
# call evaluate to compute the asymptotic covariance
asymptotic_covariance = model_object.evaluate(total_null_design,true_param,eval_options)
#compute the asymptotic upper and lower 95% bounds
asymptotic_lower_bound = fit_params - 2*np.sqrt(np.diag(asymptotic_covariance))
asymptotic_upper_bound = fit_params + 2*np.sqrt(np.diag(asymptotic_covariance))
#***hidden_start****
fim_bnds_array = np.vstack((asymptotic_lower_bound,
                            fit_params,
                            asymptotic_upper_bound)).T
fim_bnds = pd.DataFrame(np.exp(fim_bnds_array),
                        index=['Alpha0','Alpha','n','K'],
                        columns=['Lower','Estimate','Upper'])
print('')
print(fim_bnds)
print('')




#request contours and use a simple initial search
fit_options={'Confidence':'Contours'}
#fit the model to the initial data
fit_opt = model_object.fit(combined_data,start_param = fit_params, options=fit_options)
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
lik_bnds_array = np.hstack((fit_info['Lower'].to_numpy().T,
                        fit_info['Estimate'].to_numpy().T,
                        fit_info['Upper'].to_numpy().T))
lik_bnds = pd.DataFrame(np.exp(lik_bnds_array),
                        index=['Alpha0','Alpha','n','K'],
                        columns=['Lower','Estimate','Upper'])
print('')
print(lik_bnds)
print('')
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
                        index=['Alpha0','Alpha','n','K'],
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
prediction_inputs = pd.DataFrame({'Light':np.linspace(0.1,10,100),
                                  'Variable':['GFP']*100})
#request prediction and observation intervals
prediction_options = {'PredictionInterval':True,
                      'ObservationInterval':True}
#generate predictions and intervals
predictions = model_object.predict(prediction_inputs,
                                   fit_params,
                                   covariance_matrix = covariance_matrix,
                                   options=prediction_options)
#create plot
fig, ax = plt.subplots()
#plot observation interval
ax.fill_between(predictions['Inputs','Light'],
                predictions['Observation','Lower'],
                predictions['Observation','Upper'],
                alpha=0.3,
                color='C1')
#plot prediction interval
ax.fill_between(predictions['Inputs','Light'],
                predictions['Prediction','Lower'],
                predictions['Prediction','Upper'],
                alpha=0.4,
                color='C0')
#plot mean model prediction
ax.plot(predictions['Inputs','Light'], predictions['Prediction','Mean'], '-')
#plot initial dataset
ax.plot(combined_data['Light'], combined_data['Observation'], 'o', color='C1')
ax.set_xlabel('Light')
ax.set_ylabel('GFP')
plt.show()

perc0=(fim_bnds0['Upper']-fim_bnds0['Lower'])/fim_bnds0['Estimate']
perc=(fim_bnds['Upper']-fim_bnds['Lower'])/fim_bnds['Estimate']

labels = ['Alpha0','Alpha','n','K']
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, 100*perc0.to_numpy(), width, label='Initial')
rects2 = ax.bar(x + width/2, 100*perc.to_numpy(), width, label='Optimal')
ax.set_ylabel('% Error')
ax.set_title('Diameter of 95% CI Expressed in % Error')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
fig.tight_layout()
plt.show()