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
inputs = cs.SX.sym('inputs',2)
parameters = cs.SX.sym('parameters',8)

K_dr = cs.exp(parameters[0]) 
K_T = cs.exp(parameters[1])
B_o = cs.exp(parameters[2])
B = cs.exp(parameters[3])
K_m = cs.exp(parameters[4])
K_g = cs.exp(parameters[5])
Sigma_m = cs.exp(parameters[6])
Sigma_g = cs.exp(parameters[7])
m0 = 0.00002334410
g0 = 1468.31716701552

ccar = 1/( cs.sqrt(K_dr*(K_T/inputs[0]+1)**2+1) + cs.sqrt(K_dr)*(K_T/inputs[0]+1) )**2
c0 = (B_o + B*ccar/(K_m + ccar))
c1 = K_g*c0/Sigma_m
c2 = c1 - K_g*m0

gfp_mean = c1/Sigma_g +\
           c2/(Sigma_m - Sigma_g)*cs.exp(-Sigma_m*inputs[1]) +\
           (g0-c1/Sigma_g - c2/(Sigma_m - Sigma_g))*cs.exp(-Sigma_g*inputs[1])
#assume some hetroskedasticity, std_dev 5% of mean expression level
gfp_var = (0.05**2)*gfp_mean**2
#link the deterministic model to the sampling statistics (here normal mean and variance)
gfp_stats = cs.vertcat(gfp_mean, gfp_var)
#create a casadi function mapping input and parameters to sampling statistics (mean and var)
gfp_model = cs.Function('GFP',[inputs,parameters],[gfp_stats])

K_dr_val = 0.00166160831
K_T_val = 11979.25516474195
B_o_val = 0.00137146821
B_val = 0.03525605965
K_m_val = 0.28169444728
K_g_val = 68.61725640178
Sigma_m_val = 0.01180105629
Sigma_g_val = 0.01381497437

fit_params = np.log([K_dr_val, K_T_val, B_o_val, B_val, K_m_val, K_g_val, Sigma_m_val, Sigma_g_val])

# enter the function as a tuple with label indicating normal error, into observation list
observ_list = [(gfp_model,'Normal')]
#create names for inputs
input_names = ['Light','Time']
#create names for parameters
parameter_names = ['K_dr', 'K_T', 'B_o', 'B', 'K_m', 'K_g', 'Sigma_m', 'Sigma_g']
#instantiate nloed model class
model_object = Model(observ_list,input_names,parameter_names)

#set up the design algorithm to use continuous (continuous) optimization with two unique inputs points
continuous_inputs={'Inputs':['Light','Time'],
                   'Bounds':[(.01,4095),(0,6*60)],
                   'Structure':[['L1','T1'],
                                ['L2','T2'],
                                ['L3','T3'],
                                ['L4','T4'],
                                ['L5','T5'],
                                ['L6','T6'],
                                ['L7','T7'],
                                ['L8','T8'],
                                ['L9','T9'],
                                ['L10','T10']],
                    'Initial':[[0.1,350],
                                [10,350],
                                [100,350],
                                [1000,350],
                                [4090,350],
                                [4090,1],
                                [4090,10],
                                [4090,100],
                                [4090,300],
                                [4090,350]]}

#set up fixed design dictionary
#fixed_dict ={'Weight':0.5,'Design':init_design}
# generate the optimal discreteimate (relaxed) design
design_object = Design(model_object,fit_params,'D',
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