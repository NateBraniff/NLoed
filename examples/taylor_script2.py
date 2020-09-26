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
x = cs.SX.sym('x',2)
xnames = ['Light','Time']
p = cs.SX.sym('p',6)
pnames = ['alpha0','alpha','n','K','lambda1','lambda2']

#define steady state 
SS = cs.exp(p[0]) + cs.exp(p[1])*x[0]**cs.exp(p[2])/(cs.exp(p[3])**cs.exp(p[2])+x[0]**cs.exp(p[2]))
y_ss = cs.Function('SS',[x,p],[SS])
y0 = 1000 
#define the deterministic model
#gfp_model = y0 * cs.exp(-cs.exp(p[4])*x[1]) + y_ss(x,p) * (1-cs.exp(-cs.exp(p[4])*x[1]))
gfp_model = y_ss(x,p) \
            - (y_ss(x,p)-y0)*cs.exp(-cs.exp(p[4])*x[1]) \
            - (y_ss(x,p)-y0)*(cs.exp(p[4])/cs.exp(p[5]))*\
                 (cs.exp(-cs.exp(p[4])*x[1]) - cs.exp(-(cs.exp(p[4])+cs.exp(p[5]))*x[1]))

#link the deterministic model to the sampling statistics (here normal mean and variance)
mean, var = gfp_model, 100**2/3
normal_stats = cs.vertcat(mean, var)

#create a casadi function mapping input and parameters to sampling statistics (mean and var)
y = cs.Function('GFP',[x,p],[normal_stats])

# enter the function as a tuple with label indicating normal error, into observation list
observ_list = [(y,'Normal')]

#instantiate nloed model class
nloed_model = Model(observ_list, xnames, pnames)

####################################################################################################
# Fit
####################################################################################################

# obs = [ 1951,	723,	580,
#         1479,	1350,	544,
#         1516,	680,	543,
#         2057,	628,	766,
#         1360,	741,	699,
#         1670,	640,	570,
#         1164,	1135,	2064,
#         1419,	1340,	2171,
#         1330,	1615,	4598,
#         1520,	4024,	8814,
#         1657,	3898,	9919,
#         1208,	4071,	8860,
#         1784,	9213,	9584,
#         1168,	5550,	10295,
#         1739,	9006,	10672]

# init_data = pd.DataFrame({'Light':[.1]*9+[63]*9+[254]*9+[1022]*9+[4095]*9,
#                           'Time':[0.5,1.8,6]*15,
#                           'Variable':['GFP']*45,
#                           'Observation':obs}) 

# fit_options={'Confidence':'None',  
#              'InitParamBounds':[(-1,4),(-1,4),(-1,4),(-1,4),(-1,4),(-1,4)],
#              'InitSearchNumber':7,
#              'SearchBound':5.,
#              'Verbose':False}
# #fit the model to the init data
# fit_info = nloed_model.fit(init_data, options=fit_options)

# print(np.exp(fit_info))

# #fit_params = fit_info['Estimate'].to_numpy().flatten()
# fit_params = fit_info.to_numpy().flatten()

####################################################################################################
# GENERATE OPTIMAL DESIGN
####################################################################################################

#set up the design algorithm to use continuous (continuous) optimization with two unique inputs points
continuous_inputs={'Inputs':['Light','Time'],
                   'Bounds':[(.1,4095.),(1.,6.)],
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
                    'Initial':[[10,  5],
                               [200, 5],
                               [500, 5],
                               [4000,5],
                               [10,  3],
                               [200, 3],
                               [500, 3],
                               [4000,3],
                               [10,  1],
                               [4000,1]]}

# continuous_inputs={'Inputs':['Light','Time'],
#                    'Bounds':[(.1,4095.),(1.,6.)],
#                    'Structure':[['L1','T1'],
#                                 ['L2','T2'],
#                                 ['L3','T3'],
#                                 ['L4','T4'],
#                                 ['L5','T5'],
#                                 ['L6','T6'],
#                                 ['L7','T7']],
#                     'Initial':[[0.5,  5],
#                                [300,  5],
#                                [1000, 5],
#                                [4000, 5],
#                                [1000, 3],
#                                [4000, 3],
#                                [4000, 1]]}

# continuous_inputs={'Inputs':['Light','Time'],
#                    'Bounds':[(.1,4095.),(.1,6.)],
#                    'Structure':[['L1','T1'],
#                                 ['L2','T1'],
#                                 ['L3','T1'],
#                                 ['L4','T1'],
#                                 ['L5','T1'],
#                                 ['L1','T2'],
#                                 ['L2','T2'],
#                                 ['L3','T2'],
#                                 ['L4','T2'],
#                                 ['L5','T2'],
#                                 ['L1','T3'],
#                                 ['L2','T3'],
#                                 ['L3','T3'],
#                                 ['L4','T3'],
#                                 ['L5','T3'],],
#                     'Initial':[[.2,  5],
#                                [200, 5],
#                                [500, 5],
#                                [800, 5],
#                                [4000,5],
#                                [.2,  2],
#                                [200, 2],
#                                [500, 2],
#                                [800, 2],
#                                [4000,2],
#                                [.2,  1],
#                                [200, 1],
#                                [500, 1],
#                                [800, 1],
#                                [4000,1],]}

fit_params= [6.36854913687607, 9.27758542538853, 0.5923066933328027, 6.33665344125038, 0.05414838286364793, -11.91318481900083]
fit_params[5]=-4

init_design = pd.DataFrame({'Light':[.1]*9+[63]*9+[254]*9+[1022]*9+[4095]*9,
                            'Time':[0.5,1.8,6]*15,
                            'Variable':['GFP']*45,
                            'Replicats':[1]*45}) 

fixed_design = {'Weight':0.5,'Design':init_design}

opts= {'LockWeights':False}
# generate the optimal discreteimate (relaxed) design
relaxed_design = Design(nloed_model,
                        fit_params,
                        'D',
                        continuous_inputs=continuous_inputs,
                        fixed_design=fixed_design,
                        options=opts)

sample_size = 45
#generate a rounded exact design 
exact_design = relaxed_design.round(sample_size)
print(exact_design)

opt_design_tot = pd.concat([init_design, exact_design], ignore_index=True)

#get estimated covariance, bias and MSE of parameter fit (use asymptotic method here) 
opts={'Method':'MonteCarlo','Covariance':True,'Bias':True,'MSE':True,'SampleNumber':500}
diagnostic_info1 = nloed_model.evaluate(opt_design_tot, fit_params, opts)
print(diagnostic_info1)

naive_design2 = pd.DataFrame({'Light':[102]*3+[161]*3+[256]*3+[406]*3+[644]*3,
                              'Time':[0.3,0.8,3.3]*5,
                              'Variable':['GFP']*15,
                              'Replicats':[1]*15}) 

naive_design_tot = pd.concat([init_design, naive_design2], ignore_index=True)

opts={'Method':'MonteCarlo','Covariance':True,'Bias':True,'MSE':True,'SampleNumber':500}
diagnostic_info2 = nloed_model.evaluate(naive_design_tot, fit_params, opts)
print(diagnostic_info2)

# opts={'Method':'Asymptotic','Covariance':True,'Bias':True,'MSE':True}
# diagnostic_info = nloed_model.evaluate(exact_design, init_param, opts)
# print(diagnostic_info)


####################################################################################################
# EVALUATE DESIGN & PREDICT OUTPUT
####################################################################################################



# #generate predictions with error bars fdor a random selection of inputs)
# prediction_inputs = pd.DataFrame({ 'Light':np.random.uniform(0,5,10),
#                                 'Variable':['y']*10})
# cov_mat = diagnostic_info['Covariance'].to_numpy()
# pred_options = {'PredictionInterval':True,
#                 'ObservationInterval':True,
#                 'Sensitivity':True}
# predictions = nloed_model.predict(prediction_inputs,
#                                   fit_params,
#                                   covariance_matrix = cov_mat,
#                                   options=pred_options)
# print(predictions)

# t=0
