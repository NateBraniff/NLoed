""" 
Add a docstring
"""
import numpy as np
import pandas as pd
import casadi as cs
from nloed import Model
from nloed import Design
import matplotlib.pyplot as plt

####################################################################################################
# SET UP MODEL
####################################################################################################

#define input and parameters
x = cs.SX.sym('x',1)
xnames = ['Light']
p = cs.SX.sym('p',4)
pnames = ['alpha0','alpha','n','K']

#define steady state 
gfp_mean = cs.exp(p[0]) + cs.exp(p[1])*x[0]**cs.exp(p[2])/(cs.exp(p[3])**cs.exp(p[2])+x[0]**cs.exp(p[2]))
gfp_std = 0.059*gfp_mean

#link the deterministic model to the sampling statistics (here normal mean and variance)
normal_stats = cs.vertcat(gfp_mean, gfp_std**2)

#create a casadi function mapping input and parameters to sampling statistics (mean and var)
y = cs.Function('GFP',[x,p],[normal_stats])

# enter the function as a tuple with label indicating normal error, into observation list
observ_list = [(y,'Normal')]

#instantiate nloed model class
nloed_model = Model(observ_list, xnames, pnames)

####################################################################################################
# Fit
####################################################################################################

obs = [ 580,  544,   543,
        766,  699,   570,
        2064, 2171,  4598,
        8814, 9919,  8860,
        9584, 10295, 10672]

init_data = pd.DataFrame({'Light':[.1]*3+[63]*3+[254]*3+[1022]*3+[4095]*3,
                          'Variable':['GFP']*15,
                          'Observation':obs}) 

fit_options={'Confidence':'Intervals',  
             'InitParamBounds':[(3,8),(5,10),(-1,1),(3,8)],
             'InitSearchNumber':7,
             'SearchBound':5.,
             'Verbose':False}
#fit the model to the init data
# ,
#              'MaxSteps': 10000,
#              'InitialStep':0.1,
#              'SearchBound':10.
fit_info = nloed_model.fit(init_data, options=fit_options)

print(np.exp(fit_info))

fit_params = fit_info['Estimate'].to_numpy().flatten()

####################################################################################################
# EVALUATE INITIAL DESIGN
####################################################################################################

init_design = pd.DataFrame({'Light':[.1]+[63]+[254]+[1022]+[4095],
                            'Variable':['GFP']*5,
                            'Replicates':[3]*5}) 

# #get estimated covariance, bias and MSE of parameter fit (use asymptotic method here) 
# opts={'Method':'Asymptotic','FIM':True,'Covariance':True,'Bias':True,'MSE':True,'SampleNumber':500}
# diagnostic_info_initA = nloed_model.evaluate(init_design, fit_params, opts)
# print(diagnostic_info_initA)

# stderr_initA = np.sqrt(np.diag(diagnostic_info_initA['Covariance']))
# upper_bnd_init = fit_params + 2*stderr_initA
# lower_bnd_init = fit_params - 2*stderr_initA

####################################################################################################
# GENERATE OPTIMAL DESIGN
####################################################################################################

#set up the design algorithm to use continuous (continuous) optimization with two unique inputs points
continuous_inputs={'Inputs':['Light'],
                   'Bounds':[(.1,4095.)],
                   'Structure':[['L1'],
                                ['L2'],
                                ['L3'],
                                ['L4']],
                    'Initial':[[10],
                               [200],
                               [500],
                               [4000]]}

fixed_design = {'Weight':0.5,'Design':init_design}

opts= {'LockWeights':False}
# generate the optimal discreteimate (relaxed) design
design_object = Design(nloed_model,
                        fit_params,
                        'D',
                        continuous_inputs=continuous_inputs,
                        fixed_design=fixed_design,
                        options=opts)

sample_size = 16
relaxed_design = design_object.relaxed()
print(relaxed_design)
#generate a rounded exact design 
opt_design = design_object.round(sample_size)
print(opt_design)

####################################################################################################
# EVALUATE DESIGN & PREDICT OUTPUT
####################################################################################################

#############################################
#eval the optimal design + initial design
opt_design_tot = pd.concat([init_design, opt_design], ignore_index=True)

#get estimated covariance, bias and MSE of parameter fit (use asymptotic method here) 
opts={'Method':'Asymptotic','FIM':True,'Covariance':True,'Bias':True,'MSE':True,'SampleNumber':500}
diagnostic_info_optA = nloed_model.evaluate(opt_design_tot, fit_params, opts)
print(diagnostic_info_optA)

stderr_optA = np.sqrt(np.diag(diagnostic_info_optA['Covariance']))
upper_bnd_opt = fit_params + 2*stderr_optA
lower_bnd_opt = fit_params - 2*stderr_optA

#############################################
#eval the intuitive design + initial design
intuitive_design = pd.DataFrame({'Light':[110]+[192]+[335]+[585]+[2046],
                              'Variable':['GFP']*5,
                              'Replicats':[3]*5}) 

intuitive_design_tot = pd.concat([init_design, intuitive_design], ignore_index=True)

#get estimated covariance, bias and MSE of parameter fit (use asymptotic method here) 
opts={'Method':'Asymptotic','FIM':True,'Covariance':True,'Bias':True,'MSE':True,'SampleNumber':500}
diagnostic_info_intuA = nloed_model.evaluate(intuitive_design_tot, fit_params, opts)
print(diagnostic_info_intuA)

stderr_intuA = np.sqrt(np.diag(diagnostic_info_intuA['Covariance']))
upper_bnd_intu = fit_params + 2*stderr_intuA
lower_bnd_intu = fit_params - 2*stderr_intuA

#PLOT THE COMPARED CI's
exp_vals = np.tile(np.exp(fit_params),(1,3))
exp_up = np.hstack((np.exp(upper_bnd_init),np.exp(upper_bnd_intu),np.exp(upper_bnd_opt)))
exp_lw = np.hstack((np.exp(lower_bnd_init),np.exp(lower_bnd_intu),np.exp(lower_bnd_opt)))
perc_err_up = (exp_up-exp_vals)/exp_vals
perc_err_dw = (exp_lw-exp_vals)/exp_vals
perc_err_up = perc_err_up[0,[0,4,8,1,5,9,2,6,10,3,7,11]]
perc_err_dw = perc_err_dw[0,[0,4,8,1,5,9,2,6,10,3,7,11]]

labels = ['a0-Init','a0-Intu','a0-Opt', 'a-Init', 'a-Intu', 'a-Opt', 'n-Init', 'n-Intu','n-Opt', 'K-Init', 'K-Intu', 'K-Opt']
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, perc_err_up.flatten(), width, label='UpperBnd-%Err')
rects2 = ax.bar(x + width/2, perc_err_dw.flatten(), width, label='LowerBnd-%Err')
ax.set_ylabel('% Error')
ax.set_title('Upper and Lower 95% CI Bounds Expressed in % Error')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
fig.tight_layout()
plt.show()

####################################################################################################
# Simulation Study
####################################################################################################

#GEN CONTOUR PLOTS SAMPLES
###data_init = nloed_model.sample(init_design,fit_params)
#USED REAL DATA HERE
# fit_options={'Confidence':'Contours'}
# fit_info_init = nloed_model.fit(init_data, start_param=fit_params, options=fit_options)

# data_intu = nloed_model.sample(intuitive_design_tot,fit_params)
# fit_options={'Confidence':'Contours'}
# fit_info_intu = nloed_model.fit(data_intu, start_param=fit_params, options=fit_options)

# data_opt = nloed_model.sample(opt_design_tot,fit_params)
# fit_options={'Confidence':'Contours','MaxSteps':10000}
# fit_info_opt = nloed_model.fit(data_opt, start_param=fit_params, options=fit_options)

#GEN SIMULATED COV/BIAS ESTIMATES VIA SIMULATION + BACKTRANSFORMATION OF PARS
# data_init = nloed_model.sample(init_design,fit_params,design_replicats=500)
# fit_info_init = nloed_model.fit(data_init, start_param=fit_params)
# cov_init = np.cov(np.exp(fit_info_init.to_numpy().T))

# stderr_intuA = np.sqrt(np.diag(diagnostic_info_intuA['Covariance']))
# upper_bnd_intu = fit_params + 2*stderr_intuA
# lower_bnd_intu = fit_params - 2*stderr_intuA

# data_intu = nloed_model.sample(intuitive_design_tot,fit_params,design_replicats=500)
# fit_info_intu = nloed_model.fit(data_intu, start_param=fit_params)
# cov_intu = np.cov(np.exp(fit_info_intu.to_numpy().T))

# data_opt = nloed_model.sample(opt_design_tot,fit_params,design_replicats=500)
# fit_info_opt = nloed_model.fit(data_opt, start_param=fit_params)
# cov_opt = np.cov(np.exp(fit_info_opt.to_numpy().T))




