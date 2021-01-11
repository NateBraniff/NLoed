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
p = cs.SX.sym('p',5)
pnames = ['alpha0','alpha','n','K','gamma']

#define steady state 
gfp_mean = cs.exp(p[0]) + cs.exp(p[1])*x[0]**cs.exp(p[2])/(cs.exp(p[3])**cs.exp(p[2])+x[0]**cs.exp(p[2]))
gfp_var = cs.exp(p[4])*gfp_mean

geo_mean = cs.log(gfp_mean**2/cs.sqrt(gfp_mean**2 + gfp_var))
geo_var = cs.log(1 + gfp_var/gfp_mean**2)
#geo_mean = 2/3 * cs.log(gfp_mean) - 1/2 * cs.log(cs.exp(p[4]) + gfp_mean)
#2 * cs.log(gfp_mean) - 1/2 * cs.log(p[4]*gfp_mean + gfp_mean**2)
#geo_var = cs.log(cs.exp(p[4]) + gfp_mean) - cs.log(gfp_mean)

#link the deterministic model to the sampling statistics (here normal mean and variance)
lognorm_stats = cs.vertcat(geo_mean, geo_var)

#create a casadi function mapping input and parameters to sampling statistics (mean and var)
y = cs.Function('GFP',[x,p],[lognorm_stats])

# enter the function as a tuple with label indicating normal error, into observation list
observ_list = [(y,'Lognormal')]

#instantiate nloed model class
nloed_model = Model(observ_list, xnames, pnames)

nat_params = [5.53127845e+02, 9.52661655e+03, 2.41382438e+00, 3.62505725e+02, 5500]
nominal_params = np.log(nat_params)

####################################################################################################
# EVALUATE INITIAL DESIGN
####################################################################################################

init_design = pd.DataFrame({'Light':[.1]+[63]+[254]+[1022]+[4095],
                            'Variable':['GFP']*5,
                            'Replicates':[15000]*5}) 

#get estimated covariance, bias and MSE of parameter fit (use asymptotic method here) 
opts={'Method':'Asymptotic','FIM':True,'Covariance':True,'Bias':True,'MSE':True}
diagnostic_info_initA = nloed_model.evaluate(init_design, nominal_params, opts)
print(diagnostic_info_initA)

asymptotic_covariance = diagnostic_info_initA['Covariance']
asymptotic_lower_bound = nominal_params - 2*np.sqrt(np.diag(asymptotic_covariance))
asymptotic_upper_bound = nominal_params + 2*np.sqrt(np.diag(asymptotic_covariance))

fim_bnds_array = np.vstack((asymptotic_lower_bound,
                            nominal_params,
                            asymptotic_upper_bound)).T
fim_bnds0 = pd.DataFrame(np.exp(fim_bnds_array),
                        index=['Alpha0','Alpha','n','K','gamma'],
                        columns=['Lower','Estimate','Upper'])
print(fim_bnds0)

####################################################################################################
# GENERATE OPTIMAL DESIGN
####################################################################################################

#set up the design algorithm to use continuous (continuous) optimization with two unique inputs points
continuous_inputs={'Inputs':['Light'],
                   'Bounds':[(.1,4095.)],
                   'Structure':[['L1'],
                                ['L2'],
                                ['L3'],
                                ['L4'],
                                ['L5']],
                    'Initial':[[10],
                               [200],
                               [500],
                               [1000],
                               [4000]]}

fixed_design = {'Weight':0.5,'Design':init_design}

opts= {'LockWeights':False}
# generate the optimal discreteimate (relaxed) design
design_object = Design(nloed_model,
                        nominal_params,
                        'D',
                        continuous_inputs=continuous_inputs,
                        fixed_design=fixed_design,
                        options=opts)

sample_size = 75000
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
opts={'Method':'Asymptotic','FIM':True,'Covariance':True,'Bias':True,'MSE':True}
diagnostic_info_optA = nloed_model.evaluate(opt_design_tot, nominal_params, opts)
print(diagnostic_info_optA)

asymptotic_covariance = diagnostic_info_initA['Covariance']
asymptotic_lower_bound = nominal_params - 2*np.sqrt(np.diag(asymptotic_covariance))
asymptotic_upper_bound = nominal_params + 2*np.sqrt(np.diag(asymptotic_covariance))

fim_bnds_array = np.vstack((asymptotic_lower_bound,
                            nominal_params,
                            asymptotic_upper_bound)).T
fim_bnds = pd.DataFrame(np.exp(fim_bnds_array),
                        index=['Alpha0','Alpha','n','K','gamma'],
                        columns=['Lower','Estimate','Upper'])
print(fim_bnds)

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




