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
inputs = cs.SX.sym('inputs')
parameters = cs.SX.sym('parameters',5)

K_dr = cs.exp(parameters[0]) 
K_T = cs.exp(parameters[1])
B_o_prime = cs.exp(parameters[2])
B_prime = cs.exp(parameters[3])
K_m = cs.exp(parameters[4])

ccar = 1/( cs.sqrt(K_dr*(K_T/inputs+1)**2+1) + cs.sqrt(K_dr)*(K_T/inputs+1) )**2
gfp_mean = (B_o_prime + B_prime*ccar/(K_m + ccar))

#assume some hetroskedasticity, std_dev 5% of mean expression level
gfp_var = (0.05**2)*gfp_mean**2
#link the deterministic model to the sampling statistics (here normal mean and variance)
gfp_stats = cs.vertcat(gfp_mean, gfp_var)
#create a casadi function mapping input and parameters to sampling statistics (mean and var)
gfp_model = cs.Function('GFP',[inputs,parameters],[gfp_stats])

# K_g_val = 68.61725640178
# Sigma_m_val = 0.01180105629
# Sigma_g_val = 0.01381497437
# K_dr_val = 0.00166160831
# K_T_val = 11979.25516474195
# K_m_val = 0.28169444728
# B_o_val = 0.00137146821
# B_val = 0.03525605965

K_g_val = 1.6260543
Sigma_m_val = 0.00490364
Sigma_g_val = 0.046474403
K_dr_val = 4.04461938
K_T_val = 11306.84788
K_m_val = 0.000149759
B_o_val = 0.073858518
B_val = 1.8532546

B_o_prime_val = B_o_val*K_g_val/Sigma_m_val/Sigma_g_val
B_prime_val = B_val*K_g_val/Sigma_m_val/Sigma_g_val

fit_params = np.log([K_dr_val, K_T_val, B_o_prime_val, B_prime_val, K_m_val])

# enter the function as a tuple with label indicating normal error, into observation list
observ_list = [(gfp_model,'Normal')]
#create names for inputs
input_names = ['Light']
#create names for parameters
parameter_names = ['K_dr', 'K_T', 'B_o_prime', 'B_prime', 'K_m']
#instantiate nloed model class
static_model = Model(observ_list,input_names,parameter_names)

# #generate predictions with error bars fdor a random selection of inputs)
# prediction_inputs = pd.DataFrame({'Light':np.linspace(0.1,4095,100),
#                                   'Variable':['GFP']*100})
# #generate predictions and intervals
# predictions = static_model.predict(prediction_inputs,
#                                    fit_params,
#                                    options ={'Sensitivity':True})
# #create plot
# fig, ax = plt.subplots()
# #plot mean model prediction
# ax.plot(predictions['Inputs','Light'], predictions['Prediction','Mean'], '-')
# ax.set_xlabel('Light')
# ax.set_ylabel('GFP')
# plt.show()

obs1 = [1101,
        1246,
        883,
        2571,
        1768,
        1219,
        577,
        8029,
        11104,
        9792,
        16450,
        24290,
        39988,
        6480,
        33019]
lght_1 = [.1]*3+[63]*3+[254]*3+[1022]*3+[4095]*3

obs2 = [837,
        733,
        2807,
        2816,
        1857,
        4455,
        4408,
        6779,
        3325,
        4343,
        3095,
        7030,
        10583,
        11957,
        17313]
lght_2 = [.1]*2+[256]*4+[597]*5+[4095]*4

obs3 = [975,
        827,
        1103,
        1833,
        1743,
        1128,
        2459,
        3889,
        7126,
        2610,
        6837,
        10089,
        9367,
        9838]
lght_3 = [110]*3+[192]*3+[335]*2+[585]*3+[2046]*3

init_data = pd.DataFrame({'Light':lght_1 + lght_2 + lght_3,
                          'Variable':['GFP']*(15*2+14),
                          'Observation':obs1 + obs2 + obs3}) 
            
init_design = pd.DataFrame({'Light':lght_1 + lght_2 + lght_3,
                            'Variable':['GFP']*(15*2+14),
                            'Replicates':[1]*(15*2+14)}) 


#set up the design algorithm to use continuous (continuous) optimization with two unique inputs points
continuous_inputs={'Inputs':['Light'],
                   'Bounds':[(.01,4095)],
                   'Structure':[['x1'],
                                ['x2'],
                                ['x3'],
                                ['x4'],
                                ['x5']],
                    'Initial':[[0.1],
                               [100],
                               [300],
                               [1000],
                               [4000]]}
#set up fixed design dictionary
fixed_dict ={'Weight':.75,'Design':init_design}
# generate the optimal discreteimate (relaxed) design
design_object = Design(static_model,fit_params,'D',
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
sample_size = 16
#generate a rounded exact design 
exact_design = design_object.round(sample_size)
#***hidden_start****
print('')
print(exact_design)
print('')
#***hidden_end****


eval_options={'Method':'Asymptotic',
              'Covariance':True,
              'FIM':True}
# call evaluate to compute the asymptotic covariance
diagnostics0 = static_model.evaluate(init_design,fit_params,eval_options)
print(np.linalg.cond(diagnostics0['FIM']))
asymptotic_covariance = diagnostics0['Covariance']
#compute the asymptotic upper and lower 95% bounds
asymptotic_lower_bound = fit_params - 2*np.sqrt(np.diag(asymptotic_covariance))
asymptotic_upper_bound = fit_params + 2*np.sqrt(np.diag(asymptotic_covariance))
#***hidden_start****
fim_bnds_array = np.vstack((asymptotic_lower_bound,
                            fit_params,
                            asymptotic_upper_bound)).T
fim_bnds0 = pd.DataFrame(np.exp(fim_bnds_array),
                        index=parameter_names,
                        columns=['Lower','Estimate','Upper'])
print('')
print(fim_bnds0)
print('')

combined_design = pd.concat([init_design, exact_design], ignore_index=True)
# call evaluate to compute the asymptotic covariance
diagnostics_opt = static_model.evaluate(combined_design,fit_params,eval_options)
print(np.linalg.cond(diagnostics_opt['FIM']))
asymptotic_covariance = diagnostics_opt['Covariance']
#compute the asymptotic upper and lower 95% bounds
asymptotic_lower_bound = fit_params - 2*np.sqrt(np.diag(asymptotic_covariance))
asymptotic_upper_bound = fit_params + 2*np.sqrt(np.diag(asymptotic_covariance))
#***hidden_start****
fim_bnds_array = np.vstack((asymptotic_lower_bound,
                            fit_params,
                            asymptotic_upper_bound)).T
fim_bnds_opt = pd.DataFrame(np.exp(fim_bnds_array),
                        index=parameter_names,
                        columns=['Lower','Estimate','Upper'])
print('')
print(fim_bnds_opt)
print('')

####################################################################################################
# Block 2: Casadi Setup
####################################################################################################

lght_1 = [.1]*2+[256]*4+[597]*5+[4095]*4
lght_2 = [110]*3+[192]*3+[335]*2+[585]*3+[2046]*3
lght_3 = [.1]*3+[63]*3+[254]*3+[1022]*3+[4095]*3

tmpts_1 = [1*60]*3 + [2*60]*3 + [3*60]*3 + [4*60]*3 + [5*60]*3

init_design = pd.DataFrame({'Time':[6*60]*(15*2+14)+[0.5*60]*15+[1.8*60]*15+tmpts_1*2,
                            'Light1':lght_1 + lght_2 + lght_3*3 + [4095]*15 +[2046]*15,
                            'Light2':lght_1 + lght_2 + lght_3*3 + [4095]*15 +[2046]*15,
                            'Light3':lght_1 + lght_2 + lght_3*3 + [4095]*15 +[2046]*15,
                            'Variable':['GFP_1']*(15*2+14+15*2+15*2),
                            'Replicates':[1]*(15*6+14)}) 

#define input and parameter symbols
inputs = cs.SX.sym('inputs',4)
parameters = cs.SX.sym('parameters',3)

K_g = cs.exp(parameters[0])
Sigma_m = cs.exp(parameters[1])
Sigma_g = cs.exp(parameters[2])
m0 = np.mean([19.17,11.82,0.00013])
g0 = np.mean([2713,3560,2436])


# B_o_prime_val = B_o_val*K_g_val/Sigma_m_val/Sigma_g_val
# B_prime_val = B_val*K_g_val/Sigma_m_val/Sigma_g_val

observ_list=[]

indx = [0]*5 + [1]*4 + [2]*7

for i in range(16):
    ccar = 1/( cs.sqrt(K_dr_val*(K_T_val/inputs[indx[i]+1]+1)**2+1) + cs.sqrt(K_dr_val)*(K_T_val/inputs[indx[i]+1]+1) )**2
    c0 = (B_o_val + B_val*ccar/(K_m_val + ccar))
    c1 = K_g*c0/Sigma_m
    c2 = c1 - K_g*m0

    gfp_mean = c1/Sigma_g +\
            c2/(Sigma_m - Sigma_g)*cs.exp(-Sigma_m*inputs[0]) +\
            (g0-c1/Sigma_g - c2/(Sigma_m - Sigma_g))*cs.exp(-Sigma_g*inputs[0])
    #assume some hetroskedasticity, std_dev 5% of mean expression level
    gfp_var = (0.05**2)*gfp_mean**2
    #link the deterministic model to the sampling statistics (here normal mean and variance)
    gfp_stats = cs.vertcat(gfp_mean, gfp_var)
    #create a casadi function mapping input and parameters to sampling statistics (mean and var)
    gfp_model = cs.Function('GFP_'+str(i+1),[inputs,parameters],[gfp_stats])
    # enter the function as a tuple with label indicating normal error, into observation list
    observ_list.append((gfp_model,'Normal'))

fit_params = np.log([K_g_val, Sigma_m_val, Sigma_g_val])

#create names for inputs
input_names = ['Time','Light1','Light2','Light3']
#create names for parameters
parameter_names = ['K_g', 'Sigma_m', 'Sigma_g']
#instantiate nloed model class
dynamic_model = Model(observ_list,input_names,parameter_names)

discrete_inputs={'Inputs':['Light1','Light2','Light3'],
                 'Grid':[[0.01,1942,4095]]}

#set up the design algorithm to use continuous (continuous) optimization with two unique inputs points
continuous_inputs={'Inputs':['Time'],
                   'Bounds':[(0,6*60)],
                   'Structure':[['T1'],
                                ['T2'],
                                ['T3']],
                   'Initial':[[1],
                              [3*60],
                              [6*60]]}

#set up fixed design dictionary
fixed_dict ={'Weight':0.875,'Design':init_design}
# generate the optimal discreteimate (relaxed) design
opts= {'LockWeights':False}
observ_groups = [['GFP_'+str(i+1) for i in range(16)]]
design_object = Design(dynamic_model,fit_params,'D',
                       discrete_inputs = discrete_inputs,
                       continuous_inputs = continuous_inputs,
                       fixed_design = fixed_dict,
                       observ_groups = observ_groups,
                       options=opts)
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

eval_options={'Method':'Asymptotic',
              'Covariance':True,
              'FIM':True}
# call evaluate to compute the asymptotic covariance
diagnostics0 = dynamic_model.evaluate(init_design,fit_params,eval_options)
print(np.linalg.cond(diagnostics0['FIM']))
asymptotic_covariance = diagnostics0['Covariance']
#compute the asymptotic upper and lower 95% bounds
asymptotic_lower_bound = fit_params - 2*np.sqrt(np.diag(asymptotic_covariance))
asymptotic_upper_bound = fit_params + 2*np.sqrt(np.diag(asymptotic_covariance))
#***hidden_start****
fim_bnds_array = np.vstack((asymptotic_lower_bound,
                            fit_params,
                            asymptotic_upper_bound)).T
fim_bnds0 = pd.DataFrame(np.exp(fim_bnds_array),
                        index=parameter_names,
                        columns=['Lower','Estimate','Upper'])
print('')
print(fim_bnds0)
print('')

combined_design = pd.concat([init_design, exact_design], ignore_index=True)
# call evaluate to compute the asymptotic covariance
diagnostics_opt = dynamic_model.evaluate(combined_design,fit_params,eval_options)
print(np.linalg.cond(diagnostics_opt['FIM']))
asymptotic_covariance = diagnostics_opt['Covariance']
#compute the asymptotic upper and lower 95% bounds
asymptotic_lower_bound = fit_params - 2*np.sqrt(np.diag(asymptotic_covariance))
asymptotic_upper_bound = fit_params + 2*np.sqrt(np.diag(asymptotic_covariance))
#***hidden_start****
fim_bnds_array = np.vstack((asymptotic_lower_bound,
                            fit_params,
                            asymptotic_upper_bound)).T
fim_bnds_opt = pd.DataFrame(np.exp(fim_bnds_array),
                        index=parameter_names,
                        columns=['Lower','Estimate','Upper'])
print('')
print(fim_bnds_opt)
print('')