## Listing 1 ##############################################
import nloed as nl, casadi as cs
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
#CasADi input and parameter symbols
x = cs.SX.sym('x',1)
p = cs.SX.sym('p',4)
# Reparameterize to ensure positivity
q=cs.exp(p)
#Compute the sampling statistics
mean = q[0] + q[1]*x[0]**q[2]/(q[3]**q[2]+x[0]**q[2])
var = (0.059*mean)**2
#Create a sampling statistics vector
stats = cs.vertcat(mean, var)
#Create a CasADi function for f()
hill_f = cs.Function('GFP',[x,p],[stats])

## Listing 2 ##############################################
#Input and parameter name lists
xname = ['Light']
pname = ['ln_A0','ln_A','ln_N','ln_K']
#Assign a normal dist. to the model
obs_var = [(hill_f,'Normal')]
#Call the NLoed Model constructor
model=nl.Model(obs_var,xname,pname)

## Listing 3 ##############################################
#Initial green light levels
x_lvls0 = \
[.001]*3+[1.5]*3+[6.]*3+[25]*3+[100]*3
# Initial GFP observations (a.u.)
y_obs0 = [580,  544,   543,
          766,  699,   570,
          2064, 2171,  4598,
          8814, 9919,  8860,
          9584, 10295, 10672]
# Dataframe containing initial dataset
data0 = pd.DataFrame(
            {'Light':x_lvls0,
            'Variable':['GFP']*15,
            'Observation':y_obs0}) 
# Set fit options
opts={'InitParamBounds':[(3,8),(5,10),(-1,1),(1,6)],
      'InitSearchNumber':7,
      'SearchBound':5.}
#Fit the model
est = model.fit(data0, options=opts)
#Extract parameter estimate vector
param = est['Estimate'].to_numpy().flatten()

print(np.exp(est))

## Listing 4 ##############################################
#Specify input restrictions for design
inputs={'Inputs':['Light'],
        'Bounds':[(.01,100.)],
        'Structure':[['L1'],
                     ['L2'],
                     ['L3'],
                     ['L4']],
        'Initial':[[0.01],
                   [5.],
                   [20.],
                   [100.]]}
#Set objective and nominal parameters
obj = 'D'
#Declare init design dataframe
design0 = pd.DataFrame({'Light':[.01]+[1.5]+[6]+[25]+[100],
                        'Variable':['GFP']*5,
                        'Replicates':[3]*5}) 
#Set the fixed design information
init = {'Weight':0.5,'Design':design0}
#Instantiate the design object
opt_des = nl.Design(model,param,obj,
                 fixed_design=init,
                 continuous_inputs=inputs)
#Generate a rounded exact design 
design1 = opt_des.round(15)

###########################################################################################
## Re-fit #################################################################################
###########################################################################################
#Optimal green light levels
x_lvls1 = \
[.001]*4+[2.88]*5+[8.33]*4+[100]*2
# Optimal GFP observations (a.u.)
y_obs1 = [433,  477,  441,  604,
          1032, 823,  849,  792, 954,
          3555, 2039, 3384, 3740,
          10321, 11534]
# Dataframe containing initial dataset
data1 = pd.DataFrame(
            {'Light':x_lvls0+x_lvls1,
            'Variable':['GFP']*30,
            'Observation':y_obs0+y_obs1}) 

#Fit the model
est = model.fit(data1, start_param=param)
#Extract parameter estimate vector
param = est['Estimate'].to_numpy().flatten()

print(np.exp(est))


## Listing 5 ##############################################
#Merge initial and optimal designs
design_tot = pd.concat([design0, design1], ignore_index=True)
#Compute the covariance matrix
cov = model.evaluate(design_tot, param).to_numpy()
#convert the covariance matrix to numpy
#covariance_matrix_opt = covariance_opt['Covariance'].to_numpy()
#Set the list of input vectors
inputs = pd.DataFrame({'Light':np.linspace(0.1,100,100),
                       'Variable':['GFP']*100})
#set options for prediction method
opts = {'Method':'Delta',
        'ObservationInterval':True}
#Generate predictions and intervals
predictions = model.predict(inputs,
                            param,
                            covariance_matrix = cov,
                            options=opts)

###########################################################################################
## Plotting Prediction Intervals ##########################################################
###########################################################################################
#create plot
fig, ax = plt.subplots()
#plot observation interval
ax.fill_between(predictions['Inputs','Light'],
                predictions['Observation','Lower'],
                predictions['Observation','Upper'],
                alpha=0.3,
                color='C2',
                label='95% Observation Interval')
#plot mean model prediction
ax.plot(predictions['Inputs','Light'], predictions['Prediction','Mean'], '-',color='C4',label='Mean Observation')
#plot initial dataset
ax.plot(x_lvls0, y_obs0, 'o', color='C0',label='Initial Data')
ax.plot(x_lvls1, y_obs1, 'o', color='C1',label='Optimal Data')
ax.set_xlabel('Green Light %')
ax.set_ylabel('Mean Batch GFP Expression (a.u.)')
ax.set_title('Model Prediction and Observation Uncertainty Interval\n \
After Fitting to the Combined Dataset')
ax.legend(loc=4)
plt.show()

###########################################################################################
## Plotting CIs ###########################################################################
###########################################################################################


#get estimated covariance, bias and MSE of parameter fit (use asymptotic method here) 
#opts={'Method':'Asymptotic','FIM':True,'Covariance':True,'Bias':True,'MSE':True,'SampleNumber':500}
#get estimated covariance, bias and MSE of parameter fit (use asymptotic method here) 
covariance_init = model.evaluate(design0, param)
stderr_init = np.sqrt(np.diag(covariance_init['Covariance']))
upper_bnd_init = param + 2*stderr_init
lower_bnd_init = param - 2*stderr_init

#eval the optimal design + initial design
opt_design_tot = pd.concat([design0, design1], ignore_index=True)
covariance_opt = model.evaluate(opt_design_tot, param)
stderr_opt = np.sqrt(np.diag(covariance_opt['Covariance']))
upper_bnd_opt = param + 2*stderr_opt
lower_bnd_opt = param - 2*stderr_opt

#repeat init design 
rep_init_design_tot = pd.concat([design0, design0], ignore_index=True)
#get estimated covariance, bias and MSE of parameter fit (use asymptotic method here) 
covariance_rept = model.evaluate(rep_init_design_tot, param)
stderr_rept = np.sqrt(np.diag(covariance_rept['Covariance']))
upper_bnd_rept = param + 2*stderr_rept
lower_bnd_rept = param - 2*stderr_rept

#PLOT THE COMPARED CI's
exp_vals = np.exp(param)
#exp_vals = np.tile(np.exp(nom_param),(1,3))
exp_diff_opt = (np.exp(upper_bnd_opt)-np.exp(lower_bnd_opt))/exp_vals
exp_diff_init = (np.exp(upper_bnd_init)-np.exp(lower_bnd_init))/exp_vals
exp_diff_rept = (np.exp(upper_bnd_rept)-np.exp(lower_bnd_rept))/exp_vals

labels = [r'$\alpha_o$',r'$\alpha$',r'n', r'K']
x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x - width-0.01, exp_diff_init.flatten(), width, label='Initial Design')
rects2 = ax.bar(x, exp_diff_rept.flatten(), width, label='Repeated Initial Design')
rects3 = ax.bar(x + width+0.01, exp_diff_opt.flatten(), width, label='Optimal Design')
ax.set_ylabel('Interval Size as % of MLE Parameter Value')
ax.set_title('Comparison of 95% Confidence Interval Sizes\n Expressed in % Error of MLE Values')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc=2)
fig.tight_layout()
plt.show()