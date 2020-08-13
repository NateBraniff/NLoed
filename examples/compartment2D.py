import casadi as cs
import pandas as pd
import numpy as np
import re as re
import copy as cp
import matplotlib.pyplot as plt
from nloed import Model
from nloed import design


####################################################################################################
# SET UP RHS OF ODE
####################################################################################################

#states and controls
# y = cs.MX.sym('y',2)
# u = cs.MX.sym('u')
# p = cs.MX.sym('p',4)
#####  -or-  #####
y = cs.SX.sym('y',2)
u = cs.SX.sym('u')
p = cs.SX.sym('p',4)

rhs = cs.vertcat(cs.exp(p[0])*u -cs.exp(p[1])*y[0], cs.exp(p[2])*y[0]-cs.exp(p[3])*y[1])
ode = cs.Function('ode',[y,u,p],[rhs])

dt = 1

####################################################################################################
# DEFINE RK-4 STEP ALGEBRA
####################################################################################################

# # Create symbolics for RK4 integration, as shown in Casadi examples
k1 = ode(y, u, p)
k2 = ode(y + dt/2.0*k1, u, p)
k3 = ode(y + dt/2.0*k2, u, p)
k4 = ode(y + dt*k3, u, p)
y_step = y+dt/6.0*(k1+2*k2+2*k3+k4)
# Create a function to perform one step of the RK integration
step = cs.Function('step',[y, u, p],[y_step])
#####  -or-  #####
# ode_sys = {'x':y, 'p':cs.vertcat(u,p), 'ode':rhs}
# opts = {'tf':dt}
# step = cs.integrator('F', 'cvodes', ode_sys, opts)

####################################################################################################
# DEFINE TIME DISCRETIZATION, AND CONSTRUCT DETERMINISTIC MODEL VIA INTEGRATION
####################################################################################################

steps_per_sample = [1,2,3,5] 
#steps_per_sample = [1,9] 
samples_per_cntrl = 1
cntrls_per_run = 3

# y0 = cs.MX.sym('y0',2)
# uvec = cs.MX.sym('uvec',3)
#####  -or-  #####
y0 = cs.SX.sym('y0',2)
uvec = cs.SX.sym('uvec',3)
x = cs.vertcat(y0,uvec)

#Loop over the ste function to create symbolics for integrating across a sample interval
y_sample = y0
sample_list=[]
times=[]
cntr=0
for i in range(cntrls_per_run):
  for num_stps in steps_per_sample:
    for k in range(num_stps):
      y_sample = step(y_sample, uvec[i], p)
      #####  -or-  #####
      #y_sample = step(x0=y_sample, p=cs.vertcat(uvec[i], p))['xf']
      cntr+=1
    sample_list.append(y_sample)
    times.append(cntr*dt)

####################################################################################################
# CONSTRUCT MRNA and PROTEIN MEAN/VAR STATS, CREATE INTO MODEL
####################################################################################################

ode_response = []
response_names=[]
replicates=[]
for i in range(len(sample_list)):
  reps=10
  mrna_name = 'mrna_'+'t'+"{0:0=2d}".format(times[i])
  mrna_stats = cs.vertcat(sample_list[i][0], 0.001)
  mrna_func = cs.Function(mrna_name,[x,p],[mrna_stats])
  ode_response.append((mrna_func,'Normal'))
  response_names.append(mrna_name)
  replicates.append(reps)

  prot_name = 'prot_'+'t'+"{0:0=2d}".format(times[i])
  prot_stats = cs.vertcat(sample_list[i][1], 0.001)
  prot_func = cs.Function(prot_name,[x,p],[prot_stats])
  ode_response.append((prot_func,'Normal'))
  response_names.append(prot_name)
  replicates.append(reps)

xnames = ['mrna_ic','prot_ic','cntrl_1','cntrl_2','cntrl_3']
pnames = ['alpha','delta','beta','gamma']

ode_model = Model(ode_response,xnames,pnames)

####################################################################################################
# MANUALLY CREATE A DESIGN FOR TESTING
####################################################################################################

design = pd.DataFrame({ 'mrna_ic':[1],
                        'prot_ic':[1],
                        'cntrl_1':[0.1],
                        'cntrl_2':[1],
                        'cntrl_3':[0.1]})

design=design.reindex(design.index.repeat(len(response_names)))
design['Variable'] = response_names
design['Replicats'] = replicates
design = design.sort_values(by='Variable').reset_index()

####################################################################################################
# TESTING PREDICTIONS AND EVAL
####################################################################################################

predict_inputs = pd.DataFrame({ 'mrna_ic':[1]*len(response_names),
                                'prot_ic':[1]*len(response_names),
                                'cntrl_1':[0.1]*len(response_names),
                                'cntrl_2':[1.0]*len(response_names),
                                'cntrl_3':[0.1]*len(response_names),
                                'Variable':response_names})

true_pars = [np.log(0.5),np.log(1.1),np.log(2.1),np.log(0.3)]
predictions = ode_model.predict(predict_inputs,true_pars)

digit_re=re.compile('[a-z]+_t(\d+)')
type_re=re.compile('([a-z]+)_t\d+')

predictions['Prediction','Time'] = predictions['Inputs','Variable'].apply(lambda x: int(digit_re.search(x).group(1)))
predictions['Prediction','Type'] = predictions['Inputs','Variable'].apply(lambda x: type_re.search(x).group(1))

mrna_pred = predictions.loc[predictions['Prediction','Type'] == 'mrna']
prot_pred = predictions.loc[predictions['Prediction','Type'] == 'prot']

ax1 = mrna_pred.plot.scatter(x=('Prediction','Time'),y=('Prediction','Mean'),c='Red')
ax2 = prot_pred.plot.scatter(x=('Prediction','Time'),y=('Prediction','Mean'),c='Red')

ode_data = ode_model.sample(design,true_pars,design_replicats=3)

data=cp.deepcopy(ode_data[0])

data['Time'] = data['Variable'].apply(lambda x: int(digit_re.search(x).group(1)))
data['Type'] = data['Variable'].apply(lambda x: type_re.search(x).group(1))

mrna_obs = data.loc[data['Type'] == 'mrna']
prot_obs = data.loc[data['Type'] == 'prot']

mrna_obs.plot.scatter(x=('Time'),y=('Observation'),ax=ax1)
prot_obs.plot.scatter(x=('Time'),y=('Observation'),ax=ax2)
plt.show()


fit_options={'Confidence':'Intervals',
            'InitParamBounds':[(-3,2)]*4,
            'InitSearchNumber':3,
            'MaxSteps':100000}
ode_fit = ode_model.fit(ode_data, options=fit_options)


opts={'Covariance':True,'Bias':True,'MSE':True}
eval_dat = ode_model.evaluate(design,true_pars,opts)

t=0

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


