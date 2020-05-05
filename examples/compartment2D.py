import casadi as cs
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from nloed import Model
from nloed import design

#states and controls
y = cs.SX.sym('y',2)
u = cs.SX.sym('u')
p = cs.SX.sym('p',4)
rhs = cs.vertcat(cs.exp(p[0])*u -cs.exp(p[1])*y[0], cs.exp(p[2])*y[0]-cs.exp(p[3])*y[1])
ode = cs.Function('ode',[y,u,p],[rhs])

dt = 1
# Create symbolics for RK4 integration, as shown in Casadi examples
k1 = ode(y, u, p)
k2 = ode(y + dt/2.0*k1, u, p)
k3 = ode(y + dt/2.0*k2, u, p)
k4 = ode(y + dt*k3, u, p)
y_step = y+dt/6.0*(k1+2*k2+2*k3+k4)
# Create a function to perform one step of the RK integration
step = cs.Function('step',[y, u, p],[y_step])

##########################################################################################

steps_per_sample = [1,2,3,5] 
samples_per_cntrl = 1
cntrls_per_run = 3

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
      cntr+=1
    sample_list.append(y_sample)
    times.append(cntr*dt)

ode_response = []
response_names=[]
for i in range(len(sample_list)):

  mrna_stats = cs.vertcat(sample_list[i][0], 0.01)
  mrna_func = cs.Function('mrna_t'+str(times[i]),[x,p],[mrna_stats])
  ode_response.append((mrna_func,'Normal'))
  response_names.append('mrna_t'+str(times[i]))

  prot_stats = cs.vertcat(sample_list[i][1], 0.01)
  prot_func = cs.Function('prot_t'+str(times[i]),[x,p],[prot_stats])
  ode_response.append((prot_func,'Normal'))
  response_names.append('prot_t'+str(times[i]))

xnames = ['mrna_ic','prot_ic','cntrl_1','cntrl_2','cntrl_3']
pnames = ['alpha','delta','beta','gamma']

ode_model = Model(ode_response,xnames,pnames)

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

mrna_pred.plot.scatter(x=('Prediction','Time'),y=('Prediction','Mean'))
prot_pred.plot.scatter(x=('Prediction','Time'),y=('Prediction','Mean'))
plt.show()

design = pd.DataFrame({ 'mrna_ic':[0.5,1,2],
                        'prot_ic':[1,0.1,2],
                        'cntrl_1':[0.1,1,1],
                        'cntrl_2':[0.2,0,0.5],
                        'cntrl_3':[0.3,2,0.1],
                        'mrna_t1':[10,10,10],
                        'prot_t1':[10,10,10],
                        'mrna_t2':[10,10,10],
                        'prot_t2':[10,10,10],
                        'mrna_t3':[10,10,10],
                        'prot_t3':[10,10,10]})

ode_data = ode_model.sample(design,true_pars)
#
fit_options={'Confidence':'Intervals',
            'InitParamBounds':[(-3,2)]*4,
            'InitSearchNumber':11,
            'MaxSteps':100000}
ode_fit = ode_model.fit(ode_data, options=fit_options)

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


