## Listing 1 ##############################################
import nloed as nl, casadi as cs
import numpy as np, pandas as pd
#CasADi input and parameter symbols
x = cs.SX.sym('x',1)
theta = cs.SX.sym('theta',4)
#Transform; ensures parameter positivity
q=cs.exp(theta)
#Expressions for the sampling statistics
mean = (q[0] + q[1]*x[0]**q[2]/
              (q[3]**q[2] + x[0]**q[2]))
s = 0.059
var = (s*mean)**2
#Create a sampling statistics vector
stats = cs.vertcat(mean, var)
#Create a CasADi function for f(.)
func = cs.Function('GFP',[x,theta],[stats])

## Listing 2 ##############################################
#Input and parameter name lists
xname = ['Light']
tname = ['ln_A0','ln_A','ln_N','ln_K']
#Assign a normal dist. to the output
obs_var = [(func,'Normal')]
#Call the NLoed Model constructor
model = nl.Model(obs_var,xname,tname)

## Listing 3 ##############################################
#Initial green light levels (%)
levels0 = [0.01, 0.01, 0.01,
           1.5,  1.5,  1.5,
           6,    6,    6,
           25,   25,   25,
           100,  100,  100]
#Initial GFP observations (a.u.)
obs0 = [ 580,  544,   543,
         766,  699,   570,
         2064, 2171,  4598,
         8814, 9919,  8860,
         9584, 10295, 10672]
#Dataframe containing initial dataset
data0 = pd.DataFrame(
           {'Light':levels0,
            'Variable':['GFP']*15,
            'Observation':obs0}) 
#Set fit options
opts={'InitParamBounds':[(3,8),(5,10),
                         (-1,1),(1,6)],
      'InitSearchNumber':7,
      'SearchBound':5.}
#Fit the model to get an estimate
est = model.fit(data0, options=opts)
#Extract parameter estimate vector
param = est.to_numpy().flatten()

## Listing 4 ##############################################
#Specify input dictionary for the design
inp = {'Inputs':['Light'],
      'Bounds':[(.01,100.)],
      'Structure':[['L1'],
                    ['L2'],
                    ['L3'],
                    ['L4']],
      'Initial':[[0.01],
                  [5.],
                  [20.],
                  [100.]]}
#Declare init design dataframe
design0 = pd.DataFrame({
            'Light':[.01,1.5,6,25,100],
            'Variable':['GFP']*5,
            'Replicates':[3]*5}) 
#Set the fixed design information
init = {'Weight':0.5,'Design':design0}
#Set objective type
obj = 'D'
#Instantiate the design object
opt_des = nl.Design(model,param,obj,
                  fixed_design=init,
                  continuous_inputs=inp)
#Generate a rounded exact design 
design1 = opt_des.round(15)

## Listing 5 ##############################################
#Merge initial and optimal designs
total = pd.concat([design0, design1],
                    ignore_index=True)
#Compute the design metrics
metrics = model.evaluate(total, param)
#Extract the covariance matrix
cov = metrics['Covariance'].to_numpy()
#Set the list of input vectors
inputs = pd.DataFrame({
        'Light':np.linspace(.1,100,100),
        'Variable':['GFP']*100})
#set options for prediction method
opts = {'Method':'Delta',
        'ObservationInterval':True}
#Generate predictions and intervals
pred = model.predict(inputs, param,
                covariance_matrix = cov,
                options=opts)

## Supplemental Listing 1 ##############################################
#optimal green light levels
levels1 = \
[.001]*4+[2.88]*5+[8.33]*4+[100]*2
#optimal GFP observations (a.u.)
obs1 = [433,  477,  441,  604,   1032,
        823,  849,  792,  954,   3555,
        2039, 3384, 3740, 10321, 11534]
#dataframe containing initial dataset
data1 = pd.DataFrame(
            {'Light':levels0+levels1,
            'Variable':['GFP']*30,
            'Observation':obs0+obs1}) 
#fit the model
est = model.fit(data1, start_param=param)
#extract parameter estimate vector
param = est['Estimate'].to_numpy().flatten()

## Supplemental Listing 2 ##############################################
import matplotlib.pyplot as plt
import scipy.stats as st
#compute  # of std. deviations for a 95% interval
thresh = st.norm.ppf(0.95)
#compute the covariance for the initial design
covariance_init = model.evaluate(design0, param)
#compute the standard parameter error for initial design
stderr_init = np.sqrt(np.diag(covariance_init['Covariance']))
#compute upper/lower confidence bounds for initial design
upper_bnd_init = param + thresh*stderr_init
lower_bnd_init = param - thresh*stderr_init
#combine the initial and optimal designs
opt_design_tot = pd.concat([design0, design1], ignore_index=True)
#compute the covariance for the combined design
covariance_opt = model.evaluate(opt_design_tot, param)
#compute the parameter std. error for the combined design
stderr_opt = np.sqrt(np.diag(covariance_opt['Covariance']))
#compute the upper/lower confidence bounds for combined design
upper_bnd_opt = param + thresh*stderr_opt
lower_bnd_opt = param - thresh*stderr_opt
#replicate the initial design for the duplicated scenario
rep_init_design_tot = pd.concat([design0, design0], ignore_index=True)
#compute the covariance for the replicated initial design
covariance_rept = model.evaluate(rep_init_design_tot, param)
#compute the parameter std. error for replicated initial design
stderr_rept = np.sqrt(np.diag(covariance_rept['Covariance']))
#compute the upper/lower confidence bounds for replicated design
upper_bnd_rept = param + thresh*stderr_rept
lower_bnd_rept = param - thresh*stderr_rept

## Supplemental Listing 3 ##############################################
#exponentiate the final parameter estimates to convert
#from log scale 
exp_vals = np.exp(param)
#exponentiate the confidence bounds for each scenario and compute
#the normalized size of the intervals in the original scale
exp_diff_opt = (np.exp(upper_bnd_opt)-np.exp(lower_bnd_opt))/exp_vals
exp_diff_init = (np.exp(upper_bnd_init)-np.exp(lower_bnd_init))/exp_vals
exp_diff_rept = (np.exp(upper_bnd_rept)-np.exp(lower_bnd_rept))/exp_vals
#set the width of the bars
width = 0.25 
#create plot x labels names
labels = [r'$\alpha_o$',r'$\alpha$',r'n', r'K']
#set the x axis settings
x = np.arange(len(labels)) 
# create the figure
fig, ax = plt.subplots()
#plot the initial design CI interval sizes
rects1 = ax.bar(x - width-0.01, exp_diff_init.flatten(), width, label='Initial Design')
#plot the duplicated initial design CI interval sizes
rects2 = ax.bar(x, exp_diff_rept.flatten(), width, label='Repeated Initial Design')
#plot the combined (optimal + initial) design CI interval sizes
rects3 = ax.bar(x + width+0.01, exp_diff_opt.flatten(), width, label='Optimal Design')
#set y axis title
ax.set_ylabel('Interval Size as \% of MLE Parameter Value')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc=2)
fig.tight_layout()
#generate plot
plt.show()

## Supplemental Listing 4 ##############################################
#create a plot
fig, ax = plt.subplots()
#plot observation interval
ax.fill_between(pred['Inputs','Light'],
                pred['Observation','Lower'],
                pred['Observation','Upper'],
                alpha=0.3,
                color='C2',
                label='95% Observation Interval')
#plot mean model prediction
ax.plot(pred['Inputs','Light'], pred['Prediction','Mean'], '-',color='C4',label='Mean Observation')
#plot initial dataset
ax.plot(levels0, obs0, 'o', color='C0',label='Initial Data')
ax.plot(levels1, obs1, 'o', color='C1',label='Optimal Data')
ax.set_xlabel('Green Light %')
ax.set_ylabel('Mean Batch GFP Expression (a.u.)')
ax.legend(loc=4)
plt.show()

#######################
# Dynamic model example
#######################

## Supplemental Listing 5 ##############################################
# import casadi as cs
# import nloed as nl
#create state variable vector
states = cs.SX.sym('states',2)
#create control input symbol
inducer = cs.SX.sym('inducer')
#create parameter symbol vector
parameters = cs.SX.sym('parameters',6)
#log-transformed parameters
alpha = cs.exp(parameters[0]) 
K = cs.exp(parameters[1]) 
delta = cs.exp(parameters[2])
beta = cs.exp(parameters[3]) 
L = cs.exp(parameters[4]) 
gamma = cs.exp(parameters[5])
#create symbolic RHS
rhs = cs.vertcat(alpha*inducer/(K + inducer) - delta*states[0],
                 beta*states[0]/(L + states[0]) - gamma*states[1])
#create casadi RHS function
rhs_func = cs.Function('rhs_func',[states,inducer,parameters],[rhs])

## Supplemental Listing 6 ##############################################
#time step size
dt = 1
# Create symbolics for RK4 integration
k1 = rhs_func(states, inducer, parameters)
k2 = rhs_func(states + dt/2.0*k1, inducer, parameters)
k3 = rhs_func(states + dt/2.0*k2, inducer, parameters)
k4 = rhs_func(states + dt*k3, inducer, parameters)
state_step = states + dt/6.0 * (k1 + 2*k2 + 2*k3 + k4)
# Create a function to perform one step of the RK integration
step_func = cs.Function('step_func',[states, inducer, parameters],[state_step])

## Supplemental Listing 7 ##############################################
#create a symbol for the initial inducer level
initial_inducer = cs.SX.sym('init_inducer')
#define the steady state initial states in terms of the initial inducer
init_mrna = (alpha/delta)*initial_inducer/(K+initial_inducer)
ini_prot = (beta/gamma)*init_mrna/(L+init_mrna)
# zip the initial states into a vector
initial_states = cs.vertcat(init_mrna, ini_prot)

## Supplemental Listing 8 ##############################################
#3 samples per cntrl interval, 4 cntrl intervals
#control intervals are 1+2+3=6 steps long:
# cntrl_int1    cntrl_int2    cntrl_int3
#|-1---2-----3||-1---2-----3||-1---2-----3||-1---2-----3|
#set number of control intervals
num_cntrl_intervals = 4
#create a vector for inducer levels in each control interval
inducer_vector = cs.SX.sym('inducer_vec',4)
#define a sample pattern to apply in each control interval
sample_pattern= [1,2,3] 
#lists to store symbols for each sample point, and times of each sample
sample_list, times = [], []
# set the initial states and initialize the step counter
current_state, step_counter = initial_states, 0
#loop over control invervals
for interval in range(num_cntrl_intervals):
  # loop over sample pattern
  for num_stps in sample_pattern:
    #iterate steps indicated by sample pattern
    for k in range(num_stps):
      #propagate the state variables via integration
      current_state = step_func(current_state, inducer_vector[interval], parameters)
      step_counter+=1
    #save the state symbols and times of each sample
    sample_list.append(current_state)
    times.append(step_counter*dt)

## Supplemental Listing 9 ##############################################
#merge all inducer levels into a single inputs vector
inputs = cs.vertcat(initial_inducer,inducer_vector)
# create list for observation structure 
observation_list = []
#create list to store response names
observ_names, observ_type, observ_times = [], [], []
# loop over samples (time points)
for i in range(len(sample_list)):
  #create a unique name for mrna and prot samples
  mrna_name = 'mrna_'+'t'+"{0:0=2d}".format(times[i])
  prot_name = 'prot_'+'t'+"{0:0=2d}".format(times[i])
  #create mean and var tuple for mrna and prot observ.
  mrna_stats = cs.vertcat(sample_list[i][0], 0.005)
  prot_stats = cs.vertcat(sample_list[i][1], 0.005)
  #create casadi function for mrna and prot stats
  mrna_func = cs.Function(mrna_name,[inputs,parameters],[mrna_stats])
  prot_func = cs.Function(prot_name,[inputs,parameters],[prot_stats])
  #append the casadi function and distribution type to obs struct
  observation_list.extend([(mrna_func,'Normal'), (prot_func,'Normal')])
  #store observation names, useful for plotting
  observ_names.extend([mrna_name,prot_name])
  #store observation type
  observ_type.extend(['RNA','Prot'])
  #store observation time
  observ_times.extend([times[i]]*2)

## Supplemental Listing 10 ##############################################
#list the inpit and parameter names
input_names = ['Init_Inducer','Inducer_1','Inducer_2','Inducer_3','Inducer_4']
parameter_names = ['log_Alpha','log_K','log_Delta','log_Beta','log_L','log_Gamma']
#instantiate the model object
model_object = nl.Model(observation_list, input_names, parameter_names)
