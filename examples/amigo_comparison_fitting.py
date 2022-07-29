#######################
# AMIGO example
#######################

import casadi as cs
import nloed as nl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Supplemental Listing 11 ##############################################
#create state variable vector
states = cs.SX.sym('states',2)
#create parameter symbol vector
parameters = cs.SX.sym('parameters',4)
# names states
c_b=states[0]
c_s=states[1]
#log-transformed parameters
mumax=cs.exp(parameters[0]) 
ks=cs.exp(parameters[1]) 
kd=cs.exp(parameters[2]) 
yld=cs.exp(parameters[3])  
#create symbolic RHS
rhs = cs.vertcat((mumax*c_s*c_b)/(ks+c_s)-kd*c_b,
                -((mumax*c_s*c_b)/(ks+c_s))/yld)
#create casadi RHS function
rhs_func = cs.Function('rhs_func',[states,parameters],[rhs])

## Supplemental Listing 12 ##############################################
#time step size
dt = 0.1
# Create symbolics for RK4 integration
k1 = rhs_func(states, parameters)
k2 = rhs_func(states + dt/2.0*k1, parameters)
k3 = rhs_func(states + dt/2.0*k2, parameters)
k4 = rhs_func(states + dt*k3, parameters)
state_step = states + dt/6.0 * (k1 + 2*k2 + 2*k3 + k4)
# Create a function to perform one step of the RK integration
step_func = cs.Function('step_func',[states, parameters],[state_step])

## Supplemental Listing 13 ##############################################
#create a symbol for the initial conditions
init_biomass = 2#cs.exp(parameters[4])#cs.SX.sym('init_biomass')
init_substrate = 30#cs.exp(parameters[5])#cs.SX.sym('init_substrate')
# zip the initial states into a vector
initial_states = cs.vertcat(init_biomass, init_substrate)

## Supplemental Listing 14 ##############################################
#10 sampling intervals, each 10 steps long
#set number of control intervals
num_cntrl_intervals = 10
#set number of numerical steps in each interval
num_stps = 10
#lists to store symbols for each sample point, and times of each sample
sample_list, times = [], []
# set the initial states and initialize the step counter
current_state, step_counter = initial_states, 0
#add ICs
sample_list.append(current_state)
times.append(step_counter*dt)
#loop over control invervals
for interval in range(num_cntrl_intervals):
    #iterate steps indicated by sample pattern
    for k in range(num_stps):
      #propagate the state variables via integration
      current_state = step_func(current_state, parameters)
      step_counter+=1
    #save the state symbols and times of each sample
    sample_list.append(current_state)
    times.append(step_counter*dt)

## Supplemental Listing 15 ##############################################
#merge all inducer levels into a single inputs vector
inputs = cs.vertcat([])
# create list for observation structure 
observation_list = []
#create list to store response names
observ_names, observ_type, observ_times = [], [], []
# loop over samples (time points)
for i in range(len(sample_list)):
  #create a unique name for mrna and prot samples
  biomass_name = 'biomass_'+'t'+"{:05.2F}".format(times[i]).replace(".", "_")
  substrate_name = 'substrate_'+'t'+"{:05.2F}".format(times[i]).replace(".", "_")
  #create mean and var tuple for mrna and prot observ.
  biomass_stats = cs.vertcat(sample_list[i][0], 0.1)
  substrat_stats = cs.vertcat(sample_list[i][1], 0.1)
  #create casadi function for mrna and prot stats
  biomass_func = cs.Function(biomass_name,[inputs,parameters],[biomass_stats])
  substrat_func = cs.Function(substrate_name,[inputs,parameters],[substrat_stats])
  #append the casadi function and distribution type to obs struct
  observation_list.extend([(biomass_func,'Normal'), (substrat_func,'Normal')])
  #store observation names, useful for plotting
  observ_names.extend([biomass_name,substrate_name])
  #store observation type
  observ_type.extend(['Biomass','Substrate'])
  #store observation time
  observ_times.extend([times[i]]*2)

## Supplemental Listing 16 ##############################################
#list the inpit and parameter names
input_names = []
parameter_names = ['log_muMax','log_Ks','log_Kd','yield']
#instantiate the model object
model_object = nl.Model(observation_list, input_names, parameter_names)

## Supplemental Listing 17 ##############################################

#0.411636590271240 6.744866335047434 0.039525613375224 0.482112704417673
# mumax=np.log(0.4)
# ks=np.log(5.0)
# kd=np.log(0.05)
# yld=np.log(0.5)
# cb0=np.log(0.5)
# cs0=np.log(1.5)
mumax=np.log(0.41)
ks=np.log(6.7)
kd=np.log(0.039)
yld=np.log(0.48)
#cb0=np.log(0.5)
#cs0=np.log(1.5)

fit_params = [0.017232973289168913, 2727931987.1861606, 0.25462881536561893, 4.233072590363622e-12]#[mumax,ks,kd,yld]

init_inputs = pd.DataFrame({#'Init_Biomass':[0.5]*20,
                            #'Init_Substrate':[1.5]*20,
                            'Variable':['biomass_t'+"{0:0=2d}".format(i)+"_00" for i in range(0,11)]
                                        +['substrate_t'+"{0:0=2d}".format(i)+"_00" for i in range(0,11)]})


predictions = model_object.predict(init_inputs,fit_params)

biomass_dat =  [0.642563, 
                0.750319, 
                0.755610,
                0.831913, 
                0.658943, 
                0.370148, 
                0.848267, 
                0.920692, 
                1.140131, 
                0.092449, 
                0.850306] 

substrate_dat =[1.418734,
                1.656654,
                1.668335,
                1.836808,
                1.454901,
                0.817263,
                1.872915,
                2.032825,
                2.517332,
                0.204122,
                1.877418]

time_sim = [float(i) for i in range (0,11)]
time_dat = [float(i) for i in range (0,11)]

init_design = pd.DataFrame({#'Init_Biomass':[3]*20,
                            #'Init_Substrate':[20]*20,
                            'Variable':['biomass_t'+"{0:0=2d}".format(i)+"_00" for i in range(0,11)]
                                        +['substrate_t'+"{0:0=2d}".format(i)+"_00" for i in range(0,11)],
                            'Replicates':[1]*22})

samp_dat=model_object.sample(init_design,fit_params)

#plot biomass simulation
plt.plot(time_sim,predictions[('Prediction','Mean')].iloc[0:11])
#plot biomass data
plt.scatter(time_dat,biomass_dat)
#plot biomass sim dat
plt.plot(time_sim,samp_dat[('Observation')].iloc[0:11],'.')

#plot substrate simulation
plt.plot(time_sim,predictions[('Prediction','Mean')].iloc[11:22])
#plot substrate data
plt.scatter(time_dat,substrate_dat)
#plot substrate sim dat
plt.plot(time_sim,samp_dat[('Observation')].iloc[11:22],'x')
plt.show()

init_data = init_inputs.copy()
init_data["Observation"] = biomass_dat + substrate_dat

fit_vals=model_object.fit(init_data,start_param = fit_params)


t=0




#model_object.


# 
# true_param = np.log([2,10,2,3])
# init_data = model_object.sample(init_design,true_param)
# init_data['Observation'] = [1.933326,
#                             1.961151,
#                             2.042274,
#                             7.501225,
#                             7.143189,
#                             6.694389,
#                             9.898560,
#                             9.361815,
#                             10.197202,
#                             11.238234,
#                             11.163177,
#                             11.946090]