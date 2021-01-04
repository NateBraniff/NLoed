""" 
Add a docstring
"""
import numpy as np
import pandas as pd
import casadi as cs
from nloed import Design

###block 1
from nloed import Model
#create casadi symbols for the inputs
x = cs.SX.sym('x',2)
#create casadi symbols for the parameters
theta = cs.SX.sym('theta',4)
#define y1 sampling statistics; mean and variance
mean_y1 = theta[0] + theta[1]*x[0] + theta[3]*x[0]*x[1] 
var_y1 = 0.1
#define y2 sampling statistics; mean and variance
rate_y2 = cs.exp(theta[0] + theta[2]*x[1] + theta[3]*x[0]*x[1])
#create a casadi function for y1 stats
eta_y1 = cs.vertcat(mean_y1, var_y1)
func_y1 = cs.Function('y1',[x,theta],[eta_y1])
#create a casadi function for y2 stats
eta_y2 = rate_y2
func_y2 = cs.Function('y2',[x,theta],[eta_y2])

###block 2
#create observation list
observ_list = [(func_y1,'Normal'),(func_y2,'Poisson')]
#creat input name list
input_names = ['x1','x2']
#create parameter name list
parameter_names = ['Theta0','Theta1','Theta2','Theta3']
#create NLOED Model
model_object = Model(observ_list, input_names, parameter_names)

###block 3
design = pd.DataFrame({ 'x1':[0,-1,2,3]*2,
                        'x2':[1,1,-1,0]*2,
                        'Variable':['y1']*4 + ['y2']*4,
                        'Replicates':[3,1,2,2]*2})
print('')
print('')
print(design)
print('')
print('')

###block 4
design = pd.DataFrame({ 'x1':[0,-1,2,3]*2,
                        'x2':[1,1,-1,0]*2,
                        'Variable':['y1']*4 + ['y2']*4,
                        'Replicates':[3,1,2,2]*2})
#set nominal parameter values
param = [0.1, 2, 0.4, 1.3]
#declare specific options
eval_opts={'Method':'MonteCarlo',
           'FIM':True,
           'Covariance':True,
           'Bias':True,
           'MSE':True,
           'SampleNumber':100}
#call the evaluate() function
eval_info = model_object.evaluate(design, param, eval_opts)
#print the resulting evaluation
#print(eval_info)
print('')
print('')
print(eval_info)
print('')
print('')

### block 5
#define a design
design = pd.DataFrame({ 'x1':[0,-1,2,3]*2,
                        'x2':[1,1,-1,0]*2,
                        'Variable':['y1']*4 + ['y2']*4,
                        'Replicates':[3,1,2,2]*2})
#set nominal parameter values
param = [0.1, 2, 0.4, 1.3]
#call the sample() function
dataset = model_object.sample(design, param, design_replicates=1)
#print the resulting dataset
#print(dataset)
print('')
print('')
print(dataset)
print('')
print('')

###block 6
# set specific fitting options
fit_opts={'Confidence':'None',
             'InitParamBounds':[(-1,1),(-1,1),(-1,1),(-1,1)],
             'InitSearchNumber':7,
             'RadialNumber':60}
#call the fit() function
fit_info = model_object.fit(dataset, options=fit_opts)
#print the fitting information
#print(fit_info)
print('')
print('')
print(fit_info)
print('')
print('')

### block 7
#create three datasets
data1, data2, data3 = dataset = model_object.sample(design, param, design_replicates=3)
# set up specific options for fitting
fit_opts={'Confidence':'Intervals'}
#combine datasets into a single list
datasets = [data1, data2, data3]
#call the fitting procedure
fit_info = model_object.fit(datasets, options=fit_opts)
#print the fitting information
print(fit_info)
print('')
print('')
print(fit_info)
print('')
print('')

#define the inputs for predict()
predict_inputs = pd.DataFrame({ 'x1':[-1,1,-1,1]*2,
                                'x2':[-1,-1,1,1]*2,
                               'Variable':['y1']*4 + ['y2']*4})
#specify the parameter values
params = [0.1, 2, 0.4, 1.3]
#generate the predictions
predictions = model_object.predict(predict_inputs, params)
#display the predicted outputs
print(predictions)
print('')
print('')
print(predict_inputs)
print('')
print('')
print(predictions)
print('')
print('')


#define a covariance matrix for the parameters
cov_mat = np.diag(np.array(param)*0.05)
#specify desired option values
predict_opts = {'Method':'MonteCarlo',
                'PredictionInterval':True,
                'ObservationInterval':True,
                'Sensitivity':True}
#generate the predictions
predictions = model_object.predict(predict_inputs, params, cov_mat, predict_opts)
#display the predicted outputs
print(predictions)
print('')
print('')
print(predictions)
print('')
print('')

####################################################################################################
# GENERATE DESIGN
####################################################################################################

true_param = [1,1,1,1]

#discrete_inputs={'Inputs':['x1'],'Candidates':[[-1],[-.5],[0],[.5],[1]]}
#continuous_inputs={'Inputs':['x2'],'Bounds':[(-1,1)],'Structure':[['level1'],['level2']]}
discrete_inputs={'Inputs':['x1','x2'],'Candidates':[[-1, 0, 1],[-1, 0, 1]]}
#discrete_inputs={'Inputs':['x1','x2'],'Grid':[[-1,-1],[0,-1],[1,-1],[-1,0],[0,0],[1,0],[-1,1],[0,1],[1,1],[3,3]]}
## continuous_inputs={'Inputs':['x1','x2'],'Bounds':[(-1,1),(-1,1)],'Structure':[['x1_lvl1','x2_lvl1'],['x1_lvl1','x2_lvl2'],['x1_lvl2','x2_lvl2']]}
design_object = Design(lin_model, true_param, 'D', discrete_inputs)


relaxed_design = design_object.relaxed()
print('')
print('')
print(relaxed_design)
print('')
print('')


sample_size = 10
exact_design = design_object.round(sample_size)

print('')
print('')
print(exact_design)
print('')
print('')

####################################################################################################
# GENERATE SAMPLE DATA & FIT MODEL
####################################################################################################

#generate some data, to stand in for an initial experiment
data = lin_model.sample(exact_design,true_param, design_replicats=3)

print('')
print('')
print(data)
print('')
print('')

#pass some additional options to fitting alglorithm, including Profile likelihood
fit_options={'Confidence':'Intervals',
             'InitParamBounds':[(-1,1),(-1,1),(-1,1),(-1,1)],
             'InitSearchNumber':7,
             'SearchBound':5.}
#fit the model to the init data
fit_info = lin_model.fit(data, options=fit_options)

print('')
print('')
print(fit_info)
print('')
print('')

fit_params = fit_info['Estimate'].to_numpy()[0].flatten()

fit_info = fit_params

print('')
print('')
print(fit_info)
print('')
print('')


design = pd.DataFrame({'x1':[-1,1,-1,1]*2,
                       'x2':[-1,-1,1,1]*2,
                       'Variable':['y1']*4 + ['y2']*4,
                       'Replicats':[3]*8})
print('')
print('')
print(design)
print('')
print('')


####################################################################################################
# EVALUATE DESIGN & PREDICT OUTPUT
####################################################################################################

#get estimated covariance, bias and MSE of parameter fit (use asymptotic method here) 
opts={'Method':'MonteCarlo','Covariance':True,'Bias':True,'MSE':True,'SampleNumber':100}
diagnostic_info = lin_model.evaluate(exact_design,fit_params,opts)

print('')
print('')
print(diagnostic_info)
print('')
print('')

#generate predictions with error bars fdor a random selection of inputs)
prediction_inputs = pd.DataFrame({ 'x1':np.random.uniform(0,5,10),
                                   'x2':np.random.uniform(0,5,10),
                                   'Variable':['y']*10})
cov_mat = diagnostic_info['Covariance'].to_numpy()
pred_options = {'Method':'MonteCarlo',
                'PredictionInterval':True,
                'ObservationInterval':True,
                'Sensitivity':True}
predictions = lin_model.predict(prediction_inputs,
                                  fit_params,
                                  covariance_matrix = cov_mat,
                                  options=pred_options)
print(predictions)

t=0


input_info = {'x1':{'Type':'Continuous','Bounds':(-1,1)}
                }

# design = pd.DataFrame({ 'x1':[-1,-1,-1,0,0,0,1,1,1]*3,
#                         'x2':[-1,0,1,-1,0,1,-1,0,1]*3,
#                         'Variable':['y_norm']*9+['y_bern']*9+['y_pois']*9,
#                         'Replicats':[5]*9*3})

# predict_inputs = pd.DataFrame({ 'x1':[-1,-1,-1,0,0,0,1,1,1]*3,
#                                 'x2':[-1,0,1,-1,0,1,-1,0,1]*3,
#                                 'Variable':['y_norm']*9+['y_bern']*9+['y_pois']*9})

# #mixed model


# rate = cs.exp(lin_predictor)
# poisson_stats = rate
# y_pois_func = cs.Function('y_pois',[xs,ps],[poisson_stats])

# prob = cs.exp(lin_predictor)/(1+cs.exp(lin_predictor))
# bern_stats = prob
# y_bern_func = cs.Function('y_bern',[xs,ps],[bern_stats])

# mixed_response = [  (y_norm_func,'Normal'),
#                     (y_bern_func,'Bernoulli'),
#                     (y_pois_func,'Poisson')]

# mixed_model = Model(mixed_response,xnames,pnames)

# opts={'Covariance':True,'Bias':True,'MSE':True,'SampleNumber':100}
# eval_dat = mixed_model.evaluate(design,[0.5,1.1,2.1,0.3],opts)

# opts={'Method':'MonteCarlo','Covariance':True,'Bias':True,'MSE':True,'SampleNumber':100}
# eval_dat = mixed_model.evaluate(design,[0.5,1.1,2.1,0.3],opts)

# mixed_data = mixed_model.sample(design,[0.5,1.1,2.1,0.3])
# fit_options={'Confidence':'Profiles',
#             'InitParamBounds':[(-5,5),(-5,5),(-5,5),(-5,5)],
#             'InitSearchNumber':7}
# mixed_fit = mixed_model.fit(mixed_data, options=fit_options)



# fit_pars = mixed_fit['Estimate'].to_numpy().flatten()
# print(fit_pars)

# #evaluate goes here

# cov_mat = np.diag([0.1,0.1,0.1,0.1])
# pred_options = {'Method':'MonteCarlo',
#                 'PredictionInterval':True,
#                 'ObservationInterval':True,
#                 'Sensitivity':True}
# predictions_dlta = mixed_model.predict(predict_inputs,fit_pars,covariance_matrix = cov_mat,options=pred_options)



#Declare model inputs
# x=cs.SX.sym('x',4)

# #Declare the parameters-of-interest
# beta=cs.SX.sym('beta',6)

# #Define a model t
# # theta1 = cs.Function('theta1',[beta,x],[beta[0] + x[0]*beta[1] + x[1]*beta[2],1])
# # theta2 = cs.Function('theta2',[beta,x],[beta[3] + x[0]*beta[4] + x[1]**2*beta[5],1])
# theta1 = cs.Function('theta1',[beta,x],[beta[0] + x[0]*beta[2] + x[1]*x[3]*beta[4],1])
# theta2 = cs.Function('theta2',[beta,x],[beta[1] + x[1]*beta[3] + x[0]*x[2]*beta[5],1])

# #Enter the above model into the list of reponse variables
# response= [('y1','Normal',theta1),('y2','Normal',theta2)]
# #xnames=['x1','x2']
# xnames=['x1','x2','x3','x4']
# #betanames=['b1','b2','b3','b4']
# #betanames=['b1','b2','b3']
# betanames=['b1','b2','b3','b4','b5','b6']

# #Instantiate class
# linear2dx2d=model(response,xnames,betanames)

# model1={'Model':linear2dx2d, 'Parameters': [0.5,1,2,0.5,1,2],'Objective':'D'}
# #model2={'Model':linear2dx2d, 'Parameters': [2, 2, 2, 2],'Objective':'D'}

# #models=[model1,model2]
# models=[model1]

# approx={'Inputs':['x1','x2','x3','x4'],'Bounds':[(-1,1),(-1,1),(-1,1),(-1,1)]}
# obs={'Observations':[['y1','y2']]}


# design1=design(models,approx,observgroups=obs)
# print(design1)

# design1=design(models,approx)
# print(design1)


#NOTE: Add to model later: Weight, Prior (cov, norm only, user can transform var), POI (parameters-of-interest)
#NOTE: Add to approx later: Constraints, Resolution (stepsize), Grid (user defined set of possible input levels, or full grid?)
#NOTE: Add to exact later: Replicates (code to default with full reps of all unique vec), Constraints, Start, 
#NOTE: Add to obs later: Weights, if not passed, treat each observation as individual option


# #Instantiate class
# linear2dx2d=model(response,xnames,betanames)

# model1={'Model':linear2dx2d, 'Parameters': [0.5,1,2],'Objective':'D'}
# #model2={'Model':linear2dx2d, 'Parameters': [2, 2, 2, 2],'Objective':'D'}
# #NOTE: Add to model later: Weight, Prior (cov, norm only, user can transform var), POI (parameters-of-interest)
# #models=[model1,model2]
# models=[model1]

# approx={'Inputs':['x3','x4'],'Bounds':[(-1,1),(-1,1)]}
# #NOTE: Add to approx later: Constraints, Resolution (stepsize), Grid (user defined set of possible input levels, or full grid?)
# struct=[['x1_lvl1', 'x2_lvl1'],['x1_lvl1', 'x2_lvl2'],['x1_lvl2', 'x2_lvl1'],['x1_lvl2', 'x2_lvl2']]
# exact={'Inputs':['x1','x2'],'Bounds':[(-1,1),(-1,1)],'Structure':struct}
# #NOTE: Add to exact later: Replicates (code to default with full reps of all unique vec), Constraints, Start, 
# obs={'Observations':[['y1','y2']]}
# #NOTE: Add to obs later: Weights, if not passed, treat each observation as individual option

# design1=design(models,approx,exact,obs)
# print(design1)

# design1=design(models,approx,exact)
# print(design1)

