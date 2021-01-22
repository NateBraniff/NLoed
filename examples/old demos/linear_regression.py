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
x = cs.SX.sym('x',1)
xnames = ['x']
p = cs.SX.sym('p',2)
pnames = ['Intercept','Slope']

#define the deterministic model
lin_predictor = p[0] + p[1]*x[0] 

#link the deterministic model to the sampling statistics (here normal mean and variance)
mean, var = lin_predictor, 0.1
normal_stats = cs.vertcat(mean, var)

#create a casadi function mapping input and parameters to sampling statistics (mean and var)
y = cs.Function('y',[x,p],[normal_stats])

# enter the function as a tuple with label indicating normal error, into observation list
observ_list = [(y,'Normal')]

#instantiate nloed model class
nloed_model = Model(observ_list,xnames,pnames)

####################################################################################################
# GENERATE OPTIMAL DESIGN
####################################################################################################

#set up the design algorithm to use continuous (continuous) optimization with two unique inputs points
continuous_inputs={'Inputs':['x'],'Bounds':[(-1,1)],'Structure':[['x1'],['x2']]}
#set up the design algorithm to use continuous (continuous) optimization with two unique inputs points
#discrete_inputs={'Inputs':['x'],'Grid':[[-1.5,-0.1,0.1,1.5]]}

# generate the optimal approximate (relaxed) design
relaxed_design = Design(nloed_model,[1,1],'D',continuous_inputs=continuous_inputs)
# generate the optimal approximate (relaxed) design
#relaxed_design = Design(nloed_model,[1,1],'D',discrete_inputs=discrete_inputs)

sample_size = 10
#generate a rounded exact design 
exact_design = relaxed_design.round(sample_size)

print(exact_design)

####################################################################################################
# GENERATE SAMPLE DATA & FIT MODEL
####################################################################################################

#generate some data, to stand in for an initial experiment
data = nloed_model.sample(exact_design,[1,1])

print(data)

#pass some additional options to fitting alglorithm, including Profile likelihood
fit_options={'Confidence':'Profiles',
             'InitParamBounds':[(-5,5),(-5,5)],
             'InitSearchNumber':7}
#fit the model to the init data
fit_info = nloed_model.fit(data, options=fit_options)

print(fit_info)

fit_params = fit_info['Estimate'].to_numpy().flatten()

####################################################################################################
# EVALUATE DESIGN & PREDICT OUTPUT
####################################################################################################

#get estimated covariance, bias and MSE of parameter fit (use asymptotic method here) 
opts={'Covariance':True,'Bias':True,'MSE':True,'SampleNumber':100}
diagnostic_info = nloed_model.evaluate(exact_design,fit_params,opts)
print(diagnostic_info)

#generate predictions with error bars fdor a random selection of inputs)
prediction_inputs = pd.DataFrame({ 'x':np.random.uniform(-1,1,10),
                                'Variable':['y']*10})
cov_mat = diagnostic_info['Covariance'].to_numpy()
pred_options = {'PredictionInterval':True,
                'ObservationInterval':True,
                'Sensitivity':True}
predictions = nloed_model.predict(prediction_inputs,
                                  fit_params,
                                  covariance_matrix = cov_mat,
                                  options=pred_options)
print(predictions)

t=0
# design = pd.DataFrame({ 'x1':[-1,-1,-1,0,0,0,1,1,1]*3,
#                         'x2':[-1,0,1,-1,0,1,-1,0,1]*3,
#                         'Variable':['y_norm']*9+['y_bern']*9+['y_pois']*9,
#                         'Replicats':[5]*9*3})

# predict_inputs = pd.DataFrame({ 'x1':[-1,-1,-1,0,0,0,1,1,1]*3,
#                                 'x2':[-1,0,1,-1,0,1,-1,0,1]*3,
#                                 'Variable':['y_norm']*9+['y_bern']*9+['y_pois']*9})

# #mixed model
# mean, var = lin_predictor, 0.1
# normal_stats = cs.vertcat(mean, var)
# y_norm_func = cs.Function('y_norm',[xs,ps],[normal_stats])

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
#              'InitParamBounds':[(-5,5),(-5,5),(-5,5),(-5,5)],
#              'InitSearchNumber':7}
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


# t=0

# x = cs.SX.sym('x')
# xnames = ['x']
# p = cs.SX.sym('p')
# pnames = ['p']

# design = pd.DataFrame({'x':[-2,-1,0,1,2],'y':[10,10,10,10,10]})

# # #normal model
# mean, var = x*p, x**2*p**2+.1
# normal_stats = cs.vertcat(mean, var)
# y_norm_func = cs.Function('y',[x,p],[normal_stats])
# normal_response = [(y_norm_func,'Normal')]
# normal_model = Model(normal_response,xnames,pnames)
# normal_data = normal_model.sample(design,[1])
# normal_fit = normal_model.fit(normal_data,[2])

# #poisson model
# rate = cs.exp(x*p)
# poisson_stats = rate
# y_pois_func = cs.Function('y',[x,p],[poisson_stats])
# poisson_response = [(y_pois_func,'Poisson')]
# poisson_model = Model(poisson_response,xnames,pnames)
# poisson_data = poisson_model.sample(design,[1])
# poisson_fit = poisson_model.fit(poisson_data,[2])

# #lognormal model
# geomean, geovar = x*p, cs.exp(x*p)
# logn_stats = cs.vertcat(geomean, geovar)
# y_logn_func = cs.Function('y',[x,p],[logn_stats])
# logn_response = [(y_logn_func,'Lognormal')]
# logn_model = Model(logn_response,xnames,pnames)
# logn_data = logn_model.sample(design,[1])
# logn_fit = logn_model.fit(logn_data,[2])

# #binomial model
# prob, num = cs.exp(x*p)/(1+cs.exp(x*p)), 10
# binom_stats = cs.vertcat(prob, num)
# y_binom_func = cs.Function('y',[x,p],[binom_stats])
# binom_response = [(y_binom_func,'Binomial')]
# binom_model = Model(binom_response,xnames,pnames)
# binom_data = binom_model.sample(design,[1])
# binom_fit = binom_model.fit(binom_data,[2])

# #bernoulli model
# prob = cs.exp(x*p)/(1+cs.exp(x*p))
# bern_stats = prob
# y_bern_func = cs.Function('y',[x,p],[bern_stats])
# bern_response = [(y_bern_func,'Bernoulli')]
# bern_model = Model(bern_response,xnames,pnames)
# bern_data = bern_model.sample(design,[1])
# bern_fit = bern_model.fit(bern_data,[2])

# #exponential model
# rate = cs.exp(x*p)
# exp_stats = prob
# y_pois_func = cs.Function('y',[x,p],[exp_stats])
# exp_response = [(y_pois_func,'Exponential')]
# exp_model = Model(exp_response,xnames,pnames)
# exp_data = exp_model.sample(design,[1])
# exp_fit = exp_model.fit(exp_data,[2])

# t=0
# ##########################################################################################

# xs = cs.SX.sym('xs',2)
# xnames = ['x1','x2']
# ps = cs.SX.sym('ps',4)
# pnames = ['Intercept','x1-Main','x2-Main','Interaction']

# lin_predictor = ps[0] + ps[1]*xs[0] + ps[2]*xs[1] + ps[3]*xs[0]*xs[1]

# design = pd.DataFrame({'x1':[-1,-1,-1,0,0,0,1,1,1],'x2':[-1,0,1,-1,0,1,-1,0,1],'y':[5,5,5,5,5,5,5,5,5]})

# #normal model
# mean, var = lin_predictor, lin_predictor**2 + 0.1
# normal_stats = cs.vertcat(mean, var)
# y_norm_func = cs.Function('y',[xs,ps],[normal_stats])
# normal_response = [(y_norm_func,'Normal')]
# normal_model = Model(normal_response,xnames,pnames)
# normal_data = normal_model.sample(design,[0.5,1.1,2.1,0.3])
# normal_fit = normal_model.fit(normal_data,[1,1,1,1])

# #poisson model
# rate = cs.exp(lin_predictor)
# poisson_stats = rate
# y_pois_func = cs.Function('y',[xs,ps],[poisson_stats])
# poisson_response = [(y_pois_func,'Poisson')]
# poisson_model = Model(poisson_response,xnames,pnames)
# poisson_data = poisson_model.sample(design,[0.5,1.1,2.1,0.3])
# poisson_fit = poisson_model.fit(poisson_data,[1,1,1,1])

# #lognormal model
# geomean, geovar = lin_predictor, cs.exp(lin_predictor)
# logn_stats = cs.vertcat(geomean, geovar)
# y_logn_func = cs.Function('y',[xs,ps],[logn_stats])
# logn_response = [(y_logn_func,'Lognormal')]
# logn_model = Model(logn_response,xnames,pnames)
# logn_data = logn_model.sample(design,[0.5,1.1,2.1,0.3])
# logn_fit = logn_model.fit(logn_data,[1,1,1,1])

# #binomial model
# prob, num = cs.exp(lin_predictor)/(1+cs.exp(lin_predictor)), 10
# binom_stats = cs.vertcat(prob, num)
# y_binom_func = cs.Function('y',[xs,ps],[binom_stats])
# binom_response = [(y_binom_func,'Binomial')]
# binom_model = Model(binom_response,xnames,pnames)
# binom_data = binom_model.sample(design,[0.5,1.1,2.1,0.3])
# binom_fit = binom_model.fit(binom_data,[1,1,1,1])

# #bernoulli model
# prob = cs.exp(lin_predictor)/(1+cs.exp(lin_predictor))
# bern_stats = prob
# y_bern_func = cs.Function('y',[xs,ps],[bern_stats])
# bern_response = [(y_bern_func,'Bernoulli')]
# bern_model = Model(bern_response,xnames,pnames)
# bern_data = bern_model.sample(design,[0.5,1.1,2.1,0.3])
# bern_fit = bern_model.fit(bern_data,[1,1,1,1])

# #exponential model
# rate = cs.exp(lin_predictor)
# exp_stats = prob
# y_pois_func = cs.Function('y',[xs,ps],[exp_stats])
# exp_response = [(y_pois_func,'Exponential')]
# exp_model = Model(exp_response,xnames,pnames)
# exp_data = exp_model.sample(design,[0.5,1.1,2.1,0.3])
# exp_fit = exp_model.fit(exp_data,[1,1,1,1])


# t=0

# xs = cs.SX.sym('xs',2)
# xnames = ['x1','x2']
# ps = cs.SX.sym('ps',3)
# pnames = ['Intercept','x1-Main','x2-Main']

# lin_predictor = ps[0] + ps[1]*xs[0] + ps[2]*xs[1]

# design = pd.DataFrame({ 'x1':[-1,-1,-1,0,0,0,1,1,1],
#                         'x2':[-1,0,1,-1,0,1,-1,0,1],
#                         'y_norm':[50,50,50,50,50,50,50,50,50]})

# predict_inputs = pd.DataFrame({ 'x1':[-1,-1,-1,0,0,0,1,1,1],
#                                 'x2':[-1,0,1,-1,0,1,-1,0,1],
#                                 'Variable':['y_norm']*9})

# #mixed model
# mean, var = lin_predictor, lin_predictor**2+0.1
# normal_stats = cs.vertcat(mean, var)
# y_norm_func = cs.Function('y_norm',[xs,ps],[normal_stats])

# lin_response = [ (y_norm_func,'Normal')]

# lin_model = Model(lin_response,xnames,pnames)

# lin_data = lin_model.sample(design,[-2,0.5,-1.1])
# options={'Confidence':'Contours','InitParameters':[-1,0,-1]}
# lin_fit = lin_model.fit(lin_data,options)


# pred_options = {'Method':'MonteCarlo',
#                 'PredictionInterval':True,
#                 'ObservationInterval':True,
#                 'Sensitivity':True,
#                 'PredictionSampleNumber':10000}
# cov_mat = np.diag([0.1,0.1,0.1])
# predictions_mc = lin_model.predict(predict_inputs,[-2,0.5,-1.1],covariance_matrix= cov_mat,options=pred_options)

# pred_options = {'Method':'Delta',
#                 'PredictionInterval':True,
#                 'ObservationInterval':True,
#                 'Sensitivity':True}
# cov_mat = np.diag([0.1,0.1,0.1])
# predictions_dlta = lin_model.predict(predict_inputs,[-2,0.5,-1.1],covariance_matrix = cov_mat,options=pred_options)


# options={'Confidence':'Profiles','InitParamBounds':[(-5,5),(-5,5),(-5,5)],'InitSearchNumber':7}

# lin_data = lin_model.sample(design,[-2,0.5,-1.1])
# lin_fit = lin_model.fit(lin_data,options=options)
# print(lin_fit)
# fit_param_vec=lin_fit['Estimate'].to_numpy()
# pred_options={'Method':'Exact','MeanInterval':False,'PredictionInterval':True}
# cov_mat=np.diag([0.1,0.1,0.1])
# predictions=lin_model.predict(predict_inputs,fit_param_vec,covariance_matrix= cov_mat,options=pred_options)

# lin_data = lin_model.sample(design,[0.2,1.9,3.25])
# lin_fit = lin_model.fit(lin_data,options=options)
# print(lin_fit)

# lin_data = lin_model.sample(design,[-0.5,-3.3,0.2])
# lin_fit = lin_model.fit(lin_data,options=options)
# print(lin_fit)

# lin_data = lin_model.sample(design,[4.3,0,-0.4])
# lin_fit = lin_model.fit(lin_data,options=options)
# print(lin_fit)

# xs = cs.SX.sym('xs',2)
# xnames = ['x1','x2']
# ps = cs.SX.sym('ps',4)
# pnames = ['Intercept','x1-Main','x2-Main','Interaction']

# lin_predictor = ps[0] + ps[1]*xs[0] + ps[2]*xs[1] + ps[3]*xs[0]*xs[1]

# design = pd.DataFrame({ 'x1':[-1,-1,-1,0,0,0,1,1,1],
#                         'x2':[-1,0,1,-1,0,1,-1,0,1],
#                         'y_norm':[5,5,5,5,5,5,5,5,5],
#                         'y_bern':[5,5,5,5,5,5,5,5,5],
#                         'y_pois':[5,5,5,5,5,5,5,5,5]})

# predict_inputs = pd.DataFrame({ 'x1':[-1,-1,-1,0,0,0,1,1,1]*3,
#                                 'x2':[-1,0,1,-1,0,1,-1,0,1]*3,
#                                 'Variable':['y_norm']*9+['y_bern']*9+['y_pois']*9})

# #mixed model
# mean, var = lin_predictor, 0.1
# normal_stats = cs.vertcat(mean, var)
# y_norm_func = cs.Function('y_norm',[xs,ps],[normal_stats])

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


# t=0
# stats2=cs.vertcat(mean,var)
# stats3=cs.vertcat(mean,10)

# f1 = cs.Function('f1',[x,p],[stats1])
# f2 = cs.Function('f2',[x,p],[stats2])
# f3 = cs.Function('f2',[x,p],[stats3])


# resBern = [('y','Bernoulli',f1)]
# resBin = [('y','Binomial',f3)]
# resLogn = [('y','Lognormal',f2)]
# resPois = [('y','Poisson',f1)]
# resExp = [('y','Exponential',f1)]

# norm_mod = Model(resNorm,xnames,pnames)
# logn_mod = Model(resLogn,xnames,pnames)
# bern_mod = Model(resBern,xnames,pnames)
# bin_mod = Model(resBin,xnames,pnames)
# pois_mod = Model(resPois,xnames,pnames)
# exp_mod = Model(resExp,xnames,pnames)

# design = pd.DataFrame({'x':[1,20],'y':[5,5]})

# norm_data = norm_mod.sample(design,[1])
# logn_data = logn_mod.sample(design,[1])
# bern_data = bern_mod.sample(design,[.03])
# bin_data = bin_mod.sample(design,[.03])
# pois_data = pois_mod.sample(design,[1])
# exp_data = exp_mod.sample(design,[1])

# t=0


# #Declare model inputs and parametrs
# inputs = cs.SX.sym('inputs',1)
# params = cs.SX.sym('params',2)
# #define sampling stat model for normal response
# mean = p[0] + x*p[1]
# variance = cs.SX(0.01)
# #zip the sampling stats into a vector and create a casadi function
# stats = cs.vertcat(mean, variance)
# func = cs.Function('func',[params,inputs],[stats])
# #Enter the above model into the list of reponse variables
# response= [('y1','Normal',func)]
# #give the input
# input_names = ['x1']
# param_names = ['p1','p2']

# #Instantiate class
# linear1d=Model(response,input_names,param_names)

# design1 = pd.DataFrame({'x1':[0,1,2],'y1':[5,1,5]})
# design2 = pd.DataFrame({'x1':[-1,0,1],'y1':[4,3,4]})

# dataset_list = linear1d.sample([design1,design2],[0,1],5)

# pars_info = linear1d.fit(dataset_list,[0,1],options={'Confidence':'Contours'})

# prediction_inputs = dataset_list[0][0][input_names+['Variable']]
# par_est = pars_info['Estimate'][param_names].to_numpy()

# #pred_structA=linear1d.eval_model(inputs, par_est, [[1,0],[0,1]], True,options={'ErrorMethod':'Delta'})
# #pred_structB=linear1d.eval_model(inputs, par_est, [[1,0],[0,1]], True,options={'ErrorMethod':'MonteCarlo','SampleSize':10000})
# # print(pred_structA['y1']['Mean']['Bounds'])
# # print(pred_structB['y1']['Mean']['Bounds'])

# # pred_struct4=linear1d.eval_model([1,1],[[i] for i in range(10)],[[1,0],[0,1]],True,options={'ErrorMethod':'Delta'})
# # pred_struct5=linear1d.eval_model([1,1],[[i] for i in range(10)],[[1,0],[0,1]],True,options={'ErrorMethod':'MonteCarlo'})
# # pred_struct1=linear1d.eval_model([1,1],1)
# # pred_struct2=linear1d.eval_model([1,1],1,param_covariance=[[1,0],[0,1]])
# # pred_struct3=linear1d.eval_model([1,1],1,sensitivity=True)



# # paramCov1=np.cov(pars,rowvar=False)
# # paramMean1=np.mean(pars,axis=0)

# # dataset2=linear1d.sample([Experiment,Experiment],[0,1])

# modelinfo={'Model':linear1d, 'Parameters': [0.5, 1],'Objective':'D'}
# approx={'Inputs':['x1'],'Bounds':[(-1,1)]}
# obs={'Observations':[['y1']]}

# opt_approx=design([modelinfo],approxinputs=approx,observgroups=obs)
# print(opt_approx)

# opt_approx=design([modelinfo],approx)
# print(opt_approx)

# modelinfo={'Model':linear1d, 'Parameters': [0.5, 1],'Objective':'D'}
# struct=[['x1_lvl1'],['x1_lvl2'],['x1_lvl3']]
# exact={'Inputs':['x1'],'Bounds':[(-1,1)],'Structure':struct}
# obs={'Observations':[['y1']]}

# opt_exact=design([modelinfo],exactinputs=exact,observgroups=obs)
# print(opt_exact)

# opt_exact=design([modelinfo],exactinputs=exact)
# print(opt_exact)

# print('Done')

