import casadi as cs
from nloed import Model
from nloed import Design


xs = cs.SX.sym('xs',2)
xnames = ['x1','x2']
ps = cs.SX.sym('ps',4)
pnames = ['Intercept','Slope1','Slope2','Interaction']

lin_predictor = ps[0] + ps[1]*xs[0] + ps[2]*xs[1] + ps[3]*xs[0]*xs[1] 

mean, var = lin_predictor, 0.1
normal_stats = cs.vertcat(mean, var)
y = cs.Function('y',[xs,ps],[normal_stats])

lin_model = Model([(y,'Normal')],xnames,pnames)

approx_inputs={'Inputs':['x1'],'Bounds':[(-1,1)]}
exact_inputs={'Inputs':['x2'],'Bounds':[(-1,1)],'Structure':[['A'],['B']]}
#exact_inputs={'Inputs':['x1','x2'],'Bounds':[(-1,1),(-1,1)],'Structure':[['x1_lvl1','x2_lvl1'],['x1_lvl1','x2_lvl2'],['x1_lvl2','x2_lvl2']]}
#    def __init__(self, models, parameters, objective, approx_inputs=None, exact_inputs=None, observ_groups=None, fixed_design=None, options={}):
lin_design = Design(lin_model,[1,1],'D',approx_inputs,exact_inputs)#approx_inputs



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

