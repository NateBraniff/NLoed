import casadi as cs
from nloed import model
from nloed import design

#Declare model inputs
x=cs.SX.sym('x',4)

#Declare the parameters-of-interest
beta=cs.SX.sym('beta',6)

#Define a model t
# theta1 = cs.Function('theta1',[beta,x],[beta[0] + x[0]*beta[1] + x[1]*beta[2],1])
# theta2 = cs.Function('theta2',[beta,x],[beta[3] + x[0]*beta[4] + x[1]**2*beta[5],1])
theta1 = cs.Function('theta1',[beta,x],[beta[0] + x[0]*beta[2] + x[1]*x[3]*beta[4],1])
theta2 = cs.Function('theta2',[beta,x],[beta[1] + x[1]*beta[3] + x[0]*x[2]*beta[5],1])

#Enter the above model into the list of reponse variables
response= [('y1','Normal',theta1),('y2','Normal',theta2)]
#xnames=['x1','x2']
xnames=['x1','x2','x3','x4']
#betanames=['b1','b2','b3','b4']
#betanames=['b1','b2','b3']
betanames=['b1','b2','b3','b4','b5','b6']

#Instantiate class
linear2dx2d=model(response,xnames,betanames)

model1={'Model':linear2dx2d, 'Parameters': [0.5,1,2,0.5,1,2],'Objective':'D'}
#model2={'Model':linear2dx2d, 'Parameters': [2, 2, 2, 2],'Objective':'D'}

#models=[model1,model2]
models=[model1]

approx={'Inputs':['x1','x2','x3','x4'],'Bounds':[(-1,1),(-1,1),(-1,1),(-1,1)]}
obs={'Observations':[['y1','y2']]}


design1=design(models,approx,observgroups=obs)
print(design1)

design1=design(models,approx)
print(design1)


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

