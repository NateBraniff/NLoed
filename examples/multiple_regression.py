import casadi as cs
from nloed import model
from nloed import design

#Declare independent variables (covarietes, control inputs, conditions etc.)
#These are the variables you control in the experimental design, and for which you want to find the optimal settings of
x=cs.MX.sym('x',4)

#Declare the number of parameters-of-interest
#These are the parameters you are interested in estimating as accurately as possible (or nuisance parameters, you can decide which is which later)
beta=cs.MX.sym('beta',4)

#Now we define how x and beta are linked to the distribution parameters, theta, of the normal response distribution
#Define a 1D linear model with slope and intercept for the mean
theta1 = cs.Function('theta1',[beta,x],[beta[0] + x[0]*beta[2] + x[1]*beta[3],1])
theta2 = cs.Function('theta2',[beta,x],[beta[1] + x[3]*beta[0] + x[2]*beta[3],1])

#Enter the above model into the list of reponse variables
response= [('y1','normal',theta1),('y2','normal',theta2)]
xnames=['x1','x2','x3','x4']
betanames=['b1','b2','b3','b4']

#Instantiate class
linear2dx2d=model(response,xnames,betanames)

model1={'Model':linear2dx2d, 'Parameters': [1, 1, 1, 1],'Objective':'D'}
model2={'Model':linear2dx2d, 'Parameters': [2, 2, 2, 2],'Objective':'D'}
#NOTE: Add to model later: Weight, Prior (cov, norm only, user can transform var), POI (parameters-of-interest)
models=[model1,model2]

approx={'Inputs':['x3','x4'],'Bounds':[(-1,1),(0,3)]}
#NOTE: Add to approx later: Constraints, Resolution (stepsize), Grid (user defined set of possible input levels, or full grid?)
struct=[['x1_lvl1', 'x2_lvl1'],['x1_lvl1', 'x2_lvl2'],['x1_lvl2', 'x2_lvl1'],['x1_lvl2', 'x2_lvl2']]
exact={'Inputs':['x1','x2'],'Bounds':[(5,10),(2,3)],'Structure':struct}
#NOTE: Add to exact later: Replicates (code to default with full reps of all unique vec), Constraints, Start, 
obs={'Observations':[('y1','y2')]}
#NOTE: Add to obs later: Weights

design1=design(models,exact,approx,obs)

print(design1)

