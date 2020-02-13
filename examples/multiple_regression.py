import casadi as cs
from nloed import model
from nloed import design


#Declare independent variables (covarietes, control inputs, conditions etc.)
#These are the variables you control in the experimental design, and for which you want to find the optimal settings of
x=cs.MX.sym('x',2)

#Declare the number of parameters-of-interest
#These are the parameters you are interested in estimating as accurately as possible (or nuisance parameters, you can decide which is which later)
beta=cs.MX.sym('beta',4)

#Now we define how x and beta are linked to the distribution parameters, theta, of the normal response distribution
#Define a 1D linear model with slope and intercept for the mean
theta1 = cs.Function('theta1',[beta,x],[beta[0] + x[0]*beta[2] + x[1]*beta[3],1])
theta2 = cs.Function('theta2',[beta,x],[beta[1] + x[0]*beta[2] + x[1]*beta[3],1])

#Enter the above model into the list of reponse variables
response= [('y1','normal',theta1),('y2','normal',theta2)]

xnames=['x1','x2']

#Instantiate class
linear2dx2d=model(response,xnames)

# ((response grouping), (grided x), (ungrided x), (xbounds, xconstraint list), weight)
#structure=[(('y1','y2'),('x1'),(),('x2'),(),0),(('y2'),('x1','x2'),(),(),0)]

#beta0=(0.5,1)
#xbounds=(0,10)
#xi=design(linear1d,beta0,xbounds)

#print(xi)