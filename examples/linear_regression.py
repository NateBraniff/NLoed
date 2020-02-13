import casadi as cs
from nloed import model
from nloed import design


#Declare independent variables (covarietes, control inputs, conditions etc.)
#These are the variables you control in the experimental design, and for which you want to find the optimal settings of
x=cs.SX.sym('x',1)

#Declare the number of parameters-of-interest
#These are the parameters you are interested in estimating as accurately as possible (or nuisance parameters, you can decide which is which later)
beta=cs.SX.sym('beta',2)

#Now we define how x and beta are linked to the distribution parameters, theta, of the normal response distribution
#Define a 1D linear model with slope and intercept for the mean, constant variance, 
#combine these into a function computing the response distribution parameters, theta (in this case mean and var)
theta=cs.Function('theta',[beta,x],[beta[0] + x*beta[1], 1],['beta','x'],['mu','sigma'])

#Enter the above model into the list of reponse variables
response= [('y1','normal',theta)]

xnames=['x1']

#Instantiate class
linear1d=model(response,xnames)

#beta0=(0.5,1)
#xbounds=(0,10)
#xi=design(linear1d,beta0,xbounds)

#print(xi)