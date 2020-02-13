import casadi as cs
from nloed import model
from nloed import design


#Declare independent variables (covarietes, control inputs, conditions etc.)
#These are the variables you control in the experimental design, and for which you want to find the optimal settings of
x=cs.MX.sym('x',1)

#Declare the number of parameters-of-interest
#These are the parameters you are interested in estimating as accurately as possible (or nuisance parameters, you can decide which is which later)
beta=cs.MX.sym('beta',2)

#Now we define how x and beta are linked to the distribution parameters, theta, of the normal response distribution
#Define a 1D linear model with slope and intercept for the mean
theta = cs.Function('theta',[beta,x],[cs.exp(beta[0] + x*beta[1])],['beta','x'],['lambda'])

#Enter the above model into the list of reponse variables
response= [('y1','poisson',theta)]

xnames=['x1']

#Instantiate class
poisson1D=model(response,xnames)

# beta0=(0.5,1)
# xbounds=(0,20)
# xi=design(poisson1D,beta0,xbounds)

# print(xi)