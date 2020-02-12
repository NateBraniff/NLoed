import casadi as cs
from nloed import model
from nloed import design


#Declare independent variables (covarietes, control inputs, conditions etc.)
#These are the variables you control in the experimental design, and for which you want to find the optimal settings of
x=cs.MX.sym('x',2)

#Declare the number of parameters-of-interest
#These are the parameters you are interested in estimating as accurately as possible (or nuisance parameters, you can decide which is which later)
beta=cs.MX.sym('beta',4)

# x and beta combine to form a model that predicts a response, y
#Each observed response variable, y_i, must a have a distribution, you can choose a unique distribution for each response
#Define the type of distribution for the response variable
distType='normal'

#Now we define how x and beta are linked to the distribution parameters, theta, of the normal response distribution
#Define a 1D linear model with slope and intercept for the mean
mean = beta[0] + x[0]*beta[1] + x[1]*beta[2] + x[2]*beta[3]
#Define the (known) variance as a constant
var = 1
#Combine these into a list of response distribution parameters
theta =(mean, var)

#Enter the above model into the list of reponse variables
response= [(distType,theta)]

#Instantiate class
quadraticRS=model(response, beta, x)

#beta0=(0.5,1)
#xbounds=(0,10)
#xi=design(quadraticRS,beta0,xbounds)

#print(xi)