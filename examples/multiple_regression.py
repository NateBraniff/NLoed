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
betanames=['b1','b2','b3','b4']

#Instantiate class
linear2dx2d=model(response,xnames,betanames)

# ((response grouping), (grided x), ((xgrided1min,xgrided1max),(xgrided2min,xgrided2max)), (xgrided constraint ), (ungrided x), ((xungrided1min,xungrided1max),(xungrided2min,xgunrided2max)), (xungrided constraint ) , weight)
#(response grouping), (grided x), ((xgrided1min,xgrided1max),(xgrided2min,xgrided2max))

#structure=[(('y1','y2'),('x1'),((0,1)),(),('x2'),((0,1)),(),0),(('y2'),('x1','x2'),((0,1)),(),(),(),(),0)]

obsgroup1={'Group': ('y1','y2'), 'aX': ('x1'),'aXbounds': ((0,1)), 'eX': ('x2'),'eXbounds': ((0,1))}
obsgroup2={'Group': ('y2'), 'aX': ('x1','x2'),'aXbounds': ((0,1),(0,1))}
obsstruct=[obsgroup1, obsgroup2]

beta0=(0.5,1)

xi=design((linear2dx2d),obsstruct,beta0,())


#beta0=(0.5,1)
#xbounds=(0,10)
#xi=design(linear1d,beta0,xbounds)

#print(xi)