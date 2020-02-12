import casadi as cs
from nloed import model
from nloed import design


#Declare independent variables (covarietes, control inputs, conditions etc.)
#These are the variables you control in the experimental design, and for which you want to find the optimal settings of
x=cs.MX.sym('x',4)

#Declare the number of parameters-of-interest
#These are the parameters you are interested in estimating as accurately as possible (or nuisance parameters, you can decide which is which later)
beta=cs.MX.sym('beta',4)

# The code below is based on Casadi's sysid.py example (Feb, 2020): https://github.com/casadi/casadi/blob/master/docs/examples/python/sysid.py
# define ode variables, these are just for convenience, there naming won't be important later
z1  = cs.MX.sym('z1')
z2 = cs.MX.sym('z2')
u  = cs.MX.sym('u')

#define the rhs, this is based on casadi examples
states = cs.vertcat(z1,z2)
controls = u
rhs = cs.vertcat(beta[0]*u -beta[1]*z1, beta[2]*z1-beta[3]*z2)

# Create a differential function (ode right hand side)
ode = cs.Function('ode',[states,controls,beta],[rhs])

# Set the total length fo the experiment, the time interval between samples,
# the integration steps within each interval, and the RK timestep
Tend=30
SampleInterval=10
StepsPerSample = 10
dt = SampleInterval/StepsPerSample

# Create symbolics for RK4 integration, as shown in Casadi examples
k1 = ode(states,controls,beta)
k2 = ode(states+dt/2.0*k1,controls,beta)
k3 = ode(states+dt/2.0*k2,controls,beta)
k4 = ode(states+dt*k3,controls,beta)
states_final = states+dt/6.0*(k1+2*k2+2*k3+k4)

# Create a function to perform one step of the RK integration
step = cs.Function('step',[states, controls, beta],[states_final])

#Loop over the ste function to create symbolics for integrating across a sample interval
X = states
for i in range(StepsPerSample):
  X = step(X, controls, beta)

# Create a function that simulates the whole sample interval
sample = cs.Function('sample',[states, controls, beta], [X])

X0 = cs.vertcat(x[0],0)
mean1 = step(X0, x[1], beta)
mean2 = step(X0, x[2], beta)
mean3 = step(X0, x[3], beta)

#Define the type of distribution for the response variable, and variance as a constant
distType='normal'
var = 1
#Enter the above model into the list of reponse variables
response= [(distType,(mean1, var)),(distType,(mean2, var)),(distType,(mean3, var))]

#Instantiate class
comp2D=model(response, beta, x)

#beta0=(0.5,1)
#xbounds=(0,10)
#xi=design(quadraticRS,beta0,xbounds)

#print(xi)