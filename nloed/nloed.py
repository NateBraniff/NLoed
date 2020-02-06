from casadi import *
x = MX.sym("x")
print(jacobian(sin(x),x))