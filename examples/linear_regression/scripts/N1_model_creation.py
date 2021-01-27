#!/usr/bin/env python
# coding: utf-8

# # A Simple 2-Input Linear Regression Model
# 
# This example explains how to create a simple 2-input linear regression model using NLoed's Model class.

# ### Create Casadi Symbols for the Model
# 
# In this example, we assume there is a linear relationship between two input variables the experimenter controls: $x_1$ and $x_2$, and the observation variable $y$. This relationship can be described by the equation linear equation:
# 
# $\hat{y} = \beta_{0} + \beta_{1} x_{1} + \beta_{2} x_{2}$
# 
# Here $\hat{y}$ represents the mean of the observation response. In this example we assum that $y$ follows a normal distribution with constant known variance of Var($y$)$=1$.
# 
# In order to implement this model in NLoed's Model class we start by importing the NLoed and Casadi packages. We then create creat symbols for the inputs, $x_i$, and the parameters $\beta_j$. This can be done as follows:

# In[1]:


import nloed as nl
import casadi as cs
x = cs.SX.sym('X',2)
beta = cs.SX.sym('beta',3)


# Now, we use the Casadi symbols to implement an expression for the deterministic model equation. This expression links the mean of the obeservation variable $y$ to the inputs and the parameters. This is done as follows:

# In[2]:


y_mean = beta[0] + beta[1]*x[0] + beta[2]*x[1] 


# The variance of the obervation variable, $y$ is a constant, with no relation to the model inputs or the parameters. We can therefore set it as constant in our model preparation.

# In[3]:


y_var = 0.1


# ### Instantiating an NLoed Model Object
# 
# Model class's constructor requires the model to be encoded as a Casadi function, mapping the model inputs and parameters to the sampling statistics of the observation variable. In this case, as we assume $y$ has a normal distribution, it's sampling statistics consist of the mean and variance of $y$. To create a Casadi function for this relationship, we zip the the mean and variance into a single vector and call Casadi's function constructor as follows:

# In[4]:


y_stats = cs.vertcat(y_mean, y_var)
y_func = cs.Function('Observable_y',[x,beta],[y_stats])


# Above, we passed a string 'Observable_y' to the function constructor to give the Casadi function a name attribute. This name attribute will become the observation's name in the NLoed Model instance. Casadi's constructor also requires us to provide names for the inputs and parameters. This is done usign two lists of strings as follows:

# In[5]:


input_names = ['Input_x1','Input_x2']
param_names = ['Intercept','Slope1','Slope2']


# Finally, the NLoed model constructor needs to know that the observation variable $y$ is normally distributed. To label it as such we tuple the Casadi function, *y_func*, encoding the mdoel with the string, "Normal". This tuple is then places in a list, *observ_list*. If the model had other observation variables that also depended on the same inputs and parameters, they could be added to this list as additional tuples each with their own distribution. However, this model only contains one output observable so the *observ_list* looks like this:

# In[6]:


observ_list = [(y_func,'Normal')]


# And at last, the NLoed model is declared using the constructor as follows:

# In[7]:


linear_model = nl.Model(observ_list,input_names,param_names)


# The *linear_model* object can now be used for many things. Check out some of the other notebooks for examples.
