#!/usr/bin/env python
# coding: utf-8

# # Simple Linear Model
# 
# This example explains how to create a simple linear model using NLoed's Model class. We start by importing the NLoed and CasADi:

# In[1]:


import nloed as nl
import casadi as cs


# ### Create the Casadi Symbols for the Model
# 
# We start by expressing the model equations in Casadi's SX symbols. The equation we want to implement is:
# 
# $\hat{y} = \beta_{0} + \beta_{1} x_{1} + \beta_{2} x_{2}$
# 
# We first need to creat symbols for the inputs, $x_i$, and the parameters $\beta_j$. This can be done as follows:

# In[2]:


x = cs.SX.sym('X',2)
beta = cs.SX.sym('beta',3)


# Write the equation:

# In[3]:


y_mean = beta[0] + beta[1]*x[0] + beta[2]*x[1] 


# In[4]:


y_var = 0.1
y_stats = cs.vertcat(y_mean, y_var)


# Name the inputs and paramers

# In[5]:


input_names = ['Input1','Input2']
param_names = ['Intercept','Slope1','Slope2']


# create Casadi function and observation struct, observations are named by the strings entered in the struct

# In[6]:


y_func = cs.Function('y',[x,beta],[y_stats])
observ_list = [(y_func,'Normal')]


# create the NLoed Model instance

# In[7]:


linear_model = nl.Model(observ_list,input_names,param_names)

