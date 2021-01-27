#!/usr/bin/env python
# coding: utf-8

# # Creating a Simple Hill Function Model
# 
# 

# In[8]:


import nloed as nl
import casadi as cs


# ### Model Expressions and Parameterization

# In[9]:


x = cs.SX.sym('x',1)
p = cs.SX.sym('p',4)


# 

# In[10]:


alpha_0 = cs.exp(p[0])
alpha = cs.exp(p[1])
K = cs.exp(p[2])
n = cs.exp(p[3])


# 

# In[11]:


mean = alpha_0 + alpha*x[0]**n / (K**n + x[0]**n)


# 

# In[12]:


var = 0.01*mean


# ### Preparing to Call the NLoed Model Constructor

# In[13]:


normal_stats = cs.vertcat(mean, var)
response_func = cs.Function('Response',[x,p],[normal_stats])


# 

# In[14]:


xnames = ['Inducer']
pnames = ['Basal','Rate','Hill','HalfConst']
observ_list = [(response_func,'Normal')]
hill_model = nl.Model(observ_list,xnames,pnames)


# 
