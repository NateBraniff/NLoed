#!/usr/bin/env python
# coding: utf-8

# # Creating a Simple Hill Function Model
# 
# 

# In[8]:


#import nloed and casadi
import nloed as nl
import casadi as cs


# ### Model Parameterization and Equations

# In[9]:


#create symbol for the model input
x = cs.SX.sym('x',1)
#create symbols for the model parameters
p = cs.SX.sym('p',4)


# In[10]:


#use parameter transformation to ensure named parameter remain positive
#we will fit and design for fitting p, which is exponent of the 'natural' parameters
alpha_0 = cs.exp(p[0])
alpha = cs.exp(p[1])
K = cs.exp(p[2])
n = cs.exp(p[3])


# In[11]:


#write a symbolic expression for the model, predicting mean observations from input and parameters
mean = alpha_0 + alpha*x[0]**n / (K**n + x[0]**n)


# In[12]:


#create an symbolic expression for the variance of the observations
#here variance is assumed to be proportional to the mean i.e. hetroskedastic
var = 0.01*mean


# ### Preparing to Call the NLoed Model Constructor

# In[13]:


#group the mean and variance into a single vecotor
normal_stats = cs.vertcat(mean, var)
#create a CasADi function to predict the mean and variance of the observation from the input and parameters
model_function = cs.Function('Observation',[x,p],[normal_stats])


# In[14]:


#list user-chosen names of model inputs (only one in this example)
xnames = ['Inducer']
#list user-chosen names of parameters, log_ prefix is to remind us of transformation
pnames = ['log_Basal','log_Rate','log_Hill','log_HalfConst']
#create a list of model observation variable (only one)
#the single list element is a tuple of the model function and a lable indicating the distribution type
observ_list = [(model_function,'Normal')]
#create the NLoed model using the Model constructor
hill_model = nl.Model(observ_list,xnames,pnames)

