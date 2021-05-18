#!/usr/bin/env python
# coding: utf-8

# # Optimizing a Design for the Hill Function Model
# 
# 

# In[2]:


#%%capture
import numpy as np
from N1_model_creation import *


# ### Setting the Nominal Parameter Values and Design Objective

# In[3]:


#set nominal parameter values, from previous fit or literature
nominal_param = np.log([1,5,2,1])
#set design objective, here D-optimal design is used
objective = 'D'


# ### Creating a Optimal Design with Continuous Optimized Input Levels

# In[4]:


#create a dictionary describing the design requirments for a continuous design
#here input inducer levels are optimization variables that will be selected
#four unique inducer levels are allowed for in the design optimization
continuous_inputs={'Inputs':['Inducer'],
                   'Bounds':[(.1,5)],
                   'Structure':[['x1'],['x2'],['x3'],['x4']]}


# In[14]:


#instantiate the NLoed design object
continuous_design_object = nl.Design(hill_model, nominal_param, objective, continuous_inputs=continuous_inputs)


# In[6]:


#use the relaxed method to print out the optimal relaxed design
#note, relaxed design's have continuous weights rather than a number of samples for each input-observation pair
continuous_design_object.relaxed()


# In[7]:


#set the sample size for the exact design
sample_size = 48
#use the round function to round the optimal relaxed design to an exact one
exact_continuous_design = continuous_design_object.round(sample_size)
exact_continuous_design


# ### Creating an Optimal Design with Predefined Discrete Input Levels

# In[8]:


#create a dictionary describing the design requirments for a discrete design
#here input inducer levels must be selected from the candidate list
#the design optimization will allocate the number of observations to each input level
discrete_inputs={'Inputs':['Inducer'],
                 'Candidates':[[.1,.25,.5,1,2,3,4,5]]}


# In[9]:


#instantiate the NLoed design object
discrete_design_object = nl.Design(hill_model, nominal_param, objective, discrete_inputs=discrete_inputs)


# In[10]:


#use the relaxed method to print out the optimal relaxed design
discrete_design_object.relaxed()


# In[11]:


#set the sample size for the exact design
sample_size = 48
#use the round function to round the optimal relaxed design to an exact one
exact_discrete_design = discrete_design_object.round(sample_size)
exact_discrete_design

