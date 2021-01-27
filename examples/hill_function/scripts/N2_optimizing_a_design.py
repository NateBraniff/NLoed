#!/usr/bin/env python
# coding: utf-8

# # Optimizing a Design for the Hill Function Model
# 
# 

# In[18]:


import numpy as np
from scripts.N1_model_creation import *


# ### Specifying the Design Details

# In[19]:


continuous_inputs={'Inputs':['Inducer'],
                   'Bounds':[(.1,5)],
                   'Structure':[['x1'],['x2'],['x3'],['x4']]}


# 

# In[20]:


nominal_param = np.log([1,5,2,1])
objective = 'D'


# ### Instantiating the Design and Returning Results

# In[21]:


design_object = nl.Design(hill_model, nominal_param, objective, continuous_inputs=continuous_inputs)


# 

# In[22]:


design_object.relaxed()


# 

# In[23]:


sample_size = 48
exact_design = design_object.round(sample_size)
exact_design

