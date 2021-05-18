#!/usr/bin/env python
# coding: utf-8

# # Sampling and Fitting the Hill Function Model
# 

# In[24]:


#%%capture
from N2_optimizing_a_design import *


# ### Sampling the Model using Simulated Data
# 
# 

# In[25]:


data = hill_model.sample(exact_design,nominal_param)


# 

# In[26]:


print(data)


# ### Fitting the Model with Maximum Likelihood

# In[34]:


fit_options={'Confidence':'Profiles',  
             'InitParamBounds':[(-1,4),(-1,4),(-1,4),(-1,4)],
             'InitSearchNumber':7,
             'SearchBound':5.}


# 

# In[35]:


fit_info = hill_model.fit(data, options=fit_options)
print(fit_info['Estimate'])


# In[33]:


print(fit_info['Lower'])
print(fit_info['Upper'])


# 

# In[29]:


fit_params = fit_info['Estimate'].to_numpy().flatten()
print(np.exp(fit_params))


# In[30]:


print(np.exp(nominal_param))


# In[ ]:




