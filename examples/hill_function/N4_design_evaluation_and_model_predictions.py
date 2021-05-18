#!/usr/bin/env python
# coding: utf-8

# # Design Evaluations and Model Predictions for the Hill Function Model
# 

# In[1]:


get_ipython().run_cell_magic('capture', '', 'import pandas as pd\nimport matplotlib.pyplot as plt\nfrom N3_sampling_and_fitting import *')


# ### Evaluating the Optimized Design

# In[ ]:





# In[ ]:





# ### Asymptotic Covariance and Confidence Intervals

# In[3]:


eval_options={'Method':'Asymptotic',
              'Covariance':True}
asymptotic_covariance = hill_model.evaluate(exact_design,fit_params,eval_options)
print(asymptotic_covariance)


# In[23]:


asymptotic_lower_bound = fit_params - 2*np.sqrt(np.diag(asymptotic_covariance))
asymptotic_upper_bound = fit_params + 2*np.sqrt(np.diag(asymptotic_covariance))

fim_bnds_array = np.vstack((asymptotic_lower_bound,
                            fit_params,
                            asymptotic_upper_bound)).T
fim_bnds0 = pd.DataFrame(np.exp(fim_bnds_array),
                        index=['Alpha0','Alpha','n','K'],
                        columns=['Lower','Estimate','Upper'])
print('')
print(fim_bnds0)
print('')


# ### Model Predictions

# In[24]:


#convert the covariance matrix to numpy
covariance_matrix = asymptotic_covariance.to_numpy()
#generate predictions with error bars fdor a random selection of inputs)
prediction_inputs = pd.DataFrame({'Inducer':np.linspace(0.1,10,100),
                                  'Variable':['Response']*100})
#request prediction and observation intervals
prediction_options = {'PredictionInterval':True,
                      'ObservationInterval':True}
#generate predictions and intervals
predictions = hill_model.predict(prediction_inputs,
                                   fit_params,
                                   covariance_matrix = covariance_matrix,
                                   options=prediction_options)


# In[ ]:





# ### Plotting Prediction Intervals

# In[25]:


#create plot
fig, ax = plt.subplots()
#plot observation interval
ax.fill_between(predictions['Inputs','Light'],
                predictions['Observation','Lower'],
                predictions['Observation','Upper'],
                alpha=0.3,
                color='C1')
#plot prediction interval
ax.fill_between(predictions['Inputs','Light'],
                predictions['Prediction','Lower'],
                predictions['Prediction','Upper'],
                alpha=0.4,
                color='C0')
#plot mean model prediction
ax.plot(predictions['Inputs','Light'], predictions['Prediction','Mean'], '-')
#plot initial dataset
ax.plot(init_data['Light'], init_data['Observation'], 'o', color='C1')
ax.set_xlabel('Light')
ax.set_ylabel('GFP')
plt.show()


# In[ ]:




