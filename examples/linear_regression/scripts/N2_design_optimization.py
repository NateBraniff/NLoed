#!/usr/bin/env python
# coding: utf-8

# # Optimal Design for the 2-Input Linear Regression Model

# This example demonstrates how to generate an optimal design for the 2-input linear regression model dicussed in the previous notebook. To start we, import the *N1_model_creation* notebook so that the NLoed Model instance is available here.

# In[1]:


from scripts.N1_model_creation import *


# ###  Instantiating a Design Object
# 
# To create the NLoed design object we need to specify some aspects of the design we wish to generate, related to the physical constraints and the numerical formulation that will be used. We start by specifying the nominal parameter values and the objective function we will use in this design. Here we set the nominal parameters to all be $1$ and we use a D-optimal objective:

# In[ ]:


nominal_param = [1,1,1]
objective = 'D'


# We now need to specify how the Design algorithm will treat the inputs to the model. The main choice here is whether the model inputs are handled discretly or continuously. In this example we choose a continuous approach, see the NLoed manual and background documentation for further discussion and comparison. To have the Design object handle both inputs, $x1$ and $x2$ continuously we need to creat a continuous inputs dictionary that specifies the bounds of both inputs and the number of unique levels of each input the optimization algorithm will consider. Here we create a continuous_inputs dictionary which bounds both inputs between $-1$ and $1$ and allows for three unique pairs of input vectors to be considered in the design optimization (inidcated by the '*_lv#'* suffix in the *Structure* field):

# In[ ]:


continuous_inputs={'Inputs':['Input1','Input2'],
                   'Bounds':[(-1,1),(-1,1)],
                   'Structure':[['Input1_lv1','Input2_lv1'],
                                ['Input1_lv2','Input2_lv2'],
                                ['Input1_lv3','Input2_lv3']]}


# We now pass all of the design specification along with the model object, *linear_model*, into the NLoed Design constructor in order to instantiate a design object:

# In[ ]:


optimal_design = nl.Design(linear_model, nominal_param, objective, continuous_inputs=continuous_inputs)


# ### Viewing the Resulting Design
# 
# Once the Design object is created, the object instnace contains in it an optimal relaxed design which serves as an archetype for the given model and design scenario. Relaxed design's do not specify a specific sample size as they use real-valued weights to indicated the number of observations to be taken in a given set of conditions. However, somtimes it is useful to view the relaxed design and its corresponding weights as follows using the *.relaxed()* function:

# In[ ]:


optimal_design.relaxed()


# In order to generate an implementable design, the relaxed design needs to be rounded to an exact design. This can be done using the Desin class's *.round()* function as follows:

# In[ ]:


sample_size = 9
optimal_design.round(sample_size)


# The returned objects for the both the *.relaxed()* and *.round()* functions are dataframes making them easy to export to Excel or text files. The dataframes containing designs can also be used as input to user-callable functions of NLoed's Model class for data simulation and evauation of design performance on various metrics. See the other notebook examples for more details.
