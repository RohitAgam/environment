#!/usr/bin/env python
# coding: utf-8

# In[51]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from pyearth import Earth
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import warnings
import matplotlib.pyplot as plt


# In[52]:


warnings.filterwarnings("ignore")


# In[53]:


df = pd.read_csv("Data.csv")


# In[54]:


df.head(5)


# In[55]:


target_col = "sales"
X = df.loc[:, ['radio']]
y = df.loc[:, target_col]


# # Linear Regression Model

# In[56]:


LR_model = LinearRegression()


# In[57]:


LR_Model_Fitted = LR_model.fit(X, y)


# In[58]:


print("Intercept: ", LR_Model_Fitted.intercept_)
print("Slope: ", LR_Model_Fitted.coef_)


# In[59]:


y_pred = LR_Model_Fitted.predict(X)


# In[60]:


plt.scatter(X, y,color='g')
plt.plot(X, y_pred,color='k')
plt.figure(figsize=(20,12))


# In[61]:


r2_score(y, y_pred)


# In[62]:


mean_squared_error(y, y_pred)


# # MARS Regression Model

# In[63]:


MARS_model = Earth(max_terms=500, max_degree=1)


# In[64]:


MARS_model_fitted = MARS_model.fit(X, y)


# In[65]:


y_pred_mars = MARS_model_fitted.predict(X)


# In[66]:


# print(MARS_model_fitted.trace())


# In[67]:



# Plot the model
plt.figure()
plt.plot(X, y, 'r.')
plt.plot(X, y_pred_mars, 'b.')
plt.show()


# In[68]:


r2_score(y, y_pred_mars)


# In[69]:


mean_squared_error(y, y_pred_mars)


# In[70]:


print(MARS_model_fitted.summary())

