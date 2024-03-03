#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Supress Warnings

import warnings
warnings.filterwarnings('ignore')

# Import the numpy and pandas package

import numpy as np
import pandas as pd

# Data Visualisation

import matplotlib.pyplot as plt 
import seaborn as sns


# In[2]:


housing = pd.DataFrame(pd.read_csv("Housing.csv"))


# In[3]:


# Check the head of the dataset
housing.head()


# In[4]:


housing.shape


# In[5]:


housing.info()


# In[6]:


housing.describe()


# In[7]:


# Checking Null values
housing.isnull().sum()*100/housing.shape[0]
# There are no NULL values in the dataset, hence it is clean.


# In[8]:


# Outlier Analysis
fig, axs = plt.subplots(2,3, figsize = (10,5))
plt1 = sns.boxplot(housing['price'], ax = axs[0,0])
plt2 = sns.boxplot(housing['area'], ax = axs[0,1])
plt3 = sns.boxplot(housing['bedrooms'], ax = axs[0,2])
plt1 = sns.boxplot(housing['bathrooms'], ax = axs[1,0])
plt2 = sns.boxplot(housing['stories'], ax = axs[1,1])
plt3 = sns.boxplot(housing['parking'], ax = axs[1,2])

plt.tight_layout()


# In[9]:


sns.pairplot(housing)
plt.show()


# In[10]:


plt.figure(figsize=(20, 12))
plt.subplot(2,3,1)
sns.boxplot(x = 'mainroad', y = 'price', data = housing)
plt.subplot(2,3,2)
sns.boxplot(x = 'guestroom', y = 'price', data = housing)
plt.subplot(2,3,3)
sns.boxplot(x = 'basement', y = 'price', data = housing)
plt.subplot(2,3,4)
sns.boxplot(x = 'hotwaterheating', y = 'price', data = housing)
plt.subplot(2,3,5)
sns.boxplot(x = 'airconditioning', y = 'price', data = housing)
plt.subplot(2,3,6)
sns.boxplot(x = 'furnishingstatus', y = 'price', data = housing)
plt.show()


# In[11]:


# List of variables to map

varlist =  ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

# Defining the map function
def binary_map(x):
    return x.map({'yes': 1, "no": 0})

# Applying the function to the housing list
housing[varlist] = housing[varlist].apply(binary_map)


# In[12]:


housing.head()


# In[13]:


# Get the dummy variables for the feature 'furnishingstatus' and store it in a new variable - 'status'
status = pd.get_dummies(housing['furnishingstatus'])


# In[14]:


# Check what the dataset 'status' looks like
status.head()


# In[15]:


# Let's drop the first column from status df using 'drop_first = True'

status = pd.get_dummies(housing['furnishingstatus'], drop_first = True)


# In[16]:


# Add the results to the original housing dataframe

housing = pd.concat([housing, status], axis = 1)


# In[17]:


# Now let's see the head of our dataframe.

housing.head()


# In[21]:


# Drop 'furnishingstatus' as we have created the dummies for it

housing.drop(['furnishingstatus'], axis = 1, inplace = True)


# In[22]:


housing.head()


# In[23]:


from sklearn.model_selection import train_test_split

# We specify this so that the train and test data set always have the same rows, respectively
np.random.seed(0)
df_train, df_test = train_test_split(housing, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[24]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[25]:


# Apply scaler() to all the columns except the 'yes-no' and 'dummy' variables
num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking','price']

df_train[num_vars] = scaler.fit_transform(df_train[num_vars])


# In[26]:


df_train.head()


# In[27]:


df_train.describe()


# In[28]:


# Let's check the correlation coefficients to see which variables are highly correlated

plt.figure(figsize = (16, 10))
sns.heatmap(df_train.corr(), annot = True, cmap="YlGnBu")
plt.show()


# In[29]:


y_train = df_train.pop('price')
X_train = df_train


# In[30]:


# Importing RFE and LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


# In[31]:


# Running RFE with the output number of the variable equal to 10
lm = LinearRegression()
lm.fit(X_train, y_train)


# In[32]:


rfe = RFE(lm, 6)             # running RFE
rfe = rfe.fit(X_train, y_train)


# In[33]:


list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# In[34]:


col = X_train.columns[rfe.support_]
col


# In[35]:


X_train.columns[~rfe.support_]


# In[36]:


# Creating X_test dataframe with RFE selected variables
X_train_rfe = X_train[col]


# In[37]:


import statsmodels.api as sm  
X_train_rfe = sm.add_constant(X_train_rfe)


# In[39]:


lm = sm.OLS(y_train,X_train_rfe).fit()   # Running the linear model


# In[40]:


#Let's see the summary of our linear model
print(lm.summary())


# In[41]:


# Calculate the VIFs for the model
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[42]:


vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[43]:


y_train_price = lm.predict(X_train_rfe)


# In[44]:


res = (y_train_price - y_train)


# In[45]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[46]:


# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_price), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)


# In[47]:


plt.scatter(y_train,res)
plt.show()


# In[49]:


num_vars = ['area','stories', 'bathrooms', 'airconditioning', 'prefarea','parking','price']
df_test[num_vars] = scaler.fit_transform(df_test[num_vars])


# In[50]:


y_test = df_test.pop('price')
X_test = df_test


# In[51]:


X_test = sm.add_constant(X_test)


# In[52]:


X_test_rfe = X_test[X_train_rfe.columns]


# In[53]:


y_pred = lm.predict(X_test_rfe)


# In[54]:


from sklearn.metrics import r2_score 
r2_score(y_test, y_pred)


# In[55]:


fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)                          # Y-label


# In[ ]:


#We can see that the equation of our best fitted line is:
#price=0.35×area+0.20×bathrooms+0.19×stories+0.10×airconditioning+0.10×parking+0.11×prefarea

