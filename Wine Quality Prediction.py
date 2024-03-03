#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


wine = pd.read_csv('WineQT.csv')


# In[3]:


wine.head()


# In[4]:


wine.info()


# In[7]:


wine.describe()


# In[8]:


wine.isnull().sum()*100/wine.shape[0]


# In[10]:


#sns.pairplot(wine)
#plt.show()


# In[23]:


fig = plt.figure(figsize = (15,15))

fig, axs = plt.subplots(4,3, figsize = (10,5))
plt1 = sns.boxplot(wine['fixed acidity'], ax = axs[0,0])
plt2 = sns.boxplot(wine['volatile acidity'], ax = axs[0,1])
plt3 = sns.boxplot(wine['citric acid'], ax = axs[0,2])
plt1 = sns.boxplot(wine['residual sugar'], ax = axs[1,0])
plt2 = sns.boxplot(wine['chlorides'], ax = axs[1,1])
plt3 = sns.boxplot(wine['pH'], ax = axs[1,2])
plt1 = sns.boxplot(wine['free sulfur dioxide'], ax = axs[2,0])
plt2 = sns.boxplot(wine['total sulfur dioxide'], ax = axs[2,1])
plt3 = sns.boxplot(wine['density'], ax = axs[2,2])
plt1 = sns.boxplot(wine['sulphates'], ax = axs[3,0])
plt2 = sns.boxplot(wine['alcohol'], ax = axs[3,1])
plt3 = sns.boxplot(wine['quality'], ax = axs[3,2])
plt.tight_layout()


# In[24]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'fixed acidity', data = wine)


# In[25]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'volatile acidity', data = wine)


# In[26]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'citric acid', data = wine)


# In[27]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'residual sugar', data = wine)


# In[28]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'chlorides', data = wine)


# In[29]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'free sulfur dioxide', data = wine)


# In[30]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'total sulfur dioxide', data = wine)


# In[31]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'sulphates', data = wine)


# In[32]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'alcohol', data = wine)


# In[33]:


#Dividing wine as good and bad by giving the limit for the quality
bins = (2, 6.5, 8)
group_names = ['bad', 'good']
wine['quality'] = pd.cut(wine['quality'], bins = bins, labels = group_names)


# In[34]:


#Now lets assign a labels to our quality variable
label_quality = LabelEncoder()


# In[35]:


#Bad becomes 0 and good becomes 1 
wine['quality'] = label_quality.fit_transform(wine['quality'])


# In[36]:


wine['quality'].value_counts()


# In[37]:


sns.countplot(wine['quality'])


# In[38]:


#Now seperate the dataset as response variable and feature variabes
X = wine.drop('quality', axis = 1)
y = wine['quality']


# In[39]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[40]:


#Applying Standard scaling to get optimized result
sc = StandardScaler()


# In[41]:


X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[42]:


#Applying Regression Forest Classifies to get optimized result
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)


# In[43]:


#Let's see how our model performed
print(classification_report(y_test, pred_rfc))


# In[44]:


#Confusion matrix for the random forest classification
print(confusion_matrix(y_test, pred_rfc))


# In[46]:


#Stochastic Gradient Decent Classifier
sgd = SGDClassifier(penalty=None)
sgd.fit(X_train, y_train)
pred_sgd = sgd.predict(X_test)


# In[47]:


print(classification_report(y_test, pred_sgd))


# In[48]:


print(confusion_matrix(y_test, pred_sgd))


# In[50]:


#Support Vector Classifier
svc = SVC()
svc.fit(X_train, y_train)
pred_svc = svc.predict(X_test)


# In[51]:


print(classification_report(y_test, pred_svc))


# In[52]:


#Finding best parameters for our SVC model
param = {
    'C': [0.1,0.8,0.9,1,1.1,1.2,1.3,1.4],
    'kernel':['linear', 'rbf'],
    'gamma' :[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4]
}
grid_svc = GridSearchCV(svc, param_grid=param, scoring='accuracy', cv=10)


# In[53]:


grid_svc.fit(X_train, y_train)


# In[54]:


#Best parameters for our svc model
grid_svc.best_params_


# In[55]:


#Let's run our SVC again with the best parameters.
svc2 = SVC(C = 1.2, gamma =  0.9, kernel= 'rbf')
svc2.fit(X_train, y_train)
pred_svc2 = svc2.predict(X_test)
print(classification_report(y_test, pred_svc2))


# In[56]:


#Now lets try to do some evaluation for random forest model using cross validation.
rfc_eval = cross_val_score(estimator = rfc, X = X_train, y = y_train, cv = 10)
rfc_eval.mean()


# In[ ]:


#Random forest accuracy increases from 87% to 89 % using cross validation score

