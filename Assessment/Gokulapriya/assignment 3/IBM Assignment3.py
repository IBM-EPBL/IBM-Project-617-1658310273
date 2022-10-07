#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[12]:


data = pd.read_csv('C:\\Users\\intec\\Downloads\\abalone.csv')
print(data)


# In[13]:


data.head()


# In[14]:


data.dtypes


# In[15]:


data['age'] = data['Rings']+1.5
data = data.drop('Rings', axis = 1)


# In[16]:


# 3.1. Univariate analysis (Scatter plot)
plt.scatter(data.Height,data['Whole weight'])
plt.show()


# In[17]:


# 3.1. Univariate Analysis(Histogram)
plt.hist(data['age'])


# In[18]:


# 3.2 bivariate analysis (barplot)
sns.barplot(x='Sex',y='Length',data=data)


# In[20]:


# 3.2 Bivariate analysis(countplot)
sns.countplot(x='Sex',data=data)


# In[21]:


# 3.3 Multivariate analysis
pd.plotting.scatter_matrix(data.loc[:,"Sex":"age" ],diagonal="kde",figsize=(20,15))
plt.show()


# In[22]:


# 4. Perform descriptive statistics on the dataset
d = {'Sex':pd.Series(['M','M','F','M','I','I','F','F','M']),'age':pd.Series([16.5,8.5,10.5,11.5,8.5,9.5,21.5,17.5,10.5])}
df = pd.DataFrame(d)
print (df)


# In[23]:


print (df.sum())


# In[24]:


print (df.sum(1))


# In[25]:


print (df.mean())


# In[26]:


print (df.median())


# In[27]:


print (df.mode())


# In[28]:


print (df.count())


# In[29]:


# 5. Handle the missing values
data.isnull()


# In[30]:


# 6. Find the outliers 
# Visualization using box plot
sns.boxplot(data['Shucked weight'])


# In[31]:


data = pd.get_dummies(data)
dummy_data = data


# In[32]:


var = 'Viscera weight'
plt.scatter(x = data[var], y = data['age'])
plt.grid(True)


# In[33]:


data.drop(data[(data['Viscera weight'] > 0.5) &
          (data['age'] < 20)].index, inplace = True)
data.drop(data[(data['Viscera weight']<0.5) & (
data['age'] > 25)].index, inplace = True)


# In[34]:


var = 'Shell weight'
plt.scatter(x = data[var], y = data['age'])
plt.grid(True)


# In[35]:


data.drop(data[(data['Shell weight'] > 0.6) &
          (data['age'] < 25)].index, inplace = True)
data.drop(data[(data['Shell weight']<0.8) & (
data['age'] > 25)].index, inplace = True)


# In[36]:


var = 'Shucked weight'
plt.scatter(x = data[var], y = data['age'])
plt.grid(True)


# In[37]:


data.drop(data[(data['Shucked weight'] >= 1) &
          (data['age'] < 20)].index, inplace = True)
data.drop(data[(data['Viscera weight']<1) & (
data['age'] > 20)].index, inplace = True)


# In[38]:


var = 'Whole weight'
plt.scatter(x = data[var], y = data['age'])
plt.grid(True)


# In[39]:


data.drop(data[(data['Whole weight'] >= 2.5) &
          (data['age'] < 25)].index, inplace = True)
data.drop(data[(data['Whole weight']<2.5) & (
data['age'] > 25)].index, inplace = True)


# In[40]:


var = 'Diameter'
plt.scatter(x = data[var], y = data['age'])
plt.grid(True)


# In[41]:


data.drop(data[(data['Diameter'] <0.1) &
          (data['age'] < 5)].index, inplace = True)
data.drop(data[(data['Diameter']<0.6) & (
data['age'] > 25)].index, inplace = True)
data.drop(data[(data['Diameter']>=0.6) & (
data['age'] < 25)].index, inplace = True)


# In[42]:


var = 'Height'
plt.scatter(x = data[var], y = data['age'])
plt.grid(True)


# In[43]:


data.drop(data[(data['Height'] > 0.4) &
          (data['age'] < 15)].index, inplace = True)
data.drop(data[(data['Height']<0.4) & (
data['age'] > 25)].index, inplace = True)


# In[44]:


var = 'Length'
plt.scatter(x = data[var], y = data['age'])
plt.grid(True)


# In[45]:


data.drop(data[(data['Length'] <0.1) &
          (data['age'] < 5)].index, inplace = True)
data.drop(data[(data['Length']<0.8) & (
data['age'] > 25)].index, inplace = True)
data.drop(data[(data['Length']>=0.8) & (
data['age'] < 25)].index, inplace = True)


# In[46]:


# 8.Split the dependent and independent variables
X = data.drop('age', axis = 1)
y = data['age']


# In[47]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest


# In[48]:


#9.Scale the independent variables
standardScale = StandardScaler()
standardScale.fit_transform(X)

selectkBest = SelectKBest()
X_new = selectkBest.fit_transform(X, y)

# 10.Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.25)


# In[49]:


from sklearn.linear_model import LinearRegression


# In[50]:


# 11.Build the model using LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)


# In[51]:


#12.Train the model
y_train_pred = lm.predict(X_train)

#13.Test the model
y_test_pred = lm.predict(X_test)


# In[52]:


#14.Measure the performance using Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
s = mean_squared_error(y_train, y_train_pred)
print('Mean Squared error of training set :%2f'%s)

p = mean_squared_error(y_test, y_test_pred)
print('Mean Squared error of testing set :%2f'%p)


# In[53]:


from sklearn.metrics import r2_score
s = r2_score(y_train, y_train_pred)
print('R2 Score of training set:%.2f'%s)

p = r2_score(y_test, y_test_pred)
print('R2 Score of testing set:%.2f'%p)


# In[ ]:




