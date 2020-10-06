#!/usr/bin/env python
# coding: utf-8

# # Boston house price prediction

# ## Import the required library

# In[151]:


import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from IPython.display import HTML


# ## Loading the Data set

# In[152]:


#load the dataset
boston=load_boston()

#Description of the dataset
print(boston.DESCR)


# ## Put the data into pandas DataFrames

# In[153]:



features = pd.DataFrame(boston.data,columns=boston.feature_names)
features


# ## Target variable

# In[154]:


target = pd.DataFrame(boston.target,columns=['target'])
target


# In[155]:


#To find min and max target value max(dataframe_object[coumn_name])
#max(target['target'])


# In[156]:


# min(target['target'])


# ## concatenate features and target into a singel DataFrame

# In[157]:


#axis = 1 make it concat column wise
df=pd.concat([features,target],axis=1)
df


# ## Data Visualization

# In[158]:


#use round(decimals=2) to set the precision to 2 decimal places

df.describe().round(decimals = 2)


# In[159]:


# Correlation between Target & Attributes(every column on the data)
corr = df.corr('pearson')

#take absolute values of correlation
corrs=[abs(corr[attr]['target']) for attr in list(features)]

# Make a list of pairs [(corr,feature)]
l=list(zip(corrs,list(features)))

# sort the list of pairs in reverse/descending order
# with the correlation value as the key for sorting
l.sort(key = lambda x : x[0], reverse=True)

# zip the accomplish function
corrs, labels = list(zip((*l)))

#plot correlation with respect to the target variable as a bar graph
index = np.arange(len(labels))
plt.figure(figsize=(15,5))
plt.bar(index, corrs, width=0.5)
plt.xlabel('Attributes')
plt.ylabel('Correlation with the target variable')
plt.xticks(index,labels)
plt.show()


# ## Normalize the Data

# In[160]:


X=df['LSTAT'].values
Y=df['target'].values

# Before normalization
print(Y[:5])


# In[161]:


x_scaler = MinMaxScaler()
X = x_scaler.fit_transform(X.reshape(-1,1))
X = X[:,-1]

y_scaler = MinMaxScaler()
Y = y_scaler.fit_transform(Y.reshape(-1,1))
Y = Y[:,-1]

# After Normalization
print(Y[:5])


# ## Splitting data into fixed sets

# In[162]:


xtrain, xtest, ytrain, ytest = train_test_split(X,Y,test_size = 0.2)


# ## Define Mean Squared Error (MSE)

# In[163]:


def error(m, x, c, t):
    N = x.size
    e = sum(((m * x + c) - t) ** 2)
    return  e * 1/(2 * N)
        


# In[164]:


# Update Function
def update(m, x, c, t, learning_rate):
    grad_m = sum(2*((m * x + c) - t)* x)
    grad_c = sum(2*((m * x + c) - t))
    m = m - grad_m * learning_rate
    c = c - grad_c * learning_rate
    return m, c


# In[165]:


# define gradient descent function

def gradient_descent(init_m, init_c, x, t, learning_rate, iterations, error_threshold):
    m = init_m
    c = init_c
    error_values = list()
    mc_values = list()
    for i in range(iterations):
        e = error(m, x, c, t)
        if e<error_threshold:
            print('Error less than the threshold. Stopping gradient descent')
            break
        error_values.append(e)
        m, c = update(m, x, c, t, learning_rate)
        mc_values.append((m,c))
    return m, c, error_values, mc_values


# In[166]:


get_ipython().run_cell_magic('time', '', 'init_m = 0.9\ninit_c = 0\nlearning_rate = 0.001\niterations =250\nerror_threshold = 0.001\n\nm, c, error_values, mc_values = gradient_descent(init_m, init_c, xtrain, ytrain, learning_rate, iterations, error_threshold)')


# ## Model Training Visualization

# In[167]:


# As the number of iteration increases, changes in the line are less noticable
# To reduce processing time for animation, choose smaller value

mc_values_anim = mc_values[0:250:5]


# In[168]:


fig, ax = plt.subplots()
ln, =plt.plot([],[],'ro-',animated=True)

def init():
    plt.scatter(xtest, ytest, color='g')
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    return ln,

def update_frame(frame):
    m , c = mc_values_anim[frame]
    x1 ,y1 = -0.5, m * -.5 + c
    x2, y2 = 1.5, m * 1.5 + c
    ln.set_data([x1,x2],[y1,y2])
    return ln,

anim = FuncAnimation(fig, update_frame, frames=range(len(mc_values_anim)),
                           init_func=init, blit=True)
HTML(anim.to_html5_video())


# ## Error Visualization (How regression line arise)

# In[169]:


# Plotting regression line upon the training data set
# Training data doesn't lie on straight line
plt.scatter(xtrain,ytrain,color='b')
plt.plot(xtrain,(m*xtrain+c),color='r')


# In[170]:


plt.plot(np.arange(len(error_values)),error_values)
plt.ylabel('Error')
plt.xlabel('Iterations')
# even if the iteration is set to high it will not cause any more reduction in the error value


# In[171]:


# Prediction of Price on test set as a vectorized operation
predicted = (m * xtest) + c


# In[172]:


# Compute MSE for the predicted values on the testing set
mean_squared_error(ytest, predicted)


# In[173]:


# Put xtest ytest predicted values into a single dataFrame so that we can
# see the predicted values alongside the testing set

p=pd.DataFrame(list(zip(xtest,ytest,predicted)),columns=['x','target_y','predicted_y'])
p.head()
                                                         


# In[174]:


plt.scatter(xtest,ytest,color = 'b')
plt.plot(xtest,predicted,color = 'r')


# In[175]:


# Reshape to change the shape that is required by the scales
# predicted = np.array(predicted).reshape(-1,1)

predicted = predicted.reshape(-1,1)
xtest=xtest.reshape(-1,1)
ytest=ytest.reshape(-1,1)

xtest_scaled = x_scaler.inverse_transform(xtest)
ytest_scaled = y_scaler.inverse_transform(ytest)
predicted_scaled = y_scaler.inverse_transform(predicted)

# This is to remove extra dimension 
x_test_scaled = xtest_scaled[:,-1]
y_test_scaled = ytest_scaled[:,-1]
predicted_scaled = predicted_scaled[:,-1]

p = pd.DataFrame(list(zip(xtest_scaled,ytest_scaled,predicted_scaled)),columns = ['x','target_y','predicted_y'])
p=p.round(decimals = 2)
p.head()


# ## Train Model with different Attributes and Predict
# 
# ## Instead of 'LSTAT' if we use 'RM' or 'DIS'
# 
# ## MSE 'RM' = 2* MSE 'LSTAT'
# 
# ## MSE 'DIS' = 4* MSE 'LSTAT'
