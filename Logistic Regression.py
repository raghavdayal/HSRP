#!/usr/bin/env python
# coding: utf-8

# In[3]:


import Initialization_model as im
import matplotlib.pyplot as plt


# In[5]:


import tensorflow as tf


# In[2]:


Y,images = im.read_photos()
print(Y)


# In[3]:


image_train, image_test, Y_train, Y_test = im.segre_input(images, Y)


# In[4]:


training_train, training_test,Y_training_train, Y_training_test = im.fur_seg(image_train,Y_train)
print(training_train[3].shape)


# In[5]:


input_m = im.input_f(training_train,training_test,image_test,Y_training_train,Y_training_test,Y_test)
print(Y_training_test)


# In[6]:


print(input_m["in_tr_tr"].shape)


# In[8]:


nx = input_m["in_tr_tr"].shape[0]
m = input_m["in_tr_tr"].shape[1]
Y = input_m["Y_tr_tr"].T
X = input_m["in_tr_tr"]
learning_rate = 0.0075
dims = [nx,30,30,5,1]
costs = []
params = im.init_param(dims)
AL, caches = im.L_layer_forward(X,params)
cost = im.loss_function(AL,Y)
for i in range(0,100):
    AL, caches = im.L_layer_forward(X,params)
    cost = im.loss_function(AL,Y)
    print(cost)
    grads = im.L_model_backward(AL, Y, caches)
    params = im.update_parameters(params,grads,learning_rate)


# In[6]:


def compute(input_m,param):
    test = input_m["in_tr_tr"]
    AL,caches = im.L_layer_forward(test,param)
    result = im.output_m(AL)
    acc = im.classification_test(input_m["Y_tr_tr"],result)
    return result,AL,acc


# In[7]:


##testing
result,Al, acc = compute(input_m,params)
print(Al.shape)


# In[4]:


print(acc)
print(Al)
print(result)
print(input_m["Y_tr_tr"])

