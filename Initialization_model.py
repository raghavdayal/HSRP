#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as n
import cv2 as c
import os
from scipy.special import expit
from scipy import ndimage
import random as rand

# Defining a function read_photos which returns the formated images each of resolution (width_avg X Height_avg) and the y(numpy array containing the labels)

# In[2]:

def read_photos():
    dirsp1 = [] 
    dirs1 = []
    files1 = []
    dirsp0 = [] 
    dirs0 = [] 
    files0 = []
    
    for i in os.walk("./images/hsrp_plates"):
        dirsp1, dirs1, files1 = i
        
    for j in os.walk("./images/non_hsrp_plates"):
        dirsp0, dirs0, files0 = j
    
    for i in range(0,len(files1)):
        temp = dirsp1+"/"+files1[i]
        files1[i] = temp
        
    for i in range(0,len(files0)):
        temp = dirsp0+"/"+files0[i]
        files0[i] = temp
    
        
    numy_1 = len(files1)
    numy_0 = len(files0)
    
    print(numy_1)
    print(numy_0)
#     final_images = files1 + files0
    final_images_tuple = []
    final_images =[]
    l_final=[]
    l_1=[]
    l_0=[]
    for i in range(0, numy_1):
        l_1.append(1);
    for i in range(0, numy_0):
        l_0.append(0)
        
#     l_final = l_1+l_0
    k=0
    j=0
    

    for i in range(0,numy_0):
        final_images_tuple.append((files0[i],l_0[i]))
    
        
    for i in range(0,numy_1):
        final_images_tuple.append((files1[i],l_1[i]))
        
    rand.shuffle(final_images_tuple)

    images_for_training = []
    u=1
    avg_width = 0
    avg_height = 0
    sum_w =0
    sum_h=0
#     for i in final_images:
#         temp = c.imread(i)
#         sum_w = sum_w + temp.shape[1]
#         sum_h = sum_h + temp.shape[0]
    
#     avg_width = int(n.floor(sum_w/len(final_images)))
#     avg_height = int(n.floor(sum_h/len(final_images)))
    final_tuple = []
    for f in range(len(final_images_tuple)):
        i,j = final_images_tuple[f]
        temp = c.imread(i)
        for k in range(0,6):
            final_tuple.append((c.resize(ndimage.rotate(temp,k),(128,128),interpolation = c.INTER_AREA),j))
     
    rand.shuffle(final_tuple)
    
    for i in range(len(final_tuple)):
        g,h = final_tuple[i]
        images_for_training.append(g)
        l_final.append(h)

    return l_final, images_for_training

# In[3]:


def segre_input(images, Y):
    train_len = int(0.8 * len(images))
    image_train = []
    Y_train = []
    Y_test = []
    image_test = []
    i=0
    for i in range(0,train_len):
        image_train.append(images[i])
        Y_train.append(Y[i])

    for j in range(i+1,len(images)):
        image_test.append(images[j])
        Y_test.append(Y[j])
        
    
    return image_train, image_test, Y_train, Y_test
        


# In[4]:


#further segregation of training set into training_train and trianing_test
def fur_seg(training_set, Y_training):
    training_train, training_test,Y_training_train, Y_training_test = segre_input(training_set, Y_training) 
    return training_train, training_test,Y_training_train, Y_training_test


# In[5]:


def normalize_input(image):
    result = n.divide(image,255)
    return result


# In[6]:


def input_f(training_train,training_test,testing_set,Y_training_train,Y_training_test,Y_test):
    r = 128*128*3
    y_tr_tr = len(Y_training_train)
    y_tr_t = len(Y_training_test)
    y_t = len(Y_test)
    input_training_train = normalize_input(n.reshape(training_train,(r,len(training_train))))
    input_training_test = normalize_input(n.reshape(training_test,(r,len(training_test))))
    input_testing_set = normalize_input(n.reshape(testing_set, (r,len(testing_set))))
    Y_training_train = n.reshape(Y_training_train,(y_tr_tr,1))
    Y_training_test = n.reshape(Y_training_test,(y_tr_t,1))
    Y_test = n.reshape(Y_test,(y_t,1))
    input_m = {
                "in_tr_tr":input_training_train,
                "in_tr_t":input_training_test,
                "in_t_t":input_testing_set,
                "Y_tr_tr":Y_training_train,
                "Y_tr_t":Y_training_test,
                "Y_t":Y_test
                }
    return input_m

def sigmoid_s(Z):
    temp = (1+expit(-Z))
    result = n.abs(1/temp)
    return result

def sigmoid(Z):
    temp = (1+expit(-1*Z))
    result = 1/temp
    cache = (Z)
    return result, cache

def l_relu(Z):
    cache = (Z)
    result = n.where(Z > 0, Z, Z * 0.01)
    return result, cache

def back_leaky(Z):
    Z[Z<=0] = 0.01
    Z[Z>0] = 1
    return Z
    
def relu(Z):
    cache = (Z)
    return n.maximum(0,Z), cache

def reluder(Z):
    Z[Z<=0] = 0
    Z[Z>0] = 1
    return Z

def back_relu(dA,activation_cache):
    Z = activation_cache
    #A ,cache= relu(Z)
    temp = back_leaky(Z)
    dZ = dA*temp
    return dZ

def sigmoid_backward(dA,activation_cache):
    Z = activation_cache 
    s = n.abs(1/(1+expit(-Z)))
    dZ = dA * s * (1-s)
    return dZ
def init_param(dim):
    params={}
    L = len(dim)
    for i in range(1,L):
        params['W'+str(i)] = n.random.randn(dim[i],dim[i-1]).astype(n.float64)*(2/dim[i-1])
        params['b' + str(i)] = n.zeros((dim[i],1))
    return params

def linear_forward(A, W,b):
    Z = n.dot(W,A)+b #(dimsA = n_prev,m) 
    cache = (A, W, b)
    return Z, cache

def linear_activation_forward(A_prev,W,b,activation):
    Z, linear_cache = linear_forward(A_prev,W,b)
    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        A, activation_cache = l_relu(Z)
    cache = (linear_cache,activation_cache)
    return A, cache

def L_layer_forward(X,param):
    caches = []
    A = X
    L = len(param) // 2
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, param['W' + str(l)],param['b' + str(l)], "relu")
        caches.append(cache)
    A_prev  = A
    A, cache = linear_activation_forward(A_prev , param['W' + str(L)],param['b' + str(L)], "sigmoid")
    caches.append(cache)
    
    return A, caches

################implementing back_prop in n layers#########################
def linear_backward(dZ,linear_cache):
    A_prev, W,b = linear_cache
    m = A_prev.shape[1]
    print(dZ.shape)
    dw = (n.dot(dZ,A_prev.T))/m
    db = (n.sum(dZ,axis = 1,keepdims = True))/m
    dA_prev = n.dot(W.T,dZ)
    return dA_prev, dw, db

def linear_activation_backward(dA, cache, activation):
    
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = back_relu(dA,activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA,activation_cache)
        dA_prev, dW,db = linear_backward(dZ,linear_cache)
    
    return dA_prev, dW, db
# In[9]:
def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) 
    AL = AL+1e-5
    dAL = -(n.divide(Y, AL) - n.divide(1 - Y, 1 - AL))
    current_cache = caches[L-1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL,  current_cache, "sigmoid")
    grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp
    
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)],current_cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads

def update_parameters(params, grads, learning_rate):
    parameters = params.copy()
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)]-(learning_rate*grads["dW"+str(l+1)]) 
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)]-(learning_rate*grads["db"+str(l+1)])
    return parameters

def forward_prop_s(param, input_m):
    W = param["W_s"]
    b = param["b_s"]
    X = input_m["in_tr_tr"]
    Z = n.dot(X.T,W)+b
    A= sigmoid_s(Z)
    result = {
        "Z":Z,
        "A":A
    }
    return result
    
def forward_prop(W,b,test):
    X = test
    Z = n.dot(X.T,W)+b
    A= n.abs(sigmoid_s(Z))
    result = {
        "Z":Z,
        "A":A
    }
    return result

# In[10]:


def backward_prop(result_forward, param, input_m):
    W = param["W_s"]
    b = param["b_s"]
    X = input_m["in_tr_tr"]
    A = result_forward["A"]
    Y = input_m["Y_tr_tr"]
    s_db = param["s_db"]
    s_dw = param["s_dw"]
    dZ = A - Y
    dW = (1/len(Y))*(n.dot(X,dZ))
    db = (1/len(Y))*n.sum(dZ,axis=0,keepdims = True)
    back_dict = {"dZ":dZ,"dW":dW,"db":db,"s_dw":s_dw,"s_db":s_db}
    return back_dict


# In[11]:


def loss_function(AL,Y):
    A = AL
    m = AL.shape[1]
    ep = 1e-5
    J = -1/m*(n.dot(Y,n.log(A+ep).T)+n.dot((1-Y),n.log(((1-A)+ep).T)))
    J = n.squeeze(J)
    return J


# In[12]:
def output_m(A):
    result = n.zeros(A.shape)
    for i in range(0,A.shape[1]):
        if A[0][i] >= 0.59:
            result[0][i] = 1
    return result
    
def classification_test(Y,Yhat):
    absol = n.sum(n.abs(Y-Yhat.T),axis=0,keepdims=True)
    error = (absol/len(Y))*100
    accuracy = 100-error
    return accuracy

def compute(input_m,param):
    test = input_m["in_tr_tr"]
    AL,caches = L_layer_forward(test,param)
    result = output_m(AL)
    acc = classification_test(input_m["Y_tr_tr"],result)
    return result,AL,acc