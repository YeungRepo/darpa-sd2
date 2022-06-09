#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pickle
import numpy as np
import random as rd
import tensorflow as tf 
tf.compat.v1.disable_eager_execution()
sess  = tf.compat.v1.InteractiveSession()
import scipy


# In[ ]:


def generalized_hill_function(X):
    return tf.concat([[(1 + a11_tf*X[1]**n1_tf + a12_tf*X[3]**n2_tf + a13_tf*X[5]**n3_tf)/(1 + b11_tf*X[0]**n1_tf + b12_tf*X[1]**n2_tf + b13_tf*X[1]**n3_tf)], 
                      
                      [(1 + a21_tf*X[1]**n1_tf + a22_tf*X[3]**n2_tf + a23_tf*X[5]**n3_tf)/(1 + b21_tf*X[0]**n1_tf + b22_tf*X[1]**n2_tf + b23_tf*X[1]**n3_tf)],
                     
                      [(1 + a31_tf*X[1]**n1_tf + a32_tf*X[3]**n2_tf + a33_tf*X[5]**n3_tf)/(1 + b31_tf*X[0]**n1_tf + b32_tf*X[1]**n2_tf + b33_tf*X[1]**n3_tf)]], axis = 0)


# In[3]:


with open('dsgrn_simulation_data.pickle', 'rb') as handle:
    Xp_data, Xf_data = pickle.load(handle)


# In[4]:


Xp=tf.compat.v1.placeholder(tf.compat.v1.double, shape= (6, Xp_data.shape[1]))

Xf=tf.compat.v1.placeholder(tf.compat.v1.double, shape= (6, Xf_data.shape[1]))

a11_tf  = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean=5,stddev=0.3,dtype=tf.double))
a12_tf   = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean=5,stddev=0.3,dtype=tf.double))
a13_tf   = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean=5,stddev=0.3,dtype=tf.double))

b11_tf = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean=5,stddev=0.3,dtype=tf.double))
b12_tf = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean=5,stddev=0.3,dtype=tf.double))
b13_tf = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean=5,stddev=0.3,dtype=tf.double))


n1_tf = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean=5,stddev=0.3,dtype=tf.double))
n2_tf = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean=5,stddev=0.3,dtype=tf.double))
n3_tf = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean=5,stddev=0.3,dtype=tf.double))

a21_tf  = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean=5,stddev=0.3,dtype=tf.double))
a22_tf   = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean=5,stddev=0.3,dtype=tf.double))
a23_tf   = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean=5,stddev=0.3,dtype=tf.double))

b21_tf = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean=5,stddev=0.3,dtype=tf.double))
b22_tf = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean=5,stddev=0.3,dtype=tf.double))
b23_tf = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean=5,stddev=0.3,dtype=tf.double))


a31_tf  = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean=5,stddev=0.3,dtype=tf.double))
a32_tf   = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean=5,stddev=0.3,dtype=tf.double))
a33_tf   = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean=5,stddev=0.3,dtype=tf.double))

b31_tf = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean=5,stddev=0.3,dtype=tf.double))
b32_tf = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean=5,stddev=0.3,dtype=tf.double))
b33_tf = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean=5,stddev=0.3,dtype=tf.double))

Kx_tf = tf.Variable(tf.compat.v1.truncated_normal((6, 9), mean=0.0,stddev=0.1,dtype=tf.double));
#np.abs(Y - W*b)


# ## Regularization parameter 

# In[62]:


lambda_reg = 1
lambda_reg_positive = 1
num_states = 6


# In[ ]:


training_epochs = 10
learning_rates = np.array([1/10**i for i in range(0, 5)])
reg = np.array([1/10**i for i in range(0, 5)])

for lambda_reg in reg:
    for learning_rate in learning_rates:
        c = 100
        cost_list = [0]
        epoch = 0
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.99, epsilon=1e-08, use_locking=False, name='Adam').minimize(cost)
        init = tf.compat.v1.global_variables_initializer()
        
        cost = (tf.reduce_sum(tf.pow(Xf - tf.matmul(Kx_tf, tf.concat([Xp, generalized_hill_function(Xp)], axis = 0)), 2)) + lambda_reg*tf.pow(tf.norm(Kx_tf[0: num_states, 0: num_states+3], ord = 'fro', axis = (0, 1)), 2))/Xp_data.shape[1]

        with tf.compat.v1.Session() as sesh:    
            sesh.run(init)    
            #print("Initial n1", sesh.run(n1))
            while epoch < training_epochs and np.abs(cost_list[epoch-1] - c) > 1e-8:
                c = sesh.run(cost, feed_dict = {Xp: np.array(Xp_data), Xf: np.array(Xf_data)})
                cost_list.append(c)
                if epoch % 500 == 0:
                    print("Epoch:", epoch, "{:.5f}".format(c))
                    #print("Exponent", sesh.run(n1))
                    #print("R2", sesh.run(R2, feed_dict = {Xp: np.array(Xp_data), Xf: np.array(Xf_data)}))
                sesh.run(optimizer, feed_dict = {Xp: np.array(Xp_data), Xf: np.array(Xf_data)})
                epoch+=1
            print(sesh.run(cost, feed_dict = {Xp: np.array(Xp_data), Xf: np.array(Xf_data)}))
            checkpoint_dir = "checkpoints"

            # Create the directory if it does not already exist
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            # Specify the path to the checkpoint file
            checkpoint_file = os.path.join(checkpoint_dir, "checkpoint_learning_rate"+str(learning_rate) + "regularization_"+str(lambda_reg)+".chk")

            saver = tf.compat.v1.train.Saver(name="saver")
            saver.save(sesh, checkpoint_file)

