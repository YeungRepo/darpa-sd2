#!/usr/bin/env python
# coding: utf-8

# In[1]:


from matplotlib import pyplot as plt
import matplotlib
from sklearn import preprocessing
import os
import numpy as np
import random as rd
import tensorflow as tf 
tf.compat.v1.disable_eager_execution()
sess  = tf.compat.v1.InteractiveSession()
from scipy.integrate import odeint
import pickle
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
import scipy
import joblib


# In[47]:


def dsgrn_3node_ode(x, t, Vmax1, Vmax2, Vmax3, K_Arab_araC1, K_AHL_luxR1, K_NaSal_nahR1, n1_num, n2_num, n3_num, KLac, K, dm, dp, iptg):
    
    X = x
    #iptg = 0
    return [Vmax1*(((X[1]/K_Arab_araC1)**n1_num + (X[3]/K_AHL_luxR1)**n2_num + (X[5]/K_NaSal_nahR1)**n3_num)/(1 + (X[1]/K_Arab_araC1)**n1_num + (X[3]/K_AHL_luxR1)**n2_num + (X[5]/K_NaSal_nahR1)**n3_num))*(iptg/(KLac + iptg)) - dm*X[0],
            K*X[0] - dp*X[1],
            
            Vmax2*(((X[1]/K_Arab_araC1)**n1_num + (X[5]/K_NaSal_nahR1)**n3_num)/(1 + (X[1]/K_Arab_araC1)**n1_num +  (X[5]/K_NaSal_nahR1)**n3_num)) - dm*X[2],
            
            K*X[2] - dp*X[3],
            
            Vmax3*(((X[1]/K_Arab_araC1)**n1_num)/(1 +  (X[1]/K_Arab_araC1)**n1_num)) - dm*X[4],
            
            K*X[4] - dp*X[5]
]


# In[124]:


def toggle_switch_learned(X, t, Kx_num, a11_num, a12_num, a13_num, b11_num, b12_num, b13_num, a21_num, a22_num, a23_num, b21_num, b22_num, b23_num, a31_num, a32_num, a33_num, b31_num, b32_num, b33_num, n1_num, n2_num, n3_num):

    return np.matmul(Kx_num, np.hstack([X.T, np.array([(1 + a11_num*X[1]**n1_num + a12_num*X[3]**n2_num + a13_num*X[5]**n3_num)/(1 + b11_num*X[1]**n1_num + b12_num*X[3]**n2_num + b13_num*X[5]**n3_num), 
                                                       (1 + a21_num*X[1]**n1_num + a22_num*X[3]**n2_num + a23_num*X[5]**n3_num)/(1 + b21_num*X[1]**n1_num + b22_num*X[3]**n2_num + b23_num*X[5]**n3_num),
                                                       (1 + a31_num*X[1]**n1_num + a32_num*X[3]**n2_num + a33_num*X[5]**n3_num)/(1 + b31_num*X[1]**n1_num + b32_num*X[3]**n2_num + b33_num*X[5]**n3_num)]).T[0]]))
    
    


# In[11]:


def generalized_hill_function(X):
    return tf.concat([[(1 + a11_tf*X[1]**n1_tf + a12_tf*X[3]**n2_tf + a13_tf*X[5]**n3_tf)/(1 + b11_tf*X[0]**n1_tf + b12_tf*X[1]**n2_tf + b13_tf*X[1]**n3_tf)], 
                      
                      [(1 + a21_tf*X[1]**n1_tf + a22_tf*X[3]**n2_tf + a23_tf*X[5]**n3_tf)/(1 + b21_tf*X[0]**n1_tf + b22_tf*X[1]**n2_tf + b23_tf*X[1]**n3_tf)],
                     
                      [(1 + a31_tf*X[1]**n1_tf + a32_tf*X[3]**n2_tf + a33_tf*X[5]**n3_tf)/(1 + b31_tf*X[0]**n1_tf + b32_tf*X[1]**n2_tf + b33_tf*X[1]**n3_tf)]], axis = 0)


# In[16]:


ICs = []
num_ics = 100
for i in range(0, num_ics):
    ICs.append([np.random.uniform(0, 5), np.random.uniform(0, 5), np.random.uniform(0, 5), np.random.uniform(0, 5), np.random.uniform(0, 5), np.random.uniform(0, 5)])
    
for i in range(0, num_ics):
    plt.scatter(ICs[i][0],ICs[i][1])
    
plt.style.use('classic')


# In[17]:


Vmax1 = 10
Vmax2 = 10
Vmax3 = 10

K_Arab_araC1 = 5
K_AHL_luxR1  = 10
K_NaSal_nahR1= 10

n1 = 1
n2 = 2
n3 = 2

K      = 10

dm     = 0.5
dp     = 0.5

KLac   = 1

IPTG   = np.array([0] + [4**i for i in range(0, 1)])
#params = Vmax1, Vmax2, Vmax3, K_Arab_araC1, K_AHL_luxR1, K_NaSal_nahR1, n1_num, n2_num, n3_num, KLac, K, dm, dp 


# In[30]:


T  = 10
t = np.linspace(0, T, T*80)
#x_full = odeint(michaelis_menten_full, x0, t, args = (10, 10, 0.1)).T.reshape(2,2,15000)

fig, axs = plt.subplots(6, 2, figsize = (20, 20))
fig.suptitle('Horizontally stacked subplots')

count = 0
i = 0
j = 0
for ic in ICs:
    #print(iptg)
    sol, op = odeint(dsgrn_3node_ode, np.array(ic), t, args = (Vmax1, Vmax2, Vmax3, K_Arab_araC1, K_AHL_luxR1, K_NaSal_nahR1, n1, n2, n3, KLac, K, dm, dp, 0.1), full_output=True)
    
    if not count:
        Xp_data = np.array(sol).T
        Xf_data = np.gradient(sol.T, t, axis = 1)
        #print(Xf_data)
    else:
        #print("exec")
        Xp_data = np.hstack([Xp_data, np.array(sol).T])
        Xf_data = np.hstack([Xf_data, np.gradient(sol.T, t, axis = 1)])
    axs[0, 0].plot(t, np.array(sol).T[0])
    axs[0, 1].plot(t, np.array(sol).T[1])
    axs[1, 0].plot(t, np.gradient(sol.T, t, axis = 1)[0])
    axs[1, 1].plot(t, np.gradient(sol.T, t, axis = 1)[1])
    
    axs[2, 0].plot(t, np.array(sol).T[2])
    axs[2, 1].plot(t, np.array(sol).T[3])
    axs[3, 0].plot(t, np.gradient(sol.T, t, axis = 1)[2])
    axs[3, 1].plot(t, np.gradient(sol.T, t, axis = 1)[3])
    
    axs[4, 0].plot(t, np.array(sol).T[4])
    axs[4, 1].plot(t, np.array(sol).T[5])
    axs[5, 0].plot(t, np.gradient(sol.T, t, axis = 1)[4])
    axs[5, 1].plot(t, np.gradient(sol.T, t, axis = 1)[5])
    
    count+=1
plt.style.use('default')


# In[34]:


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


# R2 = (1 - tf.divide(tf.math.reduce_sum(tf.math.square(tf.concat([Xf, [Xf[0]**n1]], axis = 0) - tf.math.reduce_mean(tf.concat([Xf, [Xf[0]**n1]], axis = 0), axis=0))), tf.math.reduce_sum(tf.math.square(tf.concat([Xf, [Xf[0]**n1]], axis = 0) - tf.matmul(Kx, tf.concat([Xf, [Xf[0]**n1]], axis = 0)))))) * 100

# ## Regularization parameter 

# In[62]:


lambda_reg = 1
lambda_reg_positive = 1
num_states = 6


# In[155]:


cost = (tf.reduce_sum(tf.pow(Xf - tf.matmul(Kx_tf, tf.concat([Xp, generalized_hill_function(Xp)], axis = 0)), 2)) + lambda_reg*tf.pow(tf.norm(Kx_tf[0: num_states, 0: num_states+3], ord = 'fro', axis = (0, 1)), 2))/Xp_data.shape[1]


# #optimizer = tf.compat.v1.train.AdagradOptimizer(
#     learning_rate,
#     initial_accumulator_value=0.1,
#     use_locking=False,
#     name='Adagrad'
# ).minimize(cost)
# 

# In[156]:


cost_list[epoch-1] - c


# In[ ]:


training_epochs = 400000
learning_rate = 0.003
error_threshold = 1e-6
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.99, epsilon=1e-08, use_locking=False, name='Adam').minimize(cost)


init = tf.compat.v1.global_variables_initializer()
c = 100
cost_list = [0]
epoch = 0
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
    checkpoint_dir = "checkpoints2"

    # Create the directory if it does not already exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Specify the path to the checkpoint file
    checkpoint_file = os.path.join(checkpoint_dir, "checkpoint2.chk")
    
    saver = tf.compat.v1.train.Saver(name="saver")
    saver.save(sesh, checkpoint_file)


# In[142]:


cost_list[-1]


# In[143]:


cost_list[epoch-1]


# In[144]:



saver = tf.compat.v1.train.Saver(name="saver")

sess=tf.compat.v1.Session()    
saver.restore(sess, checkpoint_file)


# In[145]:



a11_num = sess.run(a11_tf)
a12_num = sess.run(a12_tf)
a13_num = sess.run(a13_tf)


b11_num = sess.run(b11_tf)
b12_num = sess.run(b12_tf)
b13_num = sess.run(b13_tf)


a21_num = sess.run(a21_tf)
a22_num = sess.run(a22_tf)
a23_num = sess.run(a23_tf)

b21_num = sess.run(b21_tf)
b22_num = sess.run(b22_tf)
b23_num = sess.run(b23_tf)

a31_num = sess.run(a31_tf)
a32_num = sess.run(a32_tf)
a33_num = sess.run(a33_tf)

b31_num = sess.run(b31_tf)
b32_num = sess.run(b32_tf)
b33_num = sess.run(b33_tf)


n1_num  = sess.run(n1_tf)
n2_num  = sess.run(n2_tf)
n3_num  = sess.run(n3_tf)

Kx_num  = sess.run(Kx_tf)


# In[146]:




import matplotlib
import seaborn as sns;
matplotlib.rcParams.update({'font.size': 20})
plt.figure(figsize=(20.0,10.0))
ax = sns.heatmap((Kx_num), linewidths=2.0,cmap='RdYlGn',vmin=-1,vmax=1,annot=True,fmt='.1g',cbar=False, annot_kws = {"Size": 30})
bottom, top = ax.get_ylim()
#ax.set_xticklabels(FullInputDictionary_Symbolic,fontsize=25)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.rc('text', usetex=True);
plt.rc('font', family='serif',size=40);
sns.set(font_scale=10) 
bottom, top = ax.get_ylim()
# plt.axis('equal')

ax.set_ylim(bottom + 0.5, top - 0.5)


# In[154]:


Xp_final_learned = []
Xf_final_learned = []
Xp_final_actual = []
Xf_final_actual = []
#t = np.array([i for i in range(0, N+1)])
x_learned = np.zeros(6)
x_actual = np.zeros(6)
fig, axs = plt.subplots(3, 2, figsize = (20, 20))
#fig.suptitle('Horizontally stacked subplots')
plt.style.use('classic')
x_limit = 300
for ic in ICs:
    y_learned  = odeint(toggle_switch_learned, ic, t, args = (Kx_num, a11_num, a12_num, a13_num, b11_num, b12_num, b13_num, a21_num, a22_num, a23_num, b21_num, b22_num, b23_num, a31_num, a32_num, a33_num, b31_num, b32_num, b33_num, n1_num, n2_num, n3_num))
    
    y_actual = odeint(dsgrn_3node_ode, ic, t, args = (Vmax1, Vmax2, Vmax3, K_Arab_araC1, K_AHL_luxR1, K_NaSal_nahR1, n1, n2, n3, KLac, K, dm, dp, 0))
    
    axs[0, 0].scatter(t[0:x_limit], y_actual.T[0][0:x_limit], s = 1)
    axs[0, 0].plot(t[0:x_limit], y_learned.T[0][0:x_limit])
    
    axs[0, 1].scatter(t[0:x_limit], y_actual.T[1][0:x_limit], s = 1)
    axs[0, 1].plot(t[0:x_limit], y_learned.T[1][0:x_limit])
    axs[1, 0].scatter(t[0:x_limit], y_actual.T[2][0:x_limit], s = 1)
    axs[1, 0].plot(t[0:x_limit], y_learned.T[2][0:x_limit])
    axs[1, 1].scatter(t[0:x_limit], y_actual.T[3][0:x_limit], s = 1)
    axs[1, 1].plot(t[0:x_limit], y_learned.T[3][0:x_limit])
    axs[2, 0].scatter(t[0:x_limit], y_actual.T[4][0:x_limit], s = 1)
    axs[2, 0].plot(t[0:x_limit], y_learned.T[4][0:x_limit])
    axs[2, 1].scatter(t[0:x_limit], y_actual.T[5][0:x_limit], s = 1)
    axs[2, 1].plot(t[0:x_limit], y_learned.T[5][0:x_limit])
    
    


# In[ ]:





# In[ ]:




