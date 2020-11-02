#! /usr/bin/env python

### Import Packages 
import pickle  # for data I/O
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
# Math Packages
import numpy as np
from numpy.linalg import pinv;
from numpy.polynomial.legendre import legvander;
import math;
import random;
from numpy import genfromtxt
import itertools

# # Import CVXOPT Packages
# from cvxpy import Minimize, Problem, Variable,norm1,norm2,installed_solvers,lambda_max;
# from cvxpy import norm as cvxpynorm;
# import cvxpy;

# Tensorflow Packages
import tensorflow as tf;

# Plotting Tools for Visualizing Basis Functions
import matplotlib

matplotlib.use('Agg');  # for seamless execution in Linux environments with Tensorflow
import matplotlib.pyplot as plt;

matplotlib.rcParams.update({'font.size': 22})  # default font size on (legible) figures
import control

import os
import shutil
import pandas as pd

REMOVE_PREVIOUS_RUNDATA = False

### Process Control Flags : User Defined (dev-note: run as a separate instance of code?)
# with_control = 1;  # This activates the closed-loop deep Koopman learning algorithm; requires input and state data, historical model parameter.  Now it is specified along with the dataset file path below.
plot_deep_basis = 0;  # This activates plotting of deep basis functions as a function of training iterations.
single_series = 0;  # Useful if you're analyzing a system with only a single time-series (requires extremely high temporal resolution).
debug_splash = 0;
phase_space_stitching = 0;

### Support Vars:

colors = [[0.68627453, 0.12156863, 0.16470589],
          [0.96862745, 0.84705883, 0.40000001],
          [0.83137256, 0.53333336, 0.6156863],
          [0.03529412, 0.01960784, 0.14509805],
          [0.90980393, 0.59607846, 0.78039217],
          [0.69803923, 0.87843138, 0.72941178],
          [0.20784314, 0.81568629, 0.89411765]];

colors = np.asarray(colors);  # defines a color palette

# class Koopman_Model(activation_flag=2,eval_size=100,batch_size=100,step_size_val=0.1,lambda=0.0000,max_iters=50000,valid_error_stop_threshold=0.00001,test_error_threshold=0.00001):


###  Deep Learning Optimization Parameters ###

lambd = 0.00000;
step_size_val = 0.5  # .025;
regularization_lambda_val = 0

batch_size = 400  # 30#900;
eval_size = batch_size;
add_bias = True

use_crelu = 0;
activation_flag = 2;  # sets the activation function type to RELU[0], ELU[1], SELU[2] (initialized a certain way,dropout has to be done differently) , or tanh()
max_epochs = 2000
train_error_threshold = 1e-6
valid_error_threshold = 1e-6;
test_error_threshold = 1e-6;
DISPLAY_SAMPLE_RATE_EPOCH = 1000
TRAIN_PERCENT = 50
VALID_PERCENT = 20
# WEIGHT_OUTPUT_OBJECTIVE = 0.5
# Deep Learning Metaparameters ###
keep_prob = 1.0;  # keep_prob = 1-dropout probability
res_net = 0;  # Boolean condition on whether to use a resnet connection.

# Explicitly mentioning the training routine
ls_dict_training_params = []
dict_training_params = {'step_size_val': 00.5, 'regularization_lambda_val': 0.00, 'train_error_threshold': float(1e-6),
                        'valid_error_threshold': float(1e-6), 'max_epochs': 5000, 'batch_size': 45}
ls_dict_training_params.append(dict_training_params)
dict_training_params = {'step_size_val': 00.3, 'regularization_lambda_val': 0.00, 'train_error_threshold': float(1e-6),
                        'valid_error_threshold': float(1e-6), 'max_epochs': 5000, 'batch_size': 45}
ls_dict_training_params.append(dict_training_params)
# dict_training_params = {'step_size_val': 0.1, 'regularization_lambda_val': 0, 'train_error_threshold': float(1e-7), 'valid_error_threshold': float(1e-7), 'max_epochs': 3000, 'batch_size': 45 }
# ls_dict_training_params.append(dict_training_params)
# dict_training_params = {'step_size_val': 0.08, 'regularization_lambda_val': 0, 'train_error_threshold': float(1e-8), 'valid_error_threshold': float(1e-8), 'max_epochs': 5000, 'batch_size': 45 }
# ls_dict_training_params.append(dict_training_params)
# dict_training_params = {'step_size_val': 0.05, 'regularization_lambda_val': 0, 'train_error_threshold': float(1e-8), 'valid_error_threshold': float(1e-8), 'max_epochs': 5000, 'batch_size': 45 }
# ls_dict_training_params.append(dict_training_params)
# dict_training_params = {'step_size_val': 0.01, 'regularization_lambda_val': 0, 'train_error_threshold': float(1e-8), 'valid_error_threshold': float(1e-8), 'max_epochs': 5000, 'batch_size': 45 }
# ls_dict_training_params.append(dict_training_params)

###  ------------------------------ Define Neural Network Hyperparameters ------------------------------

# ---- STATE PARAMETERS -------
x_deep_dict_size = 5
x_max_nn_layers = 3  # x_max_layers 3 works well
x_max_nn_nodes_limit = 10  # max width_limit -4 works well
x_min_nn_nodes_limit = x_max_nn_nodes_limit  # use regularization and dropout to trim edges for now.

# ---- INPUT PARAMETERS -------
u_deep_dict_size = 1
u_min_nn_nodes_limit = 5
u_max_nn_layers = 3
# u_max_nn_nodes_limit = 3

# ---- STATE n INPUT COMBINED PARAMETERS -------
xu_deep_dict_size = 1
xu_max_nn_layers = 3
xu_max_nn_nodes_limit = 5
xu_min_nn_nodes_limit = 5

best_test_error = np.inf
best_depth = x_max_nn_layers
best_width = x_min_nn_nodes_limit
# # #  ------------------------------ End Neural Network Sweep Parameters ------------------------------ # # #

# sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True));
sess = tf.InteractiveSession()


def reg_term(Wlist):
    output = tf.reduce_sum(tf.abs(Wlist[0]));
    for k in range(1, len(Wlist)):
        output += tf.reduce_sum(tf.abs(Wlist[k]));
    return output;


# def quick_nstep_predict(Y_p_old,u_control_all_training,with_control,num_bas_obs,iter):
#   n_points_pred = len(Y_p_old) - test_indices[0]-1;
#   init_index = test_indices[0];
#   Yf_final_test_stack_nn = np.asarray(Y_p_old).T[:,init_index:(init_index+1)+n_points_pred]
#   Ycurr = np.asarray(Y_p_old).T[:,init_index]
#   Ycurr = np.transpose(Ycurr);
#
#   if with_control:
#     Uf_final_test_stack_nn = np.asarray(u_control_all_training).T[:,init_index:(init_index+1)+n_points_pred]
#
#   #Reshape for tensorflow, which operates using row multiplication.
#   Ycurr = Ycurr.reshape(1,num_bas_obs);
#   psiyp_Ycurr = psiyp.eval(feed_dict={yp_feed:Ycurr});
#   psiyf_Ycurr = psiyf.eval(feed_dict={yf_feed:Ycurr});
#
#   # Define a growing list of vector valued observables that is the forward prediction of the Yf snapshot matrix, initiated from an initial condition in Yp_final_test.
#   Yf_final_test_ep_nn = [];
#   Yf_final_test_ep_nn.append(psiyp_Ycurr.tolist()[0][0:num_bas_obs]); # append the initial seed state value.
#
#   for i in range(0,n_points_pred):
#     print(i)
#     if with_control:
#       if len(U_test[i,:])==1:
#         U_temp_mat = np.reshape(Uf_final_test_stack_nn[i,:],(1,1));
#         psiyp_Ycurr = sess.run(forward_prediction_control, feed_dict={yp_feed:psiyp_Ycurr[:,0:num_bas_obs],u_control:U_temp_mat});#
#       else:
#         U_temp_mat = np.reshape(Uf_final_test_stack_nn[i,:],(1,n_inputs_control));
#         psiyp_Ycurr = sess.run(forward_prediction_control, feed_dict={yp_feed:psiyp_Ycurr[:,0:num_bas_obs],u_control:U_temp_mat});#
#     else:
#       psiyp_Ycurr = sess.run(forward_prediction,feed_dict={yp_feed:psiyp_Ycurr[:,0:num_bas_obs]});
#
#     Yout = psiyp_Ycurr.tolist()[0][0:num_bas_obs];
#     Yf_final_test_ep_nn.append(Yout);
#
#
#   Yf_final_test_ep_nn = np.asarray(Yf_final_test_ep_nn);
#   Yf_final_test_ep_nn = np.transpose(Yf_final_test_ep_nn);
#
#   prediction_error = np.linalg.norm(Yf_final_test_stack_nn-Yf_final_test_ep_nn,ord='fro')/np.linalg.norm(Yf_final_test_stack_nn,ord='fro');
#   print('%s%f' % ('[INFO] Current n-step prediction error (not used for gradient descent/backprop): ',prediction_error));
#
#   plt.figure();
#   # Make a Prediction Plot
#   x_range = np.arange(0,Yf_final_test_stack_nn.shape[1],1);
#   for i in range(0,3):
#       plt.plot(x_range,Yf_final_test_ep_nn[i,:],'--',color=colors[i,:]);
#       plt.plot(x_range,Yf_final_test_stack_nn[i,:],'*',color=colors[i,:]);
#   axes = plt.gca();
#   plt.legend(loc='best');
#   plt.xlabel('t');
#   fig = plt.gcf();
#
#   target_file = 'PredictionsDuringTraining/'+data_suffix.replace('.pickle','')+'nstep_prediction' + repr(iter) + '.pdf';
#   plt.savefig(target_file);
#   plt.close();


# def compute_covarmat(U,Y):
#   U = np.asarray(U);
#   Y = np.asarray(Y);
#   n_inputs = len(U[0]);
#   n_outputs = len(Y[0]);
#   n_data = len(U);
#
#   Output_Mat = np.zeros((n_outputs,n_inputs),dtype=np.float32);
#
#   for j in range(0,n_inputs):
#     for i in range(0,n_outputs):
#       Output_Mat[i,j] = compute_covar(Y[:,i],U[:,j]);
#   return Output_Mat;

def compute_covarmat_v2(U, Y):
    # Written by Shara
    # The compute_covarmat and compute_covar create a lot of downtime when dealing with data of large number of states
    # Both these functions are compiled into one function to improve the time of the code
    # I have verified that both methods yield the same answer
    U = np.asarray(U)
    Y = np.asarray(Y)
    if U.shape[0] != Y.shape[0]:
        print('')
        return np.inf
    else:
        return np.matmul((U - np.mean(U, axis=0)).T, Y - np.mean(Y, axis=0)) / U.shape[0]


# def gen_random_unitary(sq_matrix_dim,sc_factor=1.0):
#     n = sq_matrix_dim;
#     rand_eig = sc_factor*np.random.rand(n,1);
#     Lambda = np.zeros((n,n));
#
#     for i in range(0,n):
#         Lambda[i][i] = rand_eig[i];
#
#     T= np.random.rand(n,n);
#     inv_T = np.linalg.inv(T);
#     R = np.dot(np.dot(T,Lambda),inv_T);
#     return R;

# # THIS FUNCTION NOT USED FOR OUR PURPOSE
# def jensen_term(basis_hooks,n_samples,u):#z_list,num_bas_obs,x_deep_dict_size,iter_num):
#   #basis_hooks = z_list[-2];
#   #n_samples = 1e3;
#   random_injection = 20.0*np.random.rand(np.int(n_samples),num_bas_obs)-10.0;
#   random_injection_mean = np.mean(random_injection,axis=0);
#   Epsix = np.mean(basis_hooks.eval(feed_dict={u:random_injection}),axis=0)
#   Epsix = Epsix.reshape( (1,len(Epsix)) );
#   psiEx = basis_hooks.eval(feed_dict={u:[random_injection_mean]});
#   output = np.maximum(psiEx-Epsix,0);
#   #print "psiEx.shape:" + repr(psiEx.shape);
#   #print "Epsix.shape:" + repr(Epsix.shape);
#   #print output;
#   return np.sum(output);


# Function not used even if active
def expose_deep_basis(z_list, num_bas_obs, x_deep_dict_size, iter_num, u):
    basis_hooks = z_list[-1];  # [-1] is y  = K *\phi; -2 is \phi(yk)
    x_range = np.arange(-10.0, 10.0, 0.1);

    for i in range(0, num_bas_obs):
        plt.close();
        scan_injection = np.zeros((len(x_range), num_bas_obs));
        scan_injection[:, i] = np.transpose(x_range);
        phi_j = basis_hooks.eval(feed_dict={u: scan_injection});
        fig_hand = plt.gcf()
        plt.plot(x_range, phi_j, '.-', label='\phi_i(y)');
        # plt.ylim([-2.0,2.0]);
        fig = plt.gcf();
        plt.savefig('deep_basis_images/phi_with_u' + repr(i) + '_iternum_' + repr(iter_num) + '.jpg');
    return fig_hand;


# def compute_covar(x1,x2):
#   if (len(x1)!=len(x2)):
#     print("Error: compute_covar(x1,x2) requires x1 and x2 to be the same length");
#     return np.inf;
#   else:
#     sum_x1x2 = 0.0;
#     sum_x1 = 0.0;
#     sum_x2 = 0.0;
#     for j in range(0,len(x1)):
#       sum_x1x2 += x1[j]*x2[j];
#       sum_x1 +=x1[j];
#       sum_x2 +=x2[j];
#     covar = sum_x1x2/(1.0*len(x1))- sum_x1/(1.0*len(x1))*sum_x2/(1.0*len(x1));
#     return covar;

def estimate_K_stability(Kx, print_Kx=False):
    Kx_num = sess.run(Kx)
    np.linalg.eigvals(Kx_num)
    Kx_num_eigval_mod = np.abs(np.linalg.eigvals(Kx_num))
    if print_Kx:
        print(Kx_num)
    print('Eigen values: ')
    print(Kx_num_eigval_mod)
    if np.max(Kx_num_eigval_mod) > 1:
        print('[COMP] The identified Koopman operator is UNSTABLE with ', np.sum(np.abs(Kx_num_eigval_mod) > 1),
              'eigenvalues greater than 1')
    else:
        print('[COMP] The identified Koopman operator is STABLE')
    return


def load_pickle_data(file_path, has_control, has_output):
    '''load pickle data file for deep Koopman dynamic mode decomposition.
    Args:
       file_path:

    '''
    try:
        file_obj = open(file_path, 'rb')
        output_vec = pickle.load(file_obj)
    except:
        print('ERROR! Can\'t locate file in \"' + file_path + '\"')
        exit()
    Xp = None;
    Xf = None;
    Yp = None;
    Yf = None;
    Up = None;
    if type(output_vec) == list:
        Xp = output_vec[0];  # list of baseline observables, len(Yp) = (n_samps-1)
        Xf = output_vec[1];  # list of baseline observables, len(Yf) = (n_samps-1)
        if has_control:
            Up = output_vec[2];
        if has_output:
            Yp = output_vec[3];
            Yf = output_vec[4];
    elif type(output_vec) == dict:
        Xp = output_vec['Xp'];
        Xf = output_vec['Xf'];
        if has_control:
            Up = output_vec['Up'];
        if has_output:
            try:
                Yp = output_vec['Yp'];
            except:
                Yp = None
            Yf = output_vec['Yf'];
    if len(Xp) < 2:
        print("Warning: the time-series data provided has no more than 2 points.")
    # print("DEBUG:") + repr(len(output_vec));
    # TODO - Add code to check if the input is a 1-D array and if yes, change it to an nd-array of required form
    return np.asarray(Xp), np.asarray(Xf), Up, Yp, Yf


# def xavier_init(n_inputs, n_outputs, uniform=True):
#   """Set the parameter initialization using the method described.
#   This method is designed to keep the scale of the gradients roughly the same
#   in all layers.
#   Xavier Glorot and Yoshua Bengio (2010):
#            Understanding the difficulty of training deep feedforward neural
#            networks. International conference on artificial intelligence and
#            statistics.
#   Args:
#     n_inputs: The number of input nodes into each output.
#     n_outputs: The number of output nodes for each input.
#     uniform: If true use a uniform distribution, otherwise use a normal.
#   Returns:
#     An initializer.
#   """
#   if uniform:
#     # 6 was used in the paper.
#     init_range = math.sqrt(6.0 / (n_inputs + n_outputs))
#     return tf.random_uniform_initializer(-init_range, init_range)
#   else:
#     # 3 gives us approximately the same limits as above since this repicks
#     # values greater than 2 standard deviations from the mean.
#     stddev = math.sqrt(3.0 / (n_inputs + n_outputs))
#     return tf.truncated_normal_initializer(stddev=stddev)

def weight_variable(shape):
    std_dev = math.sqrt(3.0 / (shape[0] + shape[1]))
    return tf.Variable(tf.truncated_normal(shape, mean=0.0, stddev=std_dev, dtype=tf.float32));


def bias_variable(shape):
    std_dev = math.sqrt(3.0 / shape[0])
    return tf.Variable(tf.truncated_normal(shape, mean=0.0, stddev=std_dev, dtype=tf.float32));


# def gen_next_yk(input_var,W_list,b_list,keep_prob=1.0,activation_flag=1,res_net=0):
#     n_depth = len(W_list);
#     z_temp_list = [];
#     for k in range(0,n_depth):
#
#         if (k==0):
#             W1 = W_list[0];
#             b1 = b_list[0];
#             if activation_flag==1:# RELU
#                 z1 = tf.nn.dropout(tf.nn.relu(tf.matmul(input_var,W1)+b1),keep_prob);
#             if activation_flag==2: #ELU
#                 z1 = tf.nn.dropout(tf.nn.elu(tf.matmul(input_var,W1)+b1),keep_prob);
#             if activation_flag==3: # tanh
#                 z1 = tf.nn.dropout(tf.nn.tanh(tf.matmul(input_var,W1)+b1),keep_prob);
#
#             z_temp_list.append(z1);
#
#         if not (k==0):
#             prev_layer_output = tf.matmul(z_temp_list[k-1],W_list[k])+b_list[k]
#
#             if res_net and k==(n_depth-2):
#                 prev_layer_output += tf.matmul(u,W1)+b1 #  this expression is not compatible for variable width nets (where each layer has a different width at inialization - okay with regularization and dropout afterwards though)
#
#             if activation_flag==1:
#                 z_temp_list.append(tf.nn.dropout(tf.nn.relu(prev_layer_output),keep_prob));
#             if activation_flag==2:
#                 z_temp_list.append(tf.nn.dropout(tf.nn.elu(prev_layer_output),keep_prob));
#             if activation_flag==3:
#                 z_temp_list.append(tf.nn.dropout(tf.nn.tanh(prev_layer_output),keep_prob));
#
#     if debug_splash:
#       print("[DEBUG] z_list" + repr(z_list[-1]));
#
#     #y_out = tf.concat([z_list[-1],u],axis=1); # last element of activation output list is the actual NN output
#     y_out = z_temp_list[-1];
#
#     result = sess.run(tf.global_variables_initializer())
#     return y_out;

def initialize_Wblist(n_u, hv_list):
    W_list = [];
    b_list = [];
    n_depth = len(hv_list);
    # hv_list[n_depth-1] = n_y;
    for k in range(0, n_depth):
        if k == 0:
            W1 = weight_variable([n_u, hv_list[k]]);
            b1 = bias_variable([hv_list[k]]);
            W_list.append(W1);
            b_list.append(b1);
        else:
            W_list.append(weight_variable([hv_list[k - 1], hv_list[k]]));
            b_list.append(bias_variable([hv_list[k]]));

    return W_list, b_list;


def initialize_tensorflow_graph(n_u, x_deep_dict_size, hv_list, W_list, b_list, keep_prob=1.0, activation_flag=1,
                                res_net=0, input_variable_inclusive=True, add_bias=True):
    u = tf.placeholder(tf.float32, shape=[None,
                                          n_u]);  # state/input node,# inputs = dim(input) , None indicates batch size can be any size
    z_list = [];
    n_depth = len(hv_list);
    # print("[DEBUG] n_depth" + repr(n_depth);
    # hv_list[n_depth-2] = x_deep_dict_size;
    for k in range(0, n_depth):
        if (k == 0):
            W1 = W_list[k];
            b1 = b_list[k];
            if activation_flag == 1:  # RELU
                z_list.append(tf.nn.dropout(tf.nn.relu(tf.matmul(u, W1) + b1), keep_prob));
            if activation_flag == 2:  # ELU
                z_list.append(tf.nn.dropout(tf.nn.elu(tf.matmul(u, W1) + b1), keep_prob));
            if activation_flag == 3:  # tanh
                z_list.append(tf.nn.dropout(tf.nn.tanh(tf.matmul(u, W1) + b1), keep_prob));
        else:
            # W_list.append(weight_variable([hv_list[k-1],hv_list[k]]));
            # b_list.append(bias_variable([hv_list[k]])); TODO - If you do not see any error because of commenting this, please delete these two lines
            prev_layer_output = tf.matmul(z_list[k - 1], W_list[k]) + b_list[k]
            if debug_splash:
                print("[DEBUG] prev_layer_output.get_shape() ") + repr(prev_layer_output.get_shape());
            if res_net and k == (n_depth - 2):
                prev_layer_output += tf.matmul(u,
                                               W1) + b1  # this expression is not compatible for variable width nets (where each layer has a different width at inialization - okay with regularization and dropout afterwards though)
            if activation_flag == 1:
                z_list.append(tf.nn.dropout(tf.nn.relu(prev_layer_output), keep_prob));
            if activation_flag == 2:
                z_list.append(tf.nn.dropout(tf.nn.elu(prev_layer_output), keep_prob));
            if activation_flag == 3:
                z_list.append(tf.nn.dropout(tf.nn.tanh(prev_layer_output), keep_prob));
    if input_variable_inclusive:
        y = tf.concat([u, z_list[-1]],
                      axis=1);  # [TODO] in the most general function signature, allow for default option with state/input inclusion
    else:
        y = z_list[-1]
    if add_bias:
        y = tf.concat([y, tf.ones(shape=(tf.shape(y)[0], 1))], axis=1)
    result = sess.run(tf.global_variables_initializer());
    #  print("[DEBUG] y.get_shape(): " + repr(y.get_shape()) + " y_.get_shape(): " + repr(y_.get_shape());
    return z_list, y, u;  # ,u_control;


# Output Koopman Input Objective Function that Implements S. Balakrishnan's Algorithm for Output-Constrained Deep-DMD
# def Deep_Output_KIC_Objective(all_psiXp,all_psiXf,Kx,all_psiUp,Ku,all_psiXUp,Kxu,all_Yf,all_Yp,Wh,step_size,with_control=0,mix_state_and_control =0,with_output=0,learn_controllable_Koopman=0):
#     all_psiXf_predicted = tf.matmul(all_psiXp, Kx)
#     if with_control:
#         all_psiXf_predicted = all_psiXf_predicted + tf.matmul(all_psiUp,Ku)
#         if mix_state_and_control:
#             all_psiXf_predicted = all_psiXf_predicted + tf.matmul(all_psiXUp, Kxu)
#     if learn_controllable_Koopman:
#         print('Learning Controllable Koopman is disabled!!!')
#         # n = np.int(Kx.get_shape()[0]);
#         # Kut = tf.transpose(Ku);
#         # Kxt = tf.transpose(Kx);
#         # ctrb_matrix = Kut;
#         # for ind in range(1,n):
#         #     ctrb_matrix = tf.concat([ctrb_matrix,tf.matmul(tf.pow(Kxt,ind),Kut)],axis=1);
#         #     ctrbTctrb = tf.matmul(ctrb_matrix,tf.transpose(ctrb_matrix) );
#         #     print(ctrbTctrb.get_shape())
#         #     ctrb_s,ctrb_v = tf.self_adjoint_eig(ctrbTctrb);
#         #     print(tf.norm(ctrb_s,1))
#
#     # Dealing with Output
#     if with_output:
#         all_Yf_predicted = tf.matmul(all_psiXf_predicted,Wh)
#         # all_Yf_predicted = tf.matmul(all_psiXf, Wh)
#         all_Yp_predicted = tf.matmul(all_psiXp,Wh)
#     else:
#         all_Yf_predicted = []
#         all_Yp_predicted = []
#
#     # SSE
#     if with_output:
#         N_eqns = tf.add(tf.shape(all_psiXp)[0], tf.shape(all_Yf)[0])
#     else:
#         N_eqns = tf.shape(all_psiXp)[0]
#     N_datapts = tf.shape(all_psiXp)[1]
#     state_prediction_error = all_psiXf - all_psiXf_predicted
#     # state_prediction_error = tf.math.square(tf.norm(state_prediction_error,axis=[0,1],ord='fro'))
#     state_frobenius_norm_squared = tf.math.square(tf.norm(all_psiXf,axis=[0,1],ord='fro'))
#     if with_output:
#         output_prediction_error = all_Yf - all_Yf_predicted
#         # output_prediction_error = tf.math.square(tf.norm(output_prediction_error, axis=[0, 1], ord='fro'))
#         output_frobenius_norm_squared = tf.math.square(tf.norm(all_Yf, axis=[0, 1], ord='fro'))
#         # output_prediction_error = output_prediction_error + tf.math.square(tf.norm(all_Yf - all_Yf_predicted, axis=[0, 1], ord='fro'))    # Maybe can be used later to predict Yp as well
#         # output_frobenius_norm_squared = output_frobenius_norm_squared + tf.math.square(tf.norm(all_Yf, axis=[0, 1], ord='fro'))           # Maybe can be used later to predict Yp as well
#         # tf_koopman_loss = WEIGHT_OUTPUT_OBJECTIVE*(output_prediction_error / output_frobenius_norm_squared) + (1-WEIGHT_OUTPUT_OBJECTIVE )*( state_prediction_error / state_frobenius_norm_squared)
#         # tf_koopman_loss = (output_prediction_error+state_prediction_error)/(output_frobenius_norm_squared+state_frobenius_norm_squared)
#         # tf_koopman_loss = tf.math.divide(output_prediction_error + state_prediction_error, tf.cast(tf.math.multiply(N_eqns,N_datapts),dtype = tf.float32)) # Mean Squared Error
#         tf_koopman_loss = tf.math.reduce_max(tf.math.reduce_mean(tf.math.abs(tf.concat([output_prediction_error,state_prediction_error],1)),0)) + tf.math.reduce_max(tf.math.abs(Kx)) + tf.math.reduce_max(tf.math.abs(Ku)) + tf.math.reduce_max(tf.math.abs(Kxu)) + tf.math.reduce_max(tf.math.abs(Wh))
#     else:
#         # tf_koopman_loss = state_prediction_error / state_frobenius_norm_squared
#         tf_koopman_loss = tf.math.reduce_max(tf.math.reduce_mean(tf.math.abs(state_prediction_error), 0)) + tf.math.reduce_max(tf.math.abs(Kx)) + tf.math.reduce_max(tf.math.abs(Ku)) + tf.math.reduce_max(tf.math.abs(Kxu))
#     optimizer = tf.train.AdagradOptimizer(step_size).minimize(tf_koopman_loss)
#     result = sess.run(tf.global_variables_initializer())
#     return tf_koopman_loss,optimizer,all_psiXf_predicted,all_Yf_predicted,all_Yp_predicted

def Deep_Output_KIC_Objective_v2(all_psiXp, all_psiXf, Kx, all_psiUp, Ku, all_psiXUp, Kxu, all_Yf, all_Yp, Wh,
                                 step_size, regularization_lambda, with_control=0, mix_state_and_control=0,
                                 with_output=0, learn_controllable_Koopman=0):
    all_psiXf_predicted = tf.matmul(all_psiXp, Kx)
    regularization_penalty = tf.norm(Kx, axis=[-2, -1], ord=2)

    if with_control:
        all_psiXf_predicted = all_psiXf_predicted + tf.matmul(all_psiUp, Ku)
        regularization_penalty = regularization_penalty + tf.norm(Ku, axis=[-2, -1], ord=2)
        if mix_state_and_control:
            all_psiXf_predicted = all_psiXf_predicted + tf.matmul(all_psiXUp, Kxu)
            regularization_penalty = regularization_penalty + tf.norm(Kxu, axis=[-2, -1], ord=2)
    if learn_controllable_Koopman:
        print('Learning Controllable Koopman is disabled!!!')
        # n = np.int(Kx.get_shape()[0]);
        # Kut = tf.transpose(Ku);
        # Kxt = tf.transpose(Kx);
        # ctrb_matrix = Kut;
        # for ind in range(1,n):
        #     ctrb_matrix = tf.concat([ctrb_matrix,tf.matmul(tf.pow(Kxt,ind),Kut)],axis=1);
        #     ctrbTctrb = tf.matmul(ctrb_matrix,tf.transpose(ctrb_matrix) );
        #     print(ctrbTctrb.get_shape())
        #     ctrb_s,ctrb_v = tf.self_adjoint_eig(ctrbTctrb);
        #     print(tf.norm(ctrb_s,1))
    prediction_error_all = all_psiXf - all_psiXf_predicted
    SST = tf.math.reduce_sum(tf.math.square(all_psiXf - tf.math.reduce_mean(all_psiXf, axis=0)), axis=0)
    SSE = tf.math.reduce_sum(tf.math.square(all_psiXf - all_psiXf_predicted), axis=0)
    # Dealing with Output
    if with_output:
        all_Yf_predicted = tf.matmul(all_psiXf_predicted, Wh)
        # all_Yf_predicted = tf.matmul(all_psiXf, Wh)
        # all_Yp_predicted = tf.matmul(all_psiXp,Wh)
        all_Yp_predicted = None
        output_prediction_error = all_Yf - all_Yf_predicted
        prediction_error_all = tf.concat([prediction_error_all, output_prediction_error], 1)
        # regularization_penalty = regularization_penalty + tf.norm(Wh, axis=[-2,-1], ord=2)
        SST_y = tf.math.reduce_sum(tf.math.square(all_Yf - tf.math.reduce_mean(all_Yf, axis=0)), axis=0)
        SSE_y = tf.math.reduce_sum(tf.math.square(all_Yf - all_Yf_predicted), axis=0)
        SST = tf.concat([SST, SST_y], axis=0) + 1e-2
        SSE = tf.concat([SSE, SSE_y], axis=0)
    else:
        all_Yf_predicted = None
        all_Yp_predicted = None
    tf_koopman_loss = tf.math.reduce_mean(tf.math.square(prediction_error_all)) + tf.math.multiply(
        regularization_lambda, regularization_penalty)
    # tf_koopman_loss = tf.math.reduce_max(tf.math.reduce_mean(tf.math.square(prediction_error_all),0)) + tf.math.multiply(regularization_lambda,regularization_penalty)
    tf_koopman_accuracy = (1 - tf.math.reduce_max(tf.divide(SSE, SST))) * 100
    optimizer = tf.train.AdagradOptimizer(step_size).minimize(tf_koopman_loss)
    result = sess.run(tf.global_variables_initializer())
    return tf_koopman_loss, tf_koopman_accuracy, optimizer, all_psiXf_predicted, all_Yf_predicted, all_Yp_predicted

def Deep_Output_KIC_Objective_v3(dict_feed,dict_psi,dict_K, with_control=0, mix_state_and_control=0,with_output=0):
    dict_predictions ={}
    psiXf_predicted = tf.matmul(dict_psi['xpT'], dict_K['KxT'])
    regularization_penalty = tf.norm(dict_K['KxT'], axis=[-2, -1], ord=2)

    if with_control:
        psiXf_predicted = psiXf_predicted + tf.matmul(dict_psi['upT'], dict_K['KuT'])
        # regularization_penalty = regularization_penalty + tf.norm(dict_K['KuT'], axis=[-2, -1], ord=2)
        if mix_state_and_control:
            psiXf_predicted = psiXf_predicted + tf.matmul(dict_psi['xupT'], dict_K['KxuT'])
            # regularization_penalty = regularization_penalty + tf.norm(dict_K['KxuT'], axis=[-2, -1], ord=2)


    all_prediction_error = dict_psi['xfT'] - psiXf_predicted
    SST_x = tf.math.reduce_sum(tf.math.square(dict_psi['xfT']), axis=0)
    SSE_x = tf.math.reduce_sum(tf.math.square(all_prediction_error), axis=0)
    dict_predictions['xfT'] = psiXf_predicted
    dict_predictions['xfT_error'] = dict_psi['xfT'] - psiXf_predicted
    dict_predictions['xfT_accuracy'] = (1 - tf.math.reduce_max(tf.divide(SSE_x, SST_x))) * 100
    # Dealing with Output
    if with_output:
        Yf_predicted = tf.matmul(psiXf_predicted, dict_K['WhT'])
        # Yf_predicted = tf.matmul(dict_psi['xfT'], dict_K['WhT'])
        Yf_prediction_error = dict_feed['yfT'] - Yf_predicted
        all_prediction_error = tf.concat([all_prediction_error, Yf_prediction_error], 1)
        # regularization_penalty = regularization_penalty + tf.norm(Wh, axis=[-2,-1], ord=2)
        SST_y = tf.math.reduce_sum(tf.math.square(dict_feed['yfT']), axis=0)
        SSE_y = tf.math.reduce_sum(tf.math.square(Yf_prediction_error), axis=0)
        SST = tf.concat([SST_x, SST_y], axis=0)
        SSE = tf.concat([SSE_x, SSE_y], axis=0)
        dict_predictions['yfT'] = Yf_predicted
        dict_predictions['yfT_error'] = Yf_prediction_error
        dict_predictions['yfT_accuracy'] = (1 - tf.math.reduce_max(tf.divide(SSE_y, SST_y))) * 100
        dict_predictions['model_accuracy'] = (1 - tf.math.reduce_max(tf.divide(SSE, SST))) * 100
    else:
        SST = SST_x
        SSE = SSE_x
        dict_predictions['yfT'] = 0
        dict_predictions['yfT_error'] = 0
        dict_predictions['yft_accuracy'] = 0
        dict_predictions['model_accuracy'] = dict_predictions['xfT_accuracy']

    tf_koopman_loss = tf.math.reduce_mean(tf.math.square(all_prediction_error)) + tf.math.multiply(dict_feed['regularization_lambda'], regularization_penalty)
    # tf_koopman_loss = tf.math.reduce_max(tf.math.reduce_mean(tf.math.square(prediction_error_all),0)) + tf.math.multiply(regularization_lambda,regularization_penalty)
    tf_koopman_accuracy = (1 - tf.math.reduce_max(tf.divide(SSE, SST))) * 100
    optimizer = tf.train.AdagradOptimizer(dict_feed['step_size']).minimize(tf_koopman_loss)
    result = sess.run(tf.global_variables_initializer())
    return tf_koopman_loss, optimizer,dict_predictions

# def optimize_K_given_psi(psi_data,K_all,):



# def Deep_Control_Koopman_Objective(psiyp,psiyf,Kx,psiu,Ku,step_size,learn_controllable_Koopman=0):
#
#    forward_prediction_control = (tf.matmul(psiyp,Kx) + tf.matmul(psiu,Ku));
#
#    if learn_controllable_Koopman:
#      n = np.int(Kx.get_shape()[0]);
#      Kut = tf.transpose(Ku);
#      Kxt = tf.transpose(Kx);
#      ctrb_matrix = Kut;
#      for ind in range(1,n):
#         ctrb_matrix = tf.concat([ctrb_matrix,tf.matmul(tf.pow(Kxt,ind),Kut)],axis=1);
#      ctrbTctrb = tf.matmul(ctrb_matrix,tf.transpose(ctrb_matrix) );
#      print(ctrbTctrb.get_shape())
#      ctrb_s,ctrb_v = tf.self_adjoint_eig(ctrbTctrb);
#      print(tf.norm(ctrb_s,1))
#
#    tf_koopman_loss =  tf.reduce_mean(tf.norm(psiyf - forward_prediction_control,axis=[0,1],ord='fro'))#/tf.reduce_mean(tf.norm(psiyp,axis=[0,1],ord='fro'));
#    optimizer = tf.train.AdagradOptimizer(step_size).minimize(tf_koopman_loss);
#    result = sess.run(tf.global_variables_initializer());
#
#    return tf_koopman_loss,optimizer,forward_prediction_control;

# def Deep_Direct_Koopman_Objective(psiyp,psiyf,Kx,step_size,convex_basis=0,u=None):
#
#    forward_prediction = tf.matmul(psiyp,Kx)
#    tf_koopman_loss = tf.reduce_mean(tf.norm(forward_prediction-psiyf,axis=[0,1],ord='fro') )#/tf.reduce_mean(tf.norm(psiyf,axis=[0,1],ord='fro'));
#    if convex_basis == 1:
#      lagrange_multiplier_convex = 10.0;
#      tf_koopman_loss = tf_koopman_loss + lagrange_multiplier_convex*jensen_term(psiyp,1e6,u)
#
#    optimizer = tf.train.AdagradOptimizer(step_size).minimize(tf_koopman_loss);
#    result = sess.run(tf.global_variables_initializer());
#    return tf_koopman_loss,optimizer,forward_prediction;
#


def instantiate_comp_graph(params_list):
    n_base_dict_size = params_list['no of base observables'];
    x_deep_dict_size = params_list['no of neural network observables']
    hidden_vars_list = params_list['hidden_var_list']
    W_list = params_list['W_list']
    b_list = params_list['b_list']
    keep_prob = params_list['keep_prob']
    activation_flag = params_list['activation flag']
    res_net = params_list['res_net']
    state_include = params_list['include state']
    psiyzlist, psiy, yfeed = initialize_tensorflow_graph(n_base_dict_size, x_deep_dict_size, hidden_vars_list, W_list,
                                                         b_list, keep_prob, activation_flag, res_net, state_include)
    return psiyzlist, psiy, yfeed


def instantiate_output_variables(Output_Dim, Koopman_Dim):
    Wh_var = weight_variable([Koopman_Dim, Output_Dim]);
    yp_feed_var = tf.placeholder(tf.float32, shape=[None, Output_Dim])
    yf_feed_var = tf.placeholder(tf.float32, shape=[None, Output_Dim])
    return yf_feed_var, yp_feed_var, Wh_var;


## Functions for dynamic training with user interface
def get_variable_value(variable_name, prev_variable_value, reqd_data_type, lower_bound=0):
    # Purpose: This function is mainly to
    not_valid = True
    variable_output = prev_variable_value
    while (not_valid):
        print('Current value of ', variable_name, ' = ', prev_variable_value)
        variable_input = input('Enter new ' + variable_name + ' value [-1 or ENTER to retain previous entry]: ')
        # First check for -1
        if variable_input in ['-1', '']:
            not_valid = False
        else:
            # Second check for correct data type
            try:
                variable_input = reqd_data_type(variable_input)
                # Third check for the correct bound
                if not (variable_input > lower_bound):
                    print('Error! Value is out of bounds. Please enter a value greater than ', lower_bound)
                    not_valid = True
                else:
                    variable_output = variable_input
                    not_valid = False
            except:
                print('Error! Please enter a ', reqd_data_type, ' value, -1 or ENTER')
                not_valid = True
    return variable_output


def display_train_params(dict_run_params):
    print('======================================')
    print('CURRENT TRAINING PARAMETERS')
    print('======================================')
    print('Step Size Value            : ', dict_run_params['step_size_val'])
    print('Regularization Coefficient :', dict_run_params['regularization_lambda_val'])
    print('Train Error Threshold      : ', dict_run_params['train_error_threshold'])
    print('Validation Error Threshold : ', dict_run_params['valid_error_threshold'])
    print('Maximum number of Epochs   : ', dict_run_params['max_epochs'])
    print('Batch Size   : ', dict_run_params['batch_size'])
    print('--------------------------------------')
    return


def generate_hyperparam_entry(feed_dict_train, feed_dict_valid, error_func, r2_accuracy, n_epochs_run, dict_run_params):
    training_error = error_func.eval(feed_dict=feed_dict_train)
    validation_error = error_func.eval(feed_dict=feed_dict_valid)
    # test_error = error_func.eval(feed_dict=feed_dict_test)
    training_accuracy = r2_accuracy.eval(feed_dict=feed_dict_train)
    validation_accuracy = r2_accuracy.eval(feed_dict=feed_dict_valid)
    # test_accuracy = r2_accuracy.eval(feed_dict=feed_dict_test)
    dict_hp = {}
    dict_hp['x_hidden_variable_list'] = x_hidden_vars_list
    dict_hp['u_hidden_variable_list'] = u_hidden_vars_list
    dict_hp['xu_hidden_variable_list'] = xu_hidden_vars_list
    dict_hp['activation flag'] = activation_flag
    dict_hp['activation function'] = None
    if activation_flag == 1:
        dict_hp['activation function'] = 'relu'
    elif activation_flag == 2:
        dict_hp['activation function'] = 'elu'
    elif activation_flag == 3:
        dict_hp['activation function'] = 'tanh'
    dict_hp['no of epochs'] = n_epochs_run
    dict_hp['batch size'] = dict_run_params['batch_size']
    dict_hp['step size'] = dict_run_params['step_size_val']
    dict_hp['regularization coefficient'] = dict_run_params['regularization_lambda_val']
    dict_hp['training error'] = training_error
    dict_hp['validation error'] = validation_error
    # dict_hp['test error'] = test_error
    dict_hp['r^2 training accuracy'] = training_accuracy
    dict_hp['r^2 validation accuracy'] = validation_accuracy
    # dict_hp['r^2 test accuracy'] = test_accuracy
    return dict_hp


def static_train_net(dict_train, dict_valid, dict_feed, dict_psi, dict_K, ls_dict_training_params, deep_koopman_loss, optimizer, dict_predictions,  all_histories = {'train error': [],'validation error': []}, dict_run_info = {}):
    feed_dict_train = get_fed_dict(dict_feed, dict_train,ls_dict_training_params[0]['with_u'],ls_dict_training_params[0]['with_xu'],ls_dict_training_params[0]['with_y'])
    feed_dict_valid = get_fed_dict(dict_feed, dict_valid,ls_dict_training_params[0]['with_u'],ls_dict_training_params[0]['with_xu'],ls_dict_training_params[0]['with_y'])
    # --------
    try :
        run_info_index = list(dict_run_info.keys())[-1]
    except:
        run_info_index = 0
    for dict_train_params_i in ls_dict_training_params:
        display_train_params(dict_train_params_i)
        all_histories, good_start, n_epochs_run = train_net_v2(dict_train,feed_dict_train, feed_dict_valid, dict_feed, dict_psi, dict_K,
                                                               deep_koopman_loss, optimizer,dict_train_params_i,all_histories)
        feed_dict_train.update({dict_feed['regularization_lambda']: dict_train_params_i['regularization_lambda_val']})
        feed_dict_valid.update({dict_feed['regularization_lambda']: dict_train_params_i['regularization_lambda_val']})
        dict_run_info[run_info_index] = generate_hyperparam_entry(feed_dict_train, feed_dict_valid,deep_koopman_loss, dict_predictions['model_accuracy'],n_epochs_run, dict_train_params_i)
        print('Current Training Error  :', dict_run_info[run_info_index]['training error'])
        print('Current Validation Error      :', dict_run_info[run_info_index]['validation error'])
        # print('Current Training Accuracy  :', dict_run_info[run_info_index]['r^2 training accuracy'])
        # print('Current Validation Accuracy      :', dict_run_info[run_info_index]['r^2 validation accuracy'])
        estimate_K_stability(Kx)
        run_info_index += 1
    return all_histories, good_start, dict_run_info


def get_fed_dict(dict_feed,dict_data,with_u,with_xu,with_y):
    fed_dict= {dict_feed['xpT']: dict_data['Xp'],dict_feed['xfT']: dict_data['Xf']}
    if with_u:
        fed_dict[dict_feed['upT']] = dict_data['Up']
        if with_xu:
            fed_dict[dict_feed['xupT']] = np.concatenate([dict_data['Xp'], dict_data['Up']], axis=1)
    if with_y:
        fed_dict[dict_feed['ypT']] = dict_data['Yp']
        fed_dict[dict_feed['yfT']] = dict_data['Yf']
    return fed_dict

def dynamic_train_net(dict_train, dict_valid, dict_feed, dict_psi, dict_K, dict_run_params, deep_koopman_loss, optimizer, dict_predictions, all_histories = {'train error': [], 'validation error': []}, dict_run_info={}):
    # For evaluating how the hyperparameters performed with that training
    feed_dict_train = get_fed_dict(dict_feed,dict_train,dict_run_params['with_u'],dict_run_params['with_xu'],dict_run_params['with_y'])
    feed_dict_valid = get_fed_dict(dict_feed, dict_valid,dict_run_params['with_u'],dict_run_params['with_xu'],dict_run_params['with_y'])
    # --------
    KEEP_TRAINING = True
    NO_CHOICE_VALUES = ['n', 'N', 'no', 'No', 'NO']
    try:
        run_info_index = list(dict_run_info.keys())[-1]
    except:
        run_info_index = 0
    while (KEEP_TRAINING):
        train_user_choice = input('Do you want to train the neural net [y/n]? ')
        if train_user_choice not in NO_CHOICE_VALUES:
            display_train_params(dict_run_params)
            param_user_choice = input('Do you want to change the parameters [y/n]?')
            if param_user_choice not in NO_CHOICE_VALUES:
                NOT_SATISFIED = True
                while (NOT_SATISFIED):
                    dict_run_params['step_size_val'] = get_variable_value('Step Size', dict_run_params['step_size_val'], float)
                    dict_run_params['regularization_lambda_val'] = get_variable_value('Regularization Coefficient', dict_run_params['regularization_lambda_val'], float,-1e-100)
                    dict_run_params['train_error_threshold'] = get_variable_value('Training error threshold', dict_run_params['train_error_threshold'], float)
                    dict_run_params['valid_error_threshold']= get_variable_value('Validation error threshold', dict_run_params['valid_error_threshold'],float)
                    dict_run_params['max_epochs'] = get_variable_value('Maximum number of epochs', dict_run_params['max_epochs'], int)
                    dict_run_params['batch_size'] = get_variable_value('Batch size', dict_run_params['batch_size'], int)
                    display_train_params(dict_run_params)
                    parameter_satisfaction = input('Are these parameters fine [y/n]?')
                    if parameter_satisfaction not in NO_CHOICE_VALUES:
                        NOT_SATISFIED = False
                    else:
                        print('*** Enter the parameters again ***')
            all_histories, good_start, n_epochs_run = train_net_v2(dict_train, feed_dict_train, feed_dict_valid, dict_feed, dict_psi, dict_K, deep_koopman_loss, optimizer, dict_run_params, all_histories)

            dict_run_info[run_info_index] = generate_hyperparam_entry(feed_dict_train, feed_dict_valid,deep_koopman_loss, dict_predictions['model_accuracy'],n_epochs_run, dict_run_params)
            print('Training Error  :', dict_run_info[run_info_index]['training error'])
            print('Validation Error  :', dict_run_info[run_info_index]['validation error'])
            estimate_K_stability(Kx, True)
            run_info_index += 1
        else:
            print('Training Complete')
            KEEP_TRAINING = False
            good_start = 0
    return all_histories, good_start, dict_run_info

def train_net_v2(dict_train, feed_dict_train, feed_dict_valid, dict_feed, dict_psi, dict_K, loss_func,optimizer, dict_run_params, all_histories={'train error': [], 'validation error': []},iterative_optimization = False):
    # -----------------------------
    # Initialization
    # -----------------------------
    good_start = 1
    N_train_samples = len(dict_train['Xp'])
    runs_per_epoch = int(np.ceil(N_train_samples / batch_size))
    epoch_i = 0
    training_error = 100
    validation_error = 100
    # -----------------------------
    # Actual training
    # -----------------------------
    while ((epoch_i < dict_run_params['max_epochs']) and (training_error > dict_run_params['train_error_threshold']) and (validation_error > dict_run_params['valid_error_threshold'])):
        epoch_i += 1
        # Re initializing the training indices
        all_train_indices = list(range(N_train_samples))
        # Random sort of the training indices
        random.shuffle(all_train_indices)
        for run_i in range(runs_per_epoch):
            if run_i != runs_per_epoch - 1:
                train_indices = all_train_indices[run_i * batch_size:(run_i + 1) * batch_size]
            else:
                # Last run with the residual data
                train_indices = all_train_indices[run_i * batch_size: N_train_samples]
            dict_train_i = {'Xp': dict_train['Xp'][train_indices], 'Xf': dict_train['Xf'][train_indices]}
            if dict_run_params['with_u']:
                dict_train_i['Up'] = dict_train['Up'][train_indices]
            if dict_run_params['with_y']:
                dict_train_i['Yp'] = dict_train['Yp'][train_indices]
                dict_train_i['Yf'] = dict_train['Yf'][train_indices]
            feed_dict_train_curr = get_fed_dict(dict_feed,dict_train_i,dict_run_params['with_u'],dict_run_params['with_xu'],dict_run_params['with_y'])
            feed_dict_train_curr[dict_feed['step_size']] = dict_run_params['step_size_val']
            feed_dict_train_curr[dict_feed['regularization_lambda']] = dict_run_params['regularization_lambda_val']
            optimizer.run(feed_dict=feed_dict_train_curr)
        # After training 1 epoch
        feed_dict_train[dict_feed['regularization_lambda']] = dict_run_params['regularization_lambda_val']
        feed_dict_valid[dict_feed['regularization_lambda']] = dict_run_params['regularization_lambda_val']
        training_error = loss_func.eval(feed_dict=feed_dict_train)
        validation_error = loss_func.eval(feed_dict=feed_dict_valid)
        all_histories['train error'].append(training_error)
        all_histories['validation error'].append(validation_error)
        if np.mod(epoch_i, DISPLAY_SAMPLE_RATE_EPOCH) == 0:
            print('Epoch No: ', epoch_i, ' |   Training error: ', training_error)
            print('Validation error: '.rjust(len('Epoch No: ' + str(epoch_i) + ' |   Validation error: ')),validation_error)
            estimate_K_stability(Kx)
            print('---------------------------------------------------------------------------------------------------')
    return all_histories, good_start, epoch_i


# def train_net(u_all_training,y_all_training,mean_diff_nocovar,optimizer,u_control_all_training=None,Output_p_batch=None,Output_f_batch=None,valid_error_thres=1e-2,test_error_thres=1e-2,max_iters=100000,step_size_val=0.01):
#   iter = 0;
#   samplerate = 10;
#   good_start = 1;
#   valid_error = 100.0;
#   test_error = 100.0;
#   training_error_history_nocovar = [];
#   validation_error_history_nocovar = [];
#   test_error_history_nocovar = [];
#
#   training_error_history_withcovar = [];
#   validation_error_history_withcovar = [];
#   test_error_history_withcovar = [];
#
#   covar_actual = compute_covarmat_v2(u_all_training,y_all_training);
#   covar_model_history = [];
#   covar_diff_history = [];
#   print('Training Starts now')
#   while (((test_error>test_error_thres) or (valid_error > valid_error_thres)) and iter < max_iters):
#     iter+=1;
#     all_ind = set(np.arange(0,len(u_all_training)));
#     select_ind = np.random.randint(0,len(u_all_training),size=batch_size); # select indices for training
#     valid_ind = list(all_ind -set(select_ind))[0:batch_size];  # select indices for validation
#     select_ind_test = list(all_ind - set(valid_ind) - set(select_ind))[0:batch_size]; # select indices for test
#     u_batch =[];
#     u_control_batch = [];
#     y_batch = [];
#     u_valid = [];
#     u_control_valid = [];
#     y_valid = [];
#     u_test_train = [];
#     u_control_train = [];
#     y_test_train= [];
#     u_control_test_train = [];
#     output_batch_p = [];
#     output_batch_valid_p = [];
#     output_batch_test_train_p = [];
#     output_batch_f = [];
#     output_batch_valid_f = [];
#     output_batch_test_train_f = [];
#
#     for j in range(0,len(select_ind)):
#       u_batch.append(u_all_training[select_ind[j]]);
#       y_batch.append(y_all_training[select_ind[j]]);
#       if with_control:
#           u_control_batch.append(u_control_all_training[select_ind[j]]);
#       if with_output:
#           output_batch_p.append(Output_p_batch[select_ind[j]]);
#           output_batch_f.append(Output_f_batch[select_ind[j]]);
#
#     for k in range(0,len(valid_ind)):
#       u_valid.append(u_all_training[valid_ind[k]]);
#       y_valid.append(y_all_training[valid_ind[k]]);
#       if with_control:
#           u_control_valid.append(u_control_all_training[valid_ind[k]]);
#
#       if with_output:
#           output_batch_valid_p.append(Output_p_batch[valid_ind[k]]);
#           output_batch_valid_f.append(Output_f_batch[valid_ind[k]]);
#
#
#     for k in range(0,len(select_ind_test)):
#       u_test_train.append(u_all_training[select_ind_test[k]]);
#       y_test_train.append(y_all_training[select_ind_test[k]]);
#       if with_control:
#           u_control_test_train.append(u_control_all_training[select_ind_test[k]]);
#       if with_output:
#           output_batch_test_train_p.append(Output_p_batch[select_ind_test[k]]);
#           output_batch_test_train_f.append(Output_f_batch[select_ind_test[k]]);
#
#
#
#     if with_control and (not with_output):
#       optimizer.run(feed_dict={yp_feed:u_batch,yf_feed:y_batch,u_control:u_control_batch,step_size:step_size_val});
#       valid_error = mean_diff_nocovar.eval(feed_dict={yp_feed:u_valid,yf_feed:y_valid,u_control:u_control_valid});
#       test_error = mean_diff_nocovar.eval(feed_dict={yp_feed:u_test_train,yf_feed:y_test_train,u_control:u_control_test_train});
#
#     if with_control and with_output:
#       optimizer.run(feed_dict={yp_feed:u_batch,yf_feed:y_batch,u_control:u_control_batch,step_size:step_size_val,yp_feed:output_batch_p,yf_feed:output_batch_f});
#       train_error = mean_diff_nocovar.eval(feed_dict={yp_feed:u_batch,yf_feed:y_batch,u_control:u_control_batch,step_size:step_size_val,yp_feed:output_batch_p,yf_feed:output_batch_f});
#       valid_error = mean_diff_nocovar.eval(feed_dict={yp_feed:u_valid,yf_feed:y_valid,u_control:u_control_valid,yp_feed:output_batch_valid_p,yf_feed:output_batch_valid_f});
#       test_error = mean_diff_nocovar.eval(feed_dict={yp_feed:u_test_train,yf_feed:y_test_train,u_control:u_control_test_train,yp_feed:output_batch_test_train_p,yf_feed:output_batch_test_train_f});
#
#
#     if (not with_control) and (not with_output):
#       optimizer.run(feed_dict={yp_feed:u_batch,yf_feed:y_batch,step_size:step_size_val});
#       valid_error = mean_diff_nocovar.eval(feed_dict={yp_feed:u_valid,yf_feed:y_valid});
#       test_error = mean_diff_nocovar.eval(feed_dict={yp_feed:u_test_train,yf_feed:y_test_train});
#
#
#
#     if iter%samplerate==0:
#       if with_control and (not with_output):
#         training_error_history_nocovar.append(mean_diff_nocovar.eval(feed_dict={yp_feed:u_batch,yf_feed:y_batch,u_control:u_control_batch}));
#         validation_error_history_nocovar.append(mean_diff_nocovar.eval(feed_dict={yp_feed:u_valid,yf_feed:y_valid,u_control:u_control_valid}));
#         test_error_history_nocovar.append(mean_diff_nocovar.eval(feed_dict={yp_feed:u_test_train,yf_feed:y_test_train,u_control:u_control_test_train}));
#       if with_control and with_output:
#         training_error_history_nocovar.append(train_error);
#         validation_error_history_nocovar.append(valid_error);
#         test_error_history_nocovar.append(test_error);
#
#       if (not with_control) and (not with_output):
#         training_error_history_nocovar.append(mean_diff_nocovar.eval(feed_dict={yp_feed:u_batch,yf_feed:y_batch}));
#         validation_error_history_nocovar.append(mean_diff_nocovar.eval(feed_dict={yp_feed:u_valid,yf_feed:y_valid}));
#         test_error_history_nocovar.append(mean_diff_nocovar.eval(feed_dict={yp_feed:u_test_train,yf_feed:y_test_train}));
#
#
#       if (iter%10==0) or (iter==1):
#         plt.close();
#         if plot_deep_basis:
#           fig_hand = expose_deep_basis(psiypz_list,num_bas_obs,x_deep_dict_size,iter,yp_feed);
#           fig_hand = quick_nstep_predict(Y_p_old,u_control_all_training,with_control,num_bas_obs,iter);
#
#         if with_control and (not with_output):
#           print ("step %d , validation error %g"%(iter, mean_diff_nocovar.eval(feed_dict={yp_feed:u_valid,yf_feed:y_valid,u_control:u_control_valid})));
#           print ("step %d , test error %g"%(iter, mean_diff_nocovar.eval(feed_dict={yp_feed:u_test_train,yf_feed:y_test_train,u_control:u_control_test_train})));
#
#         if with_control and with_output:
#           print ("step %d , validation error %g"%(iter, valid_error));
#           print ("step %d , test error %g"%(iter, test_error));
#
#             #print ( test_synthesis(sess.run(Kx).T,sess.run(Ku).T ))
#         if (not with_control) and (not with_output):
#           print ("step %d , validation error %g"%(iter, mean_diff_nocovar.eval(feed_dict={yp_feed:u_valid,yf_feed:y_valid})));
#           print ("step %d , test error %g"%(iter, mean_diff_nocovar.eval(feed_dict={yp_feed:u_test_train,yf_feed:y_test_train})));
#
#     if ((iter>20000) and iter%10) :
#
#       valid_gradient = np.gradient(np.asarray(validation_error_history_nocovar[np.int(iter/samplerate*3/10):]) );
#       mu_gradient = np.mean(valid_gradient);
#
#       if ((iter <1000) and (mu_gradient >= 5e-1)): # eventually update this to be 1/10th the mean of batch data, or mean of all data handed as input param to func
#         good_start = 0; # if after 10,000 iterations validation error is still above 1e0, initialization was poor.
#         print("Terminating model refinement loop with gradient:") + repr(mu_gradient) + ", validation error after " + repr(iter) + " epochs:  " + repr(valid_error);
#         iter = max_iters; # terminate while loop and return histories
#
#
#
#     # if iter > 10000 and with_control:
#     #   At = None;
#       #At = sess.run(Kx).T;
#       #Bt = sess.run(Ku).T;
#       #ctrb_rank = np.linalg.matrix_rank(control.ctrb(At,Bt));
#       #if debug_splash:
#       #  print(repr(ctrb_rank) + " : " + repr(control.ctrb(At,Bt).shape[0]));
#
#       #if ctrb_rank == control.ctrb(At,Bt).shape[0] and test_error_history_nocovar[-1] <1e-5:
#       #   iter=max_iters;
#
#   all_histories = [training_error_history_nocovar, validation_error_history_nocovar,test_error_history_nocovar, \
#                   training_error_history_withcovar,validation_error_history_withcovar,test_error_history_withcovar,covar_actual,covar_diff_history,covar_model_history];
#
#   plt.close();
#   x = np.arange(0,len(validation_error_history_nocovar),1);
#   plt.plot(x,training_error_history_nocovar,label='train. err.');
#   plt.plot(x,validation_error_history_nocovar,label='valid. err.');
#   plt.plot(x,test_error_history_nocovar,label='test err.');
#   #plt.gca().set_yscale('log');
#   plt.savefig('all_error_history.pdf');
#
#   plt.close();
#   return all_histories,good_start;

# Functions for storing the data
def remove_past_run_data():
    FOLDER_NAME = '_current_run_saved_files'
    if os.path.exists(FOLDER_NAME):
        shutil.rmtree(FOLDER_NAME)  # Delete the folder and all the contents inside it
    os.mkdir(FOLDER_NAME)  # Recreate the folder
    return

def generate_next_run_directory():
    main_folder = '_current_run_saved_files'
    highest_run_number = -1
    if os.path.exists(main_folder):
        list_folders = os.listdir(main_folder)
        for items in os.listdir(main_folder):
            if str(items[0:4]) == 'RUN_':
                highest_run_number = np.max([highest_run_number, int(items[4:])])
            else:
                list_folders.remove(items)
    else:
        os.mkdir(main_folder)
    FOLDER_NAME = main_folder + '/RUN_' + str(highest_run_number + 1)
    os.mkdir(FOLDER_NAME)
    return FOLDER_NAME

# # # END HELPER FUNCTIONS # # #


## # - - - Begin Koopman Model Script - - - # # #

pre_examples_switch = 13;
# os.chdir('/home/deepuser/Desktop/Shara/deepDMD')
# os.chdir('/Users/shara/Desktop/Shara_optictensor/darpa-sd2/deepDMD')

data_directory = 'koopman_data/'
if pre_examples_switch == 0:
    # Randomly generated oscillator system with control
    data_suffix = 'rand_osc.pickle';
    with_control = 1
    with_output = 0;
    mix_state_and_control = 0
    phase_space_stitching = 0;

if pre_examples_switch == 1:
    data_suffix = 'cons_law.pickle';
    with_control = 0;
    with_output = 0;
    mix_state_and_control = 0
    phase_space_stitching = 0;

if pre_examples_switch == 2:
    data_suffix = '50bus.pickle';
    with_control = 1;
    with_output = 0;
    mix_state_and_control = 0
    phase_space_stitching = 0;

if pre_examples_switch == 3:
    data_suffix = 'zhang_control.pickle';
    with_control = 1;
    with_output = 0;
    mix_state_and_control = 0
    phase_space_stitching = 0;

if pre_examples_switch == 4:
    data_suffix = 'deltaomega-series.pickle';  ### IEEE 39 bus swing model
    with_control = 1;
    with_output = 0;
    mix_state_and_control = 0
    phase_space_stitching = 0;

if pre_examples_switch == 5:
    data_suffix = 'glycol.pickle';
    with_control = 0;
    with_output = 0;
    mix_state_and_control = 0
    phase_space_stitching = 0;

if pre_examples_switch == 6:
    data_suffix = 'exp_toggle_switch.pickle';
    with_control = 0;
    with_output = 0;
    mix_state_and_control = 0
    phase_space_stitching = 0;

if pre_examples_switch == 7:
    data_suffix = 'MD.pickle';
    with_control = 1;
    with_output = 0;
    mix_state_and_control = 0
    phase_space_stitching = 0;

if pre_examples_switch == 8:
    data_suffix = 'phase_space_stitching/sim_toggle_switch_phase1.pickle';
    with_control = 0;
    with_output = 0;
    mix_state_and_control = 0
    phase_space_stitching = 0;

if pre_examples_switch == 9:
    data_suffix = 'phase_space_stitching/sim_toggle_switch_phase2.pickle';
    with_control = 0;
    with_output = 0;
    mix_state_and_control = 0
    phase_space_stitching = 1;

if pre_examples_switch == 10:
    data_suffix = 'exp_toggle_switch_M9CA.pickle';
    with_control = 0;
    with_output = 0;
    mix_state_and_control = 0
    phase_space_stitching = 0;

if pre_examples_switch == 11:
    data_suffix = 'fourstate_mak.pickle'
    with_control = 0;
    with_output = 0;
    mix_state_and_control = 0
    phase_space_stitching = 0;

if pre_examples_switch == 12:
    data_suffix = 'SRI_ribo.pickle';
    with_control = 1;
    with_output = 0;
    mix_state_and_control = 0
    phase_space_stitching = 0;

if pre_examples_switch == 13:
    # data_suffix = 'Pputida_GrowthHarness_RNAseq_DeepDMD.pickle'#'SIM1SHARA_Combinatorial_Promoters_with_input';#'X8SS_Pputida_RNASeqDATA.pickle';
    # data_suffix = 'oc_deepDMD_FeedForwardLoopSystem.pickle'
    data_suffix = 'oc_deepDMD_ClosedKoopmanSystem.pickle'
    with_control = 0;
    with_output = 1;
    mix_state_and_control = 0;
    phase_space_stitching = 0;

if pre_examples_switch == 14:
    data_suffix = 'incoherent_ff_loop.pickle';
    with_control = 1;
    with_output = 0;
    mix_state_and_control = 0;
    phase_space_stitching = 0;

## CMD Line Argument (Override) Inputs:

# import sys
# if len(sys.argv)>1:
#   data_suffix = sys.argv[1];
# if len(sys.argv)>2:
#   x_max_layers= np.int(sys.argv[2]);
# if len(sys.argv)>3:
#   x_max_nodes_limit = np.int(sys.argv[3]);
# if len(sys.argv)>4:
#   x_deep_dict_size = np.int(sys.argv[4]);
# if len(sys.argv)>5 and with_control:
#   u_max_layers= np.int(sys.argv[5]);
# if len(sys.argv)>6 and with_control:
#   u_deep_dict_size = np.int(sys.argv[6]);
# if len(sys.argv)>7 and with_control:
#   plot_deep_basis = np.int(sys.argv[7]);
#

data_file = data_directory + data_suffix
Xp, Xf, Up, Yp, Yf = load_pickle_data(data_file, with_control, with_output)

num_bas_obs = len(Xp[0]);
num_all_samples = len(Xp);

n_x_nn_outputs = num_bas_obs;
n_x_nn_inputs = num_bas_obs;

if with_control:
    n_u_nn_inputs = len(Up[0])
else:
    n_u_nn_inputs = 0

# Display info
print("[INFO] Number of total samples: " + repr(num_all_samples))
print("[INFO] Observable dimension of a sample: " + repr(num_bas_obs))
print("[INFO] Xp.shape (E-DMD): " + repr(Xp.shape));
print("[INFO] Yf.shape (E-DMD): " + repr(Xf.shape));

## Train/Test Split for Benchmarking Forecasting Later

num_trains = np.int(len(Xp) * TRAIN_PERCENT / 100)
num_valids = np.int(len(Xp) * VALID_PERCENT / 100)
train_indices = np.arange(0, num_trains, 1)
valid_indices = np.arange(num_trains,num_trains+num_valids,1)
test_indices = np.arange(num_trains+num_valids, len(Xp), 1)
dict_train = {}
dict_valid = {}
dict_test = {}
dict_train['Xp'] = Xp[train_indices]
dict_valid['Xp'] = Xp[valid_indices]
dict_test['Xp'] = Xp[test_indices]
dict_train['Xf'] = Xf[train_indices]
dict_valid['Xf'] = Xf[valid_indices]
dict_test['Xf'] = Xf[test_indices]

if with_control:
    dict_train['Up'] = Up[train_indices]
    dict_valid['Up'] = Up[valid_indices]
    dict_test['Up'] = Up[test_indices]

if with_output:
    dict_train['Yp'] = Yp[train_indices]
    dict_valid['Yp'] = Yp[valid_indices]
    dict_test['Yp'] = Yp[test_indices]

    dict_train['Yf'] = Yf[train_indices]
    dict_valid['Yf'] = Yf[valid_indices]
    dict_test['Yf'] = Yf[test_indices]



# Display info
print("Number of training snapshots: " + repr(len(train_indices)));
print("Number of validation snapshots: " + repr(len(valid_indices)));
print("Number of test snapshots: " + repr(len(test_indices)));
if debug_splash:
    print("[DEBUG] Xp_test.shape (E-DMD) ") + repr(dict_test['Xp'].shape);
    print("[DEBUG] Xf_test.shape (E-DMD) ") + repr(dict_test['Xf'].shape);
    if with_control:
        print("[DEBUG] U_train.shape (E-DMD) ") + repr(dict_train['Up'].shape);
    # print("[INFO] up_all_training.shape: ") + repr(Up.shape);

## Begin Sweep
# for n_depth_reciprocal in range(0,1):#max_depth-2): #2
n_layers_reciprocal = 0  # TODO - n_layers_reciprocal is used to decrease the number of nn layers with each iteration - make sure to include the for loop and start the loop from 0
n_x_nn_layers = x_max_nn_layers - n_layers_reciprocal;
n_u_nn_layers = u_max_nn_layers - n_layers_reciprocal;
n_xu_nn_layers = xu_max_nn_layers - n_layers_reciprocal;
# for x_min_nodes_reciprocal in range(0,1):#x_min_nodes_limit): #2
n_nodes_reciprocal = 0
x_min_nn_nodes = x_min_nn_nodes_limit - n_nodes_reciprocal
x_max_nn_nodes = x_min_nn_nodes  # zero gap between min and max ; use regularization and dropout to trim edges.
u_min_nn_nodes = u_min_nn_nodes_limit - n_nodes_reciprocal
xu_min_nn_nodes = xu_min_nn_nodes_limit - n_nodes_reciprocal

if x_max_nn_nodes == x_min_nn_nodes:
    x_hidden_vars_list = np.asarray([x_min_nn_nodes] * n_x_nn_layers);
else:
    x_hidden_vars_list = np.random.randint(x_min_nn_nodes, x_max_nn_nodes, size=n_x_nn_layers);
x_hidden_vars_list[-1] = x_deep_dict_size
if with_control:
    u_hidden_vars_list = np.asarray([u_min_nn_nodes] * n_u_nn_layers)
    u_hidden_vars_list[-1] = u_deep_dict_size
    if mix_state_and_control:
        xu_hidden_vars_list = np.asarray([xu_min_nn_nodes] * n_xu_nn_layers)
        xu_hidden_vars_list[-1] = xu_deep_dict_size
    else:
        xu_hidden_vars_list = []
else:
    u_hidden_vars_list = []
    xu_hidden_vars_list = []
# Display info
print("[INFO] STATE - hidden_vars_list: " + repr(x_hidden_vars_list))
print("[INFO] INPUT - hidden_vars_list: " + repr(u_hidden_vars_list))
print("[INFO] MIX STATE n INPUT - hidden_vars_list: " + repr(xu_hidden_vars_list))

good_start = 0
max_tries = 1
try_num = 0
while good_start == 0 and try_num < max_tries:
    try_num += 1;
    if debug_splash:
        print("\n Initialization attempt number: ") + repr(try_num);
        print("\n \t Initializing Tensorflow Residual ELU Network with ") + repr(n_x_nn_inputs) + (
            " inputs and ") + repr(n_x_nn_outputs) + (" outputs and ") + repr(len(x_hidden_vars_list)) + (" layers");
    with tf.device('/cpu:0'):
        dict_feed = {}
        dict_psi = {}
        dict_K ={}
        Wx_list, bx_list = initialize_Wblist(n_x_nn_inputs, x_hidden_vars_list)
        x_params_list = {'no of base observables': n_x_nn_inputs, 'no of neural network observables': x_deep_dict_size,
                         'hidden_var_list': x_hidden_vars_list, 'W_list': Wx_list, 'b_list': bx_list,
                         'keep_prob': keep_prob, 'activation flag': activation_flag, 'res_net': res_net,
                         'include state': True, 'add bias': add_bias}
        psixpz_list, psixp, xp_feed = instantiate_comp_graph(x_params_list)
        psixfz_list, psixf, xf_feed = instantiate_comp_graph(x_params_list)
        # Kx definition
        if phase_space_stitching and (not with_control):
            try:
                Kmatrix_file_obj = open('phase_space_stitching/raws/Kmatrix_file.pickle', 'rb');
                this_pickle_file_list = pickle.load(Kmatrix_file_obj);
                Kx_num = this_pickle_file_list[0];
                Kx = tf.constant(Kx_num);  # this is assuming a row space (pre-multiplication) Koopman
            except:
                print(
                    "[Warning]: No phase space prior for the Koopman Matrix Detected @ /phase_space_stitching/raws/Kmatrix_file.pickle\n . . . learning Koopman prior as a fixed variable. ");
                no_phase_space_prior = True
                Kx = weight_variable([x_deep_dict_size + n_x_nn_inputs, x_deep_dict_size + n_x_nn_inputs]);
        else:
            if add_bias:
                # Kx = weight_variable([x_deep_dict_size + n_x_nn_inputs + 1, x_deep_dict_size + n_x_nn_inputs + 1])
                Kx = weight_variable([x_deep_dict_size + n_x_nn_inputs + 1, x_deep_dict_size + n_x_nn_inputs])
                last_row = tf.constant(np.zeros(shape=(x_deep_dict_size + n_x_nn_inputs, 1)), dtype=tf.dtypes.float32)
                last_row = tf.concat([last_row, [[1.]]], axis=0)
                Kx = tf.concat([Kx, last_row], axis=1)
            else:
                Kx = weight_variable([x_deep_dict_size + n_x_nn_inputs, x_deep_dict_size + n_x_nn_inputs])
            print('Kx initiation done!')
        dict_feed ['xpT'] = xp_feed;
        dict_feed ['xfT'] = xf_feed;
        dict_psi ['xpT'] = psixp;
        dict_psi['xfT'] = psixf;
        dict_K['KxT'] = Kx;
        if with_control:
            Wu_list, bu_list = initialize_Wblist(n_u_nn_inputs, u_hidden_vars_list)
            u_params_list = {'no of base observables': n_u_nn_inputs,
                             'no of neural network observables': u_deep_dict_size,
                             'hidden_var_list': u_hidden_vars_list, 'W_list': Wu_list, 'b_list': bu_list,
                             'keep_prob': keep_prob, 'activation flag': activation_flag, 'res_net': res_net,
                             'include state': True, 'add bias': add_bias}
            psiupz_list, psiup, up_feed = instantiate_comp_graph(u_params_list);
            dict_feed['upT'] = up_feed;
            dict_psi['upT'] = psiup;
            if add_bias:
                Ku = weight_variable([u_deep_dict_size + n_u_nn_inputs + 1, x_deep_dict_size + n_x_nn_inputs + 1])
            else:
                Ku = weight_variable([u_deep_dict_size + n_u_nn_inputs, x_deep_dict_size + n_x_nn_inputs])
            dict_K['KuT'] = Ku
            if mix_state_and_control:
                Wxu_list, bxu_list = initialize_Wblist(n_x_nn_inputs + n_u_nn_inputs, xu_hidden_vars_list)
                xu_params_list = {'no of base observables': n_x_nn_inputs + n_u_nn_inputs,
                                  'no of neural network observables': xu_deep_dict_size,
                                  'hidden_var_list': xu_hidden_vars_list, 'W_list': Wxu_list, 'b_list': bxu_list,
                                  'keep_prob': keep_prob, 'activation flag': activation_flag, 'res_net': res_net,
                                  'include state': False, 'add bias': add_bias}
                psixupz_list, psixup, xup_feed = instantiate_comp_graph(xu_params_list)
                dict_feed['xupT'] = xup_feed
                dict_psi['xupT'] = psixup
                if add_bias:
                    Kxu = weight_variable([xu_deep_dict_size, x_deep_dict_size + n_x_nn_inputs])
                else:
                    Kxu = weight_variable([xu_deep_dict_size, x_deep_dict_size + n_x_nn_inputs])
                dict_K['KxuT'] = Kxu
        #     else:
        #         Wxu_list = bxu_list = psixuz_list = psixu = xup_feed = Kxu = None
        # else:
        #     Wu_list = bu_list = psiupz_list = psiup = up_feed = Ku = None
        #     Wxu_list = bxu_list = psixupz_list = psixup = xup_feed = Kxu = None
        if with_output:
            yp_feed = tf.placeholder(tf.float32, shape=[None, Yf.shape[1]])
            yf_feed = tf.placeholder(tf.float32, shape=[None, Yf.shape[1]])
            dict_feed['ypT'] = yp_feed
            dict_feed['yfT'] = yf_feed
            if add_bias:
                Wh = weight_variable([x_deep_dict_size + n_x_nn_inputs + 1, Yf.shape[1]]);
            else:
                Wh = weight_variable([x_deep_dict_size + n_x_nn_inputs, Yf.shape[1]]);
            dict_K['WhT'] = Wh
        # else:
        #     yf_feed = yp_feed = Wh = None
        dict_feed['step_size'] = tf.placeholder(tf.float32, shape=[])
        dict_feed['regularization_lambda'] = tf.placeholder(tf.float32, shape=[])
        sess.run(tf.global_variables_initializer())

        # Kx = tf.constant([[0.86,0.,0.,0.],[0.,0.8,0., 0.],[0.,0.,0.7396, 0.],[0.,0.,0.,1.]])
        # Wh = tf.constant([[0.],[0.],[1.],[0.]])

        # deep_koopman_loss, deep_koopman_accuracy, optimizer, forward_prediction, out_pred_f, out_pred_p = Deep_Output_KIC_Objective_v2(
        #     psixp, psixf, Kx, psiup, Ku, psixup, Kxu, yf_feed, yp_feed, Wh, step_size, regularization_lambda,
        #     with_control, mix_state_and_control, with_output)

        deep_koopman_loss, optimizer,dict_predictions = Deep_Output_KIC_Objective_v3(dict_feed,dict_psi,dict_K, with_control, mix_state_and_control, with_output)



        if debug_splash:
            train_vars = tf.trainable_variables()
            values = sess.run([x.name for x in train_vars])
            print("[DEBUG] # of Trainable Variables: ") + repr(len(values))
            print("[DEBUG] Trainable Variables: ") + repr([temp_var.shape for temp_var in values])
            print("[DEBUG] # of datapoints in up_all_training: ") + repr(Xp.shape)
            print("[DEBUG] # of datapoints in uf_all_training: ") + repr(Xf.shape)
        print('Training begins now!')
        all_histories = {'train error': [], 'validation error': []}
        dict_run_info ={}


        # try:
        for i in range(len(ls_dict_training_params)):
            ls_dict_training_params[i]['with_u'] = with_control
            ls_dict_training_params[i]['with_y'] = with_output
            ls_dict_training_params[i]['with_xu'] = mix_state_and_control
        all_histories, good_start, dict_run_info = static_train_net(dict_train, dict_valid, dict_feed, dict_psi, dict_K, ls_dict_training_params, deep_koopman_loss, optimizer, dict_predictions)
        # except:
        #     print('[INFO] No static training done!')
        dict_run_params = {'step_size_val': step_size_val, 'regularization_lambda_val': regularization_lambda_val,
                           'train_error_threshold': train_error_threshold,
                           'valid_error_threshold': valid_error_threshold, 'max_epochs': max_epochs,
                           'batch_size': batch_size, 'with_u': with_control, 'with_y': with_output,
                           'with_xu': mix_state_and_control}
        print('SUCCESSFULLY HERE')
        # all_histories, good_start, dict_run_info = dynamic_train_net(dict_train, dict_valid, dict_feed, dict_psi,dict_K, dict_run_params, deep_koopman_loss, optimizer, dict_predictions,all_histories,dict_run_info)
        # # # all_histories,good_start  = train_net(up_all_training,uf_all_training,deep_koopman_loss,optimizer,U_train,Out_p_train,Out_f_train,valid_error_threshold,test_error_threshold,max_iters,step_size_val);
    training_error_history_nocovar = all_histories['train error'];
    validation_error_history_nocovar = all_histories['validation error'];
    print("[INFO] Initialization was successful: " + repr(good_start == 1));

    feed_dict_train = {xp_feed: dict_train['Xp'], xf_feed: dict_train['Xf'],
                       dict_feed['regularization_lambda']: regularization_lambda_val}
    feed_dict_valid = {xp_feed: dict_valid['Xp'], xf_feed: dict_valid['Xf'],
                       dict_feed['regularization_lambda']: regularization_lambda_val}
    feed_dict_test = {xp_feed: dict_test['Xp'], xf_feed: dict_test['Xf'],
                      dict_feed['regularization_lambda']: regularization_lambda_val}
    if with_control:
        feed_dict_train[up_feed] = dict_train['Up']
        feed_dict_valid[up_feed] = dict_valid['Up']
        feed_dict_test[up_feed] = dict_test['Up']
        if mix_state_and_control:
            feed_dict_train[xup_feed] = np.concatenate([dict_train['Xp'], dict_train['Up']], axis=1)
            feed_dict_test[xup_feed] = np.concatenate([dict_test['Xp'], dict_test['Up']], axis=1)
    if with_output:
        feed_dict_train[yf_feed] = dict_train['Yf']
        feed_dict_test[yf_feed] = dict_test['Yf']
        feed_dict_train[yp_feed] = dict_train['Yp']
        feed_dict_test[yp_feed] = dict_test['Yp']
    train_accuracy = deep_koopman_loss.eval(feed_dict=feed_dict_train)
    test_accuracy = deep_koopman_loss.eval(feed_dict=feed_dict_test)

    if debug_splash:
        print("[Result]: Training Error: ");
        print(train_accuracy);
        print("[Result]: Test Error : ");
        print(test_accuracy);
### Write Vars to Checkpoint Files/MetaFiles
# Creating a folder for saving the objects of the current run
if REMOVE_PREVIOUS_RUNDATA:
    remove_past_run_data()
FOLDER_NAME = generate_next_run_directory()

Kx_num = sess.run(Kx)
Wx_list_num = [sess.run(W_temp) for W_temp in Wx_list]
bx_list_num = [sess.run(b_temp) for b_temp in bx_list]
if with_control:
    Wu_list_num = [sess.run(W_temp) for W_temp in Wu_list]
    bu_list_num = [sess.run(b_temp) for b_temp in bu_list]
    Ku_num = sess.run(Ku)
    if mix_state_and_control:
        Wxu_list_num = [sess.run(W_temp) for W_temp in Wxu_list]
        bxu_list_num = [sess.run(b_temp) for b_temp in bxu_list]
        Kxu_num = sess.run(Kxu)
if with_output:
    Wh_num = sess.run(Wh)

dict_dump = {}
dict_dump['Wx_list_num'] = Wx_list_num
dict_dump['bx_list_num'] = bx_list_num
dict_dump['Kx_num'] = Kx_num
if with_output:
    dict_dump['Wh_num'] = Wh_num
if with_control:
    dict_dump['Wu_list_num'] = Wu_list_num
    dict_dump['bu_list_num'] = bu_list_num
    dict_dump['Ku_num'] = Ku_num
    if mix_state_and_control:
        dict_dump['Wxu_list_num'] = Wxu_list_num
        dict_dump['bxu_list_num'] = bxu_list_num
        dict_dump['Kxu_num'] = Kxu_num

with open(FOLDER_NAME + '/constrainedNN-Model.pickle', 'wb') as file_obj_swing:
    pickle.dump(dict_dump, file_obj_swing)
with open(FOLDER_NAME + '/run_info.pickle', 'wb') as file_obj_swing:
    pickle.dump(dict_run_info, file_obj_swing)
with open(FOLDER_NAME + '/all_histories.pickle', 'wb') as file_obj_swing:
    pickle.dump(all_histories, file_obj_swing)
print('------ ------ -----')
print('----- Run Info ----')
print('------ ------ -----')
print(pd.DataFrame(dict_run_info))
print('------ ------ -----')
if (not phase_space_stitching) and (not with_control):
    file_obj_phase = open('phase_space_stitching/raws/Kmatrix_file.pickle', 'wb');
    pickle.dump([Kx_num], file_obj_phase);
    file_obj_phase.close();

saver = tf.compat.v1.train.Saver()

all_tf_var_names =[]
for items in dict_psi.keys():
    tf.compat.v1.add_to_collection('psi'+items, dict_psi[items])
    all_tf_var_names.append('psi'+items)
for items in dict_feed.keys():
    tf.compat.v1.add_to_collection(items+'_feed', dict_feed[items])
    all_tf_var_names.append(items+'_feed')
for items in dict_K.keys():
    tf.compat.v1.add_to_collection(items, dict_K[items])
    all_tf_var_names.append(items)

tf.compat.v1.add_to_collection('loss_func', deep_koopman_loss)
tf.compat.v1.add_to_collection('regularization_lambda_val', regularization_lambda_val)
tf.compat.v1.add_to_collection('regularization_lambda', dict_feed['regularization_lambda'])
all_tf_var_names.append('loss_func')
all_tf_var_names.append('regularization_lambda_val')
all_tf_var_names.append('regularization_lambda')
for items in list(dict_predictions.keys()):
    all_tf_var_names.append(items)
    tf.compat.v1.add_to_collection(items, dict_predictions[items])

save_path = saver.save(sess, data_suffix + '.ckpt')
saver_path_curr = saver.save(sess, FOLDER_NAME + '/' + data_suffix + '.ckpt')
with open(FOLDER_NAME + '/all_tf_var_names.pickle', 'wb') as handle:
    pickle.dump(all_tf_var_names,handle)
Koopman_dim = Kx_num.shape[0]
print("[INFO] Koopman_dim:" + repr(Kx_num.shape))

# if single_series:
#   if pre_examples_switch ==3:
#      Y_p_old,Y_f_old,u_control_all_training = load_pickle_data('koopman_data/zhang_control.pickle',with_control,with_output);
#   if pre_examples_switch == 4:
#      Y_p_old,Y_f_old,u_control_all_training = load_pickle_data('koopman_data/deltaomega-singleseries.pickle',with_control,with_output);
#
#
#   if not( Kx_num.shape[1]==Kx_num.shape[0]):
#       print("Warning! Estimated Koopman operator is not square with dimensions : " + repr(Kx_num.shape));
#
#   train_range = len(Y_p_old)/2; # define upper limits of training data
#
#   if debug_splash:
#     print("[DEBUG] train_range: " + repr(train_range));
#
#
#
#   test_range = len(Y_p_old); # define upper limits of test data
#
#   print("[DEBUG] test_range: " + repr(test_range));
#   Yp_test = Y_p_old[train_range:test_range];
#   Yf_test = Y_f_old[train_range:test_range];
#
#   if with_control:
#     U_test = u_control_all_training_old[train_range:test_range];
#     U_train = u_control_all_training_old[0:train_range];
#     U_train = np.asarray(U_train);
#     U_test = np.asarray(U_test);
#     if len(U_test.shape)==1:
#       U_train = np.reshape(U_train,(U_train.shape[0],1));
#       U_test = np.reshape(U_test,(U_test.shape[0],1));


# Yp_train = np.asarray(Yp_train);
# Yf_train = np.asarray(Yf_train);
# Yp_test = np.asarray(Yp_test);
# Yf_test = np.asarray(Yf_test);
# Yp_final_test = Yp_test;
# Yf_final_test = Yf_test;
# Yp_final_train = Yp_train;
# Yf_final_train = Yf_train;


# # # Print Evaluation Metrics -  Deep Koopman Learning # # #

# print("[DEBUG]: Yp_train.shape") + repr(Yp_train.shape);

# if with_control and (not with_output):
#   training_error = accuracy.eval(feed_dict={yp_feed:list(Yp_train),yf_feed:list(Yf_train),u_control:list(U_train)});
#   test_error = accuracy.eval(feed_dict={yp_feed:list(Yp_test),yf_feed:list(Yf_test),u_control:list(U_test)});
# if with_control and with_output:
#   training_error = accuracy.eval(feed_dict={yp_feed:up_all_training,yf_feed:uf_all_training,u_control:U_train,yf_feed:Out_f_train,yp_feed:Out_p_train});
#   test_error = accuracy.eval(feed_dict={yp_feed:Yp_test,yf_feed:Yf_test,u_control:U_test,yf_feed:Out_f_test,yp_feed:Out_p_test});
#
# if (not with_control) and (not with_output):
#   training_error = accuracy.eval(feed_dict={yp_feed:list(Yp_train),yf_feed:list(Yf_train)});
#   test_error = accuracy.eval(feed_dict={yp_feed:list(Yp_test),yf_feed:list(Yf_test)});
#
# feed_dict_train = {yp_feed: list(Yp_train), yf_feed: list(Yf_train)}
# feed_dict_test = {yp_feed: list(Yp_test), yf_feed: list(Yf_test)}
# if with_control:
#   feed_dict_train[u_control] = U_train
#   feed_dict_test[u_control] = U_test
# if with_output:
#   feed_dict_train[yf_feed] = Out_f_train
#   feed_dict_test[yf_feed] = Out_f_test
#   feed_dict_train[yp_feed] = Out_p_train
#   feed_dict_test[yp_feed] = Out_p_test
# training_error = accuracy.eval(feed_dict = feed_dict_train)
# test_error = accuracy.eval(feed_dict=feed_dict_test)
print('%s%f' % ('[COMP] Training error: ', train_accuracy));
print('%s%f' % ('[COMP] Test error: ', test_accuracy));
estimate_K_stability(Kx)














# np.linalg.eigvals(Kx_num)
# Kx_num_eigval_mod = np.abs(np.linalg.eigvals(Kx_num))
# if np.max(Kx_num_eigval_mod)>1:
#     print('[COMP] The identified Koopman operator is UNSTABLE with ',np.sum(np.abs(Kx_num_eigval_mod)>1),'eigenvalues greater than 1')
# else:
#     print('[COMP] The identified Koopman operator is STABLE')

# # # # - - - n-step Prediction Error Analysis - - - # # #
#
#
# n_points_pred = len(Y_p_old) - test_indices[0]-1;
#
# init_index = test_indices[0];
# Yf_final_test_stack_nn = np.asarray(Y_p_old).T[:,init_index:(init_index+1)+n_points_pred]
# Ycurr = np.asarray(Y_p_old).T[:,init_index]
# Ycurr = np.transpose(Ycurr);
# if with_control:
#   Uf_final_test_stack_nn = np.asarray(u_control_all_training).T[:,init_index:(init_index+1)+n_points_pred]
#
# #Reshape for tensorflow, which operates using row multiplication.
# Ycurr = Ycurr.reshape(1,num_bas_obs);
# psiyp_Ycurr = psiyp.eval(feed_dict={yp_feed:Ycurr});
# psiyf_Ycurr = psiyf.eval(feed_dict={yf_feed:Ycurr});
#
#
# ## Define a growing list of vector valued observables that is the forward prediction of the Yf snapshot matrix, initiated from an initial condition in Yp_final_test.
# Yf_final_test_ep_nn = [];
# Yf_final_test_ep_nn.append(psiyp_Ycurr.tolist()[0][0:num_bas_obs]); # append the initial seed state value.
#
# for i in range(0,n_points_pred):
#   if with_control:
#     if len(U_test[i,:])==1:
#       U_temp_mat = np.reshape(Uf_final_test_stack_nn[i,:],(1,1));
#       psiyp_Ycurr = sess.run(forward_prediction_control, feed_dict={yp_feed:psiyp_Ycurr[:,0:num_bas_obs],u_control:U_temp_mat});#
#     else:
#       U_temp_mat = np.reshape(Uf_final_test_stack_nn.T[i,:],(1,n_inputs_control));
#       psiyp_Ycurr = sess.run(forward_prediction_control, feed_dict={yp_feed:psiyp_Ycurr[:,0:num_bas_obs],u_control:U_temp_mat});#
#   else:
#     psiyp_Ycurr = sess.run(forward_prediction,feed_dict={yp_feed:psiyp_Ycurr[:,0:num_bas_obs]});
#
#   Yout = psiyp_Ycurr.tolist()[0][0:num_bas_obs];
#   Yf_final_test_ep_nn.append(Yout);
#
#
# Yf_final_test_ep_nn = np.asarray(Yf_final_test_ep_nn);
# Yf_final_test_ep_nn = np.transpose(Yf_final_test_ep_nn);
#
# prediction_error = np.linalg.norm(Yf_final_test_stack_nn-Yf_final_test_ep_nn,ord='fro')/np.linalg.norm(Yf_final_test_stack_nn,ord='fro');
# print('%s%f' % ('[RESULT] n-step Prediction error: ',prediction_error));
#
# import matplotlib
# matplotlib.rcParams.update({'font.size':20})
#
#
# ### Make a Prediction Plot
# x_range = np.arange(0,30,1)
# #x_range = np.arange(0,Yf_final_test_stack_nn.shape[1],1);
# for i in range(0,3):#num_bas_obs):
#     plt.plot(x_range,Yf_final_test_ep_nn[i,0:len(x_range)],'--',color=colors[i,:]);
#     plt.plot(x_range,Yf_final_test_stack_nn[i,0:len(x_range)],'*',color=colors[i,:]);
# axes = plt.gca();
# axes.spines['right'].set_visible(False)
# axes.spines['top'].set_visible(False)
#
# #plt.legend(loc='best');
# plt.xlabel('t');
# fig = plt.gcf();
#
# target_file = data_suffix.replace('.pickle','')+'final_nstep_prediction.pdf';
# plt.savefig(target_file);
# plt.show();


