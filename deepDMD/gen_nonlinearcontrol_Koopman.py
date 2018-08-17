
#! /usr/bin/env python

### Import Packages 
import pickle; # for data I/O

# Math Packages 
import numpy as np;
from numpy.linalg import pinv;
from numpy.polynomial.legendre import legvander;
#import cvxopt;
#from cvxpy import Minimize, Problem, Variable,norm1,norm2,installed_solvers,lambda_max;
#from cvxpy import norm as cvxpynorm;
#import cvxpy;
import tensorflow as tf
import math;
import random;

# Plotting Tools 
import matplotlib
matplotlib.rcParams.update({'font.size':22}) # default font size on (legible) figures
matplotlib.use('Agg'); # for seamless execution in Linux environments with Tensorflow 
import matplotlib.pyplot as plt;


### User flags (dev-note: run as a separate instance of code?) 
#with_control = 1;  # This activates the closed-loop deep Koopman learning algorithm; requires input and state data. 
plot_deep_basis = 0;  # This activates plotting of deep basis functions as a function of training iterations.
single_series = 0;  # Useful if you're analyzing a system with only a single time-series (requires extremely high temporal resolution). 
debug_splash = 1;

### Support Vars: 

colors =[[ 0.68627453,  0.12156863,  0.16470589],
       [ 0.96862745,  0.84705883,  0.40000001],
       [ 0.83137256,  0.53333336,  0.6156863 ],
       [ 0.03529412,  0.01960784,  0.14509805],
       [ 0.90980393,  0.59607846,  0.78039217],
       [ 0.69803923,  0.87843138,  0.72941178],
       [ 0.20784314,  0.81568629,  0.89411765]];

colors = np.asarray(colors); # defines a color palette 



###  Deep Learning Optimization Parameters ### 

lambd = 0.00001;
step_size_val = .075#.025;

batchsize =200#30#900;
eval_size = batchsize;

use_crelu = 0;
activation_flag = 2; # sets the activation function type to RELU, ELU, SELU (initialized a certain way,dropout has to be done differently) , or tanh() 
max_iters = 500000;#10000#200000 #1000000;
valid_error_threshold = .00001;
test_error_threshold = .00001;

### Deep Learning Metaparameters ###
keep_prob = 1.0; #keep_prob = 1-dropout probability 
res_net = 0;   # Boolean condition on whether to use a resnet connection. 

# # #  PARAMS END # # # 

sess = tf.InteractiveSession();

### HELPER FUNCTIONS # # #

def reg_term(Wlist):
  output = tf.reduce_sum(tf.abs(Wlist[0]));
  for k in range(1,len(Wlist)):
    output += tf.reduce_sum(tf.abs(Wlist[k]));
  return output;

def compute_covarmat(U,Y):
  U = np.asarray(U);
  Y = np.asarray(Y);
  n_inputs = len(U[0]);
  n_outputs = len(Y[0]);
  n_data = len(U);

  Output_Mat = np.zeros((n_outputs,n_inputs),dtype=np.float32);

  for j in range(0,n_inputs):
    for i in range(0,n_outputs):
      Output_Mat[i,j] = compute_covar(Y[:,i],U[:,j]);
  return Output_Mat;

def gen_random_unitary(sq_matrix_dim,sc_factor=1.0):
    n = sq_matrix_dim;
    rand_eig = sc_factor*np.random.rand(n,1);
    Lambda = np.zeros((n,n));
    
    for i in range(0,n):
        Lambda[i][i] = rand_eig[i];
    
    T= np.random.rand(n,n);
    inv_T = np.linalg.inv(T);
    R = np.dot(np.dot(T,Lambda),inv_T);
    return R;

def expose_deep_basis(z_list,num_bas_obs,deep_dict_size,iter_num):
  basis_hooks = z_list[-2]; #[-1] is y  = K *\phi; -2 is \phi(yk)
  x_range = np.arange(-10.0,10.0,0.1);

  for i in range(0,num_bas_obs):
    plt.close();
    scan_injection = np.zeros((len(x_range),num_bas_obs));
    scan_injection[:,i]= np.transpose(x_range);
    phi_j = basis_hooks.eval(feed_dict={u:scan_injection});
    fig_hand = plt.gcf()
    plt.plot(x_range,phi_j,'.-',label='\phi_i(y)');
    plt.ylim([-2.0,2.0]);
    fig = plt.gcf();
    plt.savefig('deep_basis_images/phi_with_u' + repr(i) + '_iternum_' + repr(iter_num) + '.jpg');
    
  return fig_hand;              
  
def compute_covar(x1,x2):
  if (len(x1)!=len(x2)):
    print "Error: compute_covar(x1,x2) requires x1 and x2 to be the same length";
    return Inf;
  else:
    sum_x1x2 = 0.0;
    sum_x1 = 0.0;
    sum_x2 = 0.0;
    for j in range(0,len(x1)):
      sum_x1x2 += x1[j]*x2[j];
      sum_x1 +=x1[j];
      sum_x2 +=x2[j];
    covar = sum_x1x2/(1.0*len(x1))- sum_x1/(1.0*len(x1))*sum_x2/(1.0*len(x1));
    return covar;

def load_pickle_data(file_path,has_control):
        '''load pickle data file for deep Koopman dynamic mode decomposition. 
        Args: 
           file_path: 

        '''     
        file_obj = file(file_path,'rb');
        output_vec = pickle.load(file_obj);

        Yp = output_vec[0]; # list of baseline observables, len(Yp) = (n_samps-1) 
        Yf = output_vec[1]; # list of baseline observables, len(Yf) = (n_samps-1) 

        print "DEBUG:" + repr(len(output_vec));
        if has_control:
          u_control_all_training = output_vec[2];
          #print u_control_all_training[0:10]
        else:
          u_control_all_training = None;
          
        if len(Yp)<2:
            print "Warning: the time-series data provided has no more than 2 points."
    
        Y_whole = [None]*(len(Yp)+1);
        
        for i in range(0,len(Yp)+1):
            if i == len(Yp):
                Y_whole[i] = Yf[i-1];
            else:
                Y_whole[i] = Yp[i];

        Y_whole = np.asarray(Y_whole);
        
        return np.asarray(Yp),np.asarray(Yf),Y_whole,u_control_all_training;
        
      
def xavier_init(n_inputs, n_outputs, uniform=True):
  """Set the parameter initialization using the method described.
  This method is designed to keep the scale of the gradients roughly the same
  in all layers.
  Xavier Glorot and Yoshua Bengio (2010):
           Understanding the difficulty of training deep feedforward neural
           networks. International conference on artificial intelligence and
           statistics.
  Args:
    n_inputs: The number of input nodes into each output.
    n_outputs: The number of output nodes for each input.
    uniform: If true use a uniform distribution, otherwise use a normal.
  Returns:
    An initializer.
  """
  if uniform:
    # 6 was used in the paper.
    init_range = math.sqrt(6.0 / (n_inputs + n_outputs))
    return tf.random_uniform_initializer(-init_range, init_range)
  else:
    # 3 gives us approximately the same limits as above since this repicks
    # values greater than 2 standard deviations from the mean.
    stddev = math.sqrt(3.0 / (n_inputs + n_outputs))
    return tf.truncated_normal_initializer(stddev=stddev)

def weight_variable(shape):
  std_dev = math.sqrt(3.0 /(shape[0] + shape[1]))
  return tf.Variable(tf.truncated_normal(shape, mean=0.0,stddev=std_dev,dtype=tf.float32));
  
def bias_variable(shape):
  std_dev = math.sqrt(3.0 / shape[0])
  return tf.Variable(tf.truncated_normal(shape, mean=0.0,stddev=std_dev,dtype=tf.float32));

def gen_next_yk(input_var,W_list,b_list,keep_prob=1.0,activation_flag=1,res_net=0):
    n_depth = len(W_list);
    z_temp_list = [];
    for k in range(0,n_depth):

        if (k==0):
            W1 = W_list[0];
            b1 = b_list[0];
            if activation_flag==1:# RELU
                z1 = tf.nn.dropout(tf.nn.relu(tf.matmul(input_var,W1)+b1),keep_prob);
            if activation_flag==2: #ELU 
                z1 = tf.nn.dropout(tf.nn.elu(tf.matmul(input_var,W1)+b1),keep_prob);
            if activation_flag==3: # tanh
                z1 = tf.nn.dropout(tf.nn.tanh(tf.matmul(input_var,W1)+b1),keep_prob);

            z_temp_list.append(z1);

        if not (k==0):
            prev_layer_output = tf.matmul(z_temp_list[k-1],W_list[k])+b_list[k]

            if res_net and k==(n_depth-2):
                prev_layer_output += tf.matmul(u,W1)+b1 #  this expression is not compatible for variable width nets (where each layer has a different width at inialization - okay with regularization and dropout afterwards though)

            if activation_flag==1:
                z_temp_list.append(tf.nn.dropout(tf.nn.relu(prev_layer_output),keep_prob));
            if activation_flag==2:
                z_temp_list.append(tf.nn.dropout(tf.nn.elu(prev_layer_output),keep_prob));
            if activation_flag==3:
                z_temp_list.append(tf.nn.dropout(tf.nn.tanh(prev_layer_output),keep_prob));
    #print "[DEBUG] z_list" + repr(z_list[-1]);

    #y_out = tf.concat([z_list[-1],u],axis=1); # last element of activation output list is the actual NN output
    y_out = z_temp_list[-1];

    result = sess.run(tf.initialize_all_variables());
    return y_out;

def initialize_Wblist(n_u,hv_list):
  W_list = [];
  b_list = [];
  n_depth = len(hv_list);
  #hv_list[n_depth-1] = n_y;
  for k in range(0,n_depth):
    if k==0:
      W1 = weight_variable([n_u,hv_list[k]]);
      b1 = bias_variable([hv_list[k]]);
      W_list.append(W1);
      b_list.append(b1);
    else:
      W_list.append(weight_variable([hv_list[k-1],hv_list[k]]));
      b_list.append(bias_variable([hv_list[k]]));
      
  return W_list,b_list;
    
def initialize_tailconstrained_tensorflow_variables(n_u,deep_dict_size,hv_list,W_list,b_list,keep_prob=1.0,activation_flag=1,res_net=0):

  u = tf.placeholder(tf.float32, shape=[None,n_u]); #state/input node,# inputs = dim(input) , None indicates batch size can be any size  
  z_list= [];
  n_depth = len(hv_list);
  #print "[DEBUG] n_depth" + repr(n_depth);
  hv_list[n_depth-2] = deep_dict_size;
  for k in range(0,n_depth):
      if (k==0):
        W1 = W_list[k];
        b1 = b_list[k];
        if activation_flag==1:# RELU
          z1 = tf.nn.dropout(tf.nn.relu(tf.matmul(u,W1)+b1),keep_prob);
        if activation_flag==2: #ELU 
          z1 = tf.nn.dropout(tf.nn.elu(tf.matmul(u,W1)+b1),keep_prob);
        if activation_flag==3: # tanh
          z1 = tf.nn.dropout(tf.nn.tanh(tf.matmul(u,W1)+b1),keep_prob);
            
        z_list.append(z1);
      else:
          W_list.append(weight_variable([hv_list[k-1],hv_list[k]]));
          b_list.append(bias_variable([hv_list[k]]));
          prev_layer_output = tf.matmul(z_list[k-1],W_list[k])+b_list[k]
          print "[DEBUG] prev_layer_output.get_shape() " +repr(prev_layer_output.get_shape());
          if res_net and k==(n_depth-2):
              prev_layer_output += tf.matmul(u,W1)+b1 #  this expression is not compatible for variable width nets (where each layer has a different width at inialization - okay with regularization and dropout afterwards though)              
          if activation_flag==1:
              z_list.append(tf.nn.dropout(tf.nn.relu(prev_layer_output),keep_prob));

          if activation_flag==2:
              z_list.append(tf.nn.dropout(tf.nn.elu(prev_layer_output),keep_prob));

          if activation_flag==3:
              z_list.append(tf.nn.dropout(tf.nn.tanh(prev_layer_output),keep_prob));

  y = tf.concat([u,z_list[-1]],axis=1); # [TODO] in the most general function signature, allow for default option with state/input inclusion 
  result = sess.run(tf.initialize_all_variables());
#  print "[DEBUG] y.get_shape(): " + repr(y.get_shape()) + " y_.get_shape(): " + repr(y_.get_shape());
  return z_list,y,u;#,u_control;

def Deep_Control_Koopman_Objective(psiyf,K,psi_stack,step_size):
   forward_prediction_control = (tf.matmul(psi_stack,K)); 
   tf_koopman_loss = tf.reduce_mean(tf.norm(psiyf - forward_prediction_control,axis=[0,1],ord='fro'));#/tf.reduce_mean(tf.norm(psi_stack,axis=[0,1],ord='fro'));
                                 
   optimizer = tf.train.AdagradOptimizer(step_size).minimize(tf_koopman_loss);
   result = sess.run(tf.initialize_all_variables());
   return tf_koopman_loss,optimizer,forward_prediction_control;
  
def Deep_Direct_Koopman_Objective(psiyp,psiyf,Kx,step_size):
   forward_prediction = tf.matmul(psiyp,Kx)

   #tf_koopman_loss = tf.reduce_mean(tf.nn.l2_loss(y-y_)); 
   tf_koopman_loss = tf.reduce_mean(tf.norm(forward_prediction-psiyf,axis=[0,1],ord='fro') );#/tf.reduce_mean(tf.norm(psiyf,axis=[0,1],ord='fro'));

   optimizer = tf.train.AdagradOptimizer(step_size).minimize(tf_koopman_loss);
   result = sess.run(tf.initialize_all_variables());
   return tf_koopman_loss,optimizer,forward_prediction;

def instantiate_comp_graph(params_list):
  n_outputs = params_list[0];
  deep_dict_size = params_list[1];
  hidden_vars_list = params_list[2];
  Wy_list = params_list[3];
  by_list = params_list[4];
  keep_prob = params_list[5];
  activation_flag = params_list[6];
  res_net = params_list[7];
  psiyzlist, psiy, yfeed = initialize_tailconstrained_tensorflow_variables(n_outputs,deep_dict_size,hidden_vars_list,Wy_list,by_list,keep_prob,activation_flag,res_net);
  return psiyzlist, psiy, yfeed;
 
# consider initializing Wlist, b list first then passing as arguments. 
def train_net(u_all_training,y_all_training,mean_diff_nocovar,optimizer,u_control_all_training=None,valid_error_thres=1e-2,test_error_thres=1e-2,max_iters=100000,step_size_val=0.01):
  iter = 0;
  samplerate = 1000;
  good_start = 1;
  valid_error = 100.0;
  test_error = 100.0;
  training_error_history_nocovar = [];
  validation_error_history_nocovar = [];
  test_error_history_nocovar = [];

  training_error_history_withcovar = [];
  validation_error_history_withcovar = [];
  test_error_history_withcovar = [];

  covar_actual = compute_covarmat(u_all_training,y_all_training);
  covar_model_history = [];
  covar_diff_history = [];
  while (((test_error>test_error_thres) or (valid_error > valid_error_thres)) and iter < max_iters):
    iter+=1;
    all_ind = set(np.arange(0,len(u_all_training)));
    select_ind = np.random.randint(0,len(u_all_training),size=batchsize);
    valid_ind = list(all_ind -set(select_ind))[0:batchsize];
    select_ind_test = list(all_ind - set(valid_ind) - set(select_ind))[0:batchsize];

    
    u_batch =[];
    u_control_batch = [];
    y_batch = [];
    u_valid = [];
    u_control_valid = [];
    y_valid = [];
    u_test_train = [];
    u_control_train = [];
    y_test_train= [];
    u_control_test_train = [];
    for j in range(0,len(select_ind)):
      u_batch.append(u_all_training[select_ind[j]]);
      y_batch.append(y_all_training[select_ind[j]]);
      if with_control:
          u_control_batch.append(u_control_all_training[select_ind[j]]);
          
    for k in range(0,len(valid_ind)):
      u_valid.append(u_all_training[valid_ind[k]]);
      y_valid.append(y_all_training[valid_ind[k]]);
      if with_control:
          u_control_valid.append(u_control_all_training[valid_ind[k]]);

    for k in range(0,len(select_ind_test)):
      u_test_train.append(u_all_training[select_ind_test[k]]);
      y_test_train.append(y_all_training[select_ind_test[k]]);
      if with_control:
          u_control_test_train.append(u_control_all_training[select_ind_test[k]]);

    
    

        


    if with_control:
      optimizer.run(feed_dict={yp_feed:u_batch,yf_feed:y_batch,u_control:u_control_batch,step_size:step_size_val});
      valid_error = mean_diff_nocovar.eval(feed_dict={yp_feed:u_valid,yf_feed:y_valid,u_control:u_control_valid});
      test_error = mean_diff_nocovar.eval(feed_dict={yp_feed:u_test_train,yf_feed:y_test_train,u_control:u_control_test_train});

    else:
      optimizer.run(feed_dict={yp_feed:u_batch,yf_feed:y_batch,step_size:step_size_val});
      valid_error = mean_diff_nocovar.eval(feed_dict={yp_feed:u_valid,yf_feed:y_valid});
      test_error = mean_diff_nocovar.eval(feed_dict={yp_feed:u_test_train,yf_feed:y_test_train});


    
    #if frobenius_norm.eval(feed_dict={u:u_batch,y_:y_batch}) < 0.001:
    #  print "It took " + repr(iter) + " iterations to reduce covariate error to 0.1%!"
      
    if iter%samplerate==0:
      if with_control:
        training_error_history_nocovar.append(mean_diff_nocovar.eval(feed_dict={yp_feed:u_batch,yf_feed:y_batch,u_control:u_control_batch}));
        validation_error_history_nocovar.append(mean_diff_nocovar.eval(feed_dict={yp_feed:u_valid,yf_feed:y_valid,u_control:u_control_valid}));
        test_error_history_nocovar.append(mean_diff_nocovar.eval(feed_dict={yp_feed:u_test_train,yf_feed:y_test_train,u_control:u_control_test_train}));
      else:
        training_error_history_nocovar.append(mean_diff_nocovar.eval(feed_dict={yp_feed:u_batch,yf_feed:y_batch}));
        validation_error_history_nocovar.append(mean_diff_nocovar.eval(feed_dict={yp_feed:u_valid,yf_feed:y_valid}));
        test_error_history_nocovar.append(mean_diff_nocovar.eval(feed_dict={yp_feed:u_test_train,yf_feed:y_test_train}));
        
      if (iter%1000==0) or (iter==1):
        plt.close();
        if plot_deep_basis:
          fig_hand = expose_deep_basis(psiypz_list,num_bas_obs,deep_dict_size,iter);

        if with_control:  
          print ("step %d , validation error %g"%(iter, mean_diff_nocovar.eval(feed_dict={yp_feed:u_valid,yf_feed:y_valid,u_control:u_control_valid})));
          print ("step %d , test error %g"%(iter, mean_diff_nocovar.eval(feed_dict={yp_feed:u_test_train,yf_feed:y_test_train,u_control:u_control_test_train})));
        else:
          print ("step %d , validation error %g"%(iter, mean_diff_nocovar.eval(feed_dict={yp_feed:u_valid,yf_feed:y_valid})));
          print ("step %d , test error %g"%(iter, mean_diff_nocovar.eval(feed_dict={yp_feed:u_test_train,yf_feed:y_test_train})));
          
    if ((iter>10000) and iter%10) :

      valid_gradient = np.gradient(np.asarray(validation_error_history_nocovar[iter/samplerate*7/10:]));
      mu_gradient = np.mean(valid_gradient);

      if ((iter <1000) and (mu_gradient >= 5e-1)): # eventually update this to be 1/10th the mean of batch data, or mean of all data handed as input param to func
        good_start = 0; # if after 10,000 iterations validation error is still above 1e0, initialization was poor.
        print "Terminating model refinement loop with gradient:" + repr(mu_gradient) + ", validation error after " + repr(iter) + " epochs:  " + repr(valid_error);
        iter = max_iters; # terminate while loop and return histories
  all_histories = [training_error_history_nocovar, validation_error_history_nocovar,test_error_history_nocovar, \
                   training_error_history_withcovar,validation_error_history_withcovar,test_error_history_withcovar,covar_actual,covar_diff_history,covar_model_history];
  plt.close();
  x = np.arange(0,len(validation_error_history_nocovar),1);
  plt.plot(x,training_error_history_nocovar,label='train. err.');
  plt.plot(x,validation_error_history_nocovar,label='valid. err.');
  plt.plot(x,test_error_history_nocovar,label='test err.');
  plt.gca().set_yscale('log');
  plt.savefig('all_error_history.pdf');

  plt.close();
  return all_histories,good_start;

# # # END HELPER FUNCTIONS # # #


# # # - - - Begin Script - - - # # #


pre_examples_switch = 7; 

### Randomly generated oscillator system with control

data_directory = 'koopman_data/'
if pre_examples_switch == 0:
  data_suffix = 'rand_osc.pickle';  
  with_control = 1; 

if pre_examples_switch == 1:
  data_suffix = 'cons_law.pickle';
  with_control = 0;

if pre_examples_switch == 2:
  data_suffix = '50bus.pickle';
  with_control = 1;

if pre_examples_switch == 3:
  data_suffix = 'zhang_control.pickle';
  with_control = 1;

if pre_examples_switch == 4:
  data_suffix = 'deltaomega-series.pickle'; ### IEEE 39 bus swing model 
  with_control = 1; 

if pre_examples_switch == 5:
  data_suffix = 'glycol.pickle';
  with_control = 0; 

if pre_examples_switch == 6:
  data_suffix = 'exp_toggle_switch.pickle';
  with_control = 1; 

if pre_examples_switch == 7:
  data_suffix = 'exp_toggle_switch_M9CA.pickle';
  with_control = 1;
  
  
data_file = data_directory + data_suffix;

if with_control:
  Yp,Yf,Y_whole,u_control_all_training = load_pickle_data(data_file,with_control)
else:
  Yp,Yf,Y_whole,temp_var = load_pickle_data(data_file,with_control);
    
print "[INFO] Number of total samples: " + repr(len(Yp));
print "[INFO] Observable dimension of a sample: " + repr(len(Yp[0]));
num_bas_obs = len(Yp[0]);
num_all_samples = len(Yp);

Y_p_old = Yp;
Y_f_old = Yf;

if with_control:
  u_control_all_training_old = u_control_all_training ;
rand_indices = np.arange(0,len(Yp),1).tolist();
    
for i in range(0,len(rand_indices) ):
    curr_index = rand_indices[i];
    Yp[i] = Y_p_old[curr_index];
    Yf[i] = Y_f_old[curr_index];
    if with_control:
      u_control_all_training[i] = u_control_all_training_old[curr_index];                          

print "[INFO] Yp.shape (E-DMD): " + repr(Yp.shape);
print "[INFO] Yf.shape (E-DMD): " + repr(Yf.shape);

#import sklearn.model_selection import train_test_split
train_range = len(Yp)*5/10; # define upper limits of training data 
test_range = len(Yp); # define upper limits of test data 
#Yp_test = Yp[0:test_range];
#Yf_test = Yf[0:test_range];
Yp_test_old = Yp[train_range:test_range];
Yf_test_old = Yf[train_range:test_range];
Yp_train_old = Yp[0:train_range];
Yf_train_old = Yf[0:train_range];

num_trains = len(Yp)*9/10;
train_indices = np.random.randint(0,len(Yp),num_trains)
test_indices = np.random.randint(0,len(Yp),len(Yp)-num_trains); 
Yp_train = Yp[train_indices];
Yf_train = Yf[train_indices];
Yp_test = Yp[test_indices];
Yf_test = Yf[test_indices]; 


if with_control:
  print "[INFO]: u_control_all_training.shape post-loading: " + repr(len(u_control_all_training));
  U_train = u_control_all_training[train_indices];
  U_test  = u_control_all_training[test_indices];
  
  #U_test_old = u_control_all_training[train_range:test_range];
  #U_train_old = u_control_all_training[0:train_range];
  #U_train_old = np.asarray(U_train);
  #U_test_old = np.asarray(U_test);
  if len(U_test.shape)==1:
    U_train = np.reshape(U_train,(U_train.shape[0],1));
    U_test = np.reshape(U_test,(U_test.shape[0],1));
                                                
  print "[INFO] : U_train.shape: " + repr(U_train.shape);
else:
  U_train = None;
  U_test = None;
Yp_train = np.asarray(Yp_train);
Yf_train = np.asarray(Yf_train);
Yp_test = np.asarray(Yp_test);
Yf_test = np.asarray(Yf_test);

print "[INFO] Yp_test.shape (E-DMD) " + repr(Yp_test.shape);
print "[INFO] Yf_test.shape (E-DMD) " + repr(Yf_test.shape);
if with_control:
  print "[INFO] U_train.shape (E-DMD) " + repr(U_train.shape);

Yp_final_train = Yp_train; 
Yf_final_train = Yf_train; 
Yp_final_test = Yp_test; 
Yf_final_test = Yf_test; 

up_all_training = Yp_final_train;
uf_all_training = Yf_final_train;

print "[INFO] up_all_training.shape: " + repr(up_all_training.shape);

deep_dict_size = 20;
n_outputs =num_bas_obs;
n_inputs = num_bas_obs;

if with_control:
  deep_dict_size_control = 6;
  

if with_control:
  #print "[INFO TYPE]" + repr(type(u_control_all_training_old[0]));
  if type(u_control_all_training_old[0])==np.ndarray:
    print "[DEBUG]"  + repr(u_control_all_training_old[0]);
    n_inputs_control =u_control_all_training_old[0].shape[0];
    
  else:
    n_inputs_control = 1;
else:
  n_inputs_control = 0;


### Define Neural Network Sweep Parameters: 
  
max_depth = 7;
max_width_limit = 40;

min_width_limit = max_width_limit;# use regularization and dropout to trim edges for now. 
min_width_limit_control = 10;
max_depth_control = 3;


best_test_error = 1000.0;
best_depth = max_depth;
best_width = min_width_limit;

### End Neural Network Sweep Parameters




### Begin Sweep 

for n_depth_reciprocal in range(1,2):#max_depth-2): #2
  n_depth = max_depth+1 - n_depth_reciprocal;
  n_depth_control = max_depth_control + 1 - n_depth_reciprocal;   
  for min_width_conjugate in range(1,2):#min_width_limit): #2
      min_width= min_width_limit+1-min_width_conjugate;
      max_width = min_width; #zero gap between min and max ; use regularization and dropout to trim edges. 
      min_width_control = min_width_limit_control+1-min_width_conjugate;

      
      if min_width==max_width:
        hidden_vars_list = np.asarray([min_width]*n_depth);
      else:
        hidden_vars_list = np.random.randint(min_width,max_width,size=n_depth);

      if with_control:  
        hidden_vars_list_control = np.asarray([min_width_control]*n_depth_control);
        hidden_vars_list_control[-1] = deep_dict_size_control;

      good_start = 0;
      max_tries = 5;
      try_num = 0;
      
      # # # - - - enforce Koopman last layer # # #
      
      hidden_vars_list[-1] = deep_dict_size;
      
      print "[INFO] hidden_vars_list: " +repr(hidden_vars_list);
      while good_start==0 and try_num < max_tries:
          try_num +=1;
          if debug_splash:
            print "\n Initialization attempt number: " + repr(try_num);
            print "\n \t Initializing Tensorflow Residual ELU Network with " + repr(n_inputs) + " inputs and " + repr(n_outputs) + " outputs and " + repr(len(hidden_vars_list)) + " layers";

          with tf.device('/cpu:0'):
            Wy_list,by_list = initialize_Wblist(n_inputs,hidden_vars_list);
            params_list = [ n_outputs, deep_dict_size, hidden_vars_list,Wy_list,by_list,keep_prob,activation_flag, res_net ]
            
            psiypz_list,psiyp,yp_feed =  instantiate_comp_graph(params_list); 
            psiyfz_list,psiyf,yf_feed = instantiate_comp_graph(params_list);
            
            if with_control:
              Wu_list,bu_list = initialize_Wblist(n_inputs_control,hidden_vars_list_control);
              
              
              psiuz_list, psiu,u_control = initialize_tailconstrained_tensorflow_variables(n_inputs_control,deep_dict_size_control,hidden_vars_list_control,Wu_list,bu_list,keep_prob,activation_flag,res_net);
            
            # add hooks for affine control perturbation to Deep_Direct_Koopman_Objective
            step_size = tf.placeholder(tf.float32,shape=[]);
            Kx = weight_variable([deep_dict_size+n_outputs+deep_dict_size_control+n_inputs_control,deep_dict_size+n_outputs]);
            
            if with_control:
              #Ku = weight_variable([deep_dict_size+n_outputs+deep_dict_size_control+n_inputs_control,deep_dict_size+n_outputs]);  # [NOTE: generalize to vary deep_dict_size (first dim, num of lifted inputs)

              psistack = tf.concat([psiyp,psiu],axis=1); 
              deep_koopman_loss,optimizer,forward_prediction_control = Deep_Control_Koopman_Objective(psiyf,Kx,psistack,step_size);    
            else:
              deep_koopman_loss,optimizer,forward_prediction = Deep_Direct_Koopman_Objective(psiyp,psiyf,Kx,step_size);
              
            
            if debug_splash:
              train_vars = tf.trainable_variables();
              values = sess.run([x.name for x in train_vars]);
              print "[DEBUG] # of Trainable Variables: " + repr(len(values));
              print "[DEBUG] Trainable Variables: " + repr([ temp_var.shape for temp_var in values]);

              print "[DEBUG] # of datapoints in up_all_training: " + repr(up_all_training.shape);
              print "[DEBUG] # of datapoints in uf_all_training: " + repr(uf_all_training.shape);


            
            all_histories,good_start  = train_net(up_all_training,uf_all_training,deep_koopman_loss,optimizer,U_train,valid_error_threshold,test_error_threshold,max_iters,step_size_val);
            all_histories,good_start  = train_net(up_all_training,uf_all_training,deep_koopman_loss,optimizer,U_train,valid_error_threshold*.1,test_error_threshold*.1,max_iters,step_size_val/10);
            #all_histories,good_start  = train_net(up_all_training,uf_all_training,deep_koopman_loss,optimizer,U_train,valid_error_threshold*.025,test_error_threshold*.025,max_iters,step_size_val/100);

          training_error_history_nocovar = all_histories[0];
          validation_error_history_nocovar =   all_histories[1];
          test_error_history_nocovar = all_histories[2];
          training_error_history_withcovar  = all_histories[3];
          validation_error_history_withcovar = all_histories[4];
          test_error_history_withcovar = all_histories[5];
          print "[INFO] Initialization was successful: " + repr(good_start==1);
          
          accuracy = deep_koopman_loss;#;
          if with_control:
            train_accuracy = accuracy.eval(feed_dict={yp_feed:up_all_training,yf_feed:uf_all_training,u_control:U_train});
            test_accuracy = accuracy.eval(feed_dict={yp_feed:Yp_test,yf_feed:Yf_test,u_control:U_test});
      
          else:
            train_accuracy = accuracy.eval(feed_dict={yp_feed:up_all_training,yf_feed:uf_all_training});
            test_accuracy = accuracy.eval(feed_dict={yp_feed:Yp_test,yf_feed:Yf_test});
      
          if test_accuracy <= best_test_error:
              best_test_error = test_accuracy;
              best_depth = n_depth;
              best_width = min_width;

              if debug_splash:
                print "[DEBUG]: Regularization penalty: " + repr(sess.run(reg_term(Wy_list)));
              np.set_printoptions(precision=2,suppress=True);
              if debug_splash:
                print "[DEBUG]: " + repr(np.asarray(sess.run(Wy_list[0]).tolist()));
        
          print "[Result]: Training Error: ";
          print(train_accuracy);
          print "[Result]: Test Error : ";
          print(test_accuracy);
          
### Write Vars to Checkpoint Files/MetaFiles 
          
Kx_num = sess.run(Kx);
file_obj_swing = file('constrainedNN-Model.pickle','wb');
Wy_list_num = [sess.run(W_temp) for W_temp in Wy_list];
by_list_num = [sess.run(b_temp) for b_temp in by_list];

if with_control:
  Wu_list_num = [sess.run(W_temp) for W_temp in Wu_list];
  bu_list_num = [sess.run(b_temp) for b_temp in bu_list];
  Ku_num = None;# sess.run(Ku);
  pickle.dump([Wy_list_num,by_list_num,Wu_list_num,bu_list_num,Kx_num,Ku_num],file_obj_swing);
else:
    pickle.dump([Wy_list_num,by_list_num,Kx_num],file_obj_swing);
file_obj_swing.close();


saver = tf.train.Saver()

tf.add_to_collection('psiyp',psiyp);
tf.add_to_collection('psiyf',psiyf);
tf.add_to_collection('Kx',Kx);


if with_control:
  tf.add_to_collection('forward_prediction_control',forward_prediction_control);
  tf.add_to_collection('psiu',psiu);
  tf.add_to_collection('u_control',u_control);
  #tf.add_to_collection('Ku',);
else:
  tf.add_to_collection('forward_prediction',forward_prediction);

tf.add_to_collection('yp_feed',yp_feed);
tf.add_to_collection('yf_feed',yf_feed);





save_path = saver.save(sess, data_suffix + '.ckpt')

Koopman_dim = Kx_num.shape[0];
print "[INFO] Koopman_dim:" + repr(Kx_num.shape);

if single_series:
    #Y_p_old,Y_f_old,Y_whole,u_control_all_training = load_pickle_data('koopman_data/zhang_control.pickle');#deltaomega-series.pickle'
  Y_p_old,Y_f_old,Y_whole,u_control_all_training = load_pickle_data('koopman_data/deltaomega-singleseries.pickle'); 


#K = sess.run(K);
if not( Kx_num.shape[1]==Kx_num.shape[0]):
    print "Warning! Estimated Koopman operator is not square with dimensions : " + repr(Kx_num.shape);

train_range = len(Y_p_old)/2; # define upper limits of training data 
print "[DEBUG] train_range: " + repr(train_range);
print "[DEBUG] test_range: " + repr(test_range);
print "Y_p_old.shape" + repr(Y_p_old.shape);
print "Y_f_old.shape" + repr(Y_f_old.shape);


test_range = len(Y_p_old); # define upper limits of test data 
#Yp_test = Yp[0:test_range];
#Yf_test = Yf[0:test_range];
Yp_test = Y_p_old[train_range:test_range];
Yf_test = Y_f_old[train_range:test_range];

if with_control:
  U_test = u_control_all_training_old[train_range:test_range];
  U_train = u_control_all_training_old[0:train_range];
  U_train = np.asarray(U_train);
  U_test = np.asarray(U_test);
  if len(U_test.shape)==1:
    U_train = np.reshape(U_train,(U_train.shape[0],1));
    U_test = np.reshape(U_test,(U_test.shape[0],1));
  

Yp_train = Y_p_old[0:train_range];
Yf_train = Y_f_old[0:train_range];


Yp_train = np.asarray(Yp_train);
Yf_train = np.asarray(Yf_train);
Yp_test = np.asarray(Yp_test);
Yf_test = np.asarray(Yf_test);
Yp_final_test = Yp_test;
Yf_final_test = Yf_test;
Yp_final_train = Yp_train;    
Yf_final_train = Yf_train;
# # # Print Evaluation Metrics -  Deep Koopman Learning # # #

print "[DEBUG]: Yp_train.shape" + repr(Yp_train.shape);

if with_control:
  training_error = accuracy.eval(feed_dict={yp_feed:list(Yp_train),yf_feed:list(Yf_train),u_control:list(U_train)});
  test_error = accuracy.eval(feed_dict={yp_feed:list(Yp_test),yf_feed:list(Yf_test),u_control:list(U_test)});

else:
  training_error = accuracy.eval(feed_dict={yp_feed:list(Yp_train),yf_feed:list(Yf_train)});
  test_error = accuracy.eval(feed_dict={yp_feed:list(Yp_test),yf_feed:list(Yf_test)});
print('%s%f' % ('[COMP] Training error: ',training_error));
print('%s%f' % ('[COMP] Test error: ',test_error));

# # # - - - n-step Prediction Error Analysis - - - # # # 

print "[DEBUG] Yf_final_test.shape: " + repr( Yf_final_test.shape);

n_points_pred = 18;#Yf_final_test.shape[0]-1;


print "[INFO] Yp_final_test.shape: " + repr(Yp_final_test.shape);

Yp_final_test = np.transpose(Yp_final_test);
Ycurr = Yp_final_test[:,0]; # grab one time point from the final prediction of the training data

Ycurr = np.transpose(Ycurr);
Ycurr = Ycurr.reshape(1,num_bas_obs);

psiyp_Ycurr = psiyp.eval(feed_dict={yp_feed:Ycurr});
psiyf_Ycurr = psiyf.eval(feed_dict={yf_feed:Ycurr});

### DEBUG ###
if debug_splash:
  print "[DEBUG] Ycurr.shape: " + repr(Ycurr.shape);
  print "[DEBUG]:" + " Equality check between psiyp and psiyf: " + repr( psiyp_Ycurr-psiyf_Ycurr);
  print "[DEBUG]: " + repr(psiyp_Ycurr.shape)
  print "[DEBUG]: " + "Equality check between psiyp first 7 elems and Ycurr: " + repr(psiyp_Ycurr[:,0:num_bas_obs]-Ycurr);
if with_control:
  print "[DEBUG] U_test.shape" + repr(U_test.shape);
  
Yf_final_test_ep_nn = [];
Yf_final_test_ep_nn.append(psiyp_Ycurr.tolist()[0][0:num_bas_obs]); # append the initial seed state value.

for i in range(0,n_points_pred):
  if with_control:
    if len(U_test[i,:])==1:
      U_temp_mat = np.reshape(U_test[i,:],(1,1));
      psiyp_Ycurr = sess.run(forward_prediction_control, feed_dict={yp_feed:psiyp_Ycurr[:,0:num_bas_obs],u_control:U_temp_mat});#
    else:
      U_temp_mat = np.reshape(U_test[i,:],(1,n_inputs_control));
      psiyp_Ycurr = sess.run(forward_prediction_control, feed_dict={yp_feed:psiyp_Ycurr[:,0:num_bas_obs],u_control:U_temp_mat});# 
  else:
    psiyp_Ycurr = sess.run(forward_prediction,feed_dict={yp_feed:psiyp_Ycurr[:,0:num_bas_obs]});

  Yout = psiyp_Ycurr.tolist()[0][0:num_bas_obs];
  Yf_final_test_ep_nn.append(Yout);


Yf_final_test_ep_nn = np.asarray(Yf_final_test_ep_nn);
Yf_final_test_ep_nn = np.transpose(Yf_final_test_ep_nn);
Yf_final_test = np.transpose(Yf_final_test);


print "[INFO] Yf_final_test_ep_nn.shape: " + repr(Yf_final_test_ep_nn.shape);
print "[INFO] Yf_final_test.shape: " + repr(Yf_final_test.shape);
print "[INFO] Yp_final_test.shape: " + repr(Yp_final_test.shape);

Yf_final_test_stack_nn = np.zeros((Yf_final_test_ep_nn.shape[0],Yf_final_test_ep_nn.shape[1]));

Yf_final_test_stack_nn[:,0] = Yp_final_test[:,0]
for i in range(0,Yf_final_test_ep_nn.shape[1]-1):
        Yf_final_test_stack_nn[:,i+1] = Yf_final_test[:,i];

# DEBUG Statements 

if debug_splash:
  print "[DEBUG] Yf_final_test_stack_nn: " +repr(Yf_final_test_stack_nn);
  print "[DEBUG] Yf_final_test_ep_nn:" +  repr(Yf_final_test_ep_nn);
  print "[DEBUG] Yf_final_test_stack_nn - Yf_final_test_ep_nn : " + repr(Yf_final_test_stack_nn - Yf_final_test_ep_nn);
  print "[DEBUG] Y_final_test_stack.shape: " + repr(Yf_final_test_stack_nn.shape);
  print "[INFO] Ground truth Yf_final_test_ep_stack.shape: "+ repr( Yf_final_test_stack_nn.shape);
  print "[INFO] Prediction Yf_final_test_ep_nn.shape: "+ repr(Yf_final_test_ep_nn.shape);
  print "Denominator of prediction error: " + repr(np.linalg.norm(Yf_final_test_stack_nn,ord='fro'));
  

### 

prediction_error = np.linalg.norm(Yf_final_test_stack_nn-Yf_final_test_ep_nn,ord='fro')/np.linalg.norm(Yf_final_test_stack_nn,ord='fro');
print('%s%f' % ('[RESULT] n-step Prediction error: ',prediction_error));

x_range = np.arange(0,Yf_final_test_stack_nn.shape[1],1);
for i in range(0,3):
    plt.plot(x_range,Yf_final_test_ep_nn[i,:],'--',color=colors[i,:]);
    plt.plot(x_range,Yf_final_test_stack_nn[i,:],'*',color=colors[i,:]);
axes = plt.gca();
plt.legend(loc='best');
plt.xlabel('t');
fig = plt.gcf();

target_file = data_suffix.replace('.pickle','')+'nstep_prediction.pdf';
plt.savefig(target_file);
plt.show();


