#! /usr/bin/env python
import tensorflow as tf;
import numpy as np; 
import math;
import matplotlib.pyplot as plt;
sess = tf.InteractiveSession();

def train_net(u_all_training,y_all_training,mean_diff_nocovar,optimizer,batchsize,with_control,u,y,y_,step_size,valid_error_thres=1e-2,test_error_thres=1e-2,max_iters=100000,step_size_val=0.01):
  iter = 0;
  samplerate = 1;
  good_start = 1;
  valid_error = 100.0;
  test_error = 100.0;
  training_error_history_nocovar = [];
  validation_error_history_nocovar = [];
  test_error_history_nocovar = [];

  training_error_history_withcovar = [];
  validation_error_history_withcovar = [];
  test_error_history_withcovar = [];

  #covar_actual = compute_covarmat(u_all_training,y_all_training);
  #covar_model_history = [];
  #covar_diff_history = [];
  while (((test_error>test_error_thres) or (valid_error > valid_error_thres)) and iter < max_iters):
    iter+=1;
    select_ind = np.random.randint(0,len(u_all_training),size=batchsize);
    valid_ind = list(set(np.arange(0,len(u_all_training)))-set(select_ind) )[0:batchsize];
    select_ind_test = list(set(np.arange(0,len(u_all_training))) - set(select_ind)-set(valid_ind))[0:batchsize];
    
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
    for j in range(0,batchsize):
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
        if with_control:
          u_control_test_train.append(u_control_all_training[select_ind_test[k]]);
        y_test_train.append(y_all_training[select_ind_test[k]]);


    if with_control:
      optimizer.run(feed_dict={u:u_batch,y_:y_batch,u_control:u_control_batch,step_size:step_size_val});
      valid_error = mean_diff_nocovar.eval(feed_dict={u:u_valid,y_:y_valid,u_control:u_control_valid});
      test_error = mean_diff_nocovar.eval(feed_dict={u:u_test_train,y_:y_test_train,u_control:u_control_test_train});

    else:
      #print "u_batch[0].shape: " + repr(u_batch[0].shape);
      #print yf_feed;
      #print yp_feed;
      #print y_batch;

      #print "y_batch[0].shape: " + repr(y_batch[0].shape);
      optimizer.run(feed_dict={u:u_batch,y_:y_batch,step_size:step_size_val});
      valid_error = mean_diff_nocovar.eval(feed_dict={u:u_valid,y_:y_valid});
      #print valid_error
      test_error = mean_diff_nocovar.eval(feed_dict={u:u_test_train,y_:y_test_train});
      #print test_error


    
    #if frobenius_norm.eval(feed_dict={u:u_batch,y_:y_batch}) < 0.001:
    #  print "It took " + repr(iter) + " iterations to reduce covariate error to 0.1%!"
      
    if iter%samplerate==0:
      if with_control:
        training_error_history_nocovar.append(mean_diff_nocovar.eval(feed_dict={u:u_batch,y_:y_batch,u_control:u_control_batch}));
        validation_error_history_nocovar.append(mean_diff_nocovar.eval(feed_dict={u:u_valid,y_:y_valid,u_control:u_control_valid}));
        test_error_history_nocovar.append(mean_diff_nocovar.eval(feed_dict={u:u_test_train,y_:y_test_train,u_control:u_control_test_train}));
      else:
        training_error_history_nocovar.append(mean_diff_nocovar.eval(feed_dict={u:u_batch,y_:y_batch}));
        validation_error_history_nocovar.append(mean_diff_nocovar.eval(feed_dict={u:u_valid,y_:y_valid}));
        test_error_history_nocovar.append(mean_diff_nocovar.eval(feed_dict={u:u_test_train,y_:y_test_train}));
      
      if (iter%100==0) or (iter==1):
        plt.close();
        #if plot_deep_basis:
        #  fig_hand = expose_deep_basis(psiypz_list,num_bas_obs,deep_dict_size,iter);

        if with_control:  
          print ("step %d , validation error %g"%(iter, mean_diff_nocovar.eval(feed_dict={yp_feed:u_valid,yf_feed:y_valid,u_control:u_control_valid})));
          print ("step %d , test error %g"%(iter, mean_diff_nocovar.eval(feed_dict={yp_feed:u_test_train,yf_feed:y_test_train,u_control:u_control_test_train})));
        else:
          print ("step %d , validation error %g"%(iter, mean_diff_nocovar.eval(feed_dict={u:u_valid,y_:y_valid})));
          print ("step %d , test error %g"%(iter, mean_diff_nocovar.eval(feed_dict={u:u_test_train,y_:y_test_train})));
        
    if ((iter>100) and iter%10) :

      valid_gradient = np.gradient(np.asarray(validation_error_history_nocovar[iter/samplerate*7/10:]));
      mu_gradient = np.mean(valid_gradient);

      if ((iter <1000) and (mu_gradient >= 5e-1)): #5e-1 # eventually update this to be 1/10th the mean of batch data, or mean of all data handed as input param to func
        good_start = 0; # if after 10,000 iterations validation error is still above 1e0, initialization was poor.
        print "Terminating model refinement loop with gradient:" + repr(mu_gradient) + ", validation error after " + repr(iter) + " epochs:  " + repr(valid_error);
        iter = max_iters; # terminate while loop and return histories
  all_histories = [training_error_history_nocovar, validation_error_history_nocovar,test_error_history_nocovar];

  #plt.close();
  #x = np.arange(0,len(validation_error_history_nocovar),1);
  #plt.plot(x,training_error_history_nocovar,label='train. err.');
  #plt.plot(x,validation_error_history_nocovar,label='valid. err.');
  #plt.plot(x,test_error_history_nocovar,label='test err.');
  #plt.gca().set_yscale('log');
  #plt.savefig('all_error_history.pdf');

  #plt.close();

  
  return all_histories,good_start;

def gen_pdf(dataframe,bin_count=5):
    FL1_range = (281.0,8798.0);
    FSC_range = (582897.2,1797672.1)
    SSC_range = (36772.0,337255.1)
    range_all = [np.log10(FL1_range),np.log10(FSC_range),np.log10(SSC_range)];
    temp_array = np.histogramdd(dataframe,bins=bin_count);
    total_count = np.sum(temp_array[0],axis=None);
    output = np.asarray(temp_array[0]/total_count);
    #print temp_array[0]/total_count;
    #print np.sum(temp_array[0]/total_count);
    
    return output;

def compute_stats(matrix):
                size = matrix.shape[1]
                return [ (np.mean(matrix[:,i]), np.std(matrix[:,i]))for i in range(size) ]
 
def noised(matrix, stats):
                return np.asarray([[row[i] + np.random.normal(0.0, 0.1*stats[i][1]) for i in range(len(row))] for row in matrix])
 




### TENSORLFOW support functions ###

def weight_variable(shape):
  std_dev = math.sqrt(3.0 /(shape[0] + shape[1]))
  return tf.Variable(tf.truncated_normal(shape, mean=0.02,stddev=std_dev,dtype=tf.float32));
  
def bias_variable(shape):
  std_dev = math.sqrt(3.0 / shape[0])
  return tf.Variable(tf.truncated_normal(shape, mean=0.01,stddev=std_dev,dtype=tf.float32));


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



def initialize_tailconstrained_tensorflow_variables(n_channels,n_u,hv_list,W_list,b_list,keep_prob=1.0,activation_flag=1,res_net=0):

  #n_u is the number of bins used during hist calculation. 
  shape_vec = [n_u]*(n_channels+1); # [None,5,5,5]
  shape_vec[0] = None;
  print "shape of u:" + repr(shape_vec)
  u_handle = tf.placeholder(tf.float32, shape=shape_vec); #state/input node,# inputs = dim(input) , None indicates batch size can be any size  
  z_list= [];
  n_depth = len(hv_list);
  #print "[DEBUG] n_depth" + repr(n_depth);


  
  for k in range(0,n_depth):
      if (k==0):
        W1 = W_list[k];
        b1 = b_list[k];
        
        if n_channels > 1: 
            shape_vec = [n_u]*(n_channels); 
            shape_vec[0] = None;
            temp_shape_vec = shape_vec; 
            num_times_to_map = n_channels-1; 
            squash_weight_matrices = [];
            for map_ind in range(0,num_times_to_map):
                temp_shape_vec = [n_u]*(len(shape_vec)-(map_ind));
                if len(temp_shape_vec)>1:
                    temp_shape_vec[-1] = 1; 
                print temp_shape_vec
                squash_weight_matrices.append(weight_variable(temp_shape_vec)); 
                if map_ind == 0:
                    next_temp_var_squash = tf.map_fn(lambda x1: tf.squeeze(tf.matmul(x1,squash_weight_matrices[map_ind]) ), u_handle ),
                else: 
                    next_temp_var_squash = tf.map_fn(lambda x1: tf.squeeze(tf.matmul(x1,squash_weight_matrices[map_ind] ) ) , prev_temp_var_squash ),
                prev_temp_var_squash = next_temp_var_squash; 
                #print prev_temp_var_squash.get_shape();
            temp_var_squash = next_temp_var_squash; 
            temp_var= tf.matmul(tf.squeeze(temp_var_squash),W1) + b1; 
        #u = tf.reshape(u,(5,5,5,tf.shape(u)[0]));

        #       tf.scan(lambda a, x: tf.squeeze(tf.matmul(W_left, tf.matmul(x, W_right))), stability_pred, initializer=0.)
        #temp_var_squash1 = tf.map_fn(lambda x1:tf.squeeze(tf.matmul(x1,W_squash)),u_handle);
        #print temp_var_squash1.get_shape();

        #temp_var_squash2 = tf.map_fn(lambda x2:tf.squeeze(tf.matmul(x2,W_squash2)),temp_var_squash1);
        #temp_var_squash1 = tf.scan(lambda a1,x1 : tf.matmul(x1,W_squash),u,initializer=0)        
        #temp_var_squash2 = tf.scan(lambda a2,x2: tf.squeeze(tf.matmul(x2,W1)),temp_var_squash1,initializer=0) #+ b1;
        #temp_var = tf.matmul(temp_var_squash2,W1)+b1;


        if activation_flag==1:# RELU
          z1 = tf.nn.dropout(tf.nn.relu(temp_var),keep_prob);
        if activation_flag==2: #ELU 
          z1 = tf.nn.dropout(tf.nn.elu(temp_var),keep_prob);
        if activation_flag==3: # tanh
          z1 = tf.nn.dropout(tf.nn.tanh(temp_var),keep_prob);           
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

  y = z_list[-1];#tf.concat([u,z_list[-1]],axis=1); # [TODO] in the most general function signature, allow for default option with state/input inclusion 
  result = sess.run(tf.initialize_all_variables());
#  print "[DEBUG] y.get_shape(): " + repr(y.get_shape()) + " y_.get_shape(): " + repr(y_.get_shape());
  return z_list,y,u_handle;#,u_control;








def initialize_dictionaryNN(n_u,deep_dict_size,hv_list,W_list,b_list,keep_prob=1.0,activation_flag=1,res_net=0):
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

  

