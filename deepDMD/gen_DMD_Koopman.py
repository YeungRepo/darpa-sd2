
#! /usr/bin/env python

import pickle;
import numpy as np;
from numpy.linalg import pinv;
from numpy.polynomial.legendre import legvander;
import cvxopt;
from cvxpy import Minimize, Problem, Variable,norm1,norm2,installed_solvers,lambda_max;
from cvxpy import norm as cvxpynorm;
import cvxpy;
# # #  Koopman optimization parameters # #  # 

poly_deg=2;
continuous_predict=0;
use_solver = 1;#1 is Least Squares with Moore-Penrose or PseudoInverse  #2 is CVXOPT package, #3 is CVXPY 
quadratics = 1;
use_legendre = 0 ;

colors =[[ 0.68627453,  0.12156863,  0.16470589],
       [ 0.96862745,  0.84705883,  0.40000001],
       [ 0.83137256,  0.53333336,  0.6156863 ],
       [ 0.03529412,  0.01960784,  0.14509805],
       [ 0.90980393,  0.59607846,  0.78039217],
       [ 0.69803923,  0.87843138,  0.72941178],
       [ 0.20784314,  0.81568629,  0.89411765]];

colors = np.asarray(colors);
#np.random.randint(0,255,(Yf_final_test_ep.shape[0],3))
#colors = 1.0/255.0*np.asarray(colors,dtype=np.float32);
# # #  END # # # 

#file_path = 'koopman_data/rand_osc.pickle';
file_path = 'koopman_data/glycol.pickle';
#file_path='koopman_data/deltaomega-series.pickle';
#glycol.pickle');#deltaomega-series.pickle');


file_obj = file(file_path,'rb');
output_vec = pickle.load(file_obj);

Yp = output_vec[0]; # list of baseline observables, len(Yp) = (n_samps-1) 
Yf = output_vec[1]; # list of baseline observables, len(Yf) = (n_samps-1) 

if len(Yp)<2:
    print "Warning: the time-series data provided has no more than 2 points."
    

# not sure if we need this step. 
#for i in range(0,len(Yp)):
#    temp_Yp = np.asarray(Yp[i]);
#    temp_Yf = np.asarray(Yf[i]);
#    Yp[i] = temp_Yp.flatten();
#    Yf[i] = temp_Yf.flatten();
    
Y_whole = [None]*(len(Yp)+1);
        
for i in range(0,len(Yp)+1):
    if i == len(Yp):
        Y_whole[i] = Yf[i-1];
    else:
        Y_whole[i] = Yp[i];

Y_whole = np.asarray(Y_whole);
Y_whole = np.transpose(Y_whole);
print "[INFO] Number of total samples: " + repr(Y_whole.shape[1]);
print "[INFO] Observable dimension of a sample: " + repr(Y_whole.shape[0]);

# generate quadratic terms based on observables in Yp[i] for i = 1,.., n_samps
#output_nums = np.arange(0,len(Y_whole));
#import itertools;

#if quadratics:
#    max_combo_deg = 2;
#else:
#    max_combo_deg = 1;

def poly_lib(data,channels,poly):
    if poly == 1:
        return data;
    Yt_leg_stack = list(data)
    
    for i in range(channels):
        for j in range(i,channels):
            Yt_leg_stack.append(data[i,:]*data[j,:])
            #print(i,j)
            if poly > 2:
                for k in range (j,channels):
                    Yt_leg_stack.append(data[i,:]*data[j,:]*data[k,:])
                    #print(i,k,j)
                    if poly > 3:
                        for l in range (k,channels):
                            Yt_leg_stack.append(data[i,:]*data[j,:]*data[k,:]*data[l,:])
                            #print(i,k,j,l)
                            if poly > 4:
                                for m in range (l,channels):
                                    Yt_leg_stack.append(data[i,:]*data[j,:]*data[k,:]*data[l,:]*data[m,:])
                                    #print(i,k,j,l,m)
                        
    
    return np.asarray(Yt_leg_stack)


Y_whole = poly_lib(Y_whole,Y_whole.shape[0],poly_deg);

print "[INFO] Y_whole.shape after poly_lib: " + repr(Y_whole.shape);


#for curr_combo_deg in range(1,max_combo_deg+1):
#    all_pairwise_tuples =  list(itertools.combinations(output_nums,curr_combo_deg));
#    print "[DEBUG]: all_pairwise_tuples" + repr(all_pairwise_tuples);
#    for i in range(0,len(Y_whole)):
#        Y_wholetimepoint = Y_whole[i];
#        Y_append_all_combos = [];
#            
#        if curr_combo_deg==1:
#            Y_append = Y_wholetimepoint;
#        else:
#            Y_append = [None]*len(all_pairwise_tuples);
#            for j in range(0,len(all_pairwise_tuples)):
#                temp_tuple = all_pairwise_tuples[j];
#                Y_append[j] = 1.0;
#                for k in range(0,curr_combo_deg):
#                    n_k = temp_tuple[k];
#                    Y_append[j] = Y_append[j]*Y_wholetimepoint[n_k];
#        for elem in Y_append:
#            Y_append_all_combos.append(elem);

        #print Y_append_all_combos

#     Y_whole[i]= Y_append_all_combos;

Y_whole = np.asarray(Y_whole);
if quadratics:
    print "[INFO]: Quadratic lifted observable has the shape: " + repr(Y_whole.shape);
else:
    print "[INFO]: Quadratic lifting skipped, standard observable Y_whole has the shape: " + repr(Y_whole.shape);
Y_whole = (np.asarray(Y_whole));
#U,S,V = np.linalg.svd(Y_whole);

    #print S
#Y_whole_transformed = np.matmul(np.transpose(U),Y_whole);
#P = np.zeros((1,U.shape[0]));
#P[0][0] = 1.0;    
#Y_projected = np.matmul(P,Y_whole_transformed);
#print "projected dimension";
#print Y_projected.shape;
#print Y_projected;
Y_whole = np.transpose(Y_whole);
Yp = Y_whole[0:len(Y_whole)-1];
Yf = Y_whole[1:len(Y_whole)];

print "[INFO] Yp.shape (E-DMD): " + repr(Yp.shape);
print "[INFO] Yf.shape (E-DMD): " + repr(Yf.shape);




num_trains = 430#len(Yp)*5/10
train_indices = np.arange(0,num_trains,1);#np.random.randint(0,len(Yp),num_trains)
test_indices = np.arange(num_trains,len(Yp),1);#np.random.randint(0,len(Yp),len(Yp)-num_trains);

Yp_train = Yp[train_indices];
Yf_train = Yf[train_indices];
Yp_test = Yp[test_indices];
Yf_test = Yf[test_indices]; 


#train_range = len(Yp)*2/3; # define upper limits of training data 
#test_range = len(Yp); # define upper limits of test data 

#Yp_test = Yp[train_range:test_range];
#Yf_test = Yf[train_range:test_range];
#Yp_train = Yp[0:train_range];
#Yf_train = Yf[0:train_range];

#Yp_train = np.asarray(Yp_train);
#Yf_train = np.asarray(Yf_train);
#Yp_test = np.asarray(Yp_test);
#Yf_test = np.asarray(Yf_test);

print "[INFO] Yp_test.shape (E-DMD) " + repr(Yp_test.shape);
print "[INFO] Yf_test.shape (E-DMD) " + repr(Yf_test.shape);


Yp_final_train = Yp_train; # straight up quadratics training data 0,...,n-1 
Yf_final_train = Yf_train; # straight up quadratics training data 1 ,... n  1 step forward propagation of Yp_train
Yp_final_test = Yp_test; #straight up quadratics test data -
Yf_final_test = Yf_test; # straight up quadratics test data - 1-step forward propagation of Yp_test  

# # # - - - - - -  Legendre polynomial dictionary generation  - - - - -  # # # 

if use_legendre:
    
    #print Yp.shape
    deg_polynomial = poly_deg
    Yp_leg = legvander(Yp,deg_polynomial)
    n_points,n_channels,n_deg_polyp1=  Yp_leg.shape
    Yp_leg = np.reshape(Yp_leg,(n_deg_polyp1,n_points,n_channels));
    #print Yp_leg.shape

    Yf_leg = legvander(Yf,deg_polynomial)
    n_points,n_channels,n_deg_polyp1=  Yf_leg.shape
    Yf_leg = np.reshape(Yf_leg,(n_deg_polyp1,n_points,n_channels));
    #print Yf_leg.shape
    Yp_leg_stack = np.concatenate((Yp_leg[:,:,0],Yp_leg[:,:,1])) ;
    Yf_leg_stack = np.concatenate((Yf_leg[:,:,0],Yf_leg[:,:,1])) ;
    for i in range(2,n_channels):
        if i%1000==0:
                #print i
                #print Yp_leg_stack.shape
            print Yf_leg_stack.shape

        Yp_leg_stack = np.concatenate((Yp_leg_stack,Yp_leg[:,:,i]))
        Yf_leg_stack = np.concatenate((Yf_leg_stack,Yf_leg[:,:,i]))

        #print Yp_leg_stack.shape
        #print Yf_leg_stack.shape
    Yp_final_train = Yp_leg_stack;
    Yf_final_train = Yf_leg_stack;



    Yp_leg_test = legvander(Yp_test,deg_polynomial)
    n_points,n_channels,n_deg_polyp1=  Yp_leg_test.shape
    Yp_leg_test = np.reshape(Yp_leg_test,(n_deg_polyp1,n_points,n_channels));
    #print Yp_leg.shape

    Yf_leg_test = legvander(Yf_test,deg_polynomial)
    n_points,n_channels,n_deg_polyp1=  Yf_leg_test.shape
    Yf_leg_test = np.reshape(Yf_leg_test,(n_deg_polyp1,n_points,n_channels));
    #print Yf_leg_test.shape
    Yp_leg_stack_test = np.concatenate((Yp_leg_test[:,:,0],Yp_leg_test[:,:,1])) ;
    Yf_leg_stack_test = np.concatenate((Yf_leg_test[:,:,0],Yf_leg_test[:,:,1])) ;


    for i in range(2,n_channels):
        Yp_leg_stack_test = np.concatenate((Yp_leg_stack_test,Yp_leg_test[:,:,i]))
        Yf_leg_stack_test = np.concatenate((Yf_leg_stack_test,Yf_leg_test[:,:,i]))
    
    
    Yp_final_test = Yp_leg_stack_test;
    Yf_final_test = Yf_leg_stack_test;
    


# # # - - - - -  END Legendre polynomial dictionary generation - - - - -  


# # # - - - - -  Koopman calculation - - - - - # # # 

def calc_Koopman(Yf,Yp,flag=1):
    solver_instance = cvxpy.CVXOPT;
    #solver_instance = cvxpy.ECOS;
    if flag==1: # moore penrose inverse, plain ol' least squares Koopman
        #Yp_inv = np.dot(np.transpose(Yp_final), np.linalg.inv( np.dot(Yp_final,np.transpose(Yp_final)) )   );
        Yp_inv = np.linalg.pinv(Yp);
        K = np.dot(Yf,Yp_inv);
        

    if flag ==2: # cvx optimization approach - L2 + L1 lasso 
        norm1_term = 0.0;
        all_col_handles = [None]*Yf.shape[0]
        for i in range(0,Yf.shape[0]):            
            all_col_handles[i] = Variable(Yf.shape[0],1);
            norm1_term = norm1_term + norm2(all_col_handles[i]);

        operator = all_col_handles[0];
        for i in range(1,Yf.shape[0]):
            operator = cvxpy.hstack(operator,all_col_handles[i]);

        print "[INFO]: CVXPY Koopman operator variable: " +repr(operator);
        print "[INFO]: Yf.shape in calc_Koopman: " + repr(Yf.shape);
        norm2_fit_term = norm2(norm2(Yf-operator*Yp,axis=0));
        objective = Minimize(norm2_fit_term + norm1_term)
        constraints = [];
        prob = Problem(objective,constraints);
        result = prob.solve(verbose=True,solver=solver_instance);
        print "[INFO]: Finished executing cvx solver, printing CVXPY problem status"
        print(prob.status);
        K = operator.value;

    if flag ==3:
        operator = Variable(Yf.shape[0],Yf.shape[0])
        objective = Minimize(cvxpynorm(operator,2))
        constraints = [cvxpynorm(Yf-operator*Yp,'fro')/cvxpynorm(Yf,'fro')<0.01 ]
        prob = Problem(objective, constraints)
        result = prob.solve(verbose=True)#(solver=solver_instance);
        print(prob.status);
        K = operator.value;

    return K;



Yp_final_train = np.transpose(Yp_final_train);
Yf_final_train = np.transpose(Yf_final_train);
Yp_final_test = np.transpose(Yp_final_test);
Yf_final_test = np.transpose(Yf_final_test);


K = calc_Koopman(Yf_final_train,Yp_final_train,use_solver); # use lsq Koopman 
Koopman_dim = K.shape[0]; # K should be square.

print "[INFO] Koopman_dim:" + repr(Koopman_dim);

if not( K.shape[1]==K.shape[0]):
    print "Warning! Estimated Koopman operator is not square with dimensions : " + repr(K.shape);


    
Yf_final_predicted = np.dot(K,Yp_final_train);


# # #  - - - - training error - - - # # # 
training_error =  np.linalg.norm(Yf_final_predicted-Yf_final_train,ord='fro')/np.linalg.norm(Yf_final_train,ord='fro');
print('%s%f' % ('[COMP] Training error: ',training_error));

# # # - - - - test error  - - - - # # #

Yf_final_test_predicted = np.dot(K,Yp_final_test);

test_error =  np.linalg.norm(Yf_final_test_predicted-Yf_final_test,ord='fro')/np.linalg.norm(Yf_final_test,ord='fro');
print('%s%f' % ('[COMP] Test error: ',test_error));


# # # - - - n-step Prediction Error Analysis - - - # # # 

  
n_points_pred = len(Yp) - test_indices[0]-1;
init_index = test_indices[0];
print "[INFO] Yp_final_test.shape: " + repr(Yp_final_test.shape);
Ycurr = np.asarray(Yp).T[:,init_index]
Yf_final_test_stack = np.asarray(Yp).T[:,init_index:(init_index+1)+n_points_pred]
#Yp_final_test[:,init_index]; # Yf_leg_stack[:,-1]; # grab one time point from the final prediction of the training data

Yf_final_test_ep = [];


K = np.asarray(K);
print np.linalg.eig(K)

Yf_final_test_ep.append(Ycurr); # append the initial seed state value.


for i in range(0,n_points_pred):
    if continuous_predict:
        Ycurr = Ycurr;
    else:
            Ycurr = np.dot(K,Ycurr);
    
    Yf_final_test_ep.append(Ycurr);


Yf_final_test_ep = np.asarray(Yf_final_test_ep);
print "[INFO] Ground truth Yf_final_test_ep.shape:S1 " + repr(Yf_final_test_ep.shape);

Yf_final_test_ep = np.transpose(Yf_final_test_ep);

#Yf_final_test_stack = np.zeros((Yf_final_test.shape[0],Yf_final_test.shape[1]+1));
#for i in range(-1,Yf_final_test.shape[1]):
#    if i==-1:
#        Yf_final_test_stack[:,i+1] = Yp_final_test[:,0];
#    else:
#        Yf_final_test_stack[:,i+1] = Yf_final_test[:,i];
        
print "[INFO] Ground truth Yf_final_test_ep_stack.shape: "+ repr( Yf_final_test_stack.shape);
print "[INFO] Ground truth Yf_final_test_ep.shape: "+ repr(Yf_final_test_ep.shape); 
prediction_error = np.linalg.norm(Yf_final_test_stack-Yf_final_test_ep,ord='fro')/np.linalg.norm(Yf_final_test_stack,ord='fro');
print('%s%f' % ('[COMP] n-step Prediction error: ',prediction_error));

import matplotlib;
import matplotlib.pyplot as plt;

print "[DEBUG] Y_final_test_stack.shape: " + repr(Yf_final_test_stack.shape);
x_range = np.arange(0,350,1);#Yf_final_test_stack.shape[1],1);



import matplotlib
matplotlib.rcParams.update({'font.size':20})

for i in range(0,7):
    plt.plot(x_range,Yf_final_test_ep[i,0:len(x_range)],'--',color=colors[i,:])
    plt.plot(x_range,Yf_final_test_stack[i,0:len(x_range)],'.',color=colors[i,:]);
#plt.plot(x_range,Yf_final_test_stack[0,:],'r-',label='x_1(t)');
#plt.plot(x_range,Yf_final_test_ep[1,:],'g*--',label='x_2_pred(t)');
#plt.plot(x_range,Yf_final_test_stack[1,:],'g-',label='x_2(t)');
ax = plt.gca();
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.ylim([0.0,3.0])
#plt.legend(loc='best');
#plt.xlabel('t');
plt.show();

stats_vec = [training_error,test_error,prediction_error];
# # #  - - - End n-step prediction error - - - # # # 

#file_obj = open('stats_file.txt','a');
#file_obj.write('\n%d\t%d\t%d\t%d\t%d\t%f\t%f\t%f' %(train_range,test_range,deg_polynomial,n_channels,Koopman_dim,training_error,test_error,prediction_error));
#file_obj.close();
#temp_stats = [train_range,test_range,deg_polynomial,n_channels,Koopman_dim,training_error,test_error,prediction_error];
#all_stats.append(temp_stats);#

#file_obj = open('stats_pickle.txt','wb');
#pickle.dump(all_stats,file_obj);
#file_obj.close()
