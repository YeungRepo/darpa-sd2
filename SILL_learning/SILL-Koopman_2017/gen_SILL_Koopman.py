
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



def f(x): # think about generalized function definitions. 
    dx1= np.power(x[0],2.0);
    dx2 = np.power(x[1],2.0);
    return [dx1,dx2];


continuous_predict=0;
use_cvx = 1;

colors =[[ 0.68627453,  0.12156863,  0.16470589],
       [ 0.96862745,  0.84705883,  0.40000001],
       [ 0.83137256,  0.53333336,  0.6156863 ],
       [ 0.03529412,  0.01960784,  0.14509805],
       [ 0.90980393,  0.59607846,  0.78039217],
       [ 0.69803923,  0.87843138,  0.72941178],
       [ 0.20784314,  0.81568629,  0.89411765]];

colors = np.asarray(colors);

#file_path = 'koopman_data/rand_osc.pickle';
#file_path = 'koopman_data/glycol.pickle';
#file_path='koopman_data/deltaomega-series.pickle';
#glycol.pickle');#deltaomega-series.pickle');


#file_obj = file(file_path,'rb');
#output_vec = pickle.load(file_obj);

#Yp = output_vec[0]; # list of baseline observables, len(Yp) = (n_samps-1) 
#Yf = output_vec[1]; # list of baseline observables, len(Yf) = (n_samps-1) 



point_mesh = np.mgrid[-10.0:10.0:1.0,-10.0:10.0:1.0];

x_all_points = [];
f_all_points = [];
for point in point_mesh:
    f_temp = f(point) # this code needs to be generalized to n dimensions;
    x_all_points.append(point);
    f_all_points.append(f_temp);

f_all_points = np.asarray(f_all_points,dtype=np.float32);
x_all_points = np.asarray(x_all_points,dtype=np.float32);
print "[INFO] f_all_points.shape (E-DMD): " + repr(f_all_points.shape);


solver_instance = cvxpy.CVXOPT;



f_dimensions = f_all_points.shape[1];
num_product_terms = f_dimensions;

num_logistic_functions = 3; #this hyperparameter needs to be generalized 

all_center_handles =[None]*num_logistic_functions;
all_weight_handles = [None]*num_logistic_functions;
approximator = 0.0;
for j in range(0,x_all_points.shape[0]):
    for k in range(0,num_logistic_functions):
        all_weight_handles[k] = Variable(1,1);
        all_center_handles[k] = Variable(num_product_terms,1);
        prod_log_functions = 1.0;
    
        for i in range(0,num_product_terms):
            prod_log_functions  = prod_log_functions* 1.0/(1.0+ cvxpy.exp(alpha*(x_all_points[j] - all_center_handles[k][i])) );
        
    approximator = approximator + cvxpy.norm1(f_all_points[j] - prod_log_functions);


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


K = calc_Koopman(Yf_final_train,Yp_final_train,use_cvx); # use lsq Koopman 
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

# # # - - - n-step prediction error - - - # # # 

n_points_pred = Yf_final_test.shape[1];

print "[INFO] Yp_final_test.shape: " + repr(Yp_final_test.shape);
Ycurr = Yp_final_test[:,0]; # Yf_leg_stack[:,-1]; # grab one time point from the final prediction of the training data

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

Yf_final_test_stack = np.zeros((Yf_final_test.shape[0],Yf_final_test.shape[1]+1));
for i in range(-1,Yf_final_test.shape[1]):
    if i==-1:
        Yf_final_test_stack[:,i+1] = Yp_final_test[:,0];
    else:
        Yf_final_test_stack[:,i+1] = Yf_final_test[:,i];
        
print "[INFO] Ground truth Yf_final_test_ep_stack.shape: "+ repr( Yf_final_test_stack.shape);
print "[INFO] Ground truth Yf_final_test_ep.shape: "+ repr(Yf_final_test_ep.shape); 
prediction_error = np.linalg.norm(Yf_final_test_stack-Yf_final_test_ep,ord='fro')/np.linalg.norm(Yf_final_test_stack,ord='fro');
print('%s%f' % ('[COMP] n-step Prediction error: ',prediction_error));

import matplotlib;
import matplotlib.pyplot as plt;

print "[DEBUG] Y_final_test_stack.shape: " + repr(Yf_final_test_stack.shape);
x_range = np.arange(0,Yf_final_test_stack.shape[1],1);




for i in range(0,3):
    plt.plot(x_range,Yf_final_test_ep[i,:],'--',color=colors[i,:])
    plt.plot(x_range,Yf_final_test_stack[i,:],'.',color=colors[i,:]);
#plt.plot(x_range,Yf_final_test_stack[0,:],'r-',label='x_1(t)');
#plt.plot(x_range,Yf_final_test_ep[1,:],'g*--',label='x_2_pred(t)');
#plt.plot(x_range,Yf_final_test_stack[1,:],'g-',label='x_2(t)');

plt.legend(loc='best');
#plt.xlabel('t');
import matplotlib
matplotlib.rcParams.update({'font.size':22})
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
