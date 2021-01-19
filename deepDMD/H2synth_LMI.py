#! /usr/env/bin python 

import pickle;
import numpy as np;
from numpy.linalg import pinv;
from numpy.polynomial.legendre import legvander;
import tensorflow as tf
import os

import control

import cvxopt;
import cvxpy;



def test_synthesis(A,B):
    sf = 10.0
    # Problem data.

    N_u = B.shape[1]
    N_x = A.shape[0]
    Cv =np.eye(N_u)
    Avbar = cvxpy.Variable(rows=N_u,cols=N_u);
    Bvbar = cvxpy.Variable(rows=N_u,cols=N_x);

    P1 = cvxpy.Semidef(A.shape[0]);#,cols=A.shape[0]); # 
    P2 = cvxpy.Semidef(N_u);#,cols=N_u);

    Zeros = cvxpy.Variable(rows=N_x+N_u+N_x+N_u,cols=N_x+N_u+N_x+N_u); 
    
    # Construct the problem.

    objective = cvxpy.Minimize(cvxpy.norm1(Avbar)+cvxpy.norm1(Bvbar));#+cvxpy.norm2(Bv))
    #PMatrix = cvxpy.vstack([cvxpy.hstack([P1,np.zeros(N_x,N_u)]), cvxpy.hstack([np.zeros(( N_u,N_x )),P2]) ]);
    #QMatrix = cvxpy.vstack([cvxpy.hstack([A,np.matmul(B,Cv)]), cvxpy.hstack([Bv,Av]) ]);
    #SystemMatrix = cvxpy.vstack([cvxpy.hstack([PMatrix,QMatrix]),cvxpy.hstack([QMatrix,PMatrix])])
    SystemMatrix1 = cvxpy.hstack([P1, np.zeros((N_x,N_u)), A.T*P1, Bvbar.T]);
    SystemMatrix2 = cvxpy.hstack([np.zeros(( N_u,N_x )),P2,np.matmul(Cv.T,B.T)*P1,Avbar]);
    SystemMatrix3 = cvxpy.hstack([P1*A, P1*np.matmul(B,Cv), P1, np.zeros((N_x,N_u))]);
    SystemMatrix4 = cvxpy.hstack([Bvbar, Avbar, np.zeros((N_u,N_x)), P2]); 
    SystemMatrix = cvxpy.vstack( [SystemMatrix1,SystemMatrix2,SystemMatrix3,SystemMatrix4]);
    constraints = [Zeros==1e-10, Zeros<<SystemMatrix,0<P1,0<P2]; 

    prob = cvxpy.Problem(objective, constraints)

# The optimal objective is returned by prob.solve().
    result = prob.solve(verbose=True,solver='SCS')
    
    return result, Avbar.value, Bvbar.value, Cv, P1.value, P2.value,SystemMatrix.value
# The optimal value for x is stored in x.value.
# The optimal Lagrange multiplier for a constraint
# is stored in constraint.dual_value.
#print constraints[0].dual_value


with_control = 1;

sess = tf.InteractiveSession();

saver = tf.train.import_meta_graph('/Users/yeun026/Documents/darpa-sd2/deepDMD/MD.pickle.ckpt.meta')
saver.restore(sess,tf.train.latest_checkpoint('/Users/yeun026/Documents/darpa-sd2/deepDMD/.'));

psiyp = tf.get_collection('psiyp')[0];
psiyf = tf.get_collection('psiyf')[0];

yp_feed = tf.get_collection('yp_feed')[0];
yf_feed = tf.get_collection('yf_feed')[0];

psiu = tf.get_collection('psiu')[0];
u_control = tf.get_collection('u_control')[0];

Kx = tf.get_collection('Kx')[0];
Ku = tf.get_collection('Ku')[0];

Kx_num = sess.run(Kx);
Ku_num = sess.run(Ku);
A = np.transpose(Kx_num); # Kx_num and Ku_num were defined using row multi. 
B = np.transpose(Ku_num);
C = np.eye(A.shape[0]);
D = np.zeros((A.shape[0],B.shape[1]));
sys = control.ss(A,B,C,D);
print control.ctrb(A,B)


print np.linalg.matrix_rank(control.ctrb(A,B))
print control.ctrb(A,B).shape
print yp_feed.get_shape


opt_outcome,Avbar_value,Bvbar_value,Cv,P1v,P2v,Sv = test_synthesis(A,B);

Av = np.matmul(np.linalg.inv(P2v), Avbar_value);
Bv = np.matmul(np.linalg.inv(P2v),Bvbar_value);
print Av.shape
print Bv.shape

r1 = np.hstack([A,B])
r2 = np.hstack([Bv,Av]);
cl_mat = np.vstack([r1,r2]);
np.linalg.eig(cl_mat)


import scipy.io
import pickle 
scipy.io.savemat('/Users/yeun026/Documents/darpa-sd2/deepDMD/h2synthesis/controller_matrices.mat',mdict={'Av':Av,'Bv':Bv,'Cv':Cv}) #Av,Bv,Cv
file_obj = file('/Users/yeun026/Documents/darpa-sd2/deepDMD/h2synthesis/controller_matrices.pickle','wb');
pickle.dump([Av,Bv,Cv],file_obj,);
file_obj.close();

scipy.io.savemat('/Users/yeun026/Documents/darpa-sd2/deepDMD/h2synthesis/state_space_model.mat',mdict={'A':A,'B':B,'C':C}) #Av,Bv,Cv
file_obj = file('/Users/yeun026/Documents/darpa-sd2/deepDMD/h2synthesis/controller_matrices.pickle','wb');
pickle.dump([A,B,C],file_obj,);
file_obj.close();
