#! /usr/bin/env python 

import pickle;
import numpy as np;
from numpy.linalg import pinv;
from numpy.polynomial.legendre import legvander;
import tensorflow as tf

import control
import os; 
with_control = 1;

sess = tf.InteractiveSession();


saver = tf.train.import_meta_graph('/Users/yeun026/Documents/darpa-sd2/deepDMD/MD.pickle.ckpt.meta')
saver.restore(sess,tf.train.latest_checkpoint('/Users/yeun026/Documents/darpa-sd2/deepDMD/'));

psiyp = tf.get_collection('psiyp')[0];
psiyf = tf.get_collection('psiyf')[0];
#psiu = tf.get_collection('psiu')[0];
yp_feed = tf.get_collection('yp_feed')[0];
yf_feed = tf.get_collection('yf_feed')[0];
#u_control = tf.get_collection('u_control')[0];
from scipy.io import loadmat,savemat


var_dict = loadmat('curr_x.mat');
#print var_dict
x_curr = var_dict['x'];

#print x_curr.shape
#u_curr = var_dict['u'];

x_curr = x_curr.squeeze();
#u_curr = u_curr.squeeze(); 



psi_lifting = psiyf.eval(feed_dict={yf_feed:[x_curr]});
#psi_u_lifting = psiu.eval(feed_dict={u_control:[u_curr]});
#print psi_lifting
savemat('psi_lifting.mat',mdict={'psix':psi_lifting})
