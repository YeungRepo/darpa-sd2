#! /usr/bin/env python
Y_p_global = [];
Y_f_global = [];
u_seq_global = [];
for i in range(99,100):
    file_ind = str(i+1);
    file_name = 'raw_deltaomega_data/deltaomega-' + file_ind + '.txt';
    file_obj = file(file_name,'r');
    all_lines = file_obj.readlines();
    

    for i in range(0,len(all_lines)):

        all_lines[i]= all_lines[i].strip('\n').split();

    import numpy as np

    Y_whole = np.asarray(all_lines,dtype=np.float32);
    u_impulse = Y_whole[0];
    print len(u_impulse);
    
    u_seq = [u_impulse*0.0]*len(Y_whole);
    u_seq[0] = u_impulse;
    print u_seq[0:2]
    #print len(Y_whole);
    Y_p = Y_whole[0:-2];
    Y_f = Y_whole[1:-1];
    for j in range(0,len(Y_p)):
        Y_p_global.append(Y_p[j]);
        Y_f_global.append(Y_f[j]);

    for j in range(0,len(u_seq)):
        u_seq_global.append(u_seq[j]);
        
    import pickle

file_obj = file('koopman_data/deltaomega-singleseries.pickle','wb');

print "Number of time-points: " + repr(len(u_seq_global));

pickle.dump([Y_p_global,Y_f_global,u_seq_global],file_obj);

file_obj.close()






