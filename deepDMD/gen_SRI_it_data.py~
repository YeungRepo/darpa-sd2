

import pickle
import csv
import numpy as np
import glob
from sys import exit
from sklearn import preprocessing;
with_debug =0;
scale_data = 0;
filelist=glob.glob('./dataset_NoImbalance_NoDroopControl/*.csv')

if with_debug:
	print "[DEBUG]" + repr(filelist);

Yp_global=[];
Yf_global=[];
u_global=[];

n_states = 188;
n_controls = 5; 
n_points_truncate = 200;
g = 10;
n  = 39;
m = 46; 
freq_states = np.arange(0,g);
# angle_states = np.array([],dtype=np.int32);
# angle_states = np.arange(10,49);
#gov_states = np.arange(49,54,2);
gen_states = np.arange(n+g,n+2*g,1);
load_states = np.arange(n+2*g,n+2*g+n);
# volt_states = np.arange(137,147);
flow_states = np.arange(2*n+2*g,2*n+2*g+m);

chosen_state = np.hstack([freq_states])#,gen_states,load_states,flow_states])
print chosen_state.shape
# without phase angle, num of chosen states = 42

if with_debug:
	print "[DEBUG]: chosen states vector " + repr(chosen_state)

for file in filelist[0:]:
	# filename='partMDdata/' + file
	with open(file,'rb') as datafile:




		data=csv.reader(datafile, delimiter=',')
		
		if with_debug:
			print "[DEBUG]: data ingest object" + repr(data)

		#Convert data obj to parsable format. 
		datalist=list(data)
		n_points = len(datalist); # number of time-series points. 

		Y_whole=np.asarray(datalist,dtype=np.float32)
                #print Y_whole[0,freq_states] 
                #Y_whole[:,freq_states] = Y_whole[:,freq_states]
                
                print "Y_whole.shape" + repr(Y_whole.shape)
		if with_debug:
			print "Dimensions of data array: " + repr(Y_whole.shape);
		#Extract State and Control Time-Series Matrices 
		Filtered_State_Whole = np.empty([n_points,1]);

		for i in range(0,len(chosen_state)):
			time_horizon = len(Y_whole[:,0]);
			append_vec = np.reshape(Y_whole[:,chosen_state[i]],(time_horizon,1) );
			if with_debug:
				print "[DEBUG] append_vec.shape: " + repr (append_vec.shape);
			Filtered_State_Whole = np.hstack([Filtered_State_Whole, append_vec]);


		Filtered_State_Whole= np.delete(Filtered_State_Whole,0,1)

		
		if with_debug:
			print "[DEBUG]: shape of filtered states " + repr(Filtered_State_Whole.shape)



		Control_Whole = Y_whole[:,-1-n_controls:-1];
		
		''' # degenerate code for treating state initial cond. as an input signal (PCY)
		u_impulse=Y_whole[0]
		u=[u_impulse*0.0]*len(Y_whole)
		u[0]=u_impulse
		'''
		
		#Define One-Step State and Control Data 
		Yp = Filtered_State_Whole[0:n_points-1];
		Yf = Filtered_State_Whole[1:n_points];
		Up = Control_Whole[0:n_points-1]; 

		#Append time-series data to global data arrays. 
		for j in range(0,n_points_truncate):
			Yp_global.append(Yp[j])
			Yf_global.append(Yf[j])
			u_global.append(Up[j]);

		datafile.close()

print Up

if scale_data:
	u_scaler = preprocessing.StandardScaler().fit(u_global);
	y_scaler = preprocessing.StandardScaler().fit(Yp_global);
	Yp_global = y_scaler.transform(Yp_global);
	Yf_global = y_scaler.transform(Yf_global);
	u_global = u_scaler.transform(u_global);
	scalerfile_path = './koopman_scalers/MD_scaler.pickle';
        scalerfile = open(scalerfile_path,'wb');
        pickle.dump([u_scaler,y_scaler],scalerfile);
        scalerfile.close();

		

print "[Result] Successfully generated  "
print "\n -------- \n ----  Stats ----\n --------" + "\nNum. of States: " + repr(len(Yp_global[0])) + "\nNum. of Controls: " + repr(len(u_global[0])) + "\nNum. of Time-points: " + repr(len(Yp_global));

#print "# time epoches:" + repr(len(u_global))


picklefile_path = './koopman_data/MD.pickle';
# picklefile_path= './pickledata/MD.pickle'
#picklefile_path= '/Users/youp387/Desktop/Koopman/KOdataset/pickledata/MD.pickle'

picklefile=open(picklefile_path,'wb')
pickle.dump([Yp_global,Yf_global,u_global],picklefile)

picklefile.close()






