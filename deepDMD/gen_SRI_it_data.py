

import pickle
import csv
import numpy as np
import glob
from sys import exit
from sklearn import preprocessing;
with_debug =0;
scale_data = 0;

def find_nearest(array, value):
    array = np.asarray(array)
    diff_array = (np.abs(array - value));
    idx = diff_array.argmin();
    return idx;

PT_arrays = [];
PF_arrays = [];
LC_arrays = [];
LC_all_arrays = [];
for condition_ind in [0,1,2]:
        PT_filename = 'SRI_collab/PT_Cond' + repr(condition_ind)+'.csv';
        PF_filename = 'SRI_collab/PF_Cond' + repr(condition_ind)+'.csv';
        LC_filename = 'SRI_collab/LC_Cond' + repr(condition_ind)+'.csv';
        LC_all_filename = 'SRI_collab/LC_all_Cond' + repr(condition_ind)+'.csv';

        with open(PT_filename, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            iter = 0;
            all_rows = [];
            for row in spamreader:
                if iter>1:
                    try:
                        row = [np.float(elem) for elem in row]
                        all_rows.append(row)
                    except:
                        print(row)
                    
                iter = iter+1;
            PT_arrays.append(np.asarray(all_rows));
        
        with open(PF_filename, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            iter = 0;
            all_rows = [];
            for row in spamreader:
                if iter>1:
                    try:
                        row = [np.float(elem) for elem in row]
                        all_rows.append(row)
                    except:
                        print(row)
                    
                iter = iter+1;
            PF_arrays.append(np.asarray(all_rows));
        with open(LC_filename, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            iter = 0;
            all_rows = [];
            for row in spamreader:
                if iter>1:
                    try:
                        row = [np.float(elem) for elem in row]
                        all_rows.append(row)
                    except:
                        print(row)
                    
                iter = iter+1;
            LC_arrays.append(np.asarray(all_rows));
        with open(LC_all_filename, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            iter = 0;
            all_rows = [];
            for row in spamreader:
                if iter>1:
                    try:
                        row = [np.float(elem) for elem in row]
                        all_rows.append(row)
                    except:
                        print(row)
                    
                iter = iter+1;
            LC_all_arrays.append(np.asarray(all_rows));
 
f_PF_arrays = [];
f_PT_arrays =[];
for array_ind in range(0,len(LC_arrays)):
    this_array= LC_arrays[array_ind];
    time_column = this_array[:,0]
    this_PF_array = PF_arrays[array_ind];
    this_PT_array = PT_arrays[array_ind];
    filtered_PF_array = [];
    filtered_PT_array = [];
    for time in time_column:
        array_idx = find_nearest(this_PF_array[:,0],time)
        filtered_PF_array.append(this_PF_array[array_idx,:]);

        array_idx = find_nearest(this_PT_array[:,0],time)
        filtered_PT_array.append(this_PT_array[array_idx,:]);
    f_PF_arrays.append(filtered_PF_array);
    f_PT_arrays.append(filtered_PT_array);

        

'''
if with_debug:
	print "[DEBUG]" + repr(filelist);

Yp_global=[];
Yf_global=[];
u_global=[];


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
'''





