load('/Users/yeun026/Documents/darpa-sd2/deepDMD/h2synthesis/state_space_model.mat','A','B','C')
sys = ss(A,B,C,[],1.0);
c_sys = d2c(sys,'tustin');
Ac = c_sys.A;
Bc = c_sys.B;
Cc = c_sys.C;

save('/Users/yeun026/Documents/darpa-sd2/deepDMD/h2synthesis/continuous_state_space_model.mat','Ac','Bc','Cc');