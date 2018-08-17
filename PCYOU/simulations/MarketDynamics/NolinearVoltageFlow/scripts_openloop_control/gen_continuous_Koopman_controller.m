load('/Users/yeun026/Documents/darpa-sd2/deepDMD/h2synthesis/controller_matrices.mat','Av','Bv','Cv')
sys = ss(Av,Bv,Cv,[],1.0);
c_sys = d2c(sys,'tustin');
Ac = c_sys.A;
Bc = c_sys.B;
Cc = c_sys.C;

save('/Users/yeun026/Documents/darpa-sd2/deepDMD/h2synthesis/continuous_controller_matrices.mat','Ac','Bc','Cc');