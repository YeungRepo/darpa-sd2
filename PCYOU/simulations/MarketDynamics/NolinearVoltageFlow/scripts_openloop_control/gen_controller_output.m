function[control_signal] = gen_controller_output(state_x,psiu)

load('/Users/yeun026/Documents/darpa-sd2/deepDMD/h2synthesis/controller_matrices.mat','Av','Bv','Cv')
%sys = ss(Av,Bv,Cv,[],1.0);
%c_sys = d2c(sys);
%A = c_sys.A;
%B = c_sys.B;
%C = c_sys.C; 

%disp(size(state_x))
x = state_x;
save('curr_x.mat','x'); 
commandStr = 'python gen_liftings.py';
[status,commandOut] = system(commandStr);

load('psi_lifting.mat','psix');
%size(psiu)
%size(Av)
%size(psix')
%size(B)
%size(Bv)
dpsiu = Av*psiu+Bv*psix'; 
control_signal = dpsiu;

return