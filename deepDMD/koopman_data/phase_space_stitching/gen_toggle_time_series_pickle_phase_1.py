import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# function that returns dz/dt
def model(z,t):
    dzdt = [np.power(1+z[1]**3.55, -1) - 0.5*z[0],
            np.power(1+z[0]**3.53, -1) - 0.5*z[1]]
    return dzdt

# time points
t = np.linspace(0,50,100)
# X and Y range
Xrange = np.linspace(0,5,10)
Yrange = np.linspace(0,5,10)
All_Z = None;
All_Zf = None;
All_Zp = None; 
for k1 in range(len(Xrange)):
    for k2 in range(len(Yrange)):
        if k1 <= k2:
            if k1 < k2:
                z0 = [Xrange[k1],Yrange[k2]];
                Z = odeint(model,z0,t)
                f = plt.figure(1)
                plt.plot(Z[:,0],Z[:,1])
                Z  = Z.T;
                Zf = Z[:,1:];
                Zp = Z[:,0:-1];
            elif k1 == k2 & k1 > 1 & k2 >= 1:
                z0 = [Xrange[k1],Yrange[k2]];
                Z = odeint(model,z0,t)
                f = plt.figure(1)
                plt.plot(Z[:,0],Z[:,1])
                Z  = Z.T;
                Zf = Z[:,1:];
                Zp = Z[:,0:-1];

        
                if type(All_Zf) == np.ndarray:
                    All_Zf = np.hstack((All_Zf,Zf));
                    All_Zp = np.hstack((All_Zp,Zp));
            
                else:
                    All_Zf = Zf;
                    All_Zp = Zp;  
            

Yf = All_Zf;
Yp = All_Zp;

f.show()

print("[DEBUG] Yf.shape: " + repr(Yf.shape));
print("[DEBUG] Yp.shape: " + repr(Yp.shape));

import pickle
file_obj = open('sim_toggle_switch_phase1.pickle','wb'); # 1 - k1 >= k2, 2 - k1 <= k2
pickle.dump([list(np.transpose(Yp)), list(np.transpose(Yf))], file_obj);
file_obj.close(); 



