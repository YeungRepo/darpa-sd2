import numpy as np
import pickle
import tensorflow as tf
a = np.array([1,2,3])

# with handle as
data = pickle.load(open('koopman_data/X8SS_Pputida_RNASeqDATA.pickle','rb'))
print(len(data))
