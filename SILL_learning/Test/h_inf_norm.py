import control
import numpy as np


num = np.array([1.])
den = np.array([1, -1.])
my_tf = control.tf(num, den)
print(my_tf)

print(control.hinfsyn(my_tf, 1, 1))


def hinf(tf, tol):
    """
    Computes the H-inf norm
    :param tf: 
    :return: 
    """
    best = 0
    gammal =
    gammah =
    while (gammah - gammal) / 2 > 2*tol*gammal:
        # Form M

        # check eigenvaleus
        # update gammas
    return best