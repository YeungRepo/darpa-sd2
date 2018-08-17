import numpy as np


# Start with a model of the generator
def gnrtr(H, P_m, P_e, D, DELTA_omega_r, omega_r, omega_syn, T_do_pp, T_qo_pp, E_fd, E_q_p, E_q_ppp, I_d, X_d, X_d_p, X_d_pp, E_d_pp, I_q, X_q_p, X_q_pp):
    """
    ODE to model the behavior of a power generator, _p means derivative wrt time, _pp is second derivative and so on.
    :param H: 
    :param P_m: 
    :param P_e: 
    :param D: 
    :param DELTA_omega_r: 
    :param omega_r: 
    :param omega_syn: 
    :param T_do_pp: 
    :param T_qo_pp: 
    :param E_fd: 
    :param E_q_p: 
    :param E_q_pp: 
    :param I_d: 
    :param X_d: 
    :param X_d_p: 
    :param X_d_pp: 
    :param E_d_pp: 
    :param I_q: 
    :param X_q_p: 
    :param X_q_pp: 
    :return: DELTA_omega_r_p, delta_p=DELTA_omega_r, E_q_pp, E_q_ppp, E_d_ppp, E_d (d-axis stator voltage), E_q
    (q-axis stator voltage), E_t (terminal voltage of generator), P_e (electrical power output)
    In a BIG ol'tuple.
    """
    DELTA_omega_r_p = (2 * H)**(-1) * (P_m - P_e - D * DELTA_omega_r)
    delta_p = omega_r - omega_syn
    # TODO: Pick the order to do these variable assignments in.
    E_q_pp = (T_do_pp)**(-1) * (E_fd - E_q_p + I_d * (X_d - X_d_p))
    E_d_ppp = (T_qo_pp)**(-1) * (-E_d_pp - I_q * (X_q_p - X_q_pp))
    E_q_ppp = (T_qo_pp)**(-1) * (E_q_p - E_q_ppp + I_d * (X_d_p - X_d_pp))
    E_d = E_d_pp - X_d_pp * I_d
    E_q = E_q_pp - X_q_pp * I_d
    #TODO: This (the two lines above) could be messed up from the paper...
    E_t = np.sqrt(E_d**2 + E_q**2)
    P_e = E_d * I_d + E_q * I_q
    return DELTA_omega_r_p, delta_p, E_q_pp, E_d_ppp, E_q_ppp, E_d, E_q, E_t, P_e


# Models for a water pump
# What are we to do with all of these equations?

"""
Our Variables are:


 Below is the system of equations we consider:
 
(R_0 + R_1 + j*w_r * (X_0 + X_1)) * Q_1 - (R_0 + j*w_r * X_0) * Q_2 - H_1 = 0
(R_0 + R_2 + j*w_r * (X_0 + X_2)) * Q_2 - (R_0 + j*w_r * X_0) * Q_1 - H_2 = 0

(R_0 + R_1) * Q_d1 - w_r * (X_0 + X_1) * Q_q1 - R_0 * Q_d1 + w_r * X_0 * Q_q2 - H_1d = 0
(R_0 + R_2) * Q_d2 - w_r * (X_0 + X_2) * Q_q2 - R_0 * Q_d1 + w_r * X_0 * Q_q1 - H_2d = 0

w_r * (X_0 + X_1) * Q_d1 + (R_0 + R_1) * Q_q1 - w_r * X_0 * Q_q2 - R_0 * Q_q2 - H_1q = 0
w_r * (X_0 + X_2) * Q_d2 + (R_0 + R_1) * Q_q2 - w_r * X_0 * Q_q1 - R_0 * Q_q1 - H_2q = 0

sqrt(H_1d**2 + H_1q**2) - w_r**2 * H_1n = 0

Ncp = Re(H_1.dot(Q_1.conjugate_transpose())) = H_1d * Q_1d + H_1q * Q_1q  # Mechanical Power on the Shaft of the Centrifugal Pump
N_2 = Re(H_2.dot(Q_2.conjugate_transpose())) = H_2d * Q_2d + H_2q * Q_2q  # Useful Hydraulic Power


N_m = electromagnetic power of SM * friction and windage losses = (lambda_delta_d * I_sd + lambda_delta_q * I_sq) - DELTA_N_m(w_r) # Mechanical power on the synchronous motor shaft

"""
