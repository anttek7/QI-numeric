import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog
from numpy import linalg as LA
import CHSH

def find_psi(theta):
    return np.array([np.cos(theta/2), 0, 0, np.sin(theta/2)])
def find_P(theta, a0,a1,b0,b1):
    A = [a0,a1,b0,b1]
    P = np.array([])
    for i in range(4):
        P = np.append(P, np.cos(A[i])*np.cos(theta))
    for i in range(4):
        P = np.append(P,np.cos(A[int(i/2)])*np.cos(A[2 + i%2]) + np.sin(theta)*np.sin(A[int(i/2)])*np.sin(A[2 + i%2]))
    return P
def find_dP_dtheta(theta,a0,a1,b0,b1):
    A = [a0,a1,b0,b1]
    dP = np.array([])
    for i in range(4):
        dP = np.append(dP, -2*np.cos(A[i])*np.sin(theta))
    for i in range(4):
        dP = np.append(dP,2*np.cos(theta)*np.sin(A[int(i/2)])*np.sin(A[2 + i%2]))
    return dP

def dP_dA(theta,a0,a1,b0,b1,i):
    A = [a0,a1,b0,b1]
    dP = np.zeros(8)
    dP[i] = -np.sin(A[i])*np.cos(theta)
    if i==0 or i==1:
        dP[4 + 2*(i%2)] = -np.sin(A[i%2])*np.cos(A[2]) + np.sin(theta)*np.cos(A[i%2])*np.sin(A[2])
        dP[5 + 2*(i%2)] = -np.sin(A[i%2])*np.cos(A[3]) + np.sin(theta)*np.cos(A[i%2])*np.sin(A[3])
    else:
        dP[4 + i%2] = -np.sin(A[2 + i%2])*np.cos(A[0]) + np.sin(theta)*np.cos(A[2 + i%2])*np.sin(A[0])
        dP[6 + i%2] = -np.sin(A[2 + i%2])*np.cos(A[1]) + np.sin(theta)*np.cos(A[2 + i%2])*np.sin(A[1])        
    return np.round(dP,10)

def find_dP_da0(theta,a0,a1,b0,b1):
    dP = dP_dA(theta,a0,a1,b0,b1,0)
    return dP
def find_dP_da1(theta,a0,a1,b0,b1):
    dP = dP_dA(theta,a0,a1,b0,b1,1)
    return dP
def find_dP_db0(theta,a0,a1,b0,b1):
    dP = dP_dA(theta,a0,a1,b0,b1,2)
    return dP
def find_dP_db1(theta,a0,a1,b0,b1):
    dP = dP_dA(theta,a0,a1,b0,b1,3)
    return dP
def find_deterministic_points():
    p = 16
    D = 8
    d = D*2
    M = np.zeros((D,d))
    korelatory = np.ones(4)
    for i in range(4):
        p /= 2
        x = 1
        for j in range(d):
            if j%(2*p) < p:
                M[i][j] = x
            else:
                M[i][j] = -x
    for j in range(d):
        for i in range(4):
            korelatory[i] = M[i][j]/x**0.5
        for i in range(4):
            M[i+4][j] = korelatory[int(i/2)] * korelatory[i%2 + 2]
    return np.transpose(M)    
def find_A_ub():
    return find_deterministic_points()

def find_A_eq(theta,a0,a1,b0,b1):
    dP_dtheta = find_dP_dtheta(theta,a0,a1,b0,b1)
    dP_da0 = find_dP_da0(theta,a0,a1,b0,b1)
    dP_da1 = find_dP_da1(theta,a0,a1,b0,b1)
    dP_db0 = find_dP_db0(theta,a0,a1,b0,b1)
    dP_db1 = find_dP_db1(theta,a0,a1,b0,b1)
    A_eq = np.vstack([dP_dtheta, dP_da0, dP_da1, dP_db0, dP_db1])
    return A_eq
def B_add_limit(limit, A_ub, b_ub):
    A = np.eye(8)
    A_ub = np.append(A_ub,A, axis=0)
    A_ub = np.append(A_ub, -A, axis=0)
    b = np.ones(8)*limit
    b_ub = np.append(b_ub, b, axis=0)
    b_ub = np.append(b_ub, b, axis=0)
    return A_ub, b_ub 
    
def vector_to_matrix(V):
    matrix = np.zeros((3,3))
    matrix[0][1] = V[2]
    matrix[0][2] = V[3]
    matrix[1][0] = V[0]
    matrix[1][1] = V[4]
    matrix[1][2] = V[5]
    matrix[2][0] = V[1]
    matrix[2][1] = V[6]
    matrix[2][2] = V[7]
    return matrix
def matrix_to_vector(matrix):
    V = np.zeros(8)
    V[2] = matrix[0][1]
    V[3] = matrix[0][2]
    V[0] = matrix[1][0]
    V[4] = matrix[1][1]
    V[5] = matrix[1][2]
    V[1] = matrix[2][0]
    V[6] = matrix[2][1]
    V[7] = matrix[2][2]
    return V

def Best_func(theta,a0,a1,b0,b1, limit):
    P = find_P(theta,a0,a1,b0,b1)
    c = -P    
    A_ub = find_A_ub()
    b_ub = np.ones(16)
    A_ub, b_ub = B_add_limit(limit, A_ub, b_ub)
    A_eq = find_A_eq(theta,a0,a1,b0,b1)
    b_eq = np.zeros(5)
    bound = (None, None)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq = A_eq, b_eq = b_eq, bounds=bound, method='revised simplex')
    maxi = -res.fun 
    B = res.x
    return B, maxi 
   
def Best_point(B,accuracy):
    W_max, state, alpha, beta, l1, l2, Nonlocal = CHSH.chsh2(vector_to_matrix(B), accuracy)
    #W_max, state, alpha, beta, all_maximal_eigenvalues, l1, l2 = CHSH.chsh(vector_to_matrix(B), accuracy)
    #CHSH.plot_CHSH(accuracy, all_maximal_eigenvalues,0,"-","-")
    #print(W_max, l1, l2)
    correlators = CHSH.find_correlators2(alpha, beta, state, accuracy)
    #correlators = CHSH.find_correlators(alpha, beta, state, accuracy)
    P = matrix_to_vector(correlators)
    return P, Nonlocal

def is_exposed(theta,a0,a1,b0,b1, accuracy,limit=1):

    P = find_P(theta,a0,a1,b0,b1)
    norma = LA.norm(P)
    if norma != 0:
        Pn = P/norma
    else:
        Pn = P
        print(theta/np.pi, a0/np.pi,a1/np.pi,b0/np.pi,b1/np.pi)
        print(P)

    B, maxi = Best_func(theta,a0,a1,b0,b1, limit)
    P2,_ = Best_point(B,accuracy)
    norma = LA.norm(P2)
    if norma != 0:
        P2n = P2/norma
    else:
        P2n = P2
        print('Best point is zero')
    d = LA.norm(Pn-P2n)
    if d <= accuracy*5:
#        print('B\n',B)
#        print('P\n',P)
        
        return 1
    else:
        return 0


def is_nonlocal(B):
    accuracy = 0.0001
    P, Nonlocal = Best_point(B,accuracy)
    return P, Nonlocal
