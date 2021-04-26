import numpy as np

# Pauli matrices:
Gz = np.array([(1,0),(0,-1)])
Gx = np.array([(0,1),(1,0)])
G1 = np.array([(1,0),(0,1)])

def functional(alpha, phi):
    return [alpha*np.cos(phi/2), alpha*np.sin(phi/2), 0, 0, 1, 1, 1, -1]

def CosB(alpha, phi):
    return alpha**2 * np.cos(phi)/4

def Cos2A(alpha, phi):
    l = alpha**2 * np.sin(phi) * (96 - 16 *alpha**2 - alpha**4*(1 + np.cos(2* phi)))
    m = (4 - alpha**2)* (32 - 16 *alpha**2 + alpha**4* (1 + np.cos(2* phi)))
    return l/m

def LValue(alpha, phi):
    alpha*np.cos(phi/2) + alpha*np.sin(phi/2) + 2

def NSValue(alpha, phi):
    return max(LValue(alpha, phi), 4)

def QValue(alpha, phi):
    l = (4 - alpha**2)* (32 - alpha**4*(1 + np.cos(2* phi)))
    m = 32 -16*alpha**2 + alpha**4*(1 + np.cos(2* phi))
    QV = np.sqrt(2)*np.sqrt(l/m)
    if np.abs(alpha) < 2:
        return QV
    else:
        return LValue(alpha, phi)

def BellOperator(alpha, phi, a, b):
    A0 = np.cos(a)*Gz + np.sin(a)*Gx
    A1 = np.cos(a)*Gz - np.sin(a)*Gx
    B0 = Gz
    B1 = np.cos(b)*Gz - np.sin(b)*Gx
    B = alpha*np.cos(phi/2)*np.kron(A0, G1) + alpha*np.sin(phi/2)*np.kron(A1,G1) + np.kron(A0,B0) + np.kron(A0,B1) + np.kron(A1,B0) - np.kron(A1,B1)
    return B
