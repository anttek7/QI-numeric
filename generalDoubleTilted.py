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
def find_correlator(Operator1, Operator2, state):
    rho = np.outer(state,state)
    M = np.matmul(rho, np.kron(Operator1, Operator2))
    return np.trace(M)

def quantumPoint(alpha, phi):
    cos2a = Cos2A(alpha, phi)
    if np.abs(cos2a) <= 1:
        a = np.arccos(cos2a)/2
    else:
        return np.ones(8)/4
    b = np.arccos(CosB(alpha, phi))
    B = BellOperator(alpha, phi, a, b)
    w, v = np.linalg.eig(B)
    state = v[:, np.argmax(w)]
    
    I = np.eye(2)
    A0 = np.cos(a)*Gz + np.sin(a)*Gx
    A1 = np.cos(a)*Gz - np.sin(a)*Gx
    B0 = Gz
    B1 = np.cos(b)*Gz - np.sin(b)*Gx
    mA0 = find_correlator(A0, I, state)
    mA1 = find_correlator(A1, I, state)
    mB0 = find_correlator(I, B0, state)
    mB1 = find_correlator(I, B1, state)
    A0B0 = find_correlator(A0, B0, state)
    A0B1 = find_correlator(A0, B1, state)
    A1B0 = find_correlator(A1, B0, state)
    A1B1 = find_correlator(A1, B1, state)
    return [mA0, mA1, mB0, mB1, A0B0, A0B1, A1B0, A1B1]        
