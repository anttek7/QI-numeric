import numpy as np

# Pauli matrices:
Gz = np.array([(1,0),(0,-1)])
Gx = np.array([(0,1),(1,0)])
G1 = np.array([(1,0),(0,1)])


def func_B(t,r):
    B = [t, t, r, 0, 1, 1, 1, -1]
    return B

def BellOperator(t, r, a, b):
    A0 = np.cos(a/2)*Gz + np.sin(a/2)*Gx
    A1 = np.cos(a/2)*Gz - np.sin(a/2)*Gx
    B0 = Gz
    B1 = np.cos(b)*Gz + np.sin(b)*Gx
    B = t*(np.kron(A0, G1) + np.kron(A1,G1)) + r*np.kron(G1, B0) + np.kron(A0,B0) + np.kron(A0,B1) + np.kron(A1,B0) - np.kron(A1,B1)
    return B

def LValue(t,r):
    L1 = 2*t + r + 2
    L2 = -2*t + r - 2
    L3 = -2*t - r + 2
    L4 = r + 2
    return max([L1,L2,L3,L4])

def NSValue(t,r):
    return max(LValue(t, r), 4)

def bAnal(t,r):
    return np.pi/2

def CosAhalf(t, r):
    s = (r**2 - 4*t**2 + 4)/(2 - t**2)
    l = r*t + np.sqrt(s) 
    m = 2*(1 - t**2)
    return l/m

def QValue(t,r):
    s = (4 + r**2 - 4*t**2)*(2 - t**2)
    l = r*t + np.sqrt(s)
    m = 1 - t**2
    QV = l/m
    if np.abs(t) < 1:
        return QV
    else:
        return LValue(t, r)

def cotThetaHalf(t,r):
    l = r*np.sqt(2 - t**2) + np.sqrt(4 + r**2 - 4*t**2)*(1 + t - t**2)
    m1 = -2*r*t*np.sqrt((2 - t**2) * (4 + r**2 - 4*t**2))
    m2 = -r**2 * (1 + 2*t**2 - t**4) - 4*(-1 +4*t**2 -4*t**4 + t**6)
    return l/np.sqrt(m1+m2)

def SinTheta(cot):
    return (2*cot)/(1 + cot**2)

def CosTheta(cot):
    return (cot**2 - 1)/(1 + cot**2)

def SinA(t,r):
    cos = CosAhalf(t,r)
    return np.sqrt(1-cos**2)

def quantumPoint(t,r):
    cot = cotThetaHalf(t,r)
    sinT = SinTheta(cot)
    cosT = CosTheta(cot)
    sinA = SinA(t,r)
    cosA = CosAhalf(t,r)
    return (cosA*cosT, cosA*cosT, cosT, 0, cosA, sinA*sinT, cosA, -sinA*sinT)
