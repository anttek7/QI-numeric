import numpy as np

# Pauli matrices:
Gz = np.array([(1,0),(0,-1)])
Gx = np.array([(0,1),(1,0)])
G1 = np.array([(1,0),(0,1)])


def functional(alpha):
    return [alpha, 0, 0, 0, 1, 1, 1, -1]

def fromAlpha(alpha):
    theta = np.arctan(np.sqrt(2/alpha**2 - 0.5))/2
    mu = np.arctan(np.sin(theta))
    return mu, theta

def fromMu(mu):
    theta = np.arcsin( np.tan(mu) )
    alpha = 2/(np.sqrt(1 + 2*np.tan(theta)**2))
    return alpha, theta

def fromTheta(theta):
    alpha = 2/(np.sqrt(1 + 2*np.tan( theta )**2))
    mu = np.arctan( np.sin(theta) )
    return alpha, mu

def BellOperator(mu):
    alpha, _ = fromMu(mu)
    A0 = Gz
    A1 = Gx
    B0 = np.cos(mu)*Gz + np.sin(mu)*Gx
    B1 = np.cos(mu)*Gz - np.sin(mu)*Gx
    # state = [np.cos(theta/2), 0, 0, np.sin(theta/2)]
    B = alpha*A0 + np.kron(A0,B0) + np.kron(A0,B1) + np.kron(A1,B0) - np.kron(A1,B1)
    return B

def LValue(alpha):
    return alpha + 2

def QValue(alpha):
    if alpha >= 0 and alpha < 2:
        return np.sqrt(8 + 2*alpha**2)
    else:
        return LValue(alpha)

def NSValue(alpha):
    return max(LValue(alpha), 4)
