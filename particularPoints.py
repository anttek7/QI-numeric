import numpy as np
import tools as T
Gz = np.array([(1,0),(0,-1)])
Gx = np.array([(0,1),(1,0)])
I = np.array([(1,0),(0,1)])


def hardyPoint():
    A0 = 5 - 2*np.sqrt(5)
    A1 = np.sqrt(5) - 2
    B0 = 5 - 2*np.sqrt(5)
    B1 = np.sqrt(5) - 2
    A0B0 = 6*np.sqrt(5) - 13
    A0B1 = 3*np.sqrt(5) - 6
    A1B0 = 3*np.sqrt(5) - 6
    A1B1 = 2*np.sqrt(5) - 5
    return (A0,A1,B0,B1,A0B0,A0B1, A1B0,A1B1)


def cabelloPoint(alpha, beta, phi, zeta):
    def observable0():
        return Gz
    def observable1(alpha, phi):
        v1 = [np.cos(alpha/2), np.exp(1j*phi)*np.sin(alpha/2)]
        v2 = [-np.sin(alpha/2), np.exp(1j*phi)*np.cos(alpha/2)]
        v1 = [np.cos(alpha/2), np.sin(alpha/2)]
        v2 = [-np.sin(alpha/2), np.cos(alpha/2)]
        return np.outer(v1, v1) - np.outer(v2, v2)
    
    def getStateGeneral(delta, c, alpha, beta, phi, zeta):
        a1 = np.exp(1j*delta)*np.sqrt(1 - c**2*(1 + np.tan(alpha/2)**2 +np.tan(beta/2)**2))
        a2 = -c * np.exp(-1j*phi) * np.tan(alpha/2)
        a3 = -c * np.exp(-1j*zeta) * np.tan(beta/2)
        a4 = c
        return [a1,a2,a3,a4]
    def getStateMaximal(phi, zeta):
        k00 = 1/6 * (4 - (np.cbrt((53 - 6*np.sqrt(78))**2) + 1)/(np.cbrt(53 - 6*np.sqrt(78))) )
        x = 67*np.sqrt(78) - 414
        k01 = -np.sqrt(3)/2 * (12 + (216*np.cbrt(x**2) - 31*np.cbrt(36))/np.cbrt(x) )**(-1/2)
        k10 = k01
        k11 = 1/6 * ((np.cbrt((307 + 39*np.sqrt(78))**2) - 29)/(np.cbrt(307 + 39*np.sqrt(78))) - 2)
        print(k00, k01, k10, k11)
        state = [k00*np.exp(-1j*(zeta+phi)), k01*np.exp(-1j*phi), k10*np.exp(-1j*zeta), k11]
        state2 = [k00, -0.5781, -0.5781, k11]
        print(np.linalg.norm(state2))
        return state2
    state = getStateMaximal(phi, zeta)
    A0 = observable0()
    A1 = observable1(alpha, phi)
    B0 = observable0()
    B1 = observable1(beta, zeta)
    P = T.generalPoint(state, A0, A1, B0, B1)
    return P

def getWagnerB(theta):
    x = np.sqrt((1 + 0.5*np.cos(2*theta)**2)/np.sin(2*theta)**2)
    return np.arctan(x)

def WagnerPoints(theta):
    b = getWagnerB(theta)
    state = [np.cos(theta), 0, 0, np.sin(theta)]
    A0 = Gz
    A1 = Gx
    B0 = np.cos(b)*Gx + np.sin(b)*Gz
    B1 = np.cos(b)*Gx - np.sin(b)*Gz
    P = T.generalPoint(state,A0,A1,B0,B1)
    return P

def WagnerRealisation(theta):
    b = getWagnerB(theta)
    state = [np.cos(theta), 0, 0, np.sin(theta)]
    a0 = 0
    a1 = np.pi/2
    b0 = np.pi/2 - b
    b1 = - b0
    return state, a0, a1, b0, b1