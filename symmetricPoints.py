import numpy as np

def symmetricPoint(theta, a):
    P = (np.cos(theta), np.cos(theta)*np.cos(a), np.cos(theta), np.cos(theta)*np.cos(a),
        1, np.cos(a), np.cos(a),np.cos(a)**2 - np.sin(theta)*np.sin(a)**2 )
    return P
def symmetricPoint2(theta, Cosa):
    P = (np.cos(theta), np.cos(theta)*Cosa, np.cos(theta), np.cos(theta)*Cosa,
        1, Cosa, Cosa,Cosa**2 - np.sin(theta)*(1- Cosa**2))
    return P
