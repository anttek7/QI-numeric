import numpy as np
import matplotlib.pyplot as plt
import tools as T
import satoshi as St
def symmetricPoint(theta, a):
    P = (np.cos(theta), np.cos(theta)*np.cos(a), np.cos(theta), np.cos(theta)*np.cos(a),
        1, np.cos(a), np.cos(a),np.cos(a)**2 - np.sin(theta)*np.sin(a)**2 )
    return P
def symmetricPoint2(theta, Cosa):
    P = (np.cos(theta), np.cos(theta)*Cosa, np.cos(theta), np.cos(theta)*Cosa,
        1, Cosa, Cosa,Cosa**2 - np.sin(theta)*(1- Cosa**2))
    return P

def plotExposed(D):
    acc = 1e-4
    x0 = 0
    x_end = np.pi/2
    y0 = 0
    y_end = 1
    Y = np.linspace(y0, y_end, D)
    X = np.linspace(x0, x_end, D)
    Map = np.zeros((D,D))
    for y,CosA in enumerate(Y):
        for x,theta in enumerate(X):

            a0 = 0
            b0 = 0
            a1 = np.arccos(CosA)
            b1 = -np.arccos(CosA)+2*np.pi
            P = T.find_P(theta, a0, a1, b0, b1)
            
            P2 = symmetricPoint2(theta, CosA)
            # print(np.round(P-P2,7))
            # exp1 = T.is_exposed(theta, a0, a1, b0, b1, acc)
            exp2 = T.is_exposed_hypo(theta, a0,a1,b0,b1)
            exp3 = St.satoshiTest(P)
            nonloc = T.is_nonlocalPoint(P)
            print(exp2)
            stlm = St.STLM(P, theta)
            Ptlm = T.find_P(np.pi/2, a0, a1, b0, b1)
            tlm = T.TLM(Ptlm)
            print(stlm, tlm)
            Map[x][D-y-1] = exp2
    plt.imshow(Map.T, extent=[x0, x_end, y0, y_end])
    plt.colorbar()
    plt.xlabel("theta")
    plt.ylabel("cos(a)")
    plt.savefig("symmetric.png")
    plt.show()

# D = 40
# plotExposed(D)
