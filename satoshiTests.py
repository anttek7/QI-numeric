#%%
import numpy as np
import matplotlib.pyplot as plt
import satoshi as St
import optimisation as Op
import symmetricPoints as SP
import tilted as Tl
import generalWolf as GW
import generalDoubleTilted as GDT

def compareSatoshiAndNumeric(N):
    bothEx = 0
    bothNoEx = 0
    SatEx = 0
    NumEx = 0

    for i in range(N):
        if i%100==0:
            print(f'{i+1}/{N}')
        theta = np.random.rand()*np.pi/4
        a0 = np.random.rand()*2*np.pi
        a1 = np.random.rand()*2*np.pi
        b0 = np.random.rand()*2*np.pi
        b1 = np.random.rand()*2*np.pi

        P = Op.find_P(theta, a0,a1,b0,b1)

        isExtremal = St.satoshiTest(P)

        accuracy = 0.001
        isExtremal2 = Op.is_exposed(theta,a0,a1,b0,b1, accuracy,limit=1)

        if isExtremal:
            if isExtremal2:
                bothEx += 1
            else:
                SatEx += 1
        else:
            if isExtremal2:
                NumEx += 1
            else:
                bothNoEx += 1
        print(bothNoEx, bothEx, SatEx, NumEx)


def pointsWithTwoSolutions(N):
    for i in range(N):
        if i%100==0:
            print(f'{i+1}/{N}')

        a = np.random.rand()*2-1
        b = np.random.rand()*2-1
        c = np.random.rand()*2-1

        P = [a,a,b,b,c,c,c,c]
        isExtremal = St.satoshiTest(P)
        print(isExtremal)

def generalDoubleTiltedRegion():
    eps = 0.001
    D = 101
    A = np.linspace(0,4,D)
    Fi = np.linspace(0,2*np.pi,D)
    Plane = np.zeros((D,D))

    for i,fi in enumerate(Fi):
        print(f'{i+1}/{D}')
        for j,a in enumerate(A):
            B = func_B(a,fi)
            accuracy = 0.001
            # P2,Q, W_max, alpha, beta, state = Best_point(B,accuracy)
            isExtremal = St.satoshiTest(P)
            Plane[i][j] = Q

    plt.imshow(Plane.T, extent=[0, 2*np.pi, 4, 0])
    plt.show()

N = 1000
# compareSatoshiAndNumeric(N)
# pointsWithTwoSolutions(N)
generalDoubleTiltedRegion()