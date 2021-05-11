#%%
import numpy as np
import matplotlib.pyplot as plt
import satoshi as St
import tools as T
import symmetricPoints as SP
import tilted as Tl
import generalWolf as GW
import generalDoubleTilted as GDT
import particularPoints as Pp

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

        P = T.find_P(theta, a0,a1,b0,b1)

        isExtremal = St.satoshiTest(P)

        accuracy = 0.001
        isExtremal2 = T.is_exposed(theta,a0,a1,b0,b1, accuracy,limit=1)

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

        a = np.random.rand()*0.25-0.125
        b = np.random.rand()*2-1
        c = np.random.rand()*2-1
        # a = 1/4
        # b = 1/8
        c = 1/np.sqrt(2)
        P = [a,a,b,b,c,c,c,-c]
        isExtremal = St.satoshiTest(P)
        print(isExtremal)

def generalDoubleTiltedRegion(D):
    A = np.linspace(0,2,D)
    Phi = np.linspace(0,np.pi/2,D)
    Plane = np.zeros((D,D))

    for i,phi in enumerate(Phi):
        print(f'{i+1}/{D}')
        for j,a in enumerate(A):
            B = GDT.functional(a, phi)
            accuracy = 0.001
            P, Qn = T.Best_point(B,accuracy)

            Qs = St.satoshiTest(P)
            if Qn:
                if Qs:
                    Plane[i][j] = 1
                else:
                    Plane[i][j] = 2
            else:
                if Qs:
                    Plane[i][j] = 3
                else:
                    Plane[i][j] = 0


    plt.imshow(Plane.T, extent=[0, np.pi/2, 2, 0])
    plt.colorbar()
    plt.show()

def generalWolfRegion(D):
    acc = 0.001
    D = 101
    T = np.linspace(-1,1,D)
    R = np.linspace(0,2,D)
    Plane = np.zeros((D,D))
    Plane2 = np.zeros((D,D))

    for i,t in enumerate(T):
        print(f'{i+1}/{D}')
        for j,r in enumerate(R):
            B = GW.functional(t,r)
            P = GW.quantumPoint(t,r)
            # print(P)
            Qs = St.satoshiTest(P)
            _,Qn= T.Best_point(B,acc)
            Plane[i][j] = Qs
            Plane2[i][j] = Qn
            if Qn and (not Qs):
                Qs = St.satoshiTestComment(P)
    plt.imshow(Plane, extent=[0, 2*np.pi, 4, 0])
    plt.show()
    plt.imshow(Plane2, extent=[0, 2*np.pi, 4, 0])
    plt.show()

def hardySatoshi():
    P = Pp.hardyPoint()
    Qs = St.satoshiTestComment(P)
    print("Hardy point:", Qs)
    correct1, realisation, _ = T.twoQubitRepresentationComment(P)
    if correct1:
        theta, a0, a1, b0, b1 = realisation
        Qn = T.is_exposed(theta, a0, a1, b0, b1, 0.0001)
        print(Qn)

N = 100000
D = 101
# compareSatoshiAndNumeric(N)
# pointsWithTwoSolutions(N)
# generalDoubleTiltedRegion(D)
# generalWolfRegion(D)
# GW.testWolfBQ()
# GW.testWolfPoint()
hardySatoshi()