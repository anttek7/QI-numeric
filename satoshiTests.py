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
acc = 1e-9
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
def satoshiConditions(N):
    s0STLM0 = 0
    s1STLM0 = 0
    s0STLM1 = 0
    s1STLM1 = 0

    for i in range(N):
        if i%100==0:
            print(f'{i+1}/{N}')
        theta = np.random.rand()*np.pi/4
        a0 = np.random.rand()*2*np.pi
        a1 = np.random.rand()*2*np.pi
        b0 = np.random.rand()*2*np.pi
        b1 = np.random.rand()*2*np.pi

        P = T.find_P(theta, a0, a1, b0, b1)
        correct1, realisation, wholeSp = T.twoQubitRepresentation(P)
        theta = realisation[0]
        correct2  = St.STLM(P, theta)
        if not correct1:
            print("no 2-qubit realisation????")
        if correct2 and wholeSp:
            s1STLM1 += 1
        elif correct2 and (not wholeSp):
            s0STLM1 += 1
        elif (not correct2) and wholeSp:
            s1STLM0 += 1
        elif (not correct2) and (not wholeSp):
            s0STLM0 += 1
        print(s0STLM0, s0STLM1, s1STLM0, s1STLM1)

def checkCabelloPoint():
    alpha = np.random.rand()*np.pi
    beta = np.random.rand()*np.pi
    P = Pp.cabelloPoint(alpha, beta, 0, 0)
    print(St.satoshiTest(P))

def testWagnerPoints(theta):
    P = Pp.WagnerPoints(theta)
    b = Pp.getWagnerB(theta)
    _, a0, a1, b0, b1 = Pp.WagnerRealisation(theta)
    wholeSp, theta = St.SPlusCondition(P)
    stlm = St.STLM(P, theta)
    if wholeSp and stlm:
        Qs = 1
    else:
        Qs = 0
    
    accuracy = 0.001
    Qn = T.is_exposed(theta,a0,a1,b0,b1, accuracy,limit=1)
    print(Qn, Qs)


def hypothesis(N):
    def research(theta, a0, a1, b0, b1):
        def maximum(S):
            m = 0
            for x in range(2):
                for y in range(2):
                    if S[x][y] > m:
                        m = S[x][y]
            return m
        def getThetaFromSinSquared(s):
            return np.arccos(np.sqrt(1-s))

        P = T.find_P(theta, a0, a1, b0, b1)
        S_p, S_m = St.SPlusTemp(P)
        print(S_p)
        print(S_m)
        flag = False
        for x in range(2):
                for y in range(2):
                    thetaNew = getThetaFromSinSquared(S_p[x][y])
                    Pnew = T.find_P(thetaNew, a0, a1, b0, b1)
                    stlm = St.STLM(Pnew, thetaNew)
                    print(stlm, "STLM")
                    Qs = St.satoshiTest(Pnew)
                    print(Qs, thetaNew)
                    if Qs:
                        flag = True

        for x in range(2):
                for y in range(2):
                    thetaNew = getThetaFromSinSquared(S_m[x][y])
                    Pnew = T.find_P(thetaNew, a0, a1, b0, b1)
                    stlm = St.STLM(Pnew, thetaNew)
                    print(stlm, "STLM")
                    Qs = St.satoshiTest(Pnew)
                    print(Qs, thetaNew)
                    if Qs:
                        flag = True
        print("i co?", flag)

    def research2(theta, a0, a1, b0, b1):
        def maximum(S):
            m = 0
            for x in range(2):
                for y in range(2):
                    if S[x][y] > m:
                        m = S[x][y]
            return m
        def getThetaFromSinSquared(s):
            return np.arccos(np.sqrt(1-s))
        def largerTheta(theta):
            P = T.find_P(theta, a0, a1, b0, b1)
            S_p, S_m = St.SPlusTemp(P)
            # print(S_p)
            # print(S_m)
            thetaNew = getThetaFromSinSquared(maximum(S_p))
            return thetaNew
        print("Optimal theta start")
        optTheta = optimalTheta(a0,a1,b0,b1)
        print("Optimal theta end")
        flag = False
        while (theta < optTheta - acc) and (not flag):
            Pnew = T.find_P(theta, a0, a1, b0, b1)
            stlm = St.STLM(Pnew, theta)
            Qs = St.satoshiTest(Pnew)
            print(Qs, stlm, theta)
            thetaNew = largerTheta(theta)
            if Qs:
                flag = True
            if abs(thetaNew - theta) < acc:
                flag = True
            theta = thetaNew
        print("Next")

    s0STLM0 = 0
    s1STLM0 = 0
    s0STLM1 = 0
    s1STLM1 = 0

    for i in range(N):
        if i%100==0:
            print(f'{i+1}/{N}')
        theta = np.random.rand()*np.pi/4
        a0 = np.random.rand()*2*np.pi
        a1 = np.random.rand()*2*np.pi
        b0 = np.random.rand()*2*np.pi
        b1 = np.random.rand()*2*np.pi

        P = T.find_P(theta, a0, a1, b0, b1)
        correct1, realisation, wholeSp = T.twoQubitRepresentation(P)
        theta = realisation[0]
        correct2 = St.STLM(P, theta)
        if not correct1:
            print("no 2-qubit realisation????")
        if correct2 and wholeSp:
            s1STLM1 += 1
        elif correct2 and (not wholeSp):
            research2(theta, a0, a1, b0, b1)
            s0STLM1 += 1
        elif (not correct2) and wholeSp:
            s1STLM0 += 1
        elif (not correct2) and (not wholeSp):
            s0STLM0 += 1
        # print(s0STLM0, s0STLM1, s1STLM0, s1STLM1)

def optimalTheta(a0,a1,b0,b1):
    n = 100
    Theta = np.linspace(0, np.pi/2, n)
    flag = 0
    
    for theta in Theta:
        P = T.find_P(theta, a0, a1, b0, b1)
        Q = St.satoshiTest(P)
        if Q:
            # print(theta,"treshold")
            flag = 1
            
            # return theta
        if (not Q) and flag == 1:
            print(theta, "break")
    if flag == 0:
        print("not extremal")


N = 100000
# for i in range(N):
#     if i%100==0:
#         print(f'{i+1}/{N}')
#     a0 = np.random.rand()*2*np.pi
#     a1 = np.random.rand()*2*np.pi
#     b0 = np.random.rand()*2*np.pi
#     b1 = np.random.rand()*2*np.pi
#     optimalTheta(a0,a1,b0,b1)
#     print("asdfghjklkjhfdsdfghjkc")
D = 101
compareSatoshiAndNumeric(N)
# pointsWithTwoSolutions(N)
# generalDoubleTiltedRegion(D)
# generalWolfRegion(D)
# GW.testWolfBQ()
# GW.testWolfPoint()
# hardySatoshi()
# satoshiConditions(N)
# checkCabelloPoint()
# for theta in np.linspace(0, np.pi/4, 100):
#     testWagnerPoints(theta)

# hypothesis(N)