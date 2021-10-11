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
Gz = np.array([(1,0),(0,-1)])
Gx = np.array([(0,1),(1,0)])
G1 = np.array([(1,0),(0,1)])

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

def testWagnerPoints(N):
    for theta in np.linspace(0, np.pi/4, N):
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
        print(theta, Q)
    #     if Q:
    #         print(theta,"treshold")
    #         flag = 1
            
    #         # return theta
    #     if (not Q) and flag == 1:
    #         print(theta, "break")
    # if flag == 0:
    #     print("not extremal")
    # return flag

def tresholdTheta(a0,a1,b0,b1, acc):
    l = 0
    r = np.pi/2
    P0 = T.find_P(r, a0, a1, b0, b1)
    if not T.TLM(P0):
        return np.pi/2
    while r-l > acc:
        theta = (r+l)/2
        
        Q = newTest(theta, a0, a1, b0, b1)
        # print(l,r,Q)
        
        if Q:
            r = theta
        else:
            l = theta
    return (r+l)/2

def hypothesis2(N):
    tlm0flag0 = 0
    tlm0flag1 = 0
    tlm1flag0 = 0
    tlm1flag1 = 0
    
    for i in range(N):
        if i%100==0:
            print(f'{i+1}/{N}')
        a0 = np.random.rand()*2*np.pi
        a1 = np.random.rand()*2*np.pi
        b0 = np.random.rand()*2*np.pi
        b1 = np.random.rand()*2*np.pi
        theta = np.pi/2
        P = T.find_P(theta, a0, a1, b0, b1)
        tlm = T.TLM(P)
        # print("TLM", tlm)
        flag = optimalTheta(a0,a1,b0,b1)
        # print(tlm, flag, "tutaj")
        if tlm:
            if flag:
                tlm1flag1 += 1
            else:
                tlm1flag0 += 1
        else:
            if flag:
                tlm0flag1 += 1
            else:
                tlm0flag0 += 1
        print(tlm1flag1,tlm1flag0,tlm0flag1,tlm0flag0 )

def newTest(theta,a0,a1,b0,b1):
    Pmax = T.find_P(np.pi/2, a0, a1, b0, b1)
    tlm = T.TLM(Pmax)
    if tlm:
        P = T.find_P(theta, a0, a1, b0, b1)
        wholeSp, thetaNew = St.SPlusCondition(P)
        if np.abs(thetaNew-theta) > acc:
            print("something is wrong with theta")
        if wholeSp:
            return 1
    return 0
def compareTests(N):
    St0we0 = 0
    St0we1 = 0
    St1we0 = 0
    St1we1 = 0
    
    for i in range(N):
        if i%100==0:
            print(f'{i+1}/{N}')
        a0 = np.random.rand()*2*np.pi
        a1 = np.random.rand()*2*np.pi
        b0 = np.random.rand()*2*np.pi
        b1 = np.random.rand()*2*np.pi
        theta = np.random.rand()*np.pi/2
        P = T.find_P(theta, a0, a1, b0, b1)
        Qs = St.satoshiTest(P)
        Qw = newTest(theta, a0, a1, b0, b1)

        if Qs:
            if Qw:
                St1we1 += 1
            else:
                St1we0 += 1
                print(theta, a0,a1,b0,b1)
                check(theta, a0,a1,b0,b1)
                break
        else:
            if Qw:
                St0we1 += 1
                print(theta, a0,a1,b0,b1)
                check(theta, a0,a1,b0,b1)
                break

            else:
                St0we0 += 1
        print(St1we1, St1we0, St0we1, St0we0)

def check(theta, a0,a1,b0,b1):
    P = T.find_P(theta, a0, a1, b0, b1)
    Qs = St.satoshiTestComment(P)
    tlm = T.TLM(T.find_P(np.pi/2, a0, a1, b0, b1))
    wholeSp, thetaNew = St.SPlusCondition(P)
    Qw = newTest(theta, a0, a1, b0, b1)
    # flag = optimalTheta(a0,a1,b0,b1)
    Qn = T.is_exposed(theta, a0, a1, b0, b1, 0.0001)
    print(theta, a0,a1,b0,b1)
    print(Qs, Qw, Qn,  "extremal")
    print(tlm, wholeSp)

def functionalBehaviour(N):
    for i in range(N):
        print(f'{i+1}/{N}')
        a0 = np.random.rand()*2*np.pi
        a1 = np.random.rand()*2*np.pi
        b0 = np.random.rand()*2*np.pi
        b1 = np.random.rand()*2*np.pi
        F1 = []
        Tet = []
        MAXI = []
        tlm = T.TLM(T.find_P(np.pi/2, a0, a1, b0, b1))
        print(tlm)
        
        if tlm:
            thetag = T.getThetaFromSinSquared(T.hypoTreshold(a0,a1,b0,b1)**2)
            n = 500
            Theta = np.linspace(thetag, np.pi/2, n)            
        
            for theta in Theta:
                # a0 = 3/4*np.pi
                # a1 = np.pi/4
                # b0 = np.pi/2
                # b1 = np.pi
                P = T.find_P(theta, a0, a1, b0, b1)
                wholeSp, thetaNew = St.SPlusCondition(P)
                # if wholeSp:
                # f1,m1 = T.Best_func(theta, a0, a1, b0, b1)
                f2,m2 = T.Best_func_restricted(theta, a0, a1, b0, b1)
                # print(m1-m2,"hmm")
                # print(f1)
                # print(f2)
                MAXI.append(m2)
                Tet.append(theta)        
                F1.append(f2)
            # plt.plot(Tet,F1)
            plt.plot(Tet,MAXI)
            plt.show()
def verify(theta, a0,a1,b0,b1):
    def createCorrelatorsAndMarginals(P):
        A = np.array([P[0], P[1]])
        B = np.array([P[2], P[3]])
        AB = np.array([[P[4], P[5]],[P[6], P[7]]])
        return A,B,AB
    P = T.find_P(theta, a0, a1, b0, b1)
    A,B,AB = createCorrelatorsAndMarginals(P)
    S = np.zeros((2,2))
    for x in range(2):
        for y in range(2):
            S[x][y] = AB[x][y]**2 - A[x]**2 - B[y]**2 + 1 - 2+np.sin(theta)**2
    return S
def tresholdThetaTest(N):
    for i in range(N):
        if i%100==0:
            print(f'{i+1}/{N}')
        a0 = np.random.rand()*2*np.pi
        a1 = np.random.rand()*2*np.pi
        b0 = np.random.rand()*2*np.pi
        b1 = np.random.rand()*2*np.pi
        treshold = tresholdTheta(a0,a1,b0,b1, 1e-5)
        
        sinThetaHypo = T.hypoTreshold(a0,a1,b0,b1)
        thetaHypo = T.getThetaFromSinSquared(sinThetaHypo)
        P = T.find_P(thetaHypo, a0,a1,b0,b1)
        if T.is_nonlocalPoint(P):
            print(sinThetaHypo, np.sin(treshold), "nonlocal hypo")
        else:
            print(sinThetaHypo, np.sin(treshold), "local hypo")


def localPointsBeforeTreshold(N):
    def hypoTreshold(a0,a1,b0,b1):
        def value(a,b):
            return -np.sin(a)*np.sin(b)/(np.cos(a)*np.cos(b) - 1)
        return value(a0,b0), value(a0,b1),value(a1,b0),value(a1,b1)

    for i in range(N):
        if i%100==0:
            print(f'{i+1}/{N}')
        a0 = np.random.rand()*np.pi
        a1 = np.random.rand()*np.pi
        b0 = np.random.rand()*np.pi
        b1 = np.random.rand()*np.pi
        P = T.find_P(np.pi/2, a0, a1, b0, b1)
        if T.TLM(P):
            # treshold = tresholdTheta(a0,a1,b0,b1, 1e-5)
            tresholds = hypoTreshold(a0,a1,b0,b1)
            n = 100
            # Theta = np.linspace(0, treshold,n)
            Theta = np.linspace(0, np.pi/2,n)
            for theta in Theta:
                for tresh in tresholds:
                    if np.abs(theta - tresh) < 0.03:
                        print("facet")
                P = T.find_P(theta, a0, a1, b0, b1)
                nl = T.is_nonlocalPoint(P)
                print(nl)
            print("next")
def tlmVSstlm(N):
    for i in range(N):
        if i%100==0:
            print(f'{i+1}/{N}')
        a0 = np.random.rand()*2*np.pi
        a1 = np.random.rand()*2*np.pi
        b0 = np.random.rand()*2*np.pi
        b1 = np.random.rand()*2*np.pi
        theta = np.random.rand()*np.pi/2
        P1 = T.find_P(np.pi/2, a0,a1,b0,b1)
        P2 = T.find_P(theta, a0,a1,b0,b1)
        tlm = T.TLM(P1)
        stlm = St.STLM(P2, theta)
        wholeSp, theta = St.SPlusCondition(P2)
        # print(tlm, stlm)
        if stlm and (not tlm):
            print(tlm, stlm)
        if stlm and (not tlm) and wholeSp:
            print(tlm, stlm, wholeSp)
            print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

def geometry(N):
    A = np.linspace(0,2,N)
    for a in A:
        mu, theta = Tl.fromAlpha(a)
        A0 = Gz
        A1 = Gx
        B0 = np.cos(mu)*Gz + np.sin(mu)*Gx
        B1 = np.cos(mu)*Gz - np.sin(mu)*Gx
        Psi = [np.cos(theta/2), 0, 0, np.sin(theta/2)]
        vA0 = np.kron(A0, np.identity(2))@Psi
        vA1 = np.kron(A1, np.identity(2))@Psi
        vB0 = np.kron(np.identity(2),B0)@Psi
        vB1 = np.kron(np.identity(2),B1)@Psi
        # print(vA0,vA1,vB0,vB1)
        
        M = np.vstack((vA0,vA1,vB0,vB1,Psi))
        # print(M,"ihhihi")
        print(np.linalg.matrix_rank(M))
def minimalTheta():
    eps = 1e-5
    a0 = eps**2
    b0 = eps
    a1 = np.pi-eps
    b1 = np.pi-eps**2
    theta = np.arcsin(T.hypoTreshold(a0,a1,b0,b1))
    print(theta)
    Q = T.is_exposed(theta+eps, a0, a1, b0, b1, 1e-5)
    P = T.find_P(theta+eps, a0, a1, b0, b1)
    nl = T.is_nonlocalPoint(P)
    print(Q, nl)

def zeroProbability(N):
    def sinT(a,b,ax,by):
        return - ((-1)**(a+b)*np.sin(ax)*np.sin(by))/(1 + (-1)**(a+b)*np.cos(ax)*np.cos(by))
    def cosT(a,b,ax,by):
        return -(np.cos(ax + (-1)**(a+b)*np.cos(by)))/(1 + (-1)**(a+b)*np.cos(ax)*np.cos(by))
    def expression(a,b,ax,by):
        c1 = 1 + (-1)**(a+b)*np.cos(ax)*np.cos(by)
        c2 = (-1)**(a)*np.cos(ax) + (-1)**(b)*np.cos(by)
        c3 = (-1)**(a+b)*np.sin(ax)*np.sin(by)
        return c1 + cosT(a,b,ax,by)*c2 + sinT(a, b, ax, by)*c3
    def expression2(theta,a,b,ax,by):
        c1 = 1 + (-1)**(a+b)*np.cos(ax)*np.cos(by)
        c2 = (-1)**(a)*np.cos(ax) + (-1)**(b)*np.cos(by)
        c3 = (-1)**(a+b)*np.sin(ax)*np.sin(by)
        return c1 + np.cos(theta)*c2 + np.sin(theta)*c3
    for i in range(N):
        if i%100==0:
            print(f'{i+1}/{N}')
        a0 = np.random.rand()*2*np.pi
        a1 = np.random.rand()*2*np.pi
        b0 = np.random.rand()*2*np.pi
        b1 = np.random.rand()*2*np.pi
        theta = np.arcsin(T.hypoTreshold(a0,a1,b0,b1))
        aa = [a0,a1]
        bb = [b0,b1]
        P = T.find_P(theta,a0, a1, b0, b1)
        A,B,AB = T.createCorrelatorsAndMarginals(P)
        for x in range(2):
            for y in range(2):
                v1 = 1 + A[x] + B[y] + AB[x][y]
                v2 = 1 + A[x] - B[y] - AB[x][y]
                v3 = 1 - A[x] + B[y] - AB[x][y]
                v4 = 1 - A[x] - B[y] + AB[x][y]
                print(x,y)
                print(np.sin(aa[x])*np.sin(bb[y]))
                print(v1,v2,v3,v4)
                print(expression2(theta,0,0, aa[x],bb[y]), expression2(theta,0,1, aa[x],bb[y]), expression2(theta,1,0, aa[x],bb[y]), expression2(theta,1,1, aa[x],bb[y]))
                print(expression(0,0, aa[x],bb[y]), expression(0,1, aa[x],bb[y]), expression(1,0, aa[x],bb[y]), expression(1,1, aa[x],bb[y]))
                print(np.sin(theta), sinT(0,0,aa[x],bb[y]), sinT(0,1,aa[x],bb[y]), sinT(1,0,aa[x],bb[y]), sinT(1,1,aa[x],bb[y]))
                
def SpSmTheta(N):
    Theta = np.linspace(0, np.pi/2,N)
    
    a0 = np.random.rand()*2*np.pi
    a1 = np.random.rand()*2*np.pi
    b0 = np.random.rand()*2*np.pi
    b1 = np.random.rand()*2*np.pi
    thetaB = np.arcsin(T.hypoTreshold(a0,a1,b0,b1))
    print("graniczna", thetaB)
    for theta in Theta:
        P = T.find_P(theta,a0, a1, b0, b1)
        T.SpSm(P)
        print(theta,"\n")

def SpvsThreshold(N):
    for i in range(N):
        if i%100 == 0:
            print(i)
        a0 = np.random.rand()*2*np.pi
        a1 = np.random.rand()*2*np.pi
        b0 = np.random.rand()*2*np.pi
        b1 = np.random.rand()*2*np.pi
        theta = np.random.rand()*np.pi/2
        thetaB = np.arcsin(T.hypoTreshold(a0,a1,b0,b1))
        # print("graniczna", thetaB)
        P = T.find_P(theta,a0, a1, b0, b1)
        c1, thetan = St.SPlusCondition(P)
        if theta >= thetaB:
            c2 = 1
        else:
            c2 = 0
        if c1 != c2:
            print(c1,c2,"aaaaaa")
def nonnegativitySingularity(N):
    for i in range(1):
        a0 = 0
        a1 = np.random.rand()*2*np.pi
        b0 = 0
        b1 = np.random.rand()*2*np.pi
        
        P = T.find_P(np.pi/2,a0, a1, b0, b1)
        # print(P)
        tlm = T.TLM(P)
        # Sp, Sm = St.SPlusTemp(P)
        # print(Sp)
        # print(Sm)
        # theta = np.random.rand()*np.pi/2
        # thetaB = np.arcsin(T.hypoTreshold(a0,a1,b0,b1))
        # print(thetaB)
        thetaGr =  np.arcsin(T.hypoTresholdImproved(a0,a1,b0,b1))
        
        print(thetaGr)
        Theta = np.linspace(0,np.pi/2,N)    
        for theta in Theta:
            # print(theta)
            # if i%100 == 0:
            #     print(i)
            P = T.find_P(theta,a0, a1, b0, b1)
            stlm = St.STLM(P,theta)

            c1, thetan = St.SPlusCondition(P)
            if theta >= thetaGr:
                c2 = 1
            else:
                c2 = 0
            # sat = St.satoshiTest(P)
            # ex = T.is_exposed(theta,a0,a1,b0,b1, 0.001)
            if c1 and (not stlm):
                stlm2 = St.STLMComment(P, theta)
            print(tlm, stlm, c1, c2)


def moveHardy(N):
    P = Pp.hardyPoint()
    ok, realisation, wholeSp = T.twoQubitRepresentation(P)
    print(ok, realisation, wholeSp)
    print(realisation[2]-2*np.pi,realisation[3]-2*np.pi )
    if not ok:
        print("errorrr")
    else:
        theta, a0,a1,b0,b1 = realisation
        print(np.cos(a0), np.cos(a1), np.cos(b0), np.cos(b1))
        Sp, Sm = St.SPlusTemp(P)
        print(Sp)
        print(Sm)
        # Theta = np.linspace(theta, np.pi/2, N)
        # for th in Theta:
        #     P = T.find_P(th,a0, a1, b0, b1)
        #     Sp, Sm = St.SPlusTemp(P)
        #     print(Sp)
        #     print(Sm)
        #     print("\n")

def decompositionTest(N):
    for i in range(N):
        P = np.random.rand(8)*2-1
        Deter = T.find_deterministic_points()
        # for d in Deter:
        #     print(d)
        alpha, min, status = T.decomposeP(P, Deter)
        if status == 0:
            print(P, "point")
            print(alpha, min, "alpha, min\n")
def decomposition(N):
    Deter = T.find_deterministic_points()
    a0 = 0#np.random.rand()*np.pi 
    b0 = 0#np.random.rand()*np.pi
    a1 = np.random.rand()*np.pi
    b1 = np.random.rand()*np.pi
    Theta = np.linspace(0,np.pi/2,N)
    theta_b = np.arcsin(T.hypoTresholdImproved(a0,a1,b0,b1))
    P_b =  T.find_P(theta_b, a0,a1,b0,b1)
    decomposers = np.r_[Deter, [P_b]]
    
    for theta in Theta:
        # theta = np.random.rand()*np.pi/2
        P = T.find_P(theta, a0,a1,b0,b1)
        # print(decomposers)
        alpha, min, status = T.decomposeP(P, decomposers)
        if status == 0:
            nonl = T.is_nonlocalPoint(P)
            print(theta)
            print(nonl)
            # print(P, "point")
            # print(alpha, min, "alpha, min\n")
        else:
            print('xd')
    print(theta_b)
def notUniquePoint():
    # P = [0.4445537842667646, 0.24544802147218514, 0.734354006208671, 0.734354006208671, 0.542908951128687, 0.542908951128687, 0.3966948365612542, 0.3966948365612542]
    P5 = [0.024973, 0.058992, 0.5,      0.5,      0.07546,  0.07546,  0.092469, 0.092469]
    P4 = [0.25, 0.125, -0.262176, -0.0930301, -0.731555, -0.689269, -0.698783, 0.654382]
    P2 = [0.5, 0.5, -np.sqrt(2/5),0,-np.sqrt(5/2)/2, -np.sqrt(5/2)/2 + 1/ np.sqrt(10), -np.sqrt(5/2)/2, np.sqrt(5/2)/2 - 1/ np.sqrt(10)]
    P3 = [1/4, 1/2, -np.sqrt(7/3)/3, 1/np.sqrt(21), -37/(12*np.sqrt(21)), -37/(12*np.sqrt(21)) + 1/4*(np.sqrt(7/3)/3+1/np.sqrt(21)), -np.sqrt(7/3)/12-37/(12*np.sqrt(21)), -np.sqrt(7/3)/12+43/(12*np.sqrt(21))]
    b = np.random.rand()
    a1 = np.random.rand()
    a0 = np.random.rand()
    b= 0.5
    a0 = 0.25 
    a1 = 0.5
    A0B0 = (a0 + a1 + a0*b**2 - a1*b**2)/(2*b)
    A0B1 = A0B0
    A1B0 = A0B0 - b*(a0 - a1)
    A1B1 = A0B0 - b*(a0 - a1)
    P = [a0, a1, b, b, A0B0, A0B1, A1B0, A1B1]
    # if A0B0 < 1:
    print(P, "punkt")
    correct, realisation, wholeSp = T.twoQubitRepresentation(P)
    theta, a0, a1, b0, b1 = realisation
    # print(correct, theta, a0/np.pi, a1/np.pi, b0/np.pi, b1/np.pi, wholeSp, np.sin(theta)**2, "tu")
    # print(P[0]/np.cos(theta), P[1]/np.cos(theta), P[2]/np.cos(theta), P[3]/np.cos(theta), "cosinusy prawdziwe")
    print(St.STLM(P, theta), "stlm")
    Ptlm = T.find_P(np.pi/2, a0,a1,b0,b1)
    
    print(T.find_P(theta, a0,a1,b0,b1), "Punkt zwrotny")

    print(T.TLM(Ptlm), "Tlm")
    print(T.is_exposed(theta, a0,a1,b0,b1,0.001), "exposed")

def tangentToCriticalPoint(N):
    a0 = 0 
    b0 = 0
    a1 = np.random.rand()*np.pi
    b1 = np.random.rand()*np.pi
    theta_b = np.arcsin(T.hypoTresholdImproved(a0,a1,b0,b1))

    def pointOnTangent(g):
        P0 = np.array([0, 0, 0, 0, 1, np.cos(b1), np.cos(a1), np.cos(a1)*np.cos(b1)])
        Pm = np.array([1, np.cos(a1), 1, np.cos(b1), 0, 0, 0, 0])
        Pc = np.array([0, 0, 0, 0, 0, 0, 0, np.sin(a1)*np.sin(b1)])
        return P0 + Pm*(np.cos(theta_b) - g*np.sin(theta_b)) + Pc*(np.sin(theta_b) + g*np.cos(theta_b))

    P_b =  T.find_P(theta_b, a0,a1,b0,b1)
    nonNeg = T.isOnNonNegativityFacet(P_b)
    BrokenNonNeg = T.BeyondNonNegativityFacet(P_b)
    nonLoc = T.is_nonlocalPoint(P_b)
    print(nonNeg, BrokenNonNeg, nonLoc,"\n")
    
    G = np.linspace(-1,0,N)
    for g in G:
        print(g)
        P = pointOnTangent(g)
        print(P)
        nonNeg = T.isOnNonNegativityFacet(P)
        BrokenNonNeg = T.BeyondNonNegativityFacet(P)
        nonLoc = T.is_nonlocalPoint(P)
        print(nonNeg, BrokenNonNeg, nonLoc,"\n")
    acc = 1e-6
    l = -1
    p = 0
    while p-l > acc:
        s = (l+p)/2
        P = pointOnTangent(s)
        BrokenNonNeg = T.BeyondNonNegativityFacet(P)
        if BrokenNonNeg:
            l = s
        else:
            p = s
    s = (l+p)/2
    print("\n\n",p)
    print((np.cos(theta_b)-1)/np.sin(theta_b))
    P_g = pointOnTangent(p)
    print(P_g)
    nonNeg = T.isOnNonNegativityFacet(P_g)
    BrokenNonNeg = T.BeyondNonNegativityFacet(P_g)
    nonLoc = T.is_nonlocalPoint(P_g)
    print(nonNeg, BrokenNonNeg, nonLoc,"\n")

def notUniquePointsInvestigating(N):
    for i in range(N):
        P = np.round(T.notUniquePoints(),6)
        correct, realisation, wholeSp = T.twoQubitRepresentationSpecial(P)
        if correct:
            print(P, "point")

            theta, a0, a1, b0, b1 = realisation
            # print(correct, theta, a0/np.pi, a1/np.pi, b0/np.pi, b1/np.pi, wholeSp, np.sin(theta)**2, "tu")
            # print(P[0]/np.cos(theta), P[1]/np.cos(theta), P[2]/np.cos(theta), P[3]/np.cos(theta), "cosinusy prawdziwe")
            print(St.STLM(P, theta), "stlm")
            Ptlm = T.find_P(np.pi/2, a0,a1,b0,b1)

            print(T.find_P(theta, a0,a1,b0,b1), "Punkt zwrotny")
            print(T.is_nonlocalPoint(P), "is nonlocal?")
            print(T.TLM(Ptlm), "Tlm")
            print(T.is_exposed(theta, a0,a1,b0,b1,0.001), "exposed")

            

N = 200
D = 101

# compareSatoshiAndNumeric(N)
# pointsWithTwoSolutions(N)
# generalDoubleTiltedRegion(D)
# generalWolfRegion(D)
# GW.testWolfBQ()
# GW.testWolfPoint()
# hardySatoshi()
# satoshiConditions(N)
# checkCabelloPoint()
# for theta in np.linspace(0, np.pi/4, 100):
#     testWagnerPoints(N)

# hypothesis(N)
# hypothesis2(N)
# compareTests(N)

# theta = 1.569943835383207
# a0 = 2.952741118504561
# a1 = 1.66591491427125
# b0 = 2.952396565487108
# b1 = 0.5614602836189124
# check(theta, a0,a1,b0,b1)
# functionalBehaviour(N)
# tresholdThetaTest(N)
# localPointsBeforeTreshold(N)
# geometry(N)
# zeroProbability(N)
# tlmVSstlm(N)
# SpSmTheta(N)
# SpvsThreshold(N)
nonnegativitySingularity(N)
# moveHardy(N)
# decompositionTest(N)
# decomposition(N)
# notUniquePoint()
# tangentToCriticalPoint(N)
# notUniquePointsInvestigating(N)