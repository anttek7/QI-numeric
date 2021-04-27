#%% 
import matplotlib.pyplot as plt
import numpy as np
import CHSH
import optimisation as Op

acc = 1e-9


def createCorrelatorsAndMarginals(P):
    A = np.array([P[0], P[1]])
    B = np.array([P[2], P[3]])
    AB = np.array([[P[4], P[5]],[P[6], P[7]]])
    return A,B,AB


def twoQubitRepresentation(P):
    '''
    returns (if point has 2-qubit realisation {0,1}, possible realisation(theta, a0, a1, b0, b1), if matrix Sp has all the same components)
    '''
    def checkIfProperRange(P):
        for x in P:
            if np.abs(x) > 1:
                return 0
        return 1

    def checkIfNonZeroMarginals(P):
        for i in range(4):
            if np.abs(P[i]) > acc:
                return 1
        return 0
    
    def findK(A,B,AB):
        K = np.zeros((2,2))
        for x in range(2):
            for y in range(2):
                K[x][y] = AB[x][y] - A[x]*B[y]
        return K

    def findJ(A,B,AB):
        J = np.zeros((2,2))
        for x in range(2):
            for y in range(2):
                J[x][y] = AB[x][y]**2 - A[x]**2 - B[y]**2 + 1
        return J

    def findS(K,J):
        S_p = np.zeros((2,2))
        S_m = np.zeros((2,2))
        
        for x in range(2):
            for y in range(2):
                delta = J[x][y]**2 - 4* K[x][y]**2
                if delta < 0:
                    return 0,0,0
                S_p[x][y] = (J[x][y] + np.sqrt(delta))/2
                S_m[x][y] = (J[x][y] - np.sqrt(delta))/2
        return 1,S_p, S_m
    
    def findSolutionInS(S_p, S_m):
        def checkRepetitionInBoth(s, S_p, S_m):
            for x in range(2):
                for y in range(2):
                    if ( np.abs(s - S_p[x][y]) > acc ) and (np.abs(s - S_m[x][y]) > acc ):
                        return 0
            return 1
        def checkRepetitionInOne(S):
            s = S[0][0]
            for x in range(2):
                for y in range(2):
                    if np.abs(s - S[x][y]) > acc:
                        return 0
            return 1
        def getThetaFromSinSquared(s):
            return np.arccos(np.sqrt(1-s))

        def interpret(c1,c2,c3,c4, S_p, S_m):
            # returns (is there any solution? {0,1}, all S_p {0,1}, table of solutions)
            if (not c1) and (not c2):
                # no solution
                return 0, 0, [0]
            if c1 and (not c3) and (not c4):
                # one solution but not only S_p or S_m
                return 1, 0, [getThetaFromSinSquared(S_p[0][0])]
            if c2 and (not c1) and (not c4):
                # one solution but not only S_m or S_p
                return 1, 0, [getThetaFromSinSquared(S_m[0][0])]
            if c3:
                if c4:
                    # two solutions
                    return 1, 1, [getThetaFromSinSquared(S_p[0][0]),getThetaFromSinSquared(S_m[0][0])]
                else:
                    return 1, 1,[getThetaFromSinSquared(S_p[0][0])]
            else:
                if c4:
                    return 1, 0,[getThetaFromSinSquared(S_m[0][0])]
                

        c1 = checkRepetitionInBoth(S_p[0][0], S_p, S_m)
        c2 = checkRepetitionInBoth(S_m[0][0], S_p, S_m)
        c3 = checkRepetitionInOne(S_p)
        c4 = checkRepetitionInOne(S_m)
        # print(S_p)
        # print(S_m)
        # print(c1,c2,c3,c4)
        
        return interpret(c1,c2,c3,c4, S_p, S_m)

    def getCosines(P, theta):
        CosA = [P[0]/np.cos(theta), P[1]/np.cos(theta)] 
        CosB = [P[2]/np.cos(theta), P[3]/np.cos(theta)]
        return CosA, CosB
    def verifyCosines(CosA, CosB):
        for c in CosA:
            if np.abs(c) > 1:
                return 0
        for c in CosB:
            if np.abs(c) > 1:
                return 0
        return 1
    def checkProductCondition(P, theta):
        A,B,AB = createCorrelatorsAndMarginals(P)
        R = 1
        for x in range(2):
            for y in range(2):
                R *= ( np.cos(theta)**2*AB[x][y] - A[x]*B[y])
        # print('R',R)
        if R >= -acc:
            return 1
        else:
            return 0
    def getSines(P, CosA, CosB, theta):
        A,B,AB = createCorrelatorsAndMarginals(P)
        Sxy = np.zeros((2,2))
        SinA = np.zeros(2)
        SinB = np.zeros(2)

        for x in range(2):
            for y in range(2):
                Sxy[x][y] = (AB[x][y] - CosA[x]*CosB[y])/np.sin(theta)

        SinA[0] = np.sqrt(1 - CosA[0]**2)
        SinB[0] = Sxy[0][0]/SinA[0]
        SinB[1] = Sxy[0][1]/SinA[0]
        SinA[1] = Sxy[1][0]/Sxy[0][0]*SinA[0]
        return SinA, SinB
        
    def getAnglesFromTrig(SinA, SinB, CosA, CosB):
        aAngles = np.zeros(2)
        bAngles = np.zeros(2)
        
        for x in range(2):
            aAngles[x] = np.arccos(CosA[x])
            bAngles[x] = np.arccos(CosB[x])
        for x in range(2):
            if SinA[x] < 0:
                aAngles[x] *= -1
            if SinB[x] < 0:
                bAngles[x] *= -1
        for x in range(2):
            if aAngles[x] < 0:
                aAngles[x] = aAngles[x] + 2*np.pi
            if bAngles[x] < 0:
                bAngles[x] = bAngles[x] + 2*np.pi        

        return aAngles[0], aAngles[1], bAngles[0], bAngles[1]

    garbage = np.zeros(5)
    if not checkIfProperRange(P):
        print("wrong range of the point components")
        return 0, garbage, 0
    if not checkIfNonZeroMarginals(P):
        print("Only for non-zero marginals")
        return 1, garbage, 0

    A,B,AB = createCorrelatorsAndMarginals(P)
    K = findK(A,B,AB)
    J = findJ(A,B,AB)
    correct, S_p, S_m = findS(K,J)

    if not correct:
        print("delta < 0")
        return 0, garbage, 0
    correct, wholeSp, thetaTable = findSolutionInS(S_p, S_m)

    if not correct:
        print("no mutual solution")
        return 0, garbage, 0
    if len(thetaTable) == 2:
        if np.abs(thetaTable[0]-thetaTable[1]) < acc:
            print("Sm = Sp, one mutual solution")
        else:
            print("Sm != Sp, two different solutions!")
        theta = thetaTable[0]
        # temporarly take the first one
    else:
        theta = thetaTable[0]
    
    CosA, CosB = getCosines(P, theta)
    correct = verifyCosines(CosA, CosB)
    
    # print(np.sin(theta)**2, "sin^2theta")
    if not correct:
        print("wrong ranges of the obtained cosines")
        return 0, garbage, 0
    
    correct = checkProductCondition(P, theta)
    if not correct:
        print("Product condition is not satisfied")
        return 0, garbage, 0
    SinA, SinB = getSines(P, CosA, CosB, theta)
    
    a0, a1, b0, b1 = getAnglesFromTrig(SinA, SinB, CosA, CosB)

    realisation = (theta,a0,a1,b0,b1)
    return 1, realisation, wholeSp

def STLM(P, theta):
    A,B,AB = createCorrelatorsAndMarginals(P)

    DA = np.zeros(2)
    DB = np.zeros(2)
    for x in range(2):
        DB[x] = np.sqrt(A[x]**2 + np.sin(theta)**2)
    for y in range(2):
        DA[y] = np.sqrt(B[y]**2 + np.sin(theta)**2)
    # print(DA,DB,"DA, DB")
    CA = np.zeros((2,2))
    CB = np.zeros((2,2))
    for x in range(2):
        for y in range(2):
            CA[x][y] = AB[x][y]/DA[y]
            CB[x][y] = AB[x][y]/DB[x]
    
    IEA = np.abs(CA[0][0]*CA[0][1] - CA[1][0]*CA[1][1])
    IEA -= np.sqrt(1 - CA[0][0]**2)*np.sqrt(1 - CA[0][1]**2)
    IEA -= np.sqrt(1 - CA[1][0]**2)*np.sqrt(1 - CA[1][1]**2)

    IEB = np.abs(CB[0][0]*CB[0][1] - CB[1][0]*CB[1][1])
    IEB -= np.sqrt(1 - CB[0][0]**2)*np.sqrt(1 - CB[0][1]**2)
    IEB -= np.sqrt(1 - CB[1][0]**2)*np.sqrt(1 - CB[1][1]**2)

    if np.abs(IEA) <= acc and np.abs(IEB) <= acc:
        return 1
    else:
        return 0

def satoshiTest(P):
    correct1, realisation, wholeSp = twoQubitRepresentation(P)
    theta = realisation[0]
    correct2 = STLM(P, theta)
    
    if correct1 and correct2 and wholeSp:
        # print("Extremal")
        return 1
    else:
        # print("Not extremal")
        return 0



N = 1000
for i in range(N):
    if i%100==0:
        print(f'{i+1}/{N}')
    theta = np.random.rand()*np.pi/4
    a0 = np.random.rand()*2*np.pi
    a1 = np.random.rand()*2*np.pi
    b0 = np.random.rand()*2*np.pi
    b1 = np.random.rand()*2*np.pi
    P = Op.find_P(theta, a0, a1, b0, b1)
    # P = np.random.rand(8)*2-1

    # a = np.random.rand()*2-1
    # b = np.random.rand()*2-1
    # c = np.random.rand()*2-1
    
    # P = [a,a,b,b,c,c,c,c]
    correct, realisation, wholeSp = twoQubitRepresentation(P)
    thetan, a0n, a1n, b0n, b1n = realisation
    Pn = Op.find_P(thetan, a0n, a1n, b0n, b1n)
    
    # if np.linalg.norm(P-Pn) > acc:
    #     print("error")
    # else:
    #     print("git",wholeSp)
    if satoshiTest(P):
        print("ql")

#%%






def firstCondition(P):
    A,B,AB = createCorrelatorsAndMarginals(P)
    K = findK(A,B,AB)
    J = findJ(A,B,AB)
    # print('K\n',K)
    # print('J\n',J)
    S_p, S_m = findS(K,J)
    if np.abs(S_p[0][0] - S_p[0][1]) < 1e-9 and np.abs(S_p[0][0] - S_p[1][0]) < 1e-9 and np.abs(S_p[0][0] - S_p[1][1]) < 1e-9:
        if np.abs(S_m[0][0] - S_m[0][1]) < 1e-9 and np.abs(S_m[0][0] - S_m[1][0]) < 1e-9 and np.abs(S_m[0][0] - S_m[1][1]) < 1e-9:
            print("hihihi")
        return 1, S_p, S_m
    else:
        return 0, S_p, S_m

    
def secondCondition(P,S_p):
    A,B,AB = createCorrelatorsAndMarginals(P)
    R = 1
    for x in range(2):
        for y in range(2):
            R *= ((1 - S_p[x][y])*AB[x][y] - A[x]*B[y])
    # print('R',R)
    if R >= -1e-9:
        return 1,R
    else:
        return 0,R

def thirdCondition(P,S_p):
    A,B,AB = createCorrelatorsAndMarginals(P)

    DA = np.zeros(2)
    DB = np.zeros(2)
    for x in range(2):
        DB[x] = np.sqrt(A[x]**2 + S_p[0][0])
    for y in range(2):
        DA[y] = np.sqrt(B[y]**2 + S_p[0][0])
    # print(DA,DB,"DA, DB")
    CA = np.zeros((2,2))
    CB = np.zeros((2,2))
    for x in range(2):
        for y in range(2):
            CA[x][y] = AB[x][y]/DA[y]
            CB[x][y] = AB[x][y]/DB[x]
    
    IEA = np.abs(CA[0][0]*CA[0][1] - CA[1][0]*CA[1][1])
    IEA -= np.sqrt(1 - CA[0][0]**2)*np.sqrt(1 - CA[0][1]**2)
    IEA -= np.sqrt(1 - CA[1][0]**2)*np.sqrt(1 - CA[1][1]**2)

    IEB = np.abs(CB[0][0]*CB[0][1] - CB[1][0]*CB[1][1])
    IEB -= np.sqrt(1 - CB[0][0]**2)*np.sqrt(1 - CB[0][1]**2)
    IEB -= np.sqrt(1 - CB[1][0]**2)*np.sqrt(1 - CB[1][1]**2)

    if np.abs(IEA) <= 1e-9 and np.abs(IEB) <= 1e-9:
        return 1, IEA, IEB
    else:
        return 0, IEA, IEB
def satoshiTest(P):
    f1, S_p, S_m = firstCondition(P)
    # print('first condition ',f1)
    # print("S_p\n", S_p)
    # print("S_m\n", S_m)

    f2,R = secondCondition(P,S_p)
    # print('second condition',f2)

    f3, IEA, IEB = thirdCondition(P,S_p)
    # print("third condition", f3)
    # print(IEA)
    # print(IEB)

    if f1 and f2 and f3:
        # print("Extremal")
        return 1
    else:
        # print("Not extremal")
        return 0
def satoshiTestComments(P):
    f1, S_p, S_m = firstCondition(P)
    print('first condition ',f1)
    print("S_p\n", S_p)
    print("S_m\n", S_m)

    f2,R = secondCondition(P,S_p)
    
    print('second condition',R,f2)

    f3, IEA, IEB = thirdCondition(P,S_p)
    print("third condition", f3)
    print(IEA)
    print(IEB)

    if f1 and f2 and f3:
        # print("Extremal")
        return 1
    else:
        # print("Not extremal")
        return 0


N = 10000
bothEx = 0
bothNoEx = 0
SatEx = 0
OurEx = 0

Alpha = np.linspace(0,2,N)
D = 100
map = np.zeros((D,D))
for alpha in Alpha:
    # RANDOM
    th0 = np.random.randint(100) + 100
    th1 = np.random.randint(100)+1
    an01 = np.random.randint(100) + 100
    an02 = np.random.randint(100)+1
    an11 = np.random.randint(100) + 100
    an12 = np.random.randint(100)+1
    bn01 = np.random.randint(100) + 100
    bn02 = np.random.randint(100)+1
    bn11 = np.random.randint(100) + 100
    bn12 = np.random.randint(100)+1
     
    theta = th0/th1*np.pi/4
    a0 = an01/an02*np.pi
    a1 = an11/an12*np.pi
    b0 = bn01/bn02*np.pi
    b1 = bn11/bn12*np.pi
    
    # CHSH
    # theta = np.pi/4
    # a0 = 3*np.pi/4
    # a1 = np.pi/4
    # b0 = np.pi/2
    # b1 = np.pi

    # TILTED
    # mu, theta = tiltedAlpha(alpha)
    # a0 = 0
    # a1 = np.pi/2
    # b0 = mu
    # b1 = -mu


    P = Op.find_P(theta, a0,a1,b0,b1)
    # P=[1,2,3,4,5,6,7,8]
    # print(P)
    # P = np.random.rand(8)
    isExtremal = satoshiTest(P)
    accuracy = 0.001
    isExtremal2 = Op.is_exposed(theta,a0,a1,b0,b1, accuracy,limit=1)
    if isExtremal:
        if isExtremal2:
            bothEx += 1
        else:
            SatEx += 1
            # print(th0, th1, an01, an02, an11, an12, bn01, bn02, bn11, bn12)
            # Sth0 = th0
            # Sth1 = th1 
            # San01 = an01
            # San02 = an02
            # San11 = an11
            # San12 = an12
            # Sbn01 = bn01
            # Sbn02 = bn02
            # Sbn11 = bn11
            # Sbn12 = bn12
    else:
        if isExtremal2:
            OurEx += 1
            print(th0, th1, an01, an02, an11, an12, bn01, bn02, bn11, bn12)
            Sth0 = th0
            Sth1 = th1 
            San01 = an01
            San02 = an02
            San11 = an11
            San12 = an12
            Sbn01 = bn01
            Sbn02 = bn02
            Sbn11 = bn11
            Sbn12 = bn12
        else:
            bothNoEx += 1
    print(bothNoEx, bothEx, SatEx, OurEx)
#%%


theta = Sth0/Sth1*np.pi/4
a0 = San01/San02*np.pi
a1 = San11/San12*np.pi
b0 = Sbn01/Sbn02*np.pi
b1 = Sbn11/Sbn12*np.pi
P = Op.find_P(theta, a0,a1,b0,b1)

isExtremal = satoshiTestComments(P)

accuracy = 0.001
isExtremal2 = Op.is_exposed(theta,a0,a1,b0,b1, accuracy,limit=1)
print(isExtremal, isExtremal2)
# %%
# General Wolf
def BQAnal(t,r):
    s = (4 + r**2 - 4*t**2)*(2 - t**2)
    l = r*t + np.sqrt(s)
    m = 1 - t**2
    return l/m 
def cotTheta(t,r):
    l = r*np.sqrt(2 - t**2) + np.sqrt(4 + r**2 - 4*t**2)*(1 + t - t**2)
    m1 = -2*r*t*np.sqrt((2 - t**2) * (4 + r**2 - 4*t**2))
    m2 = -r**2 * (1 + 2*t**2 - t**4) - 4*(-1 +4*t**2 -4*t**4 + t**6)
    return l/np.sqrt(m1+m2)

def getSin2Theta(cot):
    return (2*cot)/(1 + cot**2)

def getCos2Theta(cot):
    return (cot**2 - 1)/(1 + cot**2)

def getCosA(t,r):
    s = (r**2 - 4*t**2 + 4)/(2 - t**2)
    l = r*t + np.sqrt(s) 
    m = 2*(1 - t**2)
    return l/m
def getSinA(t,r):
    cos = getCosA(t,r)
    return np.sqrt(1-cos**2)
def quantumPoint(t,r):
    cot = cotTheta(t,r)
    sin2Theta = getSin2Theta(cot)
    cos2Theta = getCos2Theta(cot)
    sinA = getSinA(t,r)
    cosA = getCosA(t,r)
    return (cosA*cos2Theta, cosA*cos2Theta, cos2Theta, 0, cosA, sinA*sin2Theta, cosA, -sinA*sin2Theta)

rn = 200
tn = 200
eps = 0.001
R = np.linspace(0,2, rn)
T = np.linspace(-1+eps, 1-eps, tn)
map = np.zeros((rn,tn))
for i,r in enumerate(R):
    for j,t in enumerate(T):
        P = quantumPoint(t,r)
        if satoshiTest(P):
            map[i][tn-1-j] = 1
            
plt.imshow(map.T,extent=[0,2,-1,1])    

# %%
t = -0.5
r = 1
print(getCosA(t,r))
print(BQAnal(t,r))

B = (t, t, r,0,1,1,1,-1)

P,quantum = Op.Best_point(B,0.000001)
print(P)
print(quantumPoint(t,r))
satoshiTest(P)

# %%
# General double tilted
def isNonLocalN(t,fi):
    B = (t*np.cos(fi/2), t*np.sin(fi/2), 0,0,1,1,1,-1)
    accuracy = 0.001
    P,Q = Op.Best_point(B,accuracy)
    return P,Q
N = 100
T = np.linspace(0,2,N)
Fi = np.linspace(0,np.pi/2,N)
Map = np.zeros((N,N))
for i,t in enumerate(T):
    print(f'{i+1}/{N}')
    for j,fi in enumerate(Fi):
        P,_ = isNonLocalN(t,fi)
        Map[i][j] = satoshiTest(P)
plt.imshow(Map,extent=[0, 2, np.pi/2, 0])

# %%
# Symmetric points on the face

def symmetricPoint(theta, a):
    P = (np.cos(theta), np.cos(theta)*np.cos(a), np.cos(theta), np.cos(theta)*np.cos(a),
        1, np.cos(a), np.cos(a),np.cos(a)**2 - np.sin(theta)*np.sin(a)**2 )
    return P
def symmetricPoint2(theta, Cosa):
    P = (np.cos(theta), np.cos(theta)*Cosa, np.cos(theta), np.cos(theta)*Cosa,
        1, Cosa, Cosa,Cosa**2 - np.sin(theta)*(1- Cosa**2))
    return P

N = 100
A = np.linspace(0,1,N)
Theta = np.linspace(0,np.pi/2,N)
Map = np.zeros((N,N))
for i,t in enumerate(Theta):
    print(f'{i+1}/{N}')
    for j,a in enumerate(A):
        P = symmetricPoint2(t,a)
        Map[i][j] = satoshiTest(P)
plt.imshow(Map,extent=[0, 1, np.pi/2, 0])
plt.show()


# %%

def findWrong_P(theta, a0,a1,b0,b1):
    A = [a0,a1,b0,b1]
    P = np.array([])
    for i in range(4):
        P = np.append(P, np.cos(A[i])*np.cos(theta))
    for i in range(4):
        P = np.append(P,np.cos(A[int(i/2)])*np.cos(A[2 + i%2]) + np.sin(theta)*np.sin(A[int(i/2)])*np.sin(A[2 + i%2]))
    P[4] = np.cos(A[0]) * np.cos(A[2]) - np.sin(theta) * np.sin(A[0]) * np.sin(A[2])
    return P

def get_sinth2(P):
    A,B,AB = createCorrelatorsAndMarginals(P)
    K = findK(A,B,AB)
    J = findJ(A,B,AB)
    S_p, S_m = findS(K,J)
    # print(S_p)
    # print(S_m)
    
    plus = S_p[0][0]
    minus = S_m[0][0]
    candidates = []
    acc = 1e-9
    if np.abs(plus - S_p[0][1]) < acc or np.abs(plus - S_m[0][1]) < acc:
        if np.abs(plus - S_p[1][0]) < acc or np.abs(plus - S_m[1][0]) < acc:
            if np.abs(plus - S_p[1][1]) < acc or np.abs(plus - S_m[1][1]) < acc:
                candidates.append(plus)
    if np.abs(minus - S_p[0][1]) < acc or np.abs(minus - S_m[0][1]) < acc:
        if np.abs(minus - S_p[1][0]) < acc or np.abs(minus - S_m[1][0]) < acc:
            if np.abs(minus - S_p[1][1]) < acc or np.abs(minus - S_m[1][1]) < acc:
                if np.abs(minus - plus) > acc:
                    candidates.append(minus)
    return candidates
def get_Cos(P,sinth):
    costh = np.sqrt(1 - sinth**2)
    CosA = [P[0]/costh, P[1]/costh]
    CosB = [P[2]/costh, P[3]/costh]
    return CosA, CosB
def get_Sxy(P, sinth):
    A,B,AB = createCorrelatorsAndMarginals(P)
    CosA, CosB = get_Cos(P,sinth)
    S = np.zeros((2,2))
    for x in range(2):
        for y in range(2):
            S[x][y] = (AB[x][y] - CosA[x]*CosB[y])/sinth
    return S

def secondCondition2(P,sinth):
    A,B,AB = createCorrelatorsAndMarginals(P)
    R = 1
    for x in range(2):
        for y in range(2):
            R *= ((1 - sinth**2)*AB[x][y] - A[x]*B[y])
    return R
theta = np.pi/8
a0 = 3*np.pi/4
a1 = np.pi/3
b0 = np.pi/4
b1 = np.pi/3
for i in range(10000):
    if i%10 == 0:
        print(i)
    theta = np.random.rand()*np.pi/4
    a0 = np.random.rand()*2*np.pi
    a1 = np.random.rand()*2*np.pi
    b0 = np.random.rand()*2*np.pi
    b1 = np.random.rand()*2*np.pi

    P = findWrong_P(theta, a0,a1,b0,b1)
    # print(P, "point")
    candidates = get_sinth2(P)
    # print(candidates, "kandydaci sin^2(th)\n")
    if len(candidates) >= 1:
        sinth = np.sqrt(candidates[0])
        # print(sinth, np.sin(theta), "sin th\n")
        S = get_Sxy(P, sinth)
        # print(S)
        R = secondCondition2(P, sinth)
        if R > 0:
            print(a0,a1,b0, b1, theta)
            print(P,"point")
            print(candidates, "kandydaci sin^2(th)\n")
            print(sinth, np.sin(theta), "sin th\n")
            print(S)
        

# %%
