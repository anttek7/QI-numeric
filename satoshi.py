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
        _,_,AB = createCorrelatorsAndMarginals(P)
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

def random2QPoints(N):
    for i in range(N):
        if i%100==0:
            print(f'{i+1}/{N}')
        theta = np.random.rand()*np.pi/4
        a0 = np.random.rand()*2*np.pi
        a1 = np.random.rand()*2*np.pi
        b0 = np.random.rand()*2*np.pi
        b1 = np.random.rand()*2*np.pi
        P = Op.find_P(theta, a0, a1, b0, b1)
        
        if satoshiTest(P):
            print("extremal")

