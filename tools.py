import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog
from numpy import linalg as LA
import CHSH

acc = 1e-9

def generalPoint(state, A0, A1, B0, B1):
    def correlator(rho, Operator1, Operator2):
        M = np.matmul(rho, np.kron(Operator1, Operator2))
        return np.trace(M)
    I = np.array([(1,0),(0,1)])
    rho = np.outer(state,state)
    
    return (correlator(rho, A0, I), correlator(rho, A1, I), correlator(rho, I, B0), correlator(rho, I, B1),
     correlator(rho, A0, B0), correlator(rho, A0, B1), correlator(rho, A1, B0), correlator(rho, A1, B1))

def find_psi(theta):
    return np.array([np.cos(theta/2), 0, 0, np.sin(theta/2)])
def find_P(theta, a0,a1,b0,b1):
    A = [a0,a1,b0,b1]
    P = np.array([])
    for i in range(4):
        P = np.append(P, np.cos(A[i])*np.cos(theta))
    for i in range(4):
        P = np.append(P,np.cos(A[int(i/2)])*np.cos(A[2 + i%2]) + np.sin(theta)*np.sin(A[int(i/2)])*np.sin(A[2 + i%2]))
    return P
def find_dP_dtheta(theta,a0,a1,b0,b1):
    A = [a0,a1,b0,b1]
    dP = np.array([])
    for i in range(4):
        dP = np.append(dP, -2*np.cos(A[i])*np.sin(theta))
    for i in range(4):
        dP = np.append(dP,2*np.cos(theta)*np.sin(A[int(i/2)])*np.sin(A[2 + i%2]))
    return dP

def dP_dA(theta,a0,a1,b0,b1,i):
    A = [a0,a1,b0,b1]
    dP = np.zeros(8)
    dP[i] = -np.sin(A[i])*np.cos(theta)
    if i==0 or i==1:
        dP[4 + 2*(i%2)] = -np.sin(A[i%2])*np.cos(A[2]) + np.sin(theta)*np.cos(A[i%2])*np.sin(A[2])
        dP[5 + 2*(i%2)] = -np.sin(A[i%2])*np.cos(A[3]) + np.sin(theta)*np.cos(A[i%2])*np.sin(A[3])
    else:
        dP[4 + i%2] = -np.sin(A[2 + i%2])*np.cos(A[0]) + np.sin(theta)*np.cos(A[2 + i%2])*np.sin(A[0])
        dP[6 + i%2] = -np.sin(A[2 + i%2])*np.cos(A[1]) + np.sin(theta)*np.cos(A[2 + i%2])*np.sin(A[1])        
    return dP

def find_dP_da0(theta,a0,a1,b0,b1):
    dP = dP_dA(theta,a0,a1,b0,b1,0)
    return dP
def find_dP_da1(theta,a0,a1,b0,b1):
    dP = dP_dA(theta,a0,a1,b0,b1,1)
    return dP
def find_dP_db0(theta,a0,a1,b0,b1):
    dP = dP_dA(theta,a0,a1,b0,b1,2)
    return dP
def find_dP_db1(theta,a0,a1,b0,b1):
    dP = dP_dA(theta,a0,a1,b0,b1,3)
    return dP
def find_deterministic_points():
    p = 16
    D = 8
    d = D*2
    M = np.zeros((D,d))
    korelatory = np.ones(4)
    for i in range(4):
        p /= 2
        x = 1
        for j in range(d):
            if j%(2*p) < p:
                M[i][j] = x
            else:
                M[i][j] = -x
    for j in range(d):
        for i in range(4):
            korelatory[i] = M[i][j]/x**0.5
        for i in range(4):
            M[i+4][j] = korelatory[int(i/2)] * korelatory[i%2 + 2]
    return np.transpose(M)    
def find_A_ub():
    return find_deterministic_points()

def find_A_eq(theta,a0,a1,b0,b1):
    dP_dtheta = find_dP_dtheta(theta,a0,a1,b0,b1)
    dP_da0 = find_dP_da0(theta,a0,a1,b0,b1)
    dP_da1 = find_dP_da1(theta,a0,a1,b0,b1)
    dP_db0 = find_dP_db0(theta,a0,a1,b0,b1)
    dP_db1 = find_dP_db1(theta,a0,a1,b0,b1)
    A_eq = np.vstack([dP_dtheta, dP_da0, dP_da1, dP_db0, dP_db1])
    return A_eq

def find_eigen1(theta, a0, a1, b0,b1):
    f1 = np.sin(theta/2) * np.sin(a0)
    f2 = np.sin(theta/2) * np.sin(a1)
    f3 = np.cos(theta/2) * np.sin(b0)
    f4 = np.cos(theta/2) * np.sin(b1)
    f5 = np.cos(theta/2) * np.sin(b0) * np.cos(a0) - np.sin(theta/2) * np.cos(b0) * np.sin(a0)
    f6 = np.cos(theta/2) * np.sin(b1) * np.cos(a0) - np.sin(theta/2) * np.cos(b1) * np.sin(a0)
    f7 = np.cos(theta/2) * np.sin(b0) * np.cos(a1) - np.sin(theta/2) * np.cos(b0) * np.sin(a1)
    f8 = np.cos(theta/2) * np.sin(b1) * np.cos(a1) - np.sin(theta/2) * np.cos(b1) * np.sin(a1)
    return [f1,f2,f3,f4,f5,f6,f7,f8]

def find_eigen2(theta, a0, a1, b0,b1):
    f1 = np.cos(theta/2) * np.sin(a0)
    f2 = np.cos(theta/2) * np.sin(a1)
    f3 = np.sin(theta/2) * np.sin(b0)
    f4 = np.sin(theta/2) * np.sin(b1)
    f5 = np.cos(theta/2) * np.sin(a0) * np.cos(b0) - np.sin(theta/2) * np.cos(a0) * np.sin(b0)
    f6 = np.cos(theta/2) * np.sin(a0) * np.cos(b1) - np.sin(theta/2) * np.cos(a0) * np.sin(b1)
    f7 = np.cos(theta/2) * np.sin(a1) * np.cos(b0) - np.sin(theta/2) * np.cos(a1) * np.sin(b0)
    f8 = np.cos(theta/2) * np.sin(a1) * np.cos(b1) - np.sin(theta/2) * np.cos(a1) * np.sin(b1)
    return [f1,f2,f3,f4,f5,f6,f7,f8]

def find_A_eq_restricted(theta,a0,a1,b0,b1):
    dP_dtheta = find_dP_dtheta(theta,a0,a1,b0,b1)
    dP_da0 = find_dP_da0(theta,a0,a1,b0,b1)
    dP_da1 = find_dP_da1(theta,a0,a1,b0,b1)
    dP_db0 = find_dP_db0(theta,a0,a1,b0,b1)
    dP_db1 = find_dP_db1(theta,a0,a1,b0,b1)
    eigen1 = find_eigen1(theta, a0, a1, b0,b1)
    eigen2 = find_eigen2(theta, a0, a1, b0,b1)
    A_eq = np.vstack([dP_dtheta, dP_da0, dP_da1, dP_db0, dP_db1, eigen1, eigen2])
    print(np.linalg.matrix_rank(A_eq), "matrix A_eq rank")
    return A_eq

def B_add_limit(limit, A_ub, b_ub):
    A = np.eye(8)
    A_ub = np.append(A_ub,A, axis=0)
    A_ub = np.append(A_ub, -A, axis=0)
    b = np.ones(8)*limit
    b_ub = np.append(b_ub, b, axis=0)
    b_ub = np.append(b_ub, b, axis=0)
    return A_ub, b_ub 
    
def vector_to_matrix(V):
    matrix = np.zeros((3,3))
    matrix[0][1] = V[2]
    matrix[0][2] = V[3]
    matrix[1][0] = V[0]
    matrix[1][1] = V[4]
    matrix[1][2] = V[5]
    matrix[2][0] = V[1]
    matrix[2][1] = V[6]
    matrix[2][2] = V[7]
    return matrix

def matrix_to_vector(matrix):
    V = np.zeros(8)
    V[2] = matrix[0][1]
    V[3] = matrix[0][2]
    V[0] = matrix[1][0]
    V[4] = matrix[1][1]
    V[5] = matrix[1][2]
    V[1] = matrix[2][0]
    V[6] = matrix[2][1]
    V[7] = matrix[2][2]
    return V

def Best_func(theta,a0,a1,b0,b1, limit=1):
    P = find_P(theta,a0,a1,b0,b1)
    c = -P    
    A_ub = find_A_ub()
    b_ub = np.ones(16)
    A_ub, b_ub = B_add_limit(limit, A_ub, b_ub)
    A_eq = find_A_eq(theta,a0,a1,b0,b1)
    b_eq = np.zeros(5)
    bound = (None, None)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq = A_eq, b_eq = b_eq, bounds=bound, method='revised simplex')
    maxi = -res.fun 
    B = res.x
    return B, maxi 

def Best_func_restricted(theta,a0,a1,b0,b1, limit=1):
    P = find_P(theta,a0,a1,b0,b1)
    c = -P    
    A_ub = find_A_ub()
    b_ub = np.ones(16)
    A_ub, b_ub = B_add_limit(limit, A_ub, b_ub)
    A_eq = find_A_eq_restricted(theta,a0,a1,b0,b1)
    b_eq = np.zeros(7)
    bound = (None, None)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq = A_eq, b_eq = b_eq, bounds=bound, method='revised simplex')
    maxi = -res.fun 
    B = res.x
    return B, maxi 
   
def Best_point(B,accuracy):
    _, state, alpha, beta, _, _, Nonlocal = CHSH.chsh2(vector_to_matrix(B), accuracy)
    #W_max, state, alpha, beta, all_maximal_eigenvalues, l1, l2 = CHSH.chsh(vector_to_matrix(B), accuracy)
    #CHSH.plot_CHSH(accuracy, all_maximal_eigenvalues,0,"-","-")
    #print(W_max, l1, l2)
    correlators = CHSH.find_correlators2(alpha, beta, state, accuracy)
    #correlators = CHSH.find_correlators(alpha, beta, state, accuracy)
    P = matrix_to_vector(correlators)
    return P, Nonlocal

def is_exposed(theta,a0,a1,b0,b1, accuracy,limit=1):
    P = find_P(theta,a0,a1,b0,b1)
    norma = LA.norm(P)
    if norma != 0:
        Pn = P/norma
    else:
        Pn = P
        print(theta/np.pi, a0/np.pi,a1/np.pi,b0/np.pi,b1/np.pi)
        print(P)
    B, _ = Best_func(theta,a0,a1,b0,b1, limit)
    P2,_ = Best_point(B,accuracy)
    norma = LA.norm(P2)
    if norma != 0:
        P2n = P2/norma
    else:
        P2n = P2
        print('Best point is zero')
    d = LA.norm(Pn-P2n)
    if d <= accuracy*5:
        return 1
    else:
        return 0

def is_nonlocal(B):
    accuracy = 0.0001
    P, Nonlocal = Best_point(B,accuracy)
    return P, Nonlocal

def is_nonlocalPoint(P):
    chsh = np.array([[0,0,0,0,1,1,1,-1],[0,0,0,0,1,1,-1,1],[0,0,0,0,1,-1,1,1],[0,0,0,0,-1,1,1,1],[0,0,0,0,-1,-1,-1,1],[0,0,0,0,-1,-1,1,-1],[0,0,0,0,-1,1,-1,-1],[0,0,0,0,1,-1,-1,-1]])
    result = np.dot(chsh,np.array(P))
    if max(np.abs(result)) > 2:
        # print(max(np.abs(result)), "CHSH value")
        return 1
    else:
        return 0

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
                # print(delta, "delta")
                if delta < -acc:
                    # print(delta, "delta")
                    return 0,0,0
                if (delta < 0) and (delta > - acc):
                    delta = 0
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
        for i,c in enumerate(CosA):
            if np.abs(c) > 1+acc:
                return 0
            elif np.abs(c) > 1:
                CosA[i] = 1

        for i,c in enumerate(CosB):
            if np.abs(c) > 1+acc:
                return 0
            elif np.abs(c) > 1:
                CosB[i] = 1
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
        # print(CosA[0], "blablaba")
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
            print(P)
            print(thetaTable)
            print(S_p,'\n', S_m)
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

def TLM(P):
    A,B,AB = createCorrelatorsAndMarginals(P)
    
    I = np.abs(AB[0][0]*AB[0][1] - AB[1][0]*AB[1][1])
    I -= np.sqrt(1 - AB[0][0]**2)*np.sqrt(1 - AB[0][1]**2)
    I -= np.sqrt(1 - AB[1][0]**2)*np.sqrt(1 - AB[1][1]**2)

    if np.abs(I) <= acc:
        return 1
    else:
        return 0

def TLM2(P):
    A,B,AB = createCorrelatorsAndMarginals(P)
    
    I1 = 1
    I2 = 1
    I3 = 0
    for x in range(2):
        for y in range(2):
            I1 *= AB[x][y]
            I2 *= np.sqrt(1 - AB[x][y]**2)
            I3 -= (AB[x][y]**2)/2
    I = 1 + I1 + I2 + I3
    if np.abs(I) <= acc:
        return 1
    else:
        return 0

def hypoTreshold(a0,a1,b0,b1):
    def value1(a,b):
        return -np.sin(a)*np.sin(b)/(np.cos(a)*np.cos(b) - 1)
    def value2(a,b):
        return -np.sin(a)*np.sin(b)/(np.cos(a)*np.cos(b) + 1)
    return max(value1(a0,b0),value2(a0,b0), value1(a0,b1),value2(a0,b1),value1(a1,b0),value2(a1,b0),value1(a1,b1),value2(a1,b1))

def getThetaFromSinSquared(s):
    return np.arccos(np.sqrt(1-s))

################ function for debugging
def twoQubitRepresentationComment(P):
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
                print(delta, "delta")
                if delta < -acc:
                    print(delta, "delta")
                    return 0,0,0
                if delta < 0 and delta > - acc:
                    delta = 0
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
        for i,c in enumerate(CosA):
            if np.abs(c) > 1+acc:
                return 0
            elif np.abs(c) > 1:
                CosA[i] = 1

        for i,c in enumerate(CosB):
            if np.abs(c) > 1+acc:
                return 0
            elif np.abs(c) > 1:
                CosB[i] = 1
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
    print(S_p,"\n", S_m,"S_p, S_m")
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
    print(CosA, CosB, "obtained cosines")
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

def SpSm(P):
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
                # print(delta, "delta")
                if delta < -acc:
                    # print(delta, "delta")
                    return 0,0,0
                if (delta < 0) and (delta > - acc):
                    delta = 0
                S_p[x][y] = (J[x][y] + np.sqrt(delta))/2
                S_m[x][y] = (J[x][y] - np.sqrt(delta))/2
        return 1,S_p, S_m
    A,B,AB = createCorrelatorsAndMarginals(P)
    K = findK(A,B,AB)
    J = findJ(A,B,AB)
    correct, S_p, S_m = findS(K,J)
    print(S_p)
    print(S_m)
 