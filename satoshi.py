import matplotlib.pyplot as plt
import numpy as np
import CHSH
import tools as T

acc = 1e-9


def createCorrelatorsAndMarginals(P):
    A = np.array([P[0], P[1]])
    B = np.array([P[2], P[3]])
    AB = np.array([[P[4], P[5]],[P[6], P[7]]])
    return A,B,AB

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
    correct1, realisation, wholeSp = T.twoQubitRepresentation(P)
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
        P = T.find_P(theta, a0, a1, b0, b1)
        
        if satoshiTest(P):
            print("extremal")

def STLMComment(P, theta):
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
    print(IEA, IEB, "STLM")
    if np.abs(IEA) <= acc and np.abs(IEB) <= acc:
        return 1
    else:
        return 0

def satoshiTestComment(P):
    # function for debugging
    correct1, realisation, wholeSp = T.twoQubitRepresentationComment(P)
    theta = realisation[0]
    correct2 = STLM(P, theta)
    print(correct1, correct2, wholeSp,"what")
    if correct1 and correct2 and wholeSp:
        # print("Extremal")
        return 1
    else:
        # print("Not extremal")
        return 0
