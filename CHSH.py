# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 11:18:42 2019

@author: Antek Nuszkiewicz
"""
import matplotlib.pyplot as plt
import numpy as np
import sys
from numpy import linalg as LA
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import HTML
import matplotlib.animation as animation
from matplotlib import cm



# Pauli matrices:
Gz = np.array([(1,0),(0,-1)])
Gx = np.array([(0,1),(1,0)])
G1 = np.array([(1,0),(0,1)])

def max_eigenvalue(w,v):
    max_w = -sys.maxsize-1
    max_v = []
    for i in range(len(w)):
        if w[i] > max_w:
            max_w = w[i]
            max_v = v[:,i]
    return max_w, max_v

def count_operators_AB(i, j, a, b):
    A0 = G1
    A1 = Gz
    A2 = np.add( np.cos(a[i])*Gz, np.sin(a[i])*Gx)
    B0 = G1
    B1 = Gz
    B2 = np.add( np.cos(b[j])*Gz, np.sin(b[j])*Gx)
    Operators = (A0,A1,A2,B0,B1,B2)
    return Operators
def count_operators_AB2(a, b):
    A0 = G1
    A1 = Gz
    A2 = np.add( np.cos(a)*Gz, np.sin(a)*Gx)
    B0 = G1
    B1 = Gz
    B2 = np.add( np.cos(b)*Gz, np.sin(b)*Gx)
    Operators = (A0,A1,A2,B0,B1,B2)
    return Operators

def count_operator_W(factor, O):
    W  = np.zeros((4,4))
    for i in range(3):
        for j in range(3):
           tensor = factor[i][j] * np.kron(O[i], O[j+3])
           W  = np.add(W, tensor)
    return W

def save_new_max(i, j, W_max, V_max, w_new, v_new, alpha, beta, accuracy):
    if w_new > W_max:
        W_max = w_new
        V_max = v_new
        alpha = i*accuracy*np.pi
        beta = j*accuracy*np.pi
    return(W_max, V_max, alpha, beta)
    
def print_CHSH(W_max, V_max, alpha, beta, correlators, LHV, l1, l2):
    print("Maximal Quantum value: ", W_max)
    print("Maximal LHV value: ", LHV)
    print("Optimal state: ", V_max)
    print("Optimal angle alpha: ", alpha,"\nand beta: ", beta )
    print("All correlators:")
    print("Schmidt coefficient: ",l1,l2)
    for i in range(3):
        for j in range(3):
            print("<A"+str(i)+"B"+str(j)+">: ", correlators[i][j])            

def animacjuj(title, all_maximal_eigenvalues,accuracy):
    def init():
        ax.plot_surface(X,Y, all_maximal_eigenvalues, cmap=cm.coolwarm)
        return fig,

    def animate(i):
        ax.view_init(elev=30., azim=2.5*i)
        return fig,

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Beta')
    ax.set_zlabel('Lamda max')
    a = b = np.linspace(0.0, np.pi, int(1/accuracy))
    X, Y = np.meshgrid(a, b)    
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=144, interval=50, blit=True)
    anim.save(title+'.gif', writer='imagemagick')

def plot_CHSH(accuracy, all_maximal_eigenvalues,save, name, title):
    a = b = np.linspace(0.0, np.pi, int(1/accuracy))
    X, Y = np.meshgrid(a, b)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Alpha')
    ax.set_title(title)
    ax.set_ylabel('Beta')
    ax.set_zlabel('Lamda max')    
    ax.plot_surface(X,Y, all_maximal_eigenvalues, cmap=cm.coolwarm)
    if save:
        plt.savefig(name+'.png')
    plt.show()


def schmidt_coefficient(state):
    A = np.array([(state[0], state[1]),(state[2],state[3])])
    X = np.matmul(A,A.transpose())
    w,v = LA.eig(X)
    w = w.real
    v = v.real
    return (np.abs(w[0])**0.5, np.abs(w[1])**0.5)
 
def neighbour(a,b, acc):
    n0 = ((a+acc)%np.pi, b)
    n1 = ((a-acc)%np.pi, b)
    n2 = (a, (b+acc)%np.pi)
    n3 = (a, (b-acc)%np.pi)
    return (n0,n1,n2,n3)
def corners():
    c0 = (0,0)
    c1 = (0,np.pi)
    c2 = (np.pi,0)
    c3 = (np.pi,np.pi)
    return (c0,c1,c2,c3)
def chsh(factor,accuracy):
    a = b = np.linspace(0.0, np.pi, int(1/accuracy))
    
    all_maximal_eigenvalues = np.zeros((len(a),len(b)))

    W_max = -sys.maxsize-1
    V_max = []
    alpha= 0
    beta = 0
    #iteracja po wszystkich kątach
    for i in range(len(a)):
        for j in range(len(b)):
            O = count_operators_AB(i, j, a, b)
            W = count_operator_W(factor, O)
            w,v = LA.eig(W) # obliczenie wartosci i wektorow wlasnych operatora Bella
            w = w.real
            v = v.real
            w_new, v_new = max_eigenvalue(w,v) #Wybor najwikeszej wartosci wlasnej i odpowiadajacy jej wektor
            all_maximal_eigenvalues[i,j] = w_new
            #Sprawdzenie czy aktualna wartosc wlasna jest maksymalna z dotychczasowych
            W_max, V_max, alpha, beta = save_new_max(i,j, W_max, V_max, w_new, v_new, alpha, beta, accuracy)
    l1, l2 = schmidt_coefficient(V_max)
    return(W_max, V_max, alpha, beta, all_maximal_eigenvalues,l1,l2)
def value_from_angles(a,b,factor):
    O = count_operators_AB2(a, b)
    W = count_operator_W(factor, O)
    w,v = LA.eig(W) # obliczenie wartosci i wektorow wlasnych operatora Bella
    w = w.real
    v = v.real
    return max_eigenvalue(w,v)
def chsh2(factor,accuracy):
    a = np.pi/2
    b = np.pi/2
    w_act, v_act = value_from_angles(a,b,factor)
    acc = 0.1            
    while acc > accuracy:
        w0 = w_act
        #print(w0,a,b,acc)        
        for a_n,b_n in neighbour(a,b, acc):    
            w_new, v_new = value_from_angles(a_n,b_n,factor)
            if w_new > w_act:    
                w_act = w_new
                v_act=v_new
                a = a_n
                b = b_n
        acc = acc/2 if w0 == w_act else acc
    quantum = True
    for a_c, b_c in corners():
        w_new, v_new = value_from_angles(a_c,b_c,factor)
        if w_act < w_new:
            quantum = False
            w_act = w_new
            v_act=v_new 
            a = a_c
            b = b_c
        
    l1, l2 = schmidt_coefficient(v_act)
    return(w_act, v_act, a, b, l1,l2, quantum)
    
def chsh_fast(factor,accuracy):
    a = b = np.linspace(0.0, np.pi, int(1/accuracy))
    LHV = -sys.maxsize-1
    W_max = -sys.maxsize-1
    for i in range(len(a)):
        for j in range(len(b)):
            O = count_operators_AB(i, j, a, b)
            W = count_operator_W(factor, O)
            w,v = LA.eig(W)
            w = w.real
            v = v.real
            w_new, v_new = max_eigenvalue(w,v)
            if (i == 0 and j == 0) or (i == len(a)-1 and j == 0) or(i == 0 and j == len(b)-1) or(i == len(a)-1 and j == len(b)-1):
                if w_new > LHV:
                    LHV = w_new
            else:
                if w_new > W_max:
                    W_max = w_new
    return(W_max, LHV)

def count_correlator(Operator1, Operator2, ro):
    M = np.matmul(ro, np.kron(Operator1, Operator2))
    return np.trace(M)

def find_correlators(alpha, beta, state, accuracy):
    i = round( alpha/(np.pi*accuracy) )
    j = round( beta/(np.pi*accuracy) )
    a = b = np.linspace(0.0, np.pi, int(1/accuracy))
    correlators = [[0,0,0],[0,0,0],[0,0,0]]
    Operators = count_operators_AB(i, j, a, b)
    ro = np.outer(state,state)
    for k in range(3):
        for l in range(3):
            correlators[k][l] = round(count_correlator(Operators[k], Operators[l+3], ro), 9)
    return correlators
def find_correlators2(alpha, beta, state, accuracy):
    correlators = [[0,0,0],[0,0,0],[0,0,0]]
    Operators = count_operators_AB2(alpha, beta)
    ro = np.outer(state,state)
    for k in range(3):
        for l in range(3):
            correlators[k][l] = count_correlator(Operators[k], Operators[l+3], ro)
    return correlators

def find_LHV(all_maximal_eigenvalues, accuracy):
    bound = round(1/accuracy)-1
    candidates = [all_maximal_eigenvalues[0][0],all_maximal_eigenvalues[0][bound],all_maximal_eigenvalues[bound][0],all_maximal_eigenvalues[bound][bound]]
    max_i = -sys.maxsize-1
    for i in candidates:
        if max_i < i:
            max_i = i
    return max_i
    
############################################################################################
#
# Operators defined as:
# A1 = Gz
# A2 = cos(alpha)*Gz + sin(alpha)*Gx
# A1 = Gz
# A2 = cos(beta)*Gz + sin(beta)*Gx
# A0,B0 = 1I  <--- unit matrix
#
# W =  a00*1*1 +  a01*1*B1 +  a02*1*B2 +
#   + a10*A1*1 + a11*A1*B1 + a12*A1*B2 +
#   + a20*A2*1 + a21*A2*B1 + a22*A2*B2 +
#
############################################################################################
# INPUT:
#
# factor = [[a00, a01, a02],
#           [a10, a11, a12],
#           [a20, a21, a22]]
#
# accuracy e.g. 0.001 <---   outcome = outcome +-0.001
#
#
############################################################################################
# OUTPUT:
##############
# Functions: #
##############
# W_max, state, alpha, beta, all_maximal_eigenvalues, l1, l2 = CHSH(factor, accuracy) <-- get all values
#
# print_CHSH(W_max, V_max, alpha, beta, correlators, LHV, l1, l2)                                       <-- print all values
#
# plot_CHSH(accuracy, all_maximal_eigenvalues,save, name, title)          <-- plot3D maximal values, save={1,0}-> save or not
#                                                                             name of saved file, title of plot
#                                                                                   
# find_correlators(alpha, beta, state, accuracy)                              <-- computing all correlators
# 
# find_LHV(all_maximal_eigenvalues, accuracy)                                  <-- finding max nonquantum value
#                                                                                   of Bell functional
# animacjuj(title, all_maximal_eigenvalues,accuracy)                           <-- makeing animation of plot
####################                                                                
# Returned values: #
####################                                                                           
# W_max                   <--- Maximal eigenvalue for all angles
# state                   <--- Eigenvector for maximal eigenvalue
# alpha, beta            <--- angles for optimal opertaors A1, A2, B1, B2
# all_maximal_eigenvalues <--- array containing maximal eigenvalues for all angles 
# l1,l2                    <--- Schmidt coefficients
############################################################################################
# Example:

accuracy = 0.00000001
factor = np.zeros((3,3))
factor[1][0] = 1-2**0.5
factor[0][1] = 0
factor[2][0] = 0
factor[0][2] = 0.324
factor[1][1] = 1
factor[1][2] = 1
factor[2][1] = 1
factor[2][2] = -1
#W_max, state, alpha, beta, l1, l2 = chsh2(factor, accuracy)
#print("W1",W_max, "a",alpha,"b",beta)
    

'''
correlators = find_correlators(alpha, beta, state, accuracy)
LHV = find_LHV(all_maximal_eigenvalues, accuracy)

print_CHSH(W_max, state, alpha, beta, correlators, LHV,l1,l2)
plot_CHSH(accuracy, all_maximal_eigenvalues,0,"-","-")
#animacjuj("animacja",all_maximal_eigenvalues,accuracy)
'''



'''
 
# Sprawdzanko że wektory rzeczywiste mają rzeczywoste wektory w bazie schmidta


def Schmidt(state):
    A = np.array([(state[0], state[1]),(state[2],state[3])])
    X = np.matmul(A,A.transpose())
    w,v = LA.eig(X)
    w = w.real
    v = v
    return v
N = 1000
for i in range(N):
    
    S = np.random.normal(1,size = 4)
    v = Schmidt(S)
    if np.linalg.norm(v.imag) != 0:
        print('xd')

# Macierz W ma rzeczywiste współczynniki i jest hermitowska
        
a = 0.322
b = 1.467
print(count_operator_W(factor, count_operators_AB2(a, b)))
'''