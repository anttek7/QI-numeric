import numpy as np

def hardyPoint():
    A0 = 5 - 2*np.sqrt(5)
    A1 = np.sqrt(5) - 2
    B0 = 5 - 2*np.sqrt(5)
    B1 = np.sqrt(5) - 2
    A0B0 = 6*np.sqrt(5) - 13
    A0B1 = 3*np.sqrt(5) - 6
    A1B0 = 3*np.sqrt(5) - 6
    A1B1 = 2*np.sqrt(5) - 5
    return (A0,A1,B0,B1,A0B0,A0B1, A1B0,A1B1)

    