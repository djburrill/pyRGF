'''
Green's Function Module

Contains all methods to generate surface and bulk Green's functions.

REFERENCES
    Sancho, MP Lopez, et al. "Highly convergent schemes for the calculation of bulk and surface Green functions." Journal of Physics F: Metal Physics 15.4 (1985): 851.
'''

# Imports
import numpy as np
import numpy.linalg as nlg
import matplotlib.pyplot as plt

# Functions
def Sancho(energy,H0,S0,H1,S1):
    '''
    Generate surface Green's function through recursive Sancho method.

    INPUT
        energy: (complex float) Complex energy.
        H0: (numpy matrix) Intra-layer Hamiltonian.
        S0: (numpy matrix) Intra-layer overlap matrix.
        H1: (numpy matrix) Inter-layer coupling Hamiltonian.
        S1: (numpy matrix) Inter-layer coupling overlap matrix.

    OUTPUT
        greenSurface: (complex numpy matrix) Surface Green's function.
        greenBulk: (complex numpy matrix) Bulk Green's function.
    '''

    # Variables
    invMat = nlg.inv(energy*S0-H0)
    tol = 1e-6
    itCounter = 0
    maxIter = 50

    # Initialize recursion parameters
    epSurfaceOld = H0 + H1*invMat*H1.H
    epBulkOld = epSurfaceOld + H1.H*invMat*H1
    alphaOld = H1*invMat*H1
    betaOld = H1.H*invMat*H1.H

    # Set dummy updated matrices
    epSurface = 10*tol+epSurfaceOld
    epBulk = 10*tol+epBulkOld
    alpha = 10*tol+alphaOld
    beta = 10*tol+betaOld

    # Initial update difference norms
    diff_epSurface = nlg.norm(epSurface - epSurfaceOld)
    diff_epBulk = nlg.norm(epBulk - epBulkOld)
    diff_alpha = nlg.norm(alpha - alphaOld)
    diff_beta = nlg.norm(beta - betaOld)

    # Iterate until convergence criteria satisfied
    while ((itCounter < maxIter) and ((diff_epSurface > tol) or (diff_epBulk > tol) or (diff_alpha > tol) or (diff_beta > tol))):
        # Calculate recursion parameters
        invMat = nlg.inv(energy*S0-epBulkOld)
        alpha = alphaOld*invMat*alphaOld
        beta = betaOld*invMat*betaOld
        epBulk = epBulkOld + alphaOld*invMat*betaOld + betaOld*invMat*alphaOld
        epSurface = epSurfaceOld + alphaOld*invMat*betaOld

        # Check convergence
        diff_epSurface = nlg.norm(epSurface - epSurfaceOld)
        diff_epBulk = nlg.norm(epBulk - epBulkOld)
        diff_alpha = nlg.norm(alpha - alphaOld)
        diff_beta = nlg.norm(beta - betaOld)

        # Set new values to old
        alphaOld = alpha
        betaOld = beta
        epBulkOld = epBulk
        epSurfaceOld = epSurface

        # Update itCounter
        itCounter += 1

    # Calculate Greens functions
    greenSurface = nlg.inv(energy*S0-epSurface)
    greenBulk = nlg.inv(energy*S0-epBulk)

    return greenSurface,greenBulk

# Main
if (__name__ == '__main__'):
    # Variables
    t = 1.0
    eta = 0.000001
    eRange = np.arange(-6.0,6.0,0.01,dtype='complex128')
    eRange += eta*1.j*np.ones(len(eRange))
    eigValList = []

    # Matrices
    h = np.asmatrix([[0,-t,0,0],
                     [-t,0,-t,0],
                     [0,-t,0,-t],
                     [0,0,-t,0]])
    u = np.asmatrix([[-t,0,0,0],
                     [0,-t,0,0],
                     [0,0,-t,0],
                     [0,0,0,-t]])
    s0 = np.asmatrix(np.eye(4))
    s1 = np.asmatrix(np.zeros([4,4]))

    # Calculate eigenvalues
    for energy in eRange:
        gS,gB = Sancho(energy,h,s0,u,s1)
        eigValList.append(nlg.eigvals(gS))

    # Convert to format for scatter plotting
    eigValList = np.asarray(eigValList)
    eigValList = np.transpose(eigValList)

    # Plot real part
    for eigSet in eigValList:
        plt.scatter(np.real(eRange),np.real(eigSet),marker='.')

    plt.title(r'Real Part of $g_s$')
    plt.xlabel('Energy (a.u.)')

    plt.show()

    # Plot imaginary part
    for eigSet in eigValList:
        plt.scatter(np.real(eRange),np.imag(eigSet),marker='.')

    plt.title(r'Imaginary Part of $g_s$')
    plt.xlabel('Energy (a.u.)')

    plt.show()
