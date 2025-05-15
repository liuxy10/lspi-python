# -*- coding: utf-8 -*-
"""Contains main LSPI method and various LSTDQ solvers."""

import abc
import logging

import numpy as np

import scipy.linalg
from numpy.linalg import eig


class Solver(object):

    r"""ABC for LSPI solvers.

    Implementations of this class will implement the various LSTDQ algorithms
    with various linear algebra solving techniques. This solver will be used
    by the lspi.learn method. The instance will be called iteratively until
    the convergence parameters are satisified.

    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def solve(self, data, policy):
        r"""Return one-step update of the policy weights for the given data.

        Parameters
        ----------
        data:
            This is the data used by the solver. In most cases this will be
            a list of samples. But it can be anything supported by the specific
            Solver implementation's solve method.
        policy: Policy
            The current policy to find an improvement to.

        Returns
        -------
        numpy.array
            Return the new weights as determined by this method.

        """
        pass  # pragma: no cover


class LSTDQSolver(Solver):

    """LSTDQ Implementation with standard matrix solvers.

    Uses the algorithm from Figure 5 of the LSPI paper. If the A matrix
    turns out to be full rank then scipy's standard linalg solver is used. If
    the matrix turns out to be less than full rank then least squares method
    will be used.

    By default the A matrix will have its diagonal preconditioned with a small
    positive value. This will help to ensure that even with few samples the
    A matrix will be full rank. If you do not want the A matrix to be
    preconditioned then you can set this value to 0.

    Parameters
    ----------
    precondition_value: float
        Value to set A matrix diagonals to. Should be a small positive number.
        If you do not want preconditioning enabled then set it 0.
    """

    def __init__(self, precondition_value=.1):
        """Initialize LSTDQSolver."""
        self.precondition_value = precondition_value

    def solve(self, data, policy):
        """Run LSTDQ iteration.

        See Figure 5 of the LSPI paper for more information.
        """
        k = policy.basis.size()
        a_mat = np.zeros((k, k))
        np.fill_diagonal(a_mat, self.precondition_value)

        b_vec = np.zeros((k, 1))
        # print(f"data: {data}")

        for sample in data:
            phi_sa = (policy.basis.evaluate(np.array(sample.state), np.array(sample.action))
                      .reshape((-1, 1)))

            if not sample.absorb:
                best_action = policy.best_action(np.array(sample.next_state))
                phi_sprime = (policy.basis
                              .evaluate(np.array(sample.next_state), best_action)
                              .reshape((-1, 1)))
            else:
                phi_sprime = np.zeros((k, 1))

            a_mat += phi_sa.dot((phi_sa - policy.discount*phi_sprime).T) 
            b_vec += phi_sa*sample.reward 
        # print(f"a_mat: {a_mat}")
        a_rank = np.linalg.matrix_rank(a_mat)
        # print(f"a_rank: {a_rank}, k: {k}")
        if a_rank == k:
            # print(f"solving a_mat: {a_mat.shape}, b_vec: {b_vec.shape} using np.linalg.pinv")
            w = np.linalg.pinv(a_mat).dot(b_vec)
            # print(f"w: {w}")
        else:
            logging.warning('A matrix is not full rank. %d < %d', a_rank, k)
            w = scipy.linalg.lstsq(a_mat, b_vec)[0]
        # print(f"w: {w}, w.shape: {w.shape}, a_rank: {a_rank}, k: {k}")
        return w.reshape((-1, ))


class PICESolver(LSTDQSolver):
    """PICE Implementation with standard matrix solvers. """
    def __call__(self, *args, **kwds):
        return super().__call__(*args, **kwds)
    
    def solve(self, data, policy):
        k = policy.basis.size()
        a_mat = np.zeros((k, k))
        np.fill_diagonal(a_mat, self.precondition_value)

        b_vec = np.zeros((k, 1))
        # print(f"data: {data}")

        for sample in data:
            phi_sa = (policy.basis.evaluate(np.array(sample.state), np.array(sample.action))
                      .reshape((-1, 1)))

            if not sample.absorb:
                best_action = policy.best_action(np.array(sample.next_state))
                phi_sprime = (policy.basis
                              .evaluate(np.array(sample.next_state), best_action)
                              .reshape((-1, 1)))
            else:
                phi_sprime = np.zeros((k, 1))

            a_mat += phi_sa.dot((phi_sa - policy.discount*phi_sprime).T) 
            b_vec += phi_sa*sample.reward 
        # print(f"a_mat: {a_mat}")
        a_rank = np.linalg.matrix_rank(a_mat)

        
        j=1
        error=1
        stepsize=0.5 * 0.5
        phiw=np.zeros((k,1))
        Cphi=a_mat/len(data)
        dphi=b_vec/len(data)
        while j<=1000 and error>1e-4:
            oPhiw=phiw
            residuePhi=phiw-stepsize*(Cphi@phiw-dphi)
            phiw=proDysktra(residuePhi,100,1e-4)
            j=j+1
            error=np.linalg.norm(oPhiw-phiw)
            stepsize=1/(j+1)
        
        w=phiw

        return w.reshape((-1, ))
    

# alternating projection
def proDysktra(x0,ballR,errTol):
    """Projection onto the set of symmetric matrices."""    

    error=1
    j=1
    I= np.zeros((len(x0),2))
    oldI=np.zeros((len(x0),2))
    x=x0.reshape((-1,))
    while j<500 and error>errTol: 
        
        oldX=x
        if np.linalg.norm(x-I[:,0])>ballR:
            x=ballR*(x-I[:,0])/np.linalg.norm(x-I[:,0])                    
        else:
            x=x-I[:,0]               
        oldI[:,0]=I[:,0]
        I[:,0]=x-(oldX-I[:,0])
        
        oldX=x
        s=convertW2S(x-I[:,1])
        D, V = eig(s)  # D is diagonal matrix, V is orthogonal
        D[D>0]=0    # set negative eigenvalues to zero                          
        s=V@np.diag(D)@V.T
        x=convertS2W(s) # x is the new point
        oldI[:,1]=I[:,1]
        I[:,1]=x-(oldX-I[:,1])  
                        
        j=j+1
        error=np.linalg.norm(oldI-I)**2                
            
    return x.reshape(-1,1)  # return the projection of x0 onto the set of symmetric matrices

def convertW2S(w):
    """Convert weight vector to a symmetric matrix."""
    # Size of the symmetric matrix
    n = int((np.sqrt(1 + 8 * len(w)) - 1) / 2) 
    idx = 0
    Phat = np.zeros((n, n))
    
    # Fill the upper triangular part of the matrix
    for r in range(n):
        for c in range(r, n):
            Phat[r, c] = w[idx]
            idx += 1
    
    # Symmetrize the matrix
    S = 0.5 * (Phat + Phat.T)
    return S

def convertS2W(S):
    """Convert symmetric matrix to weight vector."""
    if not np.allclose(S, S.T):
        raise ValueError("Input error: S must be a symmetric matrix.")
    
    n = S.shape[0]
    W = []
    for i in range(n):
        for j in range(i, n):
            W.append(S[i, j])
    return np.array(W)

