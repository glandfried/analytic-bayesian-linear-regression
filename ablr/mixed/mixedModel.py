#https://colab.research.google.com/github/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Linear_Mixed_Effects_Models.ipynb#scrollTo=yDCkuCjq2DfW
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal as normal

def moments_posterior(t,Phi, C, alpha, beta):
    N, M = Phi.shape
    _, L = C.shape
    
    Sw_0_inv = (alpha) * np.eye(M)
    Sv_0_inv = (alpha) * np.eye(L)  
    
    #         (  sA   sB  )-1 
    # Sigma = (           )
    #         (  sC   sD  )
    sA = Sv_0_inv + beta*C.T.dot(C) 
    sD = Sw_0_inv + beta*Phi.T.dot(Phi)
    sB = beta*C.T.dot(Phi)
    sC = sB.T
    
    assert \
        (L,L) == sA.shape and \
        (M,M) == sD.shape and \
        (L,M) == sB.shape and \
        (M,L) == sC.shape 
        
    S_N_inv = np.concatenate((
        np.concatenate((sA,sB),axis=1),
        np.concatenate((sC,sD),axis=1)
    ))
    
    assert (L+M,L+M) == S_N_inv.shape
    
    S_N = np.linalg.inv(S_N_inv)
    
    Sv_N = S_N[:L,:L]
    Sw_N = S_N[L:,L:]
    Svw_N = S_N[:L,L:]
    Swv_N = S_N[L:,:L]
    
    assert \
        (L,L) == Sv_N.shape and \
        (M,M) == Sw_N.shape and \
        (L,M) == Svw_N.shape and \
        (M,L) == Swv_N.shape 
    
    A = np.concatenate((C,Phi),axis=1)
    
    m_N = beta*S_N.dot(A.T).dot(t)
    
    mv_N = m_N[:L] 
    mw_N = m_N[L:] 
    
    return (m_N, [mv_N, mw_N]), (S_N, [[Sv_N, Svw_N],[Swv_N, Sw_N]])

def p_de_w_dado_t(t, Phi, C, alpha, beta):
    
    N, M = Phi.shape
    _, L = C.shape
    
    Sw_0_inv = (alpha) * np.eye(M)
    Sv_0 = (1/alpha) * np.eye(L)
    St = (1/beta) * np.eye(N)
    
    St_dado_w_inv = np.linalg.inv(St + C.dot(Sv_0).dot(C.T))
    
    Sw_N_inv = Sw_0_inv + Phi.T.dot(St_dado_w_inv).dot(Phi)    
    Sw_N = np.linalg.inv(Sw_N_inv )
    m_N = Sw_N.dot(Phi.T).dot(St_dado_w_inv).dot(t)
    
    return m_N, Sw_N

def p_de_v_dado_t(t, Phi, C, alpha, beta):
    
    N, M = Phi.shape
    _, L = C.shape
    
    Sw_0 = (1/alpha) * np.eye(M)
    Sv_0_inv = (alpha) * np.eye(L)
    St = (1/beta) * np.eye(N)
    
    St_dado_v_inv = np.linalg.inv(St + Phi.dot(Sw_0).dot(Phi.T))
    
    Sv_N_inv = Sv_0_inv + C.T.dot(St_dado_v_inv).dot(C)    
    Sv_N = np.linalg.inv(Sv_N_inv )
    m_N = Sv_N.dot(C.T).dot(St_dado_v_inv).dot(t)
    
    return m_N, Sv_N    
    
