import numpy as np
import sys

def signe(x):
    if x >= 0:
        return 1
    else:
        return -1

def stabilite(w, x, tau):
 

    norm_w = np.linalg.norm(w)
    if norm_w == 0:
        return 0
    return (tau * np.dot(w, x)) / norm_w

def minim_error(X, t, w, maxIter, eta, temp):
    """
    Implements the Minimerror learning rule.
    Minimizes the cost function V = [1 - tanh(beta * gamma / 2)] / 2
    """
    beta = 1.0 / temp
    
    for _ in range(maxIter):
        for i in range(len(X)):
            # 1. Calculate Stability (gamma) for current pattern
            gamma = stabilite(w, X[i], t[i])
            
       
            
            arg = (beta * gamma) / 2.0
            gradient_factor = (beta / 4.0) * (1 - np.tanh(arg)**2)
            
            
            w += eta * gradient_factor * t[i] * X[i]
            
        

    return w



