"""
the notears_linear function provides a convient wraper around notears_linear
handeling validation data and default values
"""

import numpy as np
from sklearn.model_selection import train_test_split

try:
   from .models.notears.linear import notears_linear  as notears_linear_train 
except ImportError:
   from models.notears.linear import notears_linear  as notears_linear_train 

def notears_linear(X, lambda1, loss_type, validate=None, max_iter=100, checkpoint = 5, es_threshold=None, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3):
    """Solve the constrained optimization problem of NOTEARS.

    Args:
        X (numpy.ndarray): [n, d] data matrix.
        lambda_1 (float): Coefficient of L1 penalty.
        loss_type (str): l2, logistic, poisson
        validate (float): percent to reserve in validation
        max_iter (int): maximum iterations
        checkpoint (int): itterations befor checkpoint
        es_threshold (float): early stop threshold
        h_tol (float): limit on h_tol
        rho_max (float): limit on rho
        w_threshold (float): limit on w
    Returns:
        numpy.ndarray: [d, d] estimated weighted matrix.

    """
    #reserve validation data
    if validate is not None:
       X, validate = train_test_split(X, test_size=validate)

    #train
    B_est = notears_linear_train(X, lambda1, loss_type, validate, max_iter, es_threshold, checkpoint, h_tol, rho_max, w_threshold)

    return B_est    # Not thresholded yet
