"""
the dagma_linear function provides a convient wraper around the DagmaLinear class
"""

try:
   from .models.dagma.linear_dagma import DagmaLinear
except ImportError:
   from models.dagma.linear_dagma import DagmaLinear


import numpy as np
from sklearn.model_selection import train_test_split


def dagma_linear(X: np.ndarray,
            lambda1: float = 0.03,
            es_threshold = 1e-6,
            validate=None,
            max_iter: int = 6e4,
            checkpoint: int = 1000,
            loss_type: str = 'l2', 
            verbose: bool = False):
    """Solve the unconstrained optimization problem of NOTEARS.

    Args:
        X (numpy.ndarray): [n, d] data matrix.
        lambda_1 (float): Coefficient of L1 penalty.
        es_threshold (float): early stop threshold
        validate (float): percent to reserve in validation
        max_iter (int): maximum iterations
        checkpoint (int): itterations befor checkpoint
        loss_type (str): l2, logistic, poisson
        verbose (bool): write to stdout
    Returns:
        numpy.ndarray: [d, d] estimated weighted matrix.

    """
    #split training and validation data
    if validate is not None:
       X, validate = train_test_split(X, test_size=validate)

    #set model
    dl = DagmaLinear(loss_type, verbose)

    #train
    B_est = dl.fit(X=X, lambda1=lambda1, es_threshold=es_threshold,
            validate=validate, max_iter=max_iter, checkpoint=checkpoint)
    return B_est
