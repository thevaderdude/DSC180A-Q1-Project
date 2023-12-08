import os


try:
   from .models.golem.golem_model_torch import GolemModelTorch
   from .trainers.golem_trainer_torch import GolemTrainerTorch
except ImportError:
   from models.golem.golem_model_torch import GolemModelTorch
   from trainers.golem_trainer_torch import GolemTrainerTorch

import numpy as np
from sklearn.model_selection import train_test_split


# For logging of tensorflow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def golem_torch(X, lambda_1, lambda_2, es_threshold, equal_variances=True,
          num_iter=1e+5, learning_rate=1e-3, seed=1,
          checkpoint_iter=None, output_dir=None, B_init=None):
    """Solve the unconstrained optimization problem of GOLEM, which involves
        GolemModel and GolemTrainer.

    Args:
        X (numpy.ndarray): [n, d] data matrix.
        lambda_1 (float): Coefficient of L1 penalty.
        lambda_2 (float): Coefficient of DAG penalty.
        es_threshold (float): early stop threshold
        equal_variances (bool): Whether to assume equal noise variances
            for likelibood objective. Default: True.
        num_iter (int): Number of iterations for training.
        learning_rate (float): Learning rate of Adam optimizer. Default: 1e-3.
        seed (int): Random seed. Default: 1.
        checkpoint_iter (int): Number of iterations between each checkpoint.
            Set to None to disable. Default: None.
        output_dir (str): Output directory to save training outputs.
        B_init (numpy.ndarray or None): [d, d] weighted matrix for initialization.
            Set to None to disable. Default: None.

    Returns:
        numpy.ndarray: [d, d] estimated weighted matrix.

    Hyperparameters:
        (1) GOLEM-NV: equal_variances=False, lambda_1=2e-3, lambda_2=5.0.
        (2) GOLEM-EV: equal_variances=True, lambda_1=2e-2, lambda_2=5.0.
    """

    # TRAIN VAL SPLIT
    # TODO: Parameterize this
    X_train, X_val = train_test_split(X, test_size=0.1)

    # Center the data
    X_train = X_train - X_train.mean(axis=0, keepdims=True)
    X_val = X_val - X_train.mean(axis=0, keepdims=True)


    # Set up model
    n, d = X_train.shape
    model = GolemModelTorch(n, d, lambda_1, lambda_2, equal_variances, seed, B_init)

    # Training
    trainer = GolemTrainerTorch(learning_rate)
    B_est = trainer.train(model, X_train, X_val, num_iter, es_threshold, checkpoint_iter, output_dir)

    return B_est    # Not thresholded yet

