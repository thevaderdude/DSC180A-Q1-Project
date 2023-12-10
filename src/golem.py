"""
the golem function provides a convient wraper around golemModel and golemTrainer

Adapted from https://github.com/ignavierng/golem dec 2023

"""




import os

try:
  from .models.golem.golem_model import GolemModel
  from .trainers.golem_trainer import GolemTrainer
except ImportError:
   from models.golem.golem_model import GolemModel
   from trainers.golem_trainer import GolemTrainer


# For logging of tensorflow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def golem(X, lambda_1, lambda_2, equal_variances=True,
          num_iter=1e+5, learning_rate=1e-3, seed=1,
          checkpoint_iter=None, output_dir=None, B_init=None):
    """Solve the unconstrained optimization problem of GOLEM, which involves
        GolemModel and GolemTrainer.

    Args:
        X (numpy.ndarray): [n, d] data matrix.
        lambda_1 (float): Coefficient of L1 penalty.
        lambda_2 (float): Coefficient of DAG penalty.
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
    # Center the data
    X = X - X.mean(axis=0, keepdims=True)

    # Set up model
    n, d = X.shape
    model = GolemModel(n, d, lambda_1, lambda_2, equal_variances, seed, B_init)


    # Training
    trainer = GolemTrainer(learning_rate)
    B_est = trainer.train(model, X, num_iter, checkpoint_iter, output_dir)

    return B_est    # Not thresholded yet

