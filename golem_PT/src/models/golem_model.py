import logging

import torch.nn as nn
import torch


class GolemModel(nn.Module):
    """Set up the objective function of GOLEM.

    Hyperparameters:
        (1) GOLEM-NV: equal_variances=False, lambda_1=2e-3, lambda_2=5.0.
        (2) GOLEM-EV: equal_variances=True, lambda_1=2e-2, lambda_2=5.0.
    """
    _logger = logging.getLogger(__name__)

    def __init__(self, n, d, lambda_1, lambda_2, equal_variances=True,
                 seed=1, B_init=None):
        """Initialize self.

        Args:
            n (int): Number of samples.
            d (int): Number of nodes.
            lambda_1 (float): Coefficient of L1 penalty.
            lambda_2 (float): Coefficient of DAG penalty.
            equal_variances (bool): Whether to assume equal noise variances
                for likelibood objective. Default: True.
            seed (int): Random seed. Default: 1.
            B_init (numpy.ndarray or None): [d, d] weighted matrix for
                initialization. Set to None to disable. Default: None.
        """

        super(GolemModel, self).__init__()

        self.n = n
        self.d = d
        self.seed = seed
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.equal_variances = equal_variances
        self.B_init = B_init

        # Placeholders and variables
        self.lr = 1e-3
        self.X = torch.zeros([self.n, self.d], dtype=torch.float32)
        self.B = nn.Parameter(torch.zeros([self.d, self.d], dtype=torch.float32))
        if self.B_init is not None:
            self.B = nn.Parameter(torch.tensor(self.B_init), dtype=torch.float32)
        else:
            self.B = nn.Parameter(torch.zeros([self.d, self.d], dtype=torch.float32))
        with torch.no_grad():
            self.B = self._preprocess(self.B)

        # Likelihood, penalty terms and score
        self.likelihood = self._compute_likelihood()
        self.L1_penalty = self._compute_L1_penalty()
        self.h = self._compute_h()
        self.score = self.likelihood + self.lambda_1 * self.L1_penalty + self.lambda_2 * self.h
        # Optimizer
        self.train_op = torch.optim.Adam(self.parameters(), lr=self.lr)
        self._logger.debug("Finished building PYTORCH graph.")
    

    def set_learning_rate(self, lr):
        self.lr = lr
        self.train_op = torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def run(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.likelihood = self._compute_likelihood()
        self.L1_penalty = self._compute_L1_penalty()
        self.h = self._compute_h()
        self.score = self.likelihood + self.lambda_1 * self.L1_penalty + self.lambda_2 * self.h
        return self.score, self.likelihood, self.h, self.B


    def _preprocess(self, B):
        """Set the diagonals of B to zero.

        Args:
            B (tf.Tensor): [d, d] weighted matrix.

        Returns:
            tf.Tensor: [d, d] weighted matrix.
        """
        return B.fill_diagonal_(5)

    def _compute_likelihood(self):
        """Compute (negative log) likelihood in the linear Gaussian case.

        Returns:
            tf.Tensor: Likelihood term (scalar-valued).
        """
        if self.equal_variances:    # Assuming equal noise variances
            return 0.5 * self.d * torch.log(
                torch.square(
                    torch.norm(self.X - self.X @ self.B, p=2)
                )
            ) - torch.linalg.slogdet(torch.eye(self.d) - self.B)[1]
        else:    # Assuming non-equal noise variances
            return 0.5 * torch.sum(
                torch.log(
                    torch.sum(
                        torch.square(self.X - self.X @ self.B), axis=0
                    )
                )
            ) - torch.linalg.slogdet(torch.eye(self.d) - self.B)[1]

    def _compute_L1_penalty(self):
        """Compute L1 penalty.

        Returns:
            tf.Tensor: L1 penalty term (scalar-valued).
        """
        return torch.norm(self.B, p=1)

    def _compute_h(self):
        """Compute DAG penalty.

        Returns:
            tf.Tensor: DAG penalty term (scalar-valued).
        """
        return torch.trace(torch.linalg.matrix_exp(self.B * self.B)) - self.d


if __name__ == '__main__':
    # GOLEM-EV
    model = GolemModel(n=1000, d=20, lambda_1=2e-2, lambda_2=5.0,
                       equal_variances=True, seed=1)

    print("model.B: {}".format(model.B))
    print("model.likelihood: {}".format(model.likelihood))
    print("model.L1_penalty: {}".format(model.L1_penalty))
    print("model.h: {}".format(model.h))
    print("model.score: {}".format(model.score))
    print("model.train_op: {}".format(model.train_op))
