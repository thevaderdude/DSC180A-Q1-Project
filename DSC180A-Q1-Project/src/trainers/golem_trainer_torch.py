import logging

import numpy as np

try:
  from ..utils.dir import create_dir
except ImportError:
  from utils.dir import create_dir


import torch

class GolemTrainerTorch:
    """Set up the trainer to solve the unconstrained optimization problem of GOLEM."""
    _logger = logging.getLogger(__name__)

    def __init__(self, learning_rate=1e-3):
        """Initialize self.

        Args:
            learning_rate (float): Learning rate of Adam optimizer.
                Default: 1e-3.
        """
        self.learning_rate = learning_rate
        


    def train(self, model, X_train, X_val, num_iter, es_threshold, checkpoint_iter=None, output_dir=None):
        # TODO: fix docstring
        """Training and checkpointing.

        Args:
            model (GolemModel object): GolemModel.
            X (numpy.ndarray): [n, d] data matrix.
            num_iter (int): Number of iterations for training.
            checkpoint_iter (int): Number of iterations between each checkpoint.
                Set to None to disable. Default: None.
            output_dir (str): Output directory to save training outputs. Default: None.

        Returns:
            numpy.ndarray: [d, d] estimated weighted matrix.
        """
        model.set_learning_rate(self.learning_rate)
        self._logger.info("Started training for {} iterations.".format(num_iter))
        for i in range(0, int(num_iter) + 1):
            if i == 0:    # Do not train here, only perform evaluation
                score, likelihood, h, B_est = self.eval_iter(model, X_train)
                train_data = score, likelihood, h, B_est
                val_data = self.eval_iter(model, X_val)
                val_score, val_likelihood, val_h, val_B_est = val_data
                prev_score = val_score
            else:    # Train
                score, likelihood, h, B_est = self.train_iter(model, X_train)
                train_data = score, likelihood, h, B_est
                val_data = self.eval_iter(model, X_val)
            if checkpoint_iter is not None and i % checkpoint_iter == 0:
                # do we do early stopping?
                if es_threshold is not None:
                    val_score, val_likelihood, val_h, val_B_est = val_data
                    if i == 0:
                        pass
                    else:
                        perc = (val_score - prev_score) / prev_score
                        if np.abs(perc) < es_threshold:
                            print(f'Early Stop: p: {perc}, i: {i}, loss: {val_score}')
                            break
                        prev_score = val_score

                self.train_checkpoint(i, train_data, val_data, output_dir)

        return B_est

    def eval_iter(self, model, X):
        """Evaluation for one iteration. Do not train here.

        Args:
            model (GolemModel object): GolemModel.
            X (numpy.ndarray): [n, d] data matrix.

        Returns:
            float: value of score function.
            float: value of likelihood function.
            float: value of DAG penalty.
            numpy.ndarray: [d, d] estimated weighted matrix.
        """
        with torch.no_grad():

            score, likelihood, h, B_est  = model.run(X)

        return score, likelihood, h, B_est.detach().numpy()

    def train_iter(self, model, X):
        """Training for one iteration.

        Args:
            model (GolemModel object): GolemModel.
            X (numpy.ndarray): [n, d] data matrix.

        Returns:
            float: value of score function.
            float: value of likelihood function.
            float: value of DAG penalty.
            numpy.ndarray: [d, d] estimated weighted matrix.
        """
        
        model.train_op.zero_grad()

        score, likelihood, h, B_est  = model.run(X)
        score.backward()
        model.train_op.step()

        return score, likelihood, h, model.B.detach().numpy()

    def train_checkpoint(self, i, train_data, val_data, output_dir):
        # TODO: fix docstring
        """Log and save intermediate results/outputs.

        Args:
            i (int): i-th iteration of training.
            score (float): value of score function.
            likelihood (float): value of likelihood function.
            h (float): value of DAG penalty.
            B_est (numpy.ndarray): [d, d] estimated weighted matrix.
            output_dir (str): Output directory to save training outputs.
        """
        val_score, val_likelihood, val_h, val_B_est = val_data
        score, likelihood, h, B_est = train_data
        self._logger.info(
            "TRAINING: [Iter {}] score {:.3E}, likelihood {:.3E}, h {:.3E}".format(
                i, score, likelihood, h
            )
        )
        self._logger.info(
            "VALIDATION: [Iter {}] score {:.3E}, likelihood {:.3E}, h {:.3E}".format(
                i, val_score, val_likelihood, val_h
            )
        )

        if output_dir is not None:
           # Save the weighted matrix (without post-processing)
            create_dir('{}/checkpoints'.format(output_dir))
            np.save('{}/checkpoints/B_iteration_{}.npy'.format(output_dir, i), B_est)
            # save train and val_scores
            if i == 0:
                with open(f'{output_dir}/scores.csv', 'w') as file:
                    file.write('i,train,val\n')
            # write new line
            with open(f'{output_dir}/scores.csv', 'a') as file:
                file.write(f'{str(i)},{str(score.detach().numpy())},{str(val_score.detach().numpy())}\n')

