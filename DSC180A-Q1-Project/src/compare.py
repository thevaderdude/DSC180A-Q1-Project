from src.golem import golem
from src.data_loader.synthetic_dataset import SyntheticDataset
from src.utils.train import postprocess
from src.utils.utils import plot_method

from src.golem_torch import golem_torch

from src.models.dagma.linear_dagma import DagmaLinear

from src.models.dagma.nonlinear_dagma import DagmaNonlinear
from src.models.dagma.nonlinear_dagma import DagmaMLP

from src.models.notears.linear import notears_linear

from src.models.notears.nonlinear import NotearsMLP
from src.models.notears.nonlinear import notears_nonlinear

from time import time


def compare(dataset, lambda_1 = 2e-2, lambda_2 = 5.0,
            equal_variances = True, early_stop_threshold = 1e-4,
            loss_type = 'l2', num_iter = 1e+5,
            learning_rate=1e-3, checkpoint_iter=None,
            B_init = None, output_dir = None, out = print, 
            nonlinear=False, dims=None, seed=1, notears=True):
 

    #TODO make a better timing method   
    #original golem
    tm = time()
    B_est = golem(dataset.X, lambda_1, lambda_2, 
              equal_variances, num_iter,
              learning_rate, seed, 
              checkpoint_iter, 
              output_dir, B_init)
    
    B_processed = postprocess(B_est, 0.3)
    plot_method("original golem", 
             dataset.X, dataset.B, B_init,
             B_est, B_processed, time()-tm, out)


    #modified golem
    tm = time()
    B_est = golem_torch(dataset.X, lambda_1, lambda_2, 
              early_stop_threshold, equal_variances, num_iter,
              learning_rate, seed, 
              checkpoint_iter, 
              output_dir, B_init)


    B_processed = postprocess(B_est, 0.3)

    plot_method("modified golem",  
             dataset.X, dataset.B, B_init,
             B_est, B_processed, time()-tm, out)

    #linear dagma
    tm = time()
    dl = DagmaLinear(loss_type)

    B_est = dl.fit(dataset.X)

    B_processed = postprocess(B_est, 0.3)

    plot_method("linear dagma",  
             dataset.X, dataset.B, B_init,
             B_est, B_processed,  time()-tm,out)

    #nonlinear dagma
    if nonlinear:
    	tm = time()
    	eq_model = DagmaMLP(dims=dims, bias=True)
    	model = DagmaNonlinear(eq_model)
    	B_est = model.fit(dataset.X)

    	B_processed = postprocess(B_est, 0.3)

    	plot_method("nonlinear dagma",  
             dataset.X, dataset.B, B_init,
             B_est, B_processed, time()-tm, out)


    #linear notears
    if notears:
       tm = time()
       B_est = notears_linear(dataset.X, lambda_1, loss_type)

       B_processed = postprocess(B_est, 0.3)


       plot_method("linear notears", 
             dataset.X, dataset.B, B_init,
             B_est, B_processed, time()-tm, out)


    #nonlinear notears
    if nonlinear and notears:
       tm = time()
       model = NotearsMLP(dims=dims, bias=True)
       B_est = notears_nonlinear(model, dataset.X, lambda_1, lambda_2)

       B_processed = postprocess(B_est, 0.3)

       plot_method("nonlinear notears",  
             dataset.X, dataset.B, B_init,
             B_est, B_processed, time()-tm, out)





