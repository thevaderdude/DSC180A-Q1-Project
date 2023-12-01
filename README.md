# DSC 180A Quarter 1 Project
## Improving [GOLEM](https://github.com/ignavierng/golem)
**In order to improve GOLEM, we converted original model from Tensorflow to Pytorch, added a validation set, implemented early stopping, and used a different DAG constraint.**

GOLEM converted to Pytorch with the additions described above is located in [golem_PT/](/golem_PT/).


### Validation Set
In order to implement early stopping and examine overfitting, we modified the training procedure to have a holdout dataset that was not trained on. We also made it so the training procedure will output a csv of the training and validation losses. The analysis of the loss over training is located at [golem_PT/Loss_Analysis.ipynb](golem_PT/Loss_Analysis.ipynb).

### Early Stopping

From the analysis of the loss, we determined that a threshold of **1e-4** for the relative change between checkpoints (1000 epochs) would result in performance equivalent to that of the full run in approximately half the epochs. A run of GOLEM with early stopping is located in [golem_PT/Early_Stop_Test.ipynb](golem_PT/Early_Stop_Test.ipynb).

### Alternate DAG Constraint
(Under Construction) 



## Misc. Acknowledgements
The original work using Tensorflow is located in [golem_TF/](/golem_TF/)

Updated requirements.txt is located at [golem_PT/requirements.txt](golem_PT/requirements.txt)

Examples located at [golem_PT/GOLEM-EV.ipynb](golem_PT/GOLEM-EV.ipynb)