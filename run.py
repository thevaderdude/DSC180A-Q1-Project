"""A conviniant way to run all programs in this project"""

from subprocess import run
import sys
from glob import glob

#For Comparisonsbetween methods
from src.compare import compare
from src.data_loader.synthetic_dataset import SyntheticDataset
from src.data_loader.real_datasets import *

#For Paper Replications Synthetic data
import logging
import seaborn as sns
import matplotlib.pyplot as plt

from src.data_loader.synthetic_dataset import SyntheticDataset
from src.golem import golem
from src.golem_torch import golem_torch
from src.notears_linear import notears_linear
from src.dagma_linear import dagma_linear
from src.utils.logger import LogHelper
from src.utils.train import plotShd, plotPreds, testMultipleMethods


#For Real World Datasets
import cdt

#For Early Stopping Analyses
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.utils.utils import get_init_path

commands = {
'clear': clear,
'all': all,
'main':main_comand,
'compare':compareisons,
'replicate':replication,
'real_data':real_dataset,
'early_stop':early_stop
}

def run():
    parser = argparse.ArgumentParser()

    cmd = parser.cmd

    comands[cmd]()

def all():
    """Dynamicaly gets and runs all comands in comands"""
    for k, f in comands.values():
        print("running:{}".format(k)
        f()

def main_comand():
    """Runs a set of executions of main.py evaluating all models against
       synthetic datasets"""
    subprocess.run(['src/run_examples.bash'], std_out = sys.stdout, check=True, text=True)

def comareisons():
    """generates a set of graphs showing how the different models
       perform against eachother with the same datasets"""
    seed = 1
    num_iter = 1e+5
    learning_rate=1e-3

    output_dir = "output"

    examples = 10
    d = 6
    degree = 3


    dataset = SyntheticDataset(examples, d, 'ER', 
                           degree, 'gaussian_ev', 
                           3, seed)

   compare(dataset, nonlinear=True, dims=[d,1],output_dir=output_dir,
                equal_variances = True,num_iter = 1e+5,learning_rate=1e-3)

   examples = 1000
   d = 6
   degree = 3


   dataset = SyntheticDataset(examples, d, 'ER', 
                           degree, 'gaussian_ev', 
                           3, seed)
   compare(dataset, nonlinear=True, dims=[d,1],output_dir=output_dir,
               equal_variances = True,num_iter = 1e+5,learning_rate=1e-3)

   examples = 1000
   d = 6
   degree = 5


   dataset = SyntheticDataset(examples, d, 'ER', 
                          degree, 'gaussian_ev', 
                          3, seed)
   compare(dataset, nonlinear=True, dims=[d,1],output_dir=output_dir,
           equal_variances = True,num_iter = 1e+5,learning_rate=1e-3)

   examples = 1000
   d = 12
   degree = 5


   dataset = SyntheticDataset(examples, d, 'ER', 
                          degree, 'gaussian_ev', 
                          3, seed)
   compare(dataset, nonlinear=True, dims=[d,3,1],output_dir=output_dir,
           equal_variances = True,num_iter = 1e+5,learning_rate=1e-3)

   examples = 1000
   d = 12
   degree = 10


   dataset = SyntheticDataset(examples, d, 'ER', 
                      degree, 'gaussian_ev', 
                      3, seed)
   compare(dataset, nonlinear=True, dims=[d,3,1],output_dir=output_dir,
           equal_variances = True, num_iter = 1e+5,learning_rate=1e-3)

def replication():
    """replicates the findings of the original DAGMA
       NOTEARS, and GOLEM papers"""
    examples = 1000
    d = 10
    degree = 3
    graph_type = 'ER'###
    noise_type = 'gaussian_nv'
    seed=332

    ds = [d*i for i in range(1, 7)]

    datasets = [SyntheticDataset(examples, i, graph_type, 
                           degree, noise_type, 
                           3, seed=seed)
            for i in ds]
    vals = testMultipleMethods(lambda x: golem(x,.3,.3,num_iter=10000,
                                   equal_variances=True), datasets)
    plotShd(vals, ds)
    plt.show()
    plotPreds(vals, ds)
    vals = testMultipleMethods(lambda x: golem(x,.3,.3,num_iter=10000,
                                   equal_variances=False), datasets)
    plotShd(vals, ds)
    plt.show()
    plotPreds(vals, ds)
    vals = testMultipleMethods(lambda x: golem_torch(x,.3,.3,num_iter=10000,
                                         equal_variances=True), datasets)
    plotShd(vals, ds)
    plt.show()
    plotPreds(vals, ds)
    vals = testMultipleMethods(lambda x: golem_torch(x,.3,.3,num_iter=10000,
                                         equal_variances=False), datasets)
    plotShd(vals, ds)
    plt.show()
    plotPreds(vals, ds)
    vals = testMultipleMethods(lambda x: dagma_linear(x,.3,loss_type='logistic',max_iter=10000), datasets)
    plotShd(vals, ds)
    plt.show()
    plotPreds(vals, ds)
    vals = testMultipleMethods(lambda x: dagma_linear(x,.3,loss_type='l2',max_iter=10000), datasets)
    plotShd(vals, ds)
    plt.show()
    plotPreds(vals, ds)
    vals = testMultipleMethods(lambda x: notears_linear(x,.3,'l2',checkpoint=100,max_iter=100), datasets)
    plotShd(vals, ds)
    plt.show()
    plotPreds(vals, ds)
    examples = 1000
    d = 10
    degree = 3
    graph_type = 'ER'###
    noise_type = 'gaussian_nv'
    seed=332

    examples = [examples*i for i in range(1, 11, 2)]

    datasets = [SyntheticDataset(i, d, graph_type, 
                           degree, noise_type, 
                           3, seed=seed)
            for i in ds]
    vals = testMultipleMethods(lambda x: golem(x,.3,.3,num_iter=10000,
                                   equal_variances=True), datasets)
    plotShd(vals, ds)
    plt.show()
    plotPreds(vals, ds)
    vals = testMultipleMethods(lambda x: golem(x,.3,.3,num_iter=10000,
                                   equal_variances=False), datasets)
    plotShd(vals, ds)
    plt.show()
    plotPreds(vals, ds)
    vals = testMultipleMethods(lambda x: golem(x,.3,.3,num_iter=10000,
                                   equal_variances=False), datasets)
    plotShd(vals, ds)
    plt.show()
    plotPreds(vals, ds)
    vals = testMultipleMethods(lambda x: golem_torch(x,.3,.3,num_iter=10000,
                                         equal_variances=True), datasets)
    plotShd(vals, ds)
    plt.show()
    plotPreds(vals, ds)
    vals = testMultipleMethods(lambda x: dagma_linear(x,.3,loss_type='logistic',num_iter=10000), datasets)
    plotShd(vals, ds)
    plt.show()
    plotPreds(vals, ds)
    vals = testMultipleMethods(lambda x: dagma_linear(x,.3,loss_type='l2',num_iter=10000), datasets)
    plotShd(vals, ds)
    plt.show()
    plotPreds(vals, ds)
    vals = testMultipleMethods(lambda x: notears_linear(x,.3,'l2',checkpoint=100,max_iter=100), datasets)
    plotShd(vals, ds)
    plt.show()
    plotPreds(vals, ds)

def real_dataset():
    
    print("Tuebingen Dataset")
    s_data, s_graph = cdt.data.load_dataset("tuebingen")

    col_num = len(s_data.columns)

    for i in range(col_num):
        column = s_data[s_data.columns[i]]
        plt.subplot(1,col_num,i)
    
        sns.histplot(column, label=col)
    
    plt.subplot(1,2,1)
    plt.matshow(s_data.corr())
    plt.title('Tuebingen corrilations')

    plt.subplot(1,2,2)
    plt.matshow(s_graph)
    plt.title('Tuebingen causation')

    plt.show()

    dataset = getTuebingen()
    compare(dataset, output_dir=output_dir, notears=False)

    print("Sacks Dataset")
    s_data, s_graph = cdt.data.load_dataset("sacks")

    for i in range(col_num):
        column = s_data[s_data.columns[i]]
        plt.subplot(1,col_num,i)
    
        sns.histplot(column, label=col)
    
    plt.subplot(1,2,1)
    plt.matshow(s_data.corr())
    plt.title('Sacks corrilations')

    plt.subplot(1,2,2)
    plt.matshow(s_graph)
    plt.title('Sacks causation')

    plt.show()

    dataset = getSacksDataset()
    compare(dataset, output_dir=output_dir, notears=False)

def early_stop():
    """we generate a set of plots of how the loss function 
       performs with various early stoping values"""
    subprocess.run(['python3', 'src/main.py', 
                     '--method GOLEM_TORCH',
                     '--seed 1 ',
                     '--d 10 ',
                     '--graph_type ER', 
                     '--degree 4',
                     '--noise_type gaussian_nv',
                     '--equal_variances',
                     '--lambda_1 2e-2',
                     '--lambda_2 5.0',
                     '--checkpoint_iter 5000'], 
                     std_out = sys.stdout, check=True, text=True) 
    path = '{}/scores.csv'.format(sorted(glob('{}/*'.format('output')))[-1])

    df = pd.read_csv(path)
    plt.plot(df['i'], df['train'], label='train')
    plt.yscale("log")
    plt.show()
    plt.plot(df['i'], df['val'], label='validate')
    plt.yscale("log")
    plt.show()
    plt.plot(df['i'], df['val'] - df['train'])
    plt.show()

def clean():
    subprocess.run(['rm', 'output/*'], std_out = sys.stdout, check=True, text=True)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('cmd', choices=commands.keys())



if __name__ == '__main__':
   run()


