# data_free_KD_regression

This repository is for the study on data free knowledge distillation for regression.

### Set up environment
```
git clone https://github.com/zhoutianxun/data_free_KD_regression.git
cd data_free_KD_regression
conda env create -f environment.yml
conda activate data_free_KD
```

### Run experiments
#### Regression datasets
First, unzip CTScan and Indoorloc datasets (compressed due to Github file size limit), located under ./datasets

Types of models in this experiment
1. teacher
2. baseline (simple gaussian sampling)
3. student (1) with generator sampling, decreasing alpha
4. student (2) with sampling by direct optimization of the generator loss function, decreasing alpha
5. student (3) with generator sampling, alpha=1
6. student (4) with sampling by direct optimization of the generator loss function, alpha=1

If you would like to rerun all experiments:
```
python run_all_experiments.py
```

If you would like to view results only: 
Change line 7 in run_all_experiments.py to
```
rerun = False
```
Then,
```
python run_all_experiments.py
```

#### MNIST experiment
Experiment can be run through jupyter notebook: mnist_regression.ipynb
```
cd data_free_KD_mnist
jupyter notebook
```

#### Protein solubility case-study experiment
Experiment can be run through jupyter notebook: regression_model_protein.ipynb
```
cd protein_solubility_case_study
jupyter notebook
```