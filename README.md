# Inverse Variance Weighting in python #
 
## Description ##
 
Provides two classes for inverse-variance weighting (binary outcome & correlation-based) including two estimators for covariance among correlated effect sizes.

**Requires the following python libraries:**

* python3 (tested on 3.11.11)
* [statsmodels](https://www.statsmodels.org/stable/index.html) >= 0.14.4
* [scipy](https://scipy.org/) >= 1.14.1
* [numpy](https://numpy.org/) >= 1.26.4
* [pandas](https://pandas.pydata.org/) >= 2.2.3


**Plotting utility requires the following python libraries:**

* [matplotlib](https://matplotlib.org/) >= 3.9.2
* [seaborn](https://seaborn.pydata.org/) >= 0.13.2


If you use conda, you can use the following command to create a specific environment with the above packages:

```{bash}
conda create -n "stats" python=3.11.11 statsmodels=0.14.4 scipy=1.14.1 numpy=1.26.4 pandas=2.2.3 matplotlib=3.9.2 seaborn=0.13.2
```


## Analysis of "Gut microbes linked with metabolic health, nutrition and diet interventions" ##


### Step 1: clone or download the repository and navigate the tools folder ###

```{bash}
git clone https://github.com/SegataLab/inverse_var_weight.git
cd inverse_var_weight/tool_scripts_scores
```


### Step 2: run the fundamental analyses ###

This step will perform a set of analyses on over 7.800 public metagenomic samples linking the ZOE MB Health & ZOE MB Diet scores with BMI and multiple disease types.

First creates the necessary folders:

```{bash}
mkdir ../temporary ../bmi_pooled_analyses ../pooling_disease_meta_analyses
```

The run the Python script:

```{bash}
python ZOE_Score_Public_Data_Fundamental_Analyses.py
```


### Step 3: perform pooled (meta-)analyses on the disease-related datasets, including accounting for correlated effect sizes ###

```{bash}
python diseased_pooled_analysis_weighted_scores.py
```


### Step 4 (optional): plot disease analysis 

```{bash}
python generate_disease_centred_main_image.py
```
