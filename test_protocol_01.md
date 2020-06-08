test on loeweburg cluster 8. July 2020

# environment setup 
- create a folder
- creating a conda environment with python 3.6.1
- install thomas' package to train the shap models from gitlab
- install the SHAP package itself
- downloading the keras models from thomas' gdrive
- make some adaptions on numpy and sklearn

```bash
mkdir delme
cd delme

# create env
create conda env
conda create --name shap python=3.6.1
conda activate shap

# download + install project and shap_explainer package
git clone https://gitlab.com/tlinden/shap_explainer.git
cd shap_explainer
yes | pip install -r requirements.txt
yes | pip install . # installs shap_explainer from the current directory


# download keras disease models from gdrive
cd models/keras
yes | pip install gdown
gdown https://drive.google.com/uc?id=1U3lhjaU6rWJpWYP6K8Fa8iCxzpsDLwHY
tar zxvf cnn_finalmodels.tgz
mv results/*.h5 .
rm -rf results
cd ..
cd ..

# adaptions for conda envs
pip uninstall --yes numpy
conda install --yes numpy
python -m pip install --upgrade scikit-image

# conda install --yes -c conda-forge scikit-image
# conda install --yes libgcc
# conda install --yes scipy
```

# test1: SHAP installation
from https://github.com/slundberg/shap#tree-ensemble-example-with-treeexplainer-xgboostlightgbmcatboostscikit-learnpyspark-models 

```bash
yes | pip install xgboost
# conda install --yes xgboost 
```

```python
import xgboost
import shap

# load JS visualization code to notebook
shap.initjs()

# train XGBoost model
X,y = shap.datasets.boston()
model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
# shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:]) # can be omited
```


# test2: keras models
loading an arbitrary disease model and show the model summary

```python

# setup ========================================================================
sysargs = {
    'COMORB' : "Hypertension",
    'SUBSET' : 0.001,
    'WORKDIR' : "."
}


import os
import pickle
import shap
from shap_explainer.utils import custom_loss as custom_loss
from shap_explainer.utils import shap_boot 
# disable GPUS. don't import tf before `gpu_assignment([])`!
from shap_explainer.utils.gpu_assignment import gpu_assignment
gpu_assignment([]) # empty array -> no GPUs
import keras
import tensorflow as tf

os.chdir(sysargs['WORKDIR'])
PATH_KERAS = "models/keras/" # todo @johann: move files

PARAMS = {
    'modelfiles' : {
        "Anxiety" : PATH_KERAS + "finalmodel_Anxiety_MAXEPOCHS200.h5",
        "Bipolar.Schizophrenia" : PATH_KERAS + "finalmodel_Bipolar.Schizophrenia_MAXEPOCHS200.h5",
        "Depression" : PATH_KERAS + "finalmodel_Depression_MAXEPOCHS200.h5",
        "Diabetes" : PATH_KERAS + "finalmodel_Diabetes_MAXEPOCHS200.h5",
        "Hyperlipidemia" : PATH_KERAS + "finalmodel_Hyperlipidemia_MAXEPOCHS200.h5",
        "Hypertension" : PATH_KERAS + "finalmodel_Hypertension_MAXEPOCHS200.h5",
        "Migraine" : PATH_KERAS + "finalmodel_Migraine_MAXEPOCHS200.h5",
        "Overweight" : PATH_KERAS + "finalmodel_Overweight_MAXEPOCHS200.h5",
        "Stroke.IschemAttack" : PATH_KERAS + "finalmodel_Stroke.IschemAttack_MAXEPOCHS200.h5"
    },
    'hdf5file' : "data/processed/mh1_fu1_CNN1D_subdomain.h5", # todo: @johann move file
    'cv_file_path' : "data/processed/censored.incident.cvsamp.csv", # todo: @johann move file
    'COMORB' : sysargs['COMORB'],
    'CV_OUTER' : None,
    'CV_INNER' : None,
    'SUBSET' : sysargs['SUBSET'],
    'SUBSET_SEED' : 1, # fix seed for reproducibility
    'BITER' : 1 # fix to only 1 iteration (1 bootstrap sample)    
}
PARAMS['savepath'] = sysargs['WORKDIR'] + "/models/shap/"

# main =========================================================================

# reload models
if PARAMS['COMORB'] == "Hypertension":
    model = shap_boot.reloadModel(PARAMS, custom_loss, "CustomStandardNormal") 
else:
    model = shap_boot.reloadModel(PARAMS, custom_loss) 
    
# output model summary
model.summary()

```


# test3: SHAP values for a tiny subset of the data
```python
# setup ========================================================================
sysargs = {
    'COMORB' : "Hypertension",
    'SUBSET' : 0.001,
    'WORKDIR' : "."
}

import os
import pickle
import shap
from shap_explainer.utils import custom_loss as custom_loss
from shap_explainer.utils import shap_boot 
# disable GPUS. don't import tf before `gpu_assignment([])`!
from shap_explainer.utils.gpu_assignment import gpu_assignment
gpu_assignment([]) # empty array -> no GPUs
import keras
import tensorflow as tf

os.chdir(sysargs['WORKDIR'])
PATH_KERAS = "models/keras/" # todo @johann: move files

PARAMS = {
    'modelfiles' : {
        "Anxiety" : PATH_KERAS + "finalmodel_Anxiety_MAXEPOCHS200.h5",
        "Bipolar.Schizophrenia" : PATH_KERAS + "finalmodel_Bipolar.Schizophrenia_MAXEPOCHS200.h5",
        "Depression" : PATH_KERAS + "finalmodel_Depression_MAXEPOCHS200.h5",
        "Diabetes" : PATH_KERAS + "finalmodel_Diabetes_MAXEPOCHS200.h5",
        "Hyperlipidemia" : PATH_KERAS + "finalmodel_Hyperlipidemia_MAXEPOCHS200.h5",
        "Hypertension" : PATH_KERAS + "finalmodel_Hypertension_MAXEPOCHS200.h5",
        "Migraine" : PATH_KERAS + "finalmodel_Migraine_MAXEPOCHS200.h5",
        "Overweight" : PATH_KERAS + "finalmodel_Overweight_MAXEPOCHS200.h5",
        "Stroke.IschemAttack" : PATH_KERAS + "finalmodel_Stroke.IschemAttack_MAXEPOCHS200.h5"
    },
    'hdf5file' : "data/processed/mh1_fu1_CNN1D_subdomain.h5", # todo: @johann move file
    'cv_file_path' : "data/processed/censored.incident.cvsamp.csv", # todo: @johann move file
    'COMORB' : sysargs['COMORB'],
    'CV_OUTER' : None,
    'CV_INNER' : None,
    'SUBSET' : sysargs['SUBSET'],
    'SUBSET_SEED' : 1, # fix seed for reproducibility
    'BITER' : 1 # fix to only 1 iteration (1 bootstrap sample)    
}
PARAMS['savepath'] = sysargs['WORKDIR'] + "/models/shap/"

# main =========================================================================

# reload models
if PARAMS['COMORB'] == "Hypertension":
    model = shap_boot.reloadModel(PARAMS, custom_loss, "CustomStandardNormal") 
else:
    model = shap_boot.reloadModel(PARAMS, custom_loss) 
    
# load data + train shap explainer 
shap_explainer = shap_boot.shap_bootstrap(model, PARAMS)

# save shap explainer model
fname = PARAMS['savepath'] + "shap_explainer-" + PARAMS['COMORB'] + ".pkl"
with open(fname , 'wb') as filename:
    pickle.dump(
        {
            'PARAMS' : PARAMS,
            'shap_explainer' : shap_explainer
        }, 
        filename)
```