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
