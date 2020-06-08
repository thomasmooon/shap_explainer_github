# helper functions to compute shap values using bootstrap (subsamples of data)
# from keras.models import load_model
from tensorflow.keras.models import load_model
# import numpy as np
# import os
# import pandas as pd
import shap
# import sys
from shap_explainer.utils.Train_Test_Data_FromHDF5 import Train_Test_Data as TTD

"""
# example for `PARAMS`

modelspath = "/root/work/src/AWS_finalmodel/q30_train_finalmodel/results/"

PARAMS = {
    'modelfiles' : {
        "Anxiety" : modelspath + "finalmodel_Anxiety_MAXEPOCHS200.h5",
        "Bipolar.Schizophrenia" : modelspath + "finalmodel_Bipolar.Schizophrenia_MAXEPOCHS200.h5",
        "Depression" : modelspath + "finalmodel_Depression_MAXEPOCHS200.h5",
        "Diabetes" : modelspath + "finalmodel_Diabetes_MAXEPOCHS200.h5",
        "Hyperlipidemia" : modelspath + "finalmodel_Hyperlipidemia_MAXEPOCHS200.h5",
        "Hypertension" : modelspath + "finalmodel_Hypertension_MAXEPOCHS200.h5",
        "Migraine" : modelspath + "finalmodel_Migraine_MAXEPOCHS200.h5",
        "Overweight" : modelspath + "finalmodel_Overweight_MAXEPOCHS200.h5",
        "Stroke.IschemAttack" : modelspath + "finalmodel_Stroke.IschemAttack_MAXEPOCHS200.h5"
    },
    'hdf5file' : "/root/work/data/mh1_fu1_CNN1D_subdomain.h5",
    'cv_file_path' : "/root/work/data/censored.incident.cvsamp.csv",    
    'COMORB' : sysargs['COMORB'],
    'CV_OUTER' : None,
    'CV_INNER' : None,
    'SUBSET' : sysargs['SUBSET'],
    'SUBSET_SEED' : sysargs['BITER'],
    'BITER' : sysargs['BITER']    
}
PARAMS['savepath'] = "/root/work/src/AWS_finalmodel/q40_shap/results/" + PARAMS['COMORB'] + "/"
PARAMS['logfile_npy'] = PARAMS['savepath'] + PARAMS['COMORB'] + "_shap_log.npy"
PARAMS['logfile_txt'] = PARAMS['savepath'] + PARAMS['COMORB'] + "_shap_log.txt"
"""

           
def reloadModel(PARAMS, custom_loss, custom_init=None):
    """ 
    Load trained final comorbidity-specific model.
    """
    if custom_init == "CustomStandardNormal":
        # https://github.com/keras-team/keras/issues/3867#issuecomment-313336090
        from keras.backend import random_normal_variable as random_normal_variable
        from tensorflow.keras.utils import get_custom_objects
        
        # added `partition_info=None` in arg list
        # https://stackoverflow.com/a/41560295
        # https://github.com/tensorflow/tensorflow/issues/24573#issuecomment-451355905
        class CustomInitializer:
            def __call__(self, shape, dtype=None, partition_info=None):
                return random_normal_variable(shape, mean=0, scale=1, dtype=dtype)
    
        model = load_model(
            PARAMS['modelfiles'][PARAMS['COMORB']], 
            custom_objects = {
                # from tlmisc.utils import custom_loss as custom_loss
                'negative_log_likelih_cox' : custom_loss.negative_log_likelih_cox,
                'initializer' : CustomInitializer
            }
        )
    
    elif custom_init == None:
        model = load_model(
            PARAMS['modelfiles'][PARAMS['COMORB']], 
            custom_objects = {
                # from tlmisc.utils import custom_loss as custom_loss
                'negative_log_likelih_cox' : custom_loss.negative_log_likelih_cox
            }
        )
    return model


def shap_bootstrap(model, PARAMS):
    """
    Repeat SHAP PARAMS['BITER'] times, given the `model` and the data (query wrt `PARAMS`).
    
    Perform PARAMS['BITER'] bootstrap iterations on the model,
    given a subset of PARAMS['SUBSET'] patients
    
    # Arguments
        model (keras model): comorbidity specific keras model
        PARAMS (dict): dictionary with parameters to query the data, bootstrap, paths to save files
        
     # Returns
         shap explainer model
         
    """

    # get data
    #=========
    data = TTD(
        hdf5path = PARAMS['hdf5file'],
        comorbidity = PARAMS['COMORB'],
        cv_outer = PARAMS['CV_OUTER'],
        cv_file_path = PARAMS['cv_file_path'],
        subset=PARAMS['SUBSET'],
        subset_seed = PARAMS['SUBSET_SEED'],
        cv_inner = PARAMS['CV_INNER'],
        cv_inner_random_state = 0)
    
    data.getAllData()
    data.reSHAPe_data() # mutate data for SHAP compatibility

    # estimate shap explainer model
    explainer = shap.DeepExplainer(model, data = data.data["train"])

    return explainer

