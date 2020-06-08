import h5py
import numpy as np
import csv
from termcolor import colored
from sklearn.model_selection import KFold
import random

class Train_Test_Data:
    """
    # Methods:
    call these methods to compute the particular feature maps
        <object>.getDiagnosisData()
        <object>.getTreatmentData()
        <object>.getStaticData()
    
        will generate dictionaries like:
        <object>.data_treatment
        <object>.data_diagnosis
        <object>.data_static
        
        with content e. g. like
        <object>.data_treatment["train"] = training feature maps
        <object>.data_treatment["test"] = test feature maps
        
        
    # --- Update 2018-09-27:
        new feature: allow inner cross-validation. 5-fold inner CV is predefined, allow to query 
            patients w.r.t. the i-th fold. 

    """
    # self.SUBDOMAINS
    SUBDOMAINS = {
        "treatment": [
            "DRUG.SUBSTANCENAME",
            "DRUG.GO", 
            "DRUG.PATHWAY", 
            "DRUG.PHEWAS", 
            "DRUG.PHEWAS_HL",
            "DRUG.THRCLDS", 
            "DRUG.THRGRDS", 
            "DRUG.TISSUE", 
            "DRUG2SIDEEFFECTxFREQ"
        ],
        "diagnosis": [
            "DISEASE.BIOMARKER",
            "DISEASE.GO",
            "DISEASE.MESHDISEASE",
            "DISEASE.PATH",
            "DISEASE.PHEWAS",
            "DISEASE.PHEWAS_HL",
            "DISEASE.SYMPTOMS"
        ],
        "static": [
            "age_sex_insur_hosp", 
            "medicalrisk", 
            "region"
        ]
    }
    
    def __init__(
        self, 
        hdf5path = None, 
        comorbidity = None, 
        cv_outer = None, 
        cv_file_path = None,
        subset=1.0, 
        cv_inner = None, 
        cv_inner_random_state = 0, 
        enrolids = None,
        treat_simulation = False,
        subset_seed = None,
        verbose = False
    ):
        """ 
        # Arguments
            hdf5path: string. path to hdf5 file
            comorbidity: string. comorbidity, e. g. `Hypertension`
            cv_outer: integer 1,...,5. outer cross validation index.
            cv_inner: integer 1,...,5. outer cross validation index. 
                If default `None` applies, then no inner CV is done (only outer).
            subset: float, 0<subset<=1. Use only a subset of the outer cv-samples.
                Implemented for testing/debugging purposes.
            cv_file_path: string. path to cross-validation sample file.
            cv_inner_random_state: integer >= 0. seed for inner CV sampling.
        """
        self.hdf5path = hdf5path
        self.hdf5file = None
        self.comorbidity = comorbidity
        self.cv_outer = cv_outer
        self.cv_inner = cv_inner
        self.cv_inner_random_state = cv_inner_random_state
        self.cv_file_path = cv_file_path
        self.cv_table = None
        self.subdomain_shapes = None
        self.train_test_enrolids = None                
        self.data_treatment = None
        self.data_diagnosis = None
        self.data_static = None
        self.labels = None
        self.subset = subset
        self.subset_seed = subset_seed
        self.enrolids = enrolids
        self.treat_simulation = treat_simulation
        self.verbose = verbose
        self.__initialize()
        
        
    def __open_hdf5file(self, hdf5path):
        if self.verbose:
            print("\nOpen hdf5file %s ..." % hdf5path)
        self.hdf5file = h5py.File(hdf5path,'r')
        
    def __get_cv_table(self, cv_file_path):
        if self.verbose:
            print("\nLoad cross validation data from %s ..." % cv_file_path)
        file = []
        with open(cv_file_path) as csvDataFile:
            csvReader = csv.reader(csvDataFile, delimiter=';')
            for row in csvReader:
                file.append(row)
        cvsamp = [tuple(x) for x in file[1:]] # create structured list, omit header
        cvsamp = np.array(cvsamp, 
                              dtype = ([
                                  ("enrolid",'U21'),
                                  ("mc",'U21'),
                                  ("time",int),
                                  ("status",int),
                                  ("cv",int)]))
        if self.verbose:
            print("\nunique comorbidities:\n" + str(np.unique(cvsamp['mc'])))
            print("\nhead cross-validation table:\nENROLID, comorbidity, time, status, cv\n" + str(cvsamp[:5]))
        self.cv_table = cvsamp
    
    def __train_test_patients_and_labels_bycrossv(self):
        """
        # Output:
            Dictionary with keys "train" and "test" containing
            the corresponding enrolids
        """
        mc = self.comorbidity
        cv_outer = self.cv_outer
        cv_inner = self.cv_inner
        cvsamp = self.cv_table
        cv_inner_random_state = self.cv_inner_random_state
        subset = self.subset
        
        # --- query patients w.r.t. outer + innter cross validation index
        
        # step 1: outer cv
        # - (recall: here data was prior stratified using 5-fold CV)
        # - skip stratification for final model -> if cv_outer = None
        if cv_outer != None:
            idx_train = np.where((cvsamp['cv'] != cv_outer) & (cvsamp['mc'] == mc))
            idx_test = np.where((cvsamp['cv'] == cv_outer) & (cvsamp['mc'] == mc))
            idx_train = np.array(idx_train[0])
            idx_test = np.array(idx_test[0])
        else:
            idx_train = np.where(cvsamp['mc'] == mc)            
            idx_train = np.array(idx_train[0])
            # even though the test fold is not applicable, let it have an arbitrary dummy
            # index, e.g. 0, to avoid awkward workarounds and dependency issues
            # compared to "idx_test = None"            
            idx_test = [0] 

        if self.verbose:
            print(colored("\nQuery training/test patients subset for %s, CV outer = %s:\n-%s train samples \n-%s test samples" % 
          (mc,cv_outer, len(idx_train),len(idx_test)),"green"))
       
        # step 2.1: use inner cross validation schema if applicable
        if (cv_inner != None) & (cv_inner in [1,2,3,4,5]):
            if self.verbose:
                print(colored("\nApply 5-fold inner cross validation, i=%s" %cv_inner,"green"))
            kf = KFold(n_splits = 5, shuffle=True, random_state = cv_inner_random_state)
            idx_train_split = []
            idx_test_split = []

            for train_index, test_index in kf.split(idx_train):
                idx_train_split.append(train_index)
                idx_test_split.append(test_index)

            # step 2.2 overwrite outer-CV indices with inner-CV indices
            # take the indices corresponding to the cv_inner set (idx_<...>_iCV is an array of arrays)
            cv_inner = cv_inner-1 # cv_inner array index = cv_inner index minus 1
            train_split_k = idx_train_split[cv_inner]             
            test_split_k = idx_test_split[cv_inner]
            idx_train_ = idx_train.copy()
            idx_train = idx_train_[train_split_k]
            idx_test = idx_train_[test_split_k]

        # ---
        
        # get patients for test and training 
        enrolid_train = cvsamp['enrolid'][idx_train]
        enrolid_test = cvsamp['enrolid'][idx_test]
        if self.verbose:
            print(colored("%s, CV outer %s, CV inner %s:\n-%s train samples\n-%s test samples" % 
              (mc,cv_outer, cv_inner, len(enrolid_train),len(enrolid_test)),"green"))
        
        # query subset of patients
        if(subset<1):
            # get random subsample idx
            n1 = range(len(enrolid_train))
            k1 = int(subset*len(enrolid_train))
            random.seed(self.subset_seed)
            train_subset = sorted(random.sample(n1, k1))            
            n2 = range(len(enrolid_test))
            k2 = int(subset*len(enrolid_test))
            test_subset = sorted(random.sample(n2, k2))
            # use random subsample
            enrolid_train = enrolid_train[train_subset]
            enrolid_test = enrolid_test[test_subset]
            print(colored("Use only a subset of %d percent of patients:" %(int(subset*100)),"red"))
            print(colored("train samples: %d, test samples %d" % (len(enrolid_train),len(enrolid_test)),"red"))

        # save patients
        enrolids = {"train":enrolid_train, "test": enrolid_test}
        self.train_test_enrolids = enrolids
        
        # get labels for test and training
        train_status = cvsamp['status'][idx_train]
        train_time = cvsamp['time'][idx_train]
        test_status = cvsamp['status'][idx_test]
        test_time = cvsamp['time'][idx_test]
        
        if(subset<1):
            train_status = train_status[train_subset]
            train_time = train_time[train_subset]
            test_status = test_status[test_subset]
            test_time = test_time[test_subset]
            
        train_Y = np.array([train_status,train_time])
        train_Y = train_Y.transpose()
        test_Y = np.array([test_status,test_time])
        test_Y = test_Y.transpose()
        
        labels = {"train": train_Y, "test":test_Y}
        self.labels = labels
        
    def __train_test_patients_and_labels_byenrolids(self):
        """
        # Output:
            Dictionary with keys "train" and "test" containing
            the corresponding enrolids
        """
        
        cvsamp = self.cv_table
        enrolids = self.enrolids
#         idx_train = [cvsamp['enrolid'].tolist().index(e) for e in enrolids[0]] # dummy
#         idx_train = enrolids[0] # dummy
        idx_train = [cvsamp['enrolid'].tolist().index(enrolids[0])]  # dummy
        idx_test = [cvsamp['enrolid'].tolist().index(e) for e in enrolids]
        
        # get patients for test and training         
        enrolid_train = cvsamp['enrolid'][idx_train]
#         enrolid_train = enrolids[0] # dummy
        enrolid_test = cvsamp['enrolid'][idx_test]
        
        # save patients
        enrolids = {"train":enrolid_train, "test": enrolid_test}
        self.train_test_enrolids = enrolids
        
        # get labels for test and training
        train_status = cvsamp['status'][idx_train]
        train_time = cvsamp['time'][idx_train]
        test_status = cvsamp['status'][idx_test]
        test_time = cvsamp['time'][idx_test]
        
        train_Y = np.array([train_status,train_time])
        train_Y = train_Y.transpose()
        test_Y = np.array([test_status,test_time])
        test_Y = test_Y.transpose()
        
        labels = {"train": train_Y, "test":test_Y}
        self.labels = labels
        
    def __train_test_data(self, domain):
        """
        Parameter:
        hdf5file = pointer to hdf5file containing feature maps
        subdomains = list of subdomains to create np arrays for
        enrolids = dict with enrolids as defined in method 'train_test_data'
        case = "longitidunal" or "static"

        Output:
        Dictionary with keys "train" and "test" corresponding to train and
        test independent variables data. Within the particular feature maps for each subdomain.
        e. g. like
        output["train"][subdomain]
        output["train"]["DRUG.GO"]
        """
        
        hdf5file = self.hdf5file
        enrolids = self.train_test_enrolids
        
        subdomains = self.SUBDOMAINS[domain]
        if self.verbose:
            print("\nCreate training/test data for domain <%s> ..." % domain)
        if domain in ["treatment","diagnosis"]:
            case = "longitidunal"
        elif domain == "static":
            case = "static"
        
        def static_reformat_feature(key):
            formated_feature = np.array(list(hdf5file[key][sd][0]))
            formated_feature = formated_feature[np.newaxis,]
            return formated_feature

        tt_data = {"train":{}, "test":{}}
        if case == "longitidunal":
            for sd in subdomains:
                tt_data["train"].update({sd: [np.array(hdf5file[key][sd]) for key in enrolids["train"]]})
                tt_data["train"][sd] = np.array(tt_data["train"][sd])
                tt_data["test"].update({sd: [np.array(hdf5file[key][sd]) for key in enrolids["test"]]})
                tt_data["test"][sd] = np.array(tt_data["test"][sd])                
                # print shapes
                shape_train = tt_data["train"][sd].shape
                shape_test = tt_data["test"][sd].shape
                if self.verbose:
                    print(" %s shape train/test: (%s, %s) " % (sd, shape_train, shape_test))
            return tt_data
        elif case == "static":
            for sd in subdomains:
                train_array = np.array([static_reformat_feature(key) for key in enrolids["train"]])
                test_array = np.array([static_reformat_feature(key) for key in enrolids["test"]])
                tt_data["train"].update({sd: train_array})
                tt_data["test"].update({sd: test_array})
                # print shapes
                shape_train = tt_data["train"][sd].shape
                shape_test = tt_data["test"][sd].shape
                if self.verbose:
                    print(" %s shape train/test: (%s, %s) " % (sd, shape_train, shape_test))
            return tt_data
        else:
            ValueError("case unknown, must be in: longitidunal, static")
            
    def __getNewTreatmentData(self):
            """
            Created for the task of treatment-simulations.
            
            Similar to __train_test_data(), but 
            - domain fixed to treatment and
            - all Patients (not only w.r.t. cross validation
            - no split to test and training, instead all are in "test" category
            
            Parameter:
            hdf5file = pointer to hdf5file containing feature maps
            subdomains = list of subdomains to create np arrays for
            enrolids = dict with enrolids as defined in method 'train_test_data'
            case = "longitidunal" or "static"

            Output:
            Dictionary with key "test".
            With the particular feature maps for each subdomain for each patient.
            e. g. like
            output["test"][subdomain]
            output["test"]["DRUG.GO"]
            """
            
            enrolids = self.enrolids
            hdf5file = self.hdf5file
                         
            domain = "treatment"
            subdomains = self.SUBDOMAINS[domain]
            tt_data = {"test":{}}
            tt_data['enrolid'] = enrolids
            for sd in subdomains:
                tt_data["test"].update({sd: [np.array(hdf5file[e][sd]) for e in enrolids]})
                tt_data["test"][sd] = np.array(tt_data["test"][sd])
                shape_test = tt_data["test"][sd].shape
                if self.verbose:
                    print(" shape test: (%s, %s) " % (sd, shape_test))
            return tt_data
        
    def __infer_subdomain_shapes(self):
        """
        output:
        dictionary of domains,
        each containing a dictionary of subdomain names and subdomain shapes
        """
                
        patients = self.hdf5file.keys()
        print("\n%s patients total\n" % len(patients))
        patients_list = [patient for patient in patients]
        arbitrary_patient = 1
        pfeatures = self.hdf5file[patients_list[arbitrary_patient]]
        subdomains = pfeatures.keys()
        
        def ishape(domain, subdomain):
            if domain == "static":
                return np.array(list(pfeatures[subdomain][0]))[np.newaxis,].shape
            else:
                return pfeatures[subdomain].shape
        
        self.subdomain_shapes = {domain: {} for domain in self.SUBDOMAINS.keys()}

        for domain in self.SUBDOMAINS.keys():
            self.subdomain_shapes[domain] = {
                subdomain: ishape(domain, subdomain) for subdomain in self.SUBDOMAINS[domain] if subdomain in subdomains
        }
        
        # print content
        for domain in self.subdomain_shapes.keys():
            if self.verbose:
                print("domain <%s>:" % domain)
            for subdomain in self.subdomain_shapes[domain]:
                sd_shape = self.subdomain_shapes[domain][subdomain]
                if self.verbose:
                    print(" %s: %s" % (subdomain,str(sd_shape)) )
            
    def __initialize(self):
        self.__open_hdf5file(self.hdf5path)
        self.__get_cv_table(self.cv_file_path)
        
#         if(self.enrolids == None):
        if(self.enrolids is None):
            self.__infer_subdomain_shapes()
            self.__train_test_patients_and_labels_bycrossv()
        elif((self.enrolids != None) & self.treat_simulation == False):
            self.__infer_subdomain_shapes()
            self.__train_test_patients_and_labels_byenrolids()
        
    def getTreatmentData(self):
        if self.data_treatment == None:
            self.data_treatment = self.__train_test_data("treatment")
        else:
            print("treatment data already exists")
        
    def getDiagnosisData(self):
        if self.data_diagnosis == None:
            self.data_diagnosis = self.__train_test_data("diagnosis")
        else:
            print("diagnosis data already exists")
                
    def getStaticData(self):
        if self.data_static == None:
            self.data_static = self.__train_test_data("static")
        else:
            print("static data already exists")
            
    def getNewTreatmentData(self):
        self.new_treatment = self.__getNewTreatmentData()
            
    def getAllData(self):
        self.getTreatmentData()
        self.getDiagnosisData()
        self.getStaticData()
        
    # SHAP interface functions
    #=========================
    def __get_dom_subdom(self):
        domains = np.repeat("treatment",9).tolist() + np.repeat("diagnosis",7).tolist() + np.repeat("static",3).tolist()
        subdomains = ["DRUG.SUBSTANCENAME","DRUG.GO","DRUG.PATHWAY","DRUG.PHEWAS","DRUG.PHEWAS_HL",\
                      "DRUG.THRCLDS","DRUG.THRGRDS","DRUG.TISSUE","DRUG2SIDEEFFECTxFREQ",\
                      "DISEASE.BIOMARKER","DISEASE.GO","DISEASE.MESHDISEASE","DISEASE.PATH",\
                      "DISEASE.PHEWAS","DISEASE.PHEWAS_HL","DISEASE.SYMPTOMS","age_sex_insur_hosp",\
                      "medicalrisk","region"]
        return(np.c_[domains, subdomains])

    def __new_shapes(self):
        # infer shapes from dict, create array of shapes
        old_shapes = self.subdomain_shapes 
        d_sd = self.__get_dom_subdom()
        shapes = []
        for i in range(d_sd.shape[0]):
            _domain = d_sd[i,0]
            _subd = d_sd[i,1]        
            _shape = old_shapes[_domain][_subd]
            shapes.append(_shape)
        self.dom_subdom_names = d_sd
        self.subdomain_shapes = shapes        
        if self.verbose:
            print("transformed subdomain shape info. new shapes:")
            print((d_sd, shapes))

    def reSHAPe_data(self):
        
        if self.verbose:
            print(colored("replace origin data structure with SHAP compatible data structure","green"))
        self.__new_shapes()
        # get domain indices
        idx_treat = self.dom_subdom_names[:,0] == "treatment"
        idx_diag = self.dom_subdom_names[:,0] == "diagnosis"    
        idx_static = self.dom_subdom_names[:,0] == "static"
        # map domains to subdomains
        sd_treat = [self.dom_subdom_names[i][1] for i in np.where(idx_treat)[0] ]
        sd_diag = [self.dom_subdom_names[i][1] for i in np.where(idx_diag)[0] ]
        sd_static = [self.dom_subdom_names[i][1] for i in np.where(idx_static)[0] ]
        
        # query + combine subdomains data        
        ## train set
        train = []
        [train.append(self.data_treatment["train"][subdomain]) for subdomain in sd_treat]
        [train.append(self.data_diagnosis["train"][subdomain]) for subdomain in sd_diag]
        [train.append(self.data_static["train"][subdomain]) for subdomain in sd_static]
        ## test set
        test = []
        [test.append(self.data_treatment["test"][subdomain]) for subdomain in sd_treat]
        [test.append(self.data_diagnosis["test"][subdomain]) for subdomain in sd_diag]
        [test.append(self.data_static["test"][subdomain]) for subdomain in sd_static]
        
        # replace old data structure
        print(colored("replacing old data structure .data_diagnosis .data_static .data_treatment...","green"))
        self.data_diagnosis = None
        self.data_static = None
        self.data_treatment = None
        print(colored("... with data[\"train\"] and data[\"test\"] ","green"))
        self.data = {
            "train" : train,
            "test" : test
        }
        
        
        

