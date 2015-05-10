#!/home/admin/jingwen.hjw/software/anaconda/bin/python2.7 
# -*- coding: utf-8 -*-

import os
import time
import json
import thread
import numpy as np
import pandas as pd
from collections import OrderedDict
from itertools import product
from threading import Thread
from multiprocessing import Pool
from sklearn.grid_search import ParameterGrid
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split


class SVMModel(object):

    def __init__(self, param, X_train, X_validate, y_train, y_validate, scaler):
        
        self.param = param
        self.X_train = X_train
        self.X_validate = X_validate
        self.y_train = y_train
        self.y_validate = y_validate
        self.scaler = scaler
        self.encoder = LabelEncoder()

    def train_validate_test(self):

        self._load_train_data()
        
        self.classifier = SVC(C=self.param["C"],
                              kernel=self.param["kernel"],
                              degree=self.param["degree"],
                              gamma=self.param["gamma"],
                              coef0=self.param["coef0"],
                              probability=self.param["probability"],
                              shrinking=self.param["shrinking"],
                              tol=self.param["tol"],
                              cache_size=self.param["cache_size"],
                              class_weight=self.param["class_weight"],
                              verbose=self.param["verbose"],
                              max_iter=self.param["max_iter"],
                              random_state=self.param["random_state"])
        
        print "training ..."
        self.classifier.fit(self.X_train, self.y_train)
        print "trained."

        print "testing ..."
        self.y_proba_train = self.classifier.predict_proba(self.X_train)
        self.y_proba_validate = self.classifier.predict_proba(self.X_validate)
        self.y_pred_train = np.argmax(self.y_proba_train, axis=1)
        self.y_pred_validate = np.argmax(self.y_proba_validate, axis=1)
        print "tested."

        self.meta = OrderedDict()
        self.meta["param"] = self.param
        self.meta["acc_train"] = self.classifier.score(self.X_train, self.y_train)
        self.meta["acc_validate"] = self.classifier.score(self.X_validate, self.y_validate)
        self.meta["logloss_train"] = log_loss(self.encoder.fit_transform(self.y_train), self.y_proba_train)
        self.meta["logloss_validate"] = log_loss(self.encoder.fit_transform(self.y_validate), self.y_proba_validate)

        self._make_submission()

    def _make_submission(self):

        print "loading testing data ..."
        self._load_test_data()
        print "testing data loaded."

        print "testing ..."
        self.y_proba_test = self.classifier.predict_proba(self.X_test)
        self.y_pred_test = np.argmax(self.y_proba_test, axis=1)
        print "tested."
        
        timestamp = time.strftime("[%Y-%m-%d]_[%H-%M-%S]", time.localtime(time.time()))
        timestamp += "_[%d]_[%d]" % (os.getpid(), thread.get_ident())

        print "dumping meta ..."
        path = "../data/submission_svm_%s.json" % timestamp
        with open(path, 'w') as fp:
            json.dump(self.meta, fp, indent=4)
        print "info_dict dumped."
 
        print "dumping submission file ..."
        path = "../data/submission_svm_%s.csv" % timestamp
        with open(path, 'w') as fp:
            fp.write("id,")
            fp.write(",".join(self.encoder.classes_))
            fp.write("\n")
            for id, proba in zip(self.id_test, self.y_proba_test):
                proba = ','.join([id] + map(str, proba.tolist()))
                fp.write(proba)
                fp.write("\n")
        print "submission file dumped."

    def _load_train_data(self):

        self.X_train = self.scaler.transform(self.X_train.astype(float))
        self.X_validate = self.scaler.transform(self.X_validate.astype(float))

    def _load_test_data(self):

        print "loading training data ..."
        data_frame = pd.read_csv("../data/test.csv")
        X = data_frame.values.copy()
        X_test, id_test = X[:, 1:], X[:, 0]
        X_test = self.scaler.transform(X_test.astype(float))
        self.X_test = X_test.astype(float)
        self.id_test = id_test.astype(str)
        print "training data loaded."

class RFModel(object):

    def __init__(self, param, X_train, X_validate, y_train, y_validate):
        
        self.param = param
        self.X_train = X_train
        self.X_validate = X_validate
        self.y_train = y_train
        self.y_validate = y_validate
        self.encoder = LabelEncoder()

    def train_validate_test(self):

        self._load_train_data()
        
        self.classifier = RandomForestClassifier(n_estimators=self.param["n_estimators"],
                                                 criterion=self.param["criterion"],
                                                 max_features=self.param["max_features"],
                                                 max_depth=self.param["max_depth"],
                                                 min_samples_split=self.param["min_samples_split"],
                                                 min_samples_leaf=self.param["min_samples_leaf"],
                                                 min_weight_fraction_leaf=self.param["min_weight_fraction_leaf"],
                                                 max_leaf_nodes=self.param["max_leaf_nodes"],
                                                 bootstrap=self.param["bootstrap"],
                                                 oob_score=self.param["oob_score"],
                                                 n_jobs=self.param["n_jobs"],
                                                 random_state=self.param["random_state"],
                                                 verbose=self.param["verbose"])
        print "training ..."
        self.classifier.fit(self.X_train, self.y_train)
        print "trained."

        print "testing ..."
        self.y_proba_train = self.classifier.predict_proba(self.X_train)
        self.y_proba_validate = self.classifier.predict_proba(self.X_validate)
        self.y_pred_train = np.argmax(self.y_proba_train, axis=1)
        self.y_pred_validate = np.argmax(self.y_proba_validate, axis=1)
        print "tested."

        self.meta = OrderedDict()
        self.meta["param"] = self.param
        self.meta["acc_train"] = self.classifier.score(self.X_train, self.y_train)
        self.meta["acc_validate"] = self.classifier.score(self.X_validate, self.y_validate)
        self.meta["logloss_train"] = log_loss(self.encoder.fit_transform(self.y_train), self.y_proba_train)
        self.meta["logloss_validate"] = log_loss(self.encoder.fit_transform(self.y_validate), self.y_proba_validate)
        
        self._make_submission()

    def _make_submission(self):

        print "loading testing data ..."
        self._load_test_data()
        print "testing data loaded."

        print "testing ..."
        self.y_proba_test = self.classifier.predict_proba(self.X_test)
        self.y_pred_test = np.argmax(self.y_proba_test, axis=1)
        print "tested."

        timestamp = time.strftime("[%Y-%m-%d]_[%H-%M-%S]", time.localtime(time.time()))
        timestamp += "_[%d]_[%d]" % (os.getpid(), thread.get_ident())

        print "dumping meta ..."
        path = "../data/submission_rf_%s.json" % timestamp
        with open(path, 'w') as fp:
            json.dump(self.meta, fp, indent=4)
        print "info_dict dumped."
         
        print "dumping submission file ..."
        path = "../data/submission_rf_%s.csv" % timestamp
        with open(path, 'w') as fp:
            fp.write("id,")
            fp.write(",".join(self.encoder.classes_))
            fp.write("\n")
            for id, proba in zip(self.id_test, self.y_proba_test):
                proba = ','.join([id] + map(str, proba.tolist()))
                fp.write(proba)
                fp.write("\n")
        print "submission file dumped."

    def _load_train_data(self):

        self.X_train = self.X_train.astype(float)
        self.X_validate = self.X_validate.astype(float)

    def _load_test_data(self):

        print "loading training data ..."
        data_frame = pd.read_csv("../data/test.csv")
        X = data_frame.values.copy()
        X_test, id_test = X[:, 1:], X[:, 0]
        X_test = X_test.astype(float)
        self.X_test = X_test.astype(float)
        self.id_test = id_test.astype(str)
        print "training data loaded."

class SVMPool(Thread):

    def __init__(self, param_list, X_train, X_validate, y_train, y_validate, scaler):

        self.param_list = param_list
        self.X_train = X_train
        self.X_validate = X_validate
        self.y_train = y_train
        self.y_validate = y_validate
        self.scaler = scaler
        self.pool = Pool()
        self.model_list = []
        self.process_list = []
        super(SVMPool, self).__init__()

    def run(self):

        self.train_validate_test()

        # check
        assert len(self.param_list) == len(self.model_list)
        for param in self.param_list:
            assert self.fetch_model(param) is not None
        
    def train_validate_test(self):

        for param in self.param_list:
            model = SVMModel(param,
                             self.X_train, self.X_validate,
                             self.y_train, self.y_validate,
                             self.scaler)
            self.model_list.append(model)
            
        for model in self.model_list:
            process = self.pool.apply_async(trigger_model, args=(model, ))
            self.process_list.append(process)

        self.pool.close()
        self.pool.join()

        self.model_list = []
        for process in self.process_list:
            self.model_list.append(process.get())

    def fetch_model(self, param):

        for model in self.model_list:
            if model.param == param:
                return model
        return None
    
class RFPool(Thread):

    def __init__(self, param_list, X_train, X_validate, y_train, y_validate):

        self.param_list = param_list
        self.X_train = X_train
        self.X_validate = X_validate
        self.y_train = y_train
        self.y_validate = y_validate
        self.pool = Pool()
        self.model_list = []
        self.process_list = []
        super(RFPool, self).__init__()

    def run(self):

        self.train_validate_test()
        
        # check
        assert len(self.param_list) == len(self.model_list)
        for param in self.param_list:
            assert self.fetch_model(param) is not None
        
        
    def train_validate_test(self):

        for param in self.param_list:
            model = RFModel(param,
                            self.X_train, self.X_validate,
                            self.y_train, self.y_validate)
            self.model_list.append(model)
            
        for model in self.model_list:
            process = self.pool.apply_async(trigger_model, args=(model, ))
            self.process_list.append(process)

        self.pool.close()
        self.pool.join()

        self.model_list = []
        for process in self.process_list:
            self.model_list.append(process.get())

    def fetch_model(self, param):

        for model in self.model_list:
            if model.param == param:
                return model
        return None
    
class Merge(Thread):

    def __init__(self, svm_model, rf_model, y_train, y_validate):

        self.svm_model = svm_model
        self.rf_model = rf_model
        self.svm_param = self.svm_model.param
        self.rf_param = self.rf_model.param
        self.y_train = y_train
        self.y_validate = y_validate
        self.encoder = LabelEncoder()
        self.meta = OrderedDict()
        super(Merge, self).__init__()

    def run(self):

        self.merge()

    def merge(self):

        self.y_proba_train = np.mean([self.svm_model.y_proba_train, self.rf_model.y_proba_train], axis=0)
        self.y_proba_validate = np.mean([self.svm_model.y_proba_validate, self.rf_model.y_proba_validate], axis=0)
        self.y_pred_train = np.argmax(self.y_proba_train, axis=1)
        self.y_pred_validate = np.argmax(self.y_proba_validate, axis=1)

        self.meta["svm"] = self.svm_model.meta
        self.meta["rf"] = self.rf_model.meta
        self.meta["acc_train"] = accuracy_score(self.encoder.fit_transform(self.y_train), self.y_pred_train)
        self.meta["acc_validate"] = accuracy_score(self.encoder.fit_transform(self.y_validate), self.y_pred_validate)
        self.meta["logloss_train"] = log_loss(self.encoder.fit_transform(self.y_train), self.y_proba_train)
        self.meta["logloss_validate"] = log_loss(self.encoder.fit_transform(self.y_validate), self.y_proba_validate)

        self._make_submission()

    def _make_submission(self):

        print "loading testing data ..."
        self._load_test_data()
        print "testing data loaded."

        print "testing ..."
        self.y_proba_test = np.mean([self.svm_model.y_proba_test, self.rf_model.y_proba_test], axis=0)
        self.y_pred_test = np.argmax(self.y_proba_test, axis=1)
        print "tested."
        
        timestamp = time.strftime("[%Y-%m-%d]_[%H-%M-%S]", time.localtime(time.time()))
        timestamp += "_[%d]_[%d]" % (os.getpid(), thread.get_ident())

        print "dumping meta ..."
        path = "../data/submission_merge_%s.json" % timestamp
        with open(path, 'w') as fp:
            json.dump(self.meta, fp, indent=4)
        print "info_dict dumped."
         
        print "dumping submission file ..."
        path = "../data/submission_merge_%s.csv" % timestamp
        with open(path, 'w') as fp:
            fp.write("id,")
            fp.write(",".join(self.encoder.classes_))
            fp.write("\n")
            for id, proba in zip(self.id_test, self.y_proba_test):
                proba = ','.join([id] + map(str, proba.tolist()))
                fp.write(proba)
                fp.write("\n")
        print "submission file dumped."

    def _load_test_data(self):

        print "loading training data ..."
        data_frame = pd.read_csv("../data/test.csv")
        X = data_frame.values.copy()
        X_test, id_test = X[:, 1:], X[:, 0]
        X_test = X_test.astype(float)
        self.X_test = X_test.astype(float)
        self.id_test = id_test.astype(str)
        print "training data loaded."

def trigger_model(model):

    model.train_validate_test()
    return model

def load_train_data(train_size=0.8):

    print "loading training data ..."
    data_frame = pd.read_csv("../data/train.csv")
    X = data_frame.values.copy()
    np.random.shuffle(X)
    X_train, X_validate, y_train, y_validate = train_test_split(
        X[:, 1:-1], X[:, -1], train_size=train_size
    )

    scaler = StandardScaler()
    scaler.fit(X_train.astype(float))
    
    print "training data loaded."
    return (X_train.astype(float), X_validate.astype(float),
            y_train.astype(str), y_validate.astype(str),
            scaler)

def develop():

    svm_grid = [
        {
            "C"            : [1.0, 2.0],
            "kernel"       : ["rbf"],
            "degree"       : [3],
            "gamma"        : [0.01, 0.008],
            "coef0"        : [0.0],
            "probability"  : [True],
            "shrinking"    : [True],
            "tol"          : [1e-10],
            "cache_size"   : [1024],
            "class_weight" : ["auto"],
            "verbose"      : [1],
            "max_iter"     : [-1],
            "random_state" : [None]
        },
        {
            "C"            : [0.6, 0.8],
            "kernel"       : ["rbf"],
            "degree"       : [3],
            "gamma"        : [0.008, 0.001],
            "coef0"        : [0.0],
            "probability"  : [True],
            "shrinking"    : [True],
            "tol"          : [1e-10],
            "cache_size"   : [1024],
            "class_weight" : ["auto"],
            "verbose"      : [1],
            "max_iter"     : [-1],
            "random_state" : [None]
        }
    ]

    rf_grid = [
        {
            "n_estimators"      : [200],
            "criterion"         : ["entropy"],
            "max_features"      : ["log2"],
            "max_depth"         : [35],
            "min_samples_split" : [10],
            "min_samples_leaf"  : [20],
            "min_weight_fraction_leaf" : [0.0],
            "max_leaf_nodes"    : [None],
            "bootstrap"         : [True], 
            "oob_score"         : [True],
            "n_jobs"            : [-1],
            "random_state"      : [None], 
            "verbose"           : [0]
        },
        {
            "n_estimators"      : [200],
            "criterion"         : ["entropy"],
            "max_features"      : ["log2"],
            "max_depth"         : [None],
            "min_samples_split" : [10],
            "min_samples_leaf"  : [20],
            "min_weight_fraction_leaf" : [0.0],
            "max_leaf_nodes"    : [None],
            "bootstrap"         : [True], 
            "oob_score"         : [True],
            "n_jobs"            : [-1],
            "random_state"      : [None], 
            "verbose"           : [0]
        }
    ]

    svm_list = list(ParameterGrid(svm_grid))
    rf_list = list(ParameterGrid(rf_grid))

    print "loading training data ..."
    X_train, X_validate, y_train, y_validate, scaler = load_train_data()
    print "training data loaded."

    svm_pool = SVMPool(svm_list, X_train, X_validate, y_train, y_validate, scaler)
    rf_pool = RFPool(rf_list, X_train, X_validate, y_train, y_validate)

    svm_pool.start()
    rf_pool.start()

    svm_pool.join()
    rf_pool.join()

    merge_list = []
    for svm_param, rf_param in product(svm_list, rf_list):
        svm_model = svm_pool.fetch_model(svm_param)
        rf_model = rf_pool.fetch_model(rf_param)
        merge_list.append(Merge(svm_model, rf_model,
                                y_train, y_validate))

    for merge in merge_list:
        merge.start()
    
    for merge in merge_list:
        merge.join()    

def main():

    develop()
    
        
if __name__ == "__main__":
    
    main()
    
