#!/home/admin/jingwen.hjw/software/anaconda/bin/python2.7 
# -*- coding: utf-8 -*-

import os
import time
import json
import numpy as np
import pandas as pd
from collections import OrderedDict
from multiprocessing import Pool
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split


class SVM(object):

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
        timestamp += "_[%d]" % os.getpid()

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

class RF(object):

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
        timestamp += "_[%d]" % os.getpid()

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


class MergePlan(object):

    def __init__(self, param_svm, param_rf, X_train, X_validate, y_train, y_validate, scaler):

        self.param_rf = param_rf
        self.param_svm = param_svm
        self.X_train = X_train
        self.X_validate = X_validate
        self.y_train = y_train
        self.y_validate = y_validate
        self.scaler = scaler
        self.encoder = LabelEncoder()
        self.pool = Pool()

    def train_validate_test(self):

        self.model_svm = SVM(self.param_svm,
                             self.X_train, self.X_validate,
                             self.y_train, self.y_validate,
                             self.scaler)
        
        self.model_rf = RF(self.param_rf,
                           self.X_train, self.X_validate,
                           self.y_train, self.y_validate)

        process_svm = self.pool.apply_async(trigger_model, args=(self.model_svm, ))
        process_rf = self.pool.apply_async(trigger_model, args=(self.model_rf, ))

        self.pool.close()
        self.pool.join()

        self.model_svm = process_svm.get()
        self.model_rf = process_rf.get()

        self.y_proba_train = np.mean([self.model_svm.y_proba_train, self.model_rf.y_proba_train], axis=0)
        self.y_proba_validate = np.mean([self.model_svm.y_proba_validate, self.model_rf.y_proba_validate], axis=0)
        self.y_pred_train = np.argmax(self.y_proba_train, axis=1)
        self.y_pred_validate = np.argmax(self.y_proba_validate, axis=1)

        self.meta = OrderedDict()
        self.meta["svm"] = self.model_svm.meta
        self.meta["rf"] = self.model_rf.meta
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
        self.y_proba_test = np.mean([self.model_svm.y_proba_test, self.model_rf.y_proba_test], axis=0)
        self.y_pred_test = np.argmax(self.y_proba_test, axis=1)
        print "tested."
        
        timestamp = time.strftime("[%Y-%m-%d]_[%H-%M-%S]", time.localtime(time.time()))
        timestamp += "_[%d]" % os.getpid()

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

    param_svm = OrderedDict()
    param_svm["C"] = 1.0
    param_svm["kernel"] = "rbf"
    param_svm["degree"] = 3
    param_svm["gamma"] = 0.005
    param_svm["coef0"] = 0.0
    param_svm["probability"] = True
    param_svm["shrinking"] = True
    param_svm["tol"] = 1e-10
    param_svm["cache_size"] = 1024
    param_svm["class_weight"] = "auto"
    param_svm["verbose"] = 1
    param_svm["max_iter"] = 2
    param_svm["random_state"] = None

    param_rf = OrderedDict()
    param_rf["n_estimators"] = 200
    param_rf["criterion"] = "gini"
    param_rf["max_depth"] = None
    param_rf["max_features"] = "log2"
    param_rf["min_samples_split"] = 30
    param_rf["min_samples_leaf"] = 20
    param_rf["min_weight_fraction_leaf"] = 0.0
    param_rf["max_leaf_nodes"] = None
    param_rf["bootstrap"] = True
    param_rf["oob_score"] = True
    param_rf["n_jobs"] = -1
    param_rf["random_state"] = None
    param_rf["verbose"] = 0
                    
    print "loading training data ..."
    X_train, X_validate, y_train, y_validate, scaler = load_train_data()
    print "training data loaded."

    merge_plan = MergePlan(param_svm, param_rf, X_train, X_validate, y_train, y_validate, scaler)
    merge_plan.train_validate_test()

def main():

    develop()
    
        
if __name__ == "__main__":
    
    main()
    
