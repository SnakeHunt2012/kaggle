#!/home/admin/jingwen.hjw/software/anaconda/bin/python2.7 
# -*- coding: utf-8 -*-

import time
import json
import numpy as np
import pandas as pd
from collections import OrderedDict
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split


def logloss_mc(y_true, y_prob, epsilon=1e-15):

    # normalize
    y_prob = y_prob / y_prob.sum(axis=1).reshape(-1, 1)
    y_prob = np.maximum(epsilon, y_prob)
    y_prob = np.minimum(1 - epsilon, y_prob)
    
    # get probabilities
    y = [y_prob[i, j] for (i, j) in enumerate(y_true)]
    ll = - np.mean(np.log(y))
    return ll


def load_train_data(train_size=0.8):

    print "loading training data ..."
    data_frame = pd.read_csv("../data/train.csv")
    X = data_frame.values.copy()
    np.random.shuffle(X)
    X_train, X_validate, y_train, y_validate = train_test_split(
        X[:, 1:-1], X[:, -1], train_size=train_size
    )
    
    mu = X_train.mean(axis=0)
    sigma = X_train.max(axis=0) - X_train.min(axis=0)
    
    #X_train = (X_train - mu) / sigma
    #X_validate = (X_validate - mu) / sigma
    print "training data loaded."
    return (X_train.astype(float), X_validate.astype(float),
            y_train.astype(str), y_validate.astype(str),
            mu, sigma)


def load_test_data(mu, sigma):

    print "loading training data ..."
    data_frame = pd.read_csv("../data/test.csv")
    X = data_frame.values.copy()
    X_test, id_test = X[:, 1:], X[:, 0]
    #X_test = (X_test - mu) / sigma
    print "training data loaded."
    return X_test.astype(float), id_test.astype(str)


def make_submission(classifier, encoder, mu, sigma, info_dict, path=""):

    print "loading testing data ..."
    X_test, id_test = load_test_data(mu, sigma)
    print "testing data loaded."

    print "testing ..."
    y_proba = classifier.predict_proba(X_test)
    print "tested."

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
    
    print "dumping info_dict ..."
    path = "../data/submission_svm_%s.json" % timestamp
    with open(path, 'w') as fp:
        json.dump(info_dict, fp, indent=4)
    print "info_dict dumped."

    print "dumping submission file ..."
    path = "../data/submission_svm_%s.csv" % timestamp
    with open(path, 'w') as fp:
        fp.write("id,")
        fp.write(",".join(encoder.classes_))
        fp.write("\n")
        for id, proba in zip(id_test, y_proba):
            proba = ','.join([id] + map(str, proba.tolist()))
            fp.write(proba)
            fp.write("\n")
    print "submission file dumped."


def develop():

    parameter_dict = OrderedDict()
    parameter_dict["C"] = 0.4
    parameter_dict["kernel"] = "rbf"
    parameter_dict["degree"] = 3
    parameter_dict["gamma"] = 0.0
    parameter_dict["coef0"] = 0.0
    parameter_dict["probability"] = True
    parameter_dict["shrinking"] = True
    parameter_dict["tol"] = 1e-10
    parameter_dict["cache_size"] = 1024 * 10
    parameter_dict["class_weight"] = "auto"
    parameter_dict["verbose"] = 1
    parameter_dict["max_iter"] = -1
    parameter_dict["random_state"] = None

    X_train, X_validate, y_train, y_validate, mu, sigma = load_train_data()
    classifier = SVC(C=parameter_dict["C"],
                     kernel=parameter_dict["kernel"],
                     degree=parameter_dict["degree"],
                     gamma=parameter_dict["gamma"],
                     coef0=parameter_dict["coef0"],
                     probability=parameter_dict["probability"],
                     shrinking=parameter_dict["shrinking"],
                     tol=parameter_dict["tol"],
                     cache_size=parameter_dict["cache_size"],
                     class_weight=parameter_dict["class_weight"],
                     verbose=parameter_dict["verbose"],
                     max_iter=parameter_dict["max_iter"],
                     random_state=parameter_dict["random_state"])
    
    print "training ..."
    classifier.fit(X_train, y_train)
    print "trained."
    
    print "testing ..."
    y_train_proba = classifier.predict_proba(X_train)
    y_validate_proba = classifier.predict_proba(X_validate)
    print "tested."

    acc_train = classifier.score(X_train, y_train)
    acc_validate = classifier.score(X_validate, y_validate)
    print "mean accuracy on training set:   %s" % str(acc_train)
    print "mean accuracy on validation set: %s" % str(acc_validate)

    encoder = LabelEncoder()
    
    logloss_train = logloss_mc(encoder.fit_transform(y_train), y_train_proba)
    print "logarithmic loss on training set:    %s" % str(logloss_train)

    logloss_validate = logloss_mc(encoder.fit_transform(y_validate), y_validate_proba)
    print "logarithmic loss on validateion set: %s" % str(logloss_validate)

    info_dict = parameter_dict.copy()
    info_dict["acc_train"] = acc_train
    info_dict["acc_validate"] = acc_validate
    info_dict["logloss_train"] = logloss_train
    info_dict["logloss_validate"] = logloss_validate
    
    make_submission(classifier, encoder, mu, sigma, info_dict)
    

def main():

    develop()
    
        
if __name__ == "__main__":
    
    main()
    
