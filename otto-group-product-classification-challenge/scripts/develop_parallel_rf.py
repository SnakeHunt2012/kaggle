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
from sklearn.ensemble import RandomForestClassifier
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
    print "training data loaded."
    return (X_train.astype(float), X_validate.astype(float),
            y_train.astype(str), y_validate.astype(str))


def load_test_data():

    print "loading training data ..."
    data_frame = pd.read_csv("../data/test.csv")
    X = data_frame.values.copy()
    X_test, id_test = X[:, 1:], X[:, 0]
    print "training data loaded."
    return X_test.astype(float), id_test.astype(str)


def make_submission(classifier, encoder, info_dict, path=""):

    print "loading testing data ..."
    X_test, id_test = load_test_data()
    print "testing data loaded."

    print "testing ..."
    y_proba = classifier.predict_proba(X_test)
    print "tested."

    timestamp = time.strftime("[%Y-%m-%d]_[%H-%M-%S]_", time.localtime(time.time()))
    timestamp += "[%d]" % os.getpid()
    
    print "dumping info_dict ..."
    path = "../data/submission_rf_%s.json" % timestamp
    with open(path, 'w') as fp:
        json.dump(info_dict, fp, indent=4)
    print "info_dict dumped."

    print "dumping submission file ..."
    path = "../data/submission_rf_%s.csv" % timestamp
    with open(path, 'w') as fp:
        fp.write("id,")
        fp.write(",".join(encoder.classes_))
        fp.write("\n")
        for id, proba in zip(id_test, y_proba):
            proba = ','.join([id] + map(str, proba.tolist()))
            fp.write(proba)
            fp.write("\n")
    print "submission file dumped."

def train_validate_test(parameter_dict, X_train, X_validate, y_train, y_validate):

    random_forest_classifier = RandomForestClassifier(n_estimators=parameter_dict["n_estimators"],
                                                      criterion=parameter_dict["criterion"],
                                                      max_features=parameter_dict["max_features"],
                                                      max_depth=parameter_dict["max_depth"],
                                                      min_samples_split=parameter_dict["min_samples_split"],
                                                      min_samples_leaf=parameter_dict["min_samples_leaf"],
                                                      max_leaf_nodes=parameter_dict["max_leaf_nodes"],
                                                      min_weight_fraction_leaf=parameter_dict["min_weight_fraction_leaf"],
                                                      max_leaf_nodes=parameter_dict["max_leaf_nodes"],
                                                      bootstrap=parameter_dict["bootstrap"],
                                                      oob_score=parameter_dict["oob_score"],
                                                      n_jobs=parameter_dict["n_jobs"],
                                                      random_state=parameter_dict["random_state"],
                                                      verbose=parameter_dict["verbose"])
    
    print "training ..."
    random_forest_classifier.fit(X_train, y_train)
    print "trained."
    
    print "testing ..."
    y_train_proba = random_forest_classifier.predict_proba(X_train)
    y_validate_proba = random_forest_classifier.predict_proba(X_validate)
    print "tested."

    acc_train = random_forest_classifier.score(X_train, y_train)
    acc_validate = random_forest_classifier.score(X_validate, y_validate)
    print "mean accuracy on training set:   %s" % str(acc_train)
    print "mean accuracy on validation set: %s" % str(acc_validate)

    encoder = LabelEncoder()

    logloss_train = logloss_mc(encoder.fit_transform(y_train), y_train_proba)
    assert (encoder.classes_ == random_forest_classifier.classes_).all()
    print "logarithmic loss on training set:    %s" % str(logloss_train)

    logloss_validate = logloss_mc(encoder.fit_transform(y_validate), y_validate_proba)
    assert (encoder.classes_ == random_forest_classifier.classes_).all()
    print "logarithmic loss on validateion set: %s" % str(logloss_validate)

    info_dict = parameter_dict.copy()
    info_dict["acc_train"] = acc_train
    info_dict["acc_validate"] = acc_validate
    info_dict["logloss_train"] = logloss_train
    info_dict["logloss_validate"] = logloss_validate
    
    make_submission(random_forest_classifier, encoder, info_dict)

    return info_dict
    
def develop():

    min_samples_split_list = [10, 20, 30]
    min_samples_leaf_list = [5, 10, 15, 20]
    min_weight_fraction_leaf_list = [None]
    max_leaf_nodes_list = [None]

    parameter_dict_list = []
    for min_samples_split in min_samples_split_list:
        for min_samples_leaf in min_samples_leaf_list:
            parameter_dict = OrderedDict()
            parameter_dict["n_estimators"] = 200
            parameter_dict["criterion"] = "gini"
            parameter_dict["max_features"] = "log2"
            parameter_dict["max_depth"] = None
            parameter_dict["min_samples_split"] = min_samples_split  # tuning
            parameter_dict["min_samples_leaf"] = min_samples_leaf    # tuning
            #parameter_dict["min_weight_fraction_leaf"] = min_weight_fraction_leaf  # tuning
            #parameter_dict["max_leaf_nodes"] = max_leaf_nodes # tuning
            parameter_dict["bootstrap"] = True
            parameter_dict["oob_score"] = True
            parameter_dict["n_jobs"] = -1
            parameter_dict["random_state"] = None
            parameter_dict["verbose"] = 0
            parameter_dict_list.append(parameter_dict)

    print "loading training data ..."
    X_train, X_validate, y_train, y_validate = load_train_data()
    print "training data loaded."

    pool = Pool()

    print "deploying train_validate_test ..."
    result_list = [pool.apply_async(train_validate_test,
                                    args=(parameter_dict, X_train, X_validate, y_train, y_validate))
                   for parameter_dict in parameter_dict_list]
    print "train_validate_test deployed."

    pool.close()
    pool.join()

    print "printing results ..."
    print "===================="
    for result in result_list:
        print "Result:"
        try:
            print result.get()
        except Exception, e:
            print e
        print "===================="
    print "results printed."
    

def main():

    develop()
    
        
if __name__ == "__main__":
    
    main()
    
