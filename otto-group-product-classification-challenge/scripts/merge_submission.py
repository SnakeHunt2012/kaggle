#!/home/admin/jingwen.hjw/software/anaconda/bin/python2.7 
# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd

def load_submission(submission_path):

    print "loading submission %s ..." % submission_path
    data_frame = pd.read_csv(submission_path)
    X = data_frame.values.copy()
    print "submission %s loaded."
    return X


def merge_submission(submission_list):

    print "merging submission %s ..." % str(submission_list)
    X = np.mean(submission_list, axis=0)
    print "submission %s merged." % str(submission_list)
    return X


def dump_submission(X):

    label_list = ["Class_%d" % i for i in range(1, 10)]
    timestamp = time.strftime("[%Y-%m-%d]_[%H-%M-%S]", time.localtime(time.time()))
    path = "../sandbox/submission_merge_%s.csv" % timestamp
    print "dumping submission %s ..." % path
    with open(path, 'w') as fp:
        fp.write("id,")
        fp.write(",".join(label_list))
        fp.write("\n")
        for item in X:
            fp.write("%d," % int(item[0]))
            fp.write(",".join(map(str, item[1:].tolist())))
            fp.write("\n")
    print "submission %s dumped." % path


def main():

    X_1 = load_submission("../sandbox/submission_rf_[2015-04-26]_[15-13-13]_[6200].csv")
    X_2 = load_submission("../sandbox/submission_svm_[2015-04-30]_[00-21-32]_[6323].csv")
    X_merge = merge_submission([X_1, X_2])
    dump_submission(X_merge)


if __name__ == "__main__":

    main()
