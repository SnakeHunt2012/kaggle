# -*- coding: utf-8 -*-

import os
import sys
import json
from argparse import ArgumentParser

def extract_svm(dir, format):

    script_dir = os.getcwd() + "/" + \
               "/".join(sys.argv[0].split("/")[:-1])

    with os.popen("ls " + dir.rstrip('/') + "/*svm*.json") as fp:
        file_list = fp.read().split()

    json_list = []
    for json_file in file_list:
        with open(json_file) as fp:
            json_list.append((json_file, json.load(fp)))
    json_list.sort(key=lambda (json_file, json_dict):
                   (json_dict["logloss_validate"]))

    for json_file, json_dict in json_list:
        switch_case = {
            "human" : "%f\t%f\t%f\t%f\t|  %f\t%f\t%s",
            "csv" : "%f,%f,%f,%f,%f,%f,%s"
        }
        print switch_case[format] % \
            (json_dict["acc_train"] - json_dict["acc_validate"],
             json_dict["acc_validate"],
             json_dict["logloss_train"] - json_dict["logloss_validate"],
             json_dict["logloss_validate"],
             json_dict["C"],
             json_dict["gamma"],
             json_file)

def extract_rf(dir, format):

    script_dir = os.getcwd() + "/" + \
               "/".join(sys.argv[0].split("/")[:-1])

    with os.popen("ls " + dir.rstrip('/') + "/*rf*.json") as fp:
        file_list = fp.read().split()

    json_list = []
    for json_file in file_list:
        with open(json_file) as fp:
            json_list.append((json_file, json.load(fp)))
    json_list.sort(key=lambda (json_file, json_dict):
                   (json_dict["logloss_validate"]))

    for json_file, json_dict in json_list:
        switch_case = {
            "human" : "%f\t%f\t%f\t%f\t|  %d\t%d\t%s",
            "csv" : "%f,%f,%f,%f,%d,%d,%s"
        }
        print "json_file:", json_file
        print switch_case[format] % \
            (json_dict["acc_train"] - json_dict["acc_validate"],
             json_dict["acc_validate"],
             json_dict["logloss_train"] - json_dict["logloss_validate"],
             json_dict["logloss_validate"],
             json_dict["min_samples_split"],
             json_dict["min_samples_leaf"],
             json_file)

def extract_merge(dir, format):

    script_dir = os.getcwd() + "/" + \
               "/".join(sys.argv[0].split("/")[:-1])

    with os.popen("ls " + dir.rstrip('/') + "/*merge*.json") as fp:
        file_list = fp.read().split()

    json_list = []
    for json_file in file_list:
        with open(json_file) as fp:
            json_str = fp.read()
            if json_str != "":
                json_list.append((json_file, json.loads(json_str)))
    json_list.sort(key=lambda (json_file, json_dict):
                   (json_dict["logloss_validate"]))

    for json_file, json_dict in json_list:
        switch_case = {
            "human" : "%f  %f  %f\t|  %f  %f  %f\t|  %f  %f\t|  %f  %f\t|  %s",
            "csv" : "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%s"
        }
        print switch_case[format] % \
            (json_dict["acc_validate"] - json_dict["svm"]["acc_validate"],
             json_dict["acc_validate"] - json_dict["rf"]["acc_validate"],
             json_dict["acc_validate"],
             json_dict["logloss_validate"] - json_dict["svm"]["logloss_validate"],
             json_dict["logloss_validate"] - json_dict["rf"]["logloss_validate"],
             json_dict["logloss_validate"],
             json_dict["svm"]["acc_validate"],
             json_dict["rf"]["acc_validate"],
             json_dict["svm"]["logloss_validate"],
             json_dict["rf"]["logloss_validate"],
             json_file)

def main():

    parser = ArgumentParser()
    parser.add_argument("model", type=str, help=u"模型(svm/rf/merge)")
    parser.add_argument("dir", type=str, help=u"输出路径")
    parser.add_argument("format", type=str, help=u"打印格式(human/csv)")
    args = parser.parse_args()

    switch_case = {
        "svm" : extract_svm,
        "rf" : extract_rf,
        "merge" : extract_merge
    }
    switch_case[args.model](args.dir, args.format)

if __name__ == "__main__":

    main()
