import os
import sys
import json

def extract_svm():

    script_dir = os.getcwd() + "/" + \
               "/".join(sys.argv[0].split("/")[:-1])

    with os.popen("ls ../data/*svm*.json") as fp:
        file_list = fp.read().split()

    json_list = []
    for json_file in file_list:
        with open(json_file) as fp:
            json_list.append((json_file, json.load(fp)))
    json_list.sort(key=lambda (json_file, json_dict):
    #               json_dict["acc_train"] - json_dict["acc_validate"])
                   (json_dict["C"], json_dict["gamma"]))

    for json_file, json_dict in json_list:
        print "%f\t%f\t%f\t%f\t%s" % \
            (json_dict["acc_train"] - json_dict["acc_validate"],
             json_dict["acc_validate"],
             json_dict["C"],
             json_dict["gamma"],
             json_file)

def extract_rf():

    script_dir = os.getcwd() + "/" + \
               "/".join(sys.argv[0].split("/")[:-1])

    with os.popen("ls ../data/*rf*.json") as fp:
        file_list = fp.read().split()

    json_list = []
    for json_file in file_list:
        with open(json_file) as fp:
            json_list.append((json_file, json.load(fp)))
    json_list.sort(key=lambda (json_file, json_dict):
    #               json_dict["acc_train"] - json_dict["acc_validate"])
                   (json_dict["criterion"],
                    #json_dict["max_depth"],
                    json_dict["min_samples_split"],
                    json_dict["min_samples_leaf"]))

    for json_file, json_dict in json_list:
        print "%f\t%f\t%s\t%r\t%d\t%d\t%s" % \
            (json_dict["acc_train"] - json_dict["acc_validate"],
             json_dict["acc_validate"],
             json_dict["criterion"],
             json_dict["max_depth"],
             json_dict["min_samples_split"],
             json_dict["min_samples_leaf"],
             json_file)
    

def main():

    #extract_rf()
    extract_svm()

if __name__ == "__main__":

    main()
