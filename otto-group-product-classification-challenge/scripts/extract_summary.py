import os
import sys
import json

def main():

    script_dir = os.getcwd() + "/" + \
               "/".join(sys.argv[0].split("/")[:-1])

    with os.popen("ls ../data/*.json") as fp:
        file_list = fp.read().split()

    json_list = []
    for json_file in file_list:
        with open(json_file) as fp:
            json_list.append((json_file, json.load(fp)))
    json_list.sort(key=lambda (json_file, json_dict):
    #               json_dict["acc_train"] - json_dict["acc_validate"])
                   (json_dict["degree"], json_dict["C"], json_file.split("_")[2]))

    for json_file, json_dict in json_list:
        print "%f\t%f\t%s\t%d\t%f\t%s" % \
            (json_dict["acc_train"] - json_dict["acc_validate"],
             json_dict["acc_validate"],
             json_dict["kernel"], json_dict["degree"], json_dict["C"],
             json_file)

if __name__ == "__main__":

    main()
