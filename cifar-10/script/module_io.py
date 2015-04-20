"""
Module I/O
"""
import os
import ConfigParser
import csv

###################
# Global Constant #
###################

root_dir = os.getcwd() + "/.." 
config_dir = root_dir + "/" + "config"
global_config_path = config_dir + "/" + "global.cfg"
develop_config_path = config_dir + "/" + "develop.cfg"
train_config_path = config_dir + "/" + "train.cfg"
test_config_path = config_dir + "/" + "test.cfg"

##################
# Config Related #
##################

def load_config(path):
    '''
    Load config from xxx.cfg file.

    :type path: string
    :param path: path to the config file in
                 python cfg format, the file
                 is usually named by xxx.cfg
    '''
    config = ConfigParser.ConfigParser()
    with open(path) as config_file:
        config.readfp(config_file)
    return config

def load_label_number_dict():
    '''
    Load mapping from label to number as a
    dict from the config file containing the
    mapping.

    :type path: string
    :param path: path to the config file in
                 python cfg format, the file
                 contains the mapping from
                 label to number
    '''
    config = load_config(global_config_path)
    map_list = config.items("Label Mapping")
    map_list = [(str(key), int(value))
                for key, value in map_list]
    map_dict = dict(map_list)
    return map_dict
    

def load_number_label_dict():
    '''
    Load mapping from number to label as a
    dict from the config file containing the
    mapping.

    :type path: string
    :param path: path to the config file in
                 python cfg format, the file
                 contains the mapping from
                 label to number
    '''
    label_number_dict = load_label_number_dict()
    number_label_list = [(label_number_dict[key], key)
                         for key in label_number_dict]
    number_label_dict = dict(number_label_list)
    return number_label_dict

def load_train_proportion():
    '''
    Load train proportion in train data for
    development.

    :type path: string
    :param path: path to the config file in
                 python cfg format, the file
                 contains the development
                 configuration.
    '''
    config = load_config(develop_config_path)
    train_proportion_int = config.get("Data Preparation",
                                      "train_proportion")
    train_proportion_float = float(train_proportion_int)
    return train_proportion_float

def load_validate_proportion():
    '''
    Load validate proportion in train data for
    development.

    :type path: string
    :param path: path to the config file in
                 python cfg format, the file
                 contains the development
                 configuration.
    '''
    config = load_config(develop_config_path)
    validate_proportion_int = config.get("Data Preparation",
                                         "validate_proportion")
    validate_proportion_float = float(validate_proportion_int)
    return validate_proportion_float
    
    
################
# Data Related #
################

def load_train_data(path):
    '''
    Load train data from csv file to array,
    the file is in csv format, indicated by
    parameter path.

    :type path: string
    :param path: path to the train.csv file
                 in which contain pixels and
                 labels
    '''
    pass

def load_test_data(path):
    '''
    Load test data from csv file to array,
    the file is in csv format, indicated by
    parameter path.

    :type path: string
    :param path: path to the test.csv file
                 in which contain pixels
    '''
    pass

def load_develop_data(path):
    '''
    Load develop data from csv file to array,
    the file is in csv format, indicated by
    parameter path. This function split it to
    two parts, train set and validate set.

    :type path: string
    :param path: path to the train.csv file
                 in which contain pixels and
                 labels
    '''

    # read in train and validate proportion
    train_proportion = load_train_proportion()
    validate_proportion = load_validate_proportion()

    # read in samples from csv file to sample_list
    sample_list = []
    with open(path, 'r') as file:
        csv_reader = csv.reader(file)
        for sample in csv_reader:
            sample_list.append(sample)

    # remove header
    sample_list = sample_list[1:]

    # get label_dict
    label_dict = load_label_number_dict()

    # get label_list
    # and pop picture id from sample_list
    # and pop labels from sample_list
    label_list = []
    for sample in sample_list:
        # pop picutre id from sample_list
        picture_id = sample.pop(0)
        # pop label from sample_list
        label_string = sample.pop(-1)
        # convert from label to number
        label_id = label_dict[label_string]
        # add to lable_list
        label_list.append(label_id)

    # get picture_list
    picture_list = sample_list
    sample_amount = len(picture_list)

    # assemble train_set:
    # * train_picture_list
    # * train_label_list
    train_benchmark = int(sample_amount *
                          train_proportion)
    train_picture_list = picture_list[:train_benchmark]
    train_label_list = label_list[:train_benchmark]
    train_set = (train_picture_list, train_label_list)

    # assemble validate_set:
    # * validate_picture_list
    # * validate_label_list
    validate_benchmark = int(sample_amount *
                             (1 - validate_proportion))
    validate_picture_list = picture_list[validate_benchmark:]
    validate_label_list = label_list[validate_benchmark:]
    validate_set = (validate_picture_list, validate_label_list)

    return train_set, validate_set

def save_submit_data(data, path):
    '''
    '''
    pass

#################
# Model Related #
#################

def load_model(path):
    '''
    '''
    pass

def save_model(model, path):
    '''
    '''
    pass

