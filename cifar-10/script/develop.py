"""
Development
"""

from sys import stdout 
from sys import stderr

import time
import numpy

import theano
import theano.tensor as T

import module_io as io
import module_util as util
import module_model as model


def develop():
    '''
    Development
    '''

    # load config
    develop_config = io.load_config(io.develop_config_path)
    train_data_path = str(develop_config.get("Data Preparation", "data_path"))
    n_in = int(develop_config.get("Training Model", "n_in"))
    n_hidden = int(develop_config.get("Training Model", "n_hidden"))
    n_out = int(develop_config.get("Training Model", "n_out"))
    learning_rate = float(develop_config.get("Training Model", "learning_rate"))
    L1_weight = float(develop_config.get("Training Model", "L1_weight"))
    L2_squared_weight = float(develop_config.get("Training Model", "L2_squared_weight"))
    activation = str(develop_config.get("Training Model", "activation"))
    if activation == "tanh":
        activation = T.tanh
    elif activation == "sigmoid":
        activation = T.nnet.sigmoid
    else:
        activation = None
    batch_size = int(develop_config.get("Training Model", "batch_size"))
    n_epoch = int(develop_config.get("Training Model", "n_epoch"))

    # load data
    develop_data_set = io.load_develop_data(train_data_path)
    train_set_shared, validate_set_shared = util.share_develop_set(develop_data_set)
    train_pixel_shared, train_label_shared = train_set_shared
    validate_pixel_shared, validate_label_shared = validate_set_shared

    # compute parameters
    n_train_batch = train_pixel_shared.get_value(borrow = True).shape[0] / batch_size
    n_validate_batch = validate_pixel_shared.get_value(borrow = True).shape[0] / batch_size

    # construct model
    index = T.lscalar()
    X = T.matrix('X')
    y = T.ivector('y')
    random_state = numpy.random.RandomState(1234)
    classifier = model.Multilayer_Perceptron(X, n_in, n_hidden, n_out, random_state)
    cost = classifier.cost(y, L1_weight, L2_squared_weight)
    error = classifier.error(y)

    # construct the gradients for parameters of the model
    parameter_gradient_list = []
    for parameter in classifier.parameter_list:
        parameter_gradient = T.grad(cost, parameter)
        parameter_gradient_list.append(parameter_gradient)
        
    # construct the update list for training
    update_list = []
    for parameter, parameter_gradient in zip(classifier.parameter_list,
                                             parameter_gradient_list):
        update_list.append((parameter, parameter - learning_rate * parameter_gradient))
        
    # function train: return cost on training set
    train_model = theano.function(inputs = [index],
                                  outputs = cost,
                                  updates = update_list,
                                  givens = {
                                      X: train_pixel_shared[index * batch_size:(index + 1) * batch_size],
                                      y: train_label_shared[index * batch_size:(index + 1) * batch_size]})

    # cost on training set
    cost_on_train_set = theano.function(inputs = [index],
                                        outputs = cost,
                                        givens = {
                                            X: train_pixel_shared[index * batch_size:(index + 1) * batch_size],
                                            y: train_label_shared[index * batch_size:(index + 1) * batch_size]})

    # cost on validation set
    cost_on_validate_set = theano.function(inputs = [index],
                                           outputs = cost,
                                           givens = {
                                               X: validate_pixel_shared[index * batch_size:(index + 1) * batch_size],
                                               y: validate_label_shared[index * batch_size:(index + 1) * batch_size]})

    # error on training set
    error_on_train_set = theano.function(inputs = [index],
                                         outputs = error,
                                         givens = {
                                             X: train_pixel_shared[index * batch_size:(index + 1) * batch_size],
                                             y: train_label_shared[index * batch_size:(index + 1) * batch_size]})

    # error on validateion set
    error_on_validate_set = theano.function(inputs = [index],
                                            outputs = error,
                                            givens = {
                                                X: validate_pixel_shared[index * batch_size:(index + 1) * batch_size],
                                                y: validate_label_shared[index * batch_size:(index + 1) * batch_size]})

    # training
    header = "cost_minibatch_average" + ',' + \
             "cost_current_average" + ',' + \
             "error_current_average" + ',' + \
             "cost_train_average" + ',' + \
             "error_train_average" + ',' + \
             "cost_validate_average" + ',' + \
             "error_validate_average" + ',' + \
             "clock" + ',' + \
             "time"
    print >> stdout, header

    # start timing
    start_time = time.time()
    for epoch in xrange(n_epoch):
        for minibatch_index in xrange(n_train_batch):
            # train model
            cost_minibatch_average = train_model(minibatch_index)
            # compute cost on current minibatch
            cost_current_average = cost_on_train_set(minibatch_index)
            # compute error on current minibatch
            error_current_average = error_on_train_set(minibatch_index)
            # compute cost on training set
            cost_train_list = [cost_on_train_set(i)
                               for i in xrange(n_train_batch)]
            cost_train_average = numpy.mean(cost_train_list)
            # compute error on training set
            error_train_list = [error_on_train_set(i)
                                for i in xrange(n_train_batch)]
            error_train_average = numpy.mean(error_train_list)
            # compute cost on validation set
            cost_validate_list = [cost_on_validate_set(i)
                                  for i in xrange(n_validate_batch)]
            cost_validate_average = numpy.mean(cost_validate_list)
            # compute error on validation set
            error_validate_list = [error_on_validate_set(i)
                                   for i in xrange(n_validate_batch)]
            error_validate_average = numpy.mean(error_validate_list)
            # time
            current_time = time.time()
            used_time = current_time - start_time
            hours, minutes, seconds = util.format_time(used_time)
            # log
            print >> stdout, "%f,%f,%f,%f,%f,%f,%f,%s,%s" % (cost_minibatch_average,
                                                             cost_current_average,
                                                             error_current_average,
                                                             cost_train_average,
                                                             error_train_average,
                                                             cost_validate_average,
                                                             error_validate_average,
                                                             "\"%d:%d:%d\"" % (hours, minutes, seconds),
                                                             "\"%s\"" % time.ctime())

            
    
if __name__=='__main__':
    develop()
