"""
Module Util
"""
import numpy
import theano
import theano.tensor as T

def share_develop_set(data_set):
    '''
    Load the data_set in shared variables (for development).

    :type data_set: tuple
    :param data_set: the development set, a tuple in form
                     tuple(train_set, validate_set) in which
                     * train_set: tuple(train_picture_list,
                                        train_label_list)
                     * validate_set: tuple(validate_picture_list,
                                           validate_label_list)
    '''

    # fetch train_set and validate_set
    train_set, validate_set = data_set

    # fetch picture_list and label_list
    train_picture_list, train_label_list = train_set
    validate_picture_list, validate_label_list = validate_set

    # convert list to numpy array for sharing later
    train_pixel_matrix = numpy.array(train_picture_list,
                                     dtype = theano.config.floatX)
    train_label_vector = numpy.array(train_label_list,
                                     dtype = theano.config.floatX)
    validate_pixel_matrix = numpy.array(validate_picture_list,
                                        dtype = theano.config.floatX)
    validate_label_vector = numpy.array(validate_label_list,
                                        dtype = theano.config.floatX)

    # store datasets in shared variables
    train_pixel_shared = theano.shared(train_pixel_matrix,
                                       borrow = True)
    train_label_shared = theano.shared(train_label_vector,
                                       borrow = True)
    validate_pixel_shared = theano.shared(validate_pixel_matrix,
                                          borrow = True)
    validate_label_shared = theano.shared(validate_label_vector,
                                          borrow = True)

    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``*_shared`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``*_label_shared`` we will have to cast it to int. This little
    # hack lets ous get around this issue
    train_label_shared = T.cast(train_label_shared, 'int32')
    validate_label_shared = T.cast(validate_label_shared, 'int32')

    # assemble the shared data sets
    train_set_shared = (train_pixel_shared, train_label_shared)
    validate_set_shared = (validate_pixel_shared, validate_label_shared)

    return train_set_shared, validate_set_shared
    

def share_train_set(data_set):
    '''
    Load the data_set in shared variabels (for train).

    :type data_set: tuple
    :param data_set: the training set, a tuple in form
                     tuple(train_picture_list, train_label_list)
    '''

def share_test_set(data_set):
    '''
    Load the data_set in shared variabels (for test).

    :type data_set: tuple
    :param data_set: the testing set, a tuple in form
                     tuple(test_picture_list, test_label_list)
    '''

def format_time(seconds):
    '''
    Convert time from seconds to format "hours:minutes:seconds".

    :type seconds: float or int or string
    :param seconds: the seconds to convert
    '''
    return (int(seconds) / (60 * 60), # hours
            int(seconds) / 60 % 60,   # minutes
            int(seconds) % 60)        # seconds
