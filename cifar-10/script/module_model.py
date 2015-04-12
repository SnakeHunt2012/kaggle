"""
Module Model
"""

import numpy
import theano
import theano.tensor as T


#############################
# Logistic Regression Model #
#############################

class Logistic_Regression(object):
    """
    Multi-class Logistic Regression Class
    """

    def __init__(self, X, n_in, n_out):
        '''
        Initialize the parameters of the logistic regression.
        
        :type X: theano.tensor.TensorType
        :param X: symbolic variable that describes the input
                  of the architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of
                     the space in which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of
                      the space in which the labels lie
        '''

        # initialize with 0 the weights W as a matrix of shape
        # (n_in, n_out)
        self.W = theano.shared(value = numpy.zeros((n_in, n_out),
                                                   dtype = theano.config.floatX),
                                                   name = 'W',
                                                   borrow = True)

        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(value = numpy.zeros((n_out, ),
                                                   dtype = theano.config.floatX),
                                                   name = 'b',
                                                   borrow = True)

        # compute vector of class-membership probablilities in
        # symbolic form
        self.p_y_given_X = T.nnet.softmax(T.dot(X, self.W) + self.b)

        # compute prediction as class whose probability is maximal
        # in symbolic form
        self.y_pred = T.argmax(self.p_y_given_X, axis = 1)

        # pack parameters of the model
        self.parameter_list = [self.W, self.b]

    def cost(self, y):
        '''
        Return the mean of the negative log-likelihood of the
        prediction of this model under a given target distribution.

        .. math::
        
            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example
                  the correct label
        '''
        return -T.mean(T.log(self.p_y_given_X)[T.arange(y.shape[0]), y])

    def error(self, y):
        '''
        Return a float representing the number of errors in the minibatch,
        zero one loss over the size of the minibatch.

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example
                  the correct label
        '''

        # check if y has the same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError("y should have the same shape as self.y_pred",
                            ('y', y.type, "y_pred", self.y_pred.type))

        # check if y is of the correct datatype
        if y.dtype.startswith("int"):
            # the T.neq operator returns a vector of 0s and 1s, where
            # 1 represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


##########################
# Multi-layer perceptron #
##########################

class Hidden_Layer(object):
    """
    Typical Hidden Layer of a MLP
    """
    
    def __init__(self, input, n_in, n_out, random_state,
                 W = None, b = None, activation = None):
        '''
        Initialize the parameters of the hidden layer.

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape
                      (n_examples, n_in)
        
        :type n_in: int
        :param n_in: dimensionality of input
        
        :type n_out: int
        :param n_out: number of hidden units,
                      dimensionality of output
        '''

        self.input = input

        # initialize parameter matrix W for model
        if W is None:
            W_value = numpy.asarray(random_state.uniform(
                low = -numpy.sqrt(6. / (n_in + n_out)),
                high = numpy.sqrt(6. / (n_in + n_out)),
                size = (n_in, n_out)), dtype = theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_value *= 4
                
            W = theano.shared(value = W_value, name = 'W', borrow = True)

        # initialize parameter vector b for model
        if b is None:
            b_value = numpy.zeros((n_out,), dtype = theano.config.floatX)
            b = theano.shared(value = b_value, name = 'b', borrow = True)

        # install parameters W and b into model
        self.W = W
        self.b = b

        # generate linear output
        linear_output = T.dot(input, self.W) + self.b

        # generate output 
        if activation is None:
            self.output = linear_output
        else:
            self.output = activation(linear_output)

        # pack parameters of the hidden layer
        self.parameter_list = [self.W, self.b]

class Multilayer_Perceptron(object):
    """
    Multi-Layer Perceptron Class
    """
    
    def __init__(self, X, n_in, n_hidden, n_out, random_state,
                 activation = None):
        '''
        Initialize the parameters of the multi-layer perceptron.

        :type X: theano.tensor.TensorType
        :param X: symbolic variable that describes the input of
                  the architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the
                     space in which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden unit
                         

        :type n_out: int
        :param n_out: number of output units, the dimention of
                      the space in which the label lie

        :type random_state: numpy.random.RandomState
        :param random_state: a random number generator used to initialize
                             weights
        '''

        # equip with hidden layer
        self.hidden_layer = Hidden_Layer(
            input = X,
            n_in = n_in,
            n_out = n_hidden,
            random_state = random_state,
            activation = activation)
        
        # equip with logistict regression layer
        self.logistic_regression_layer = Logistic_Regression(
            X = self.hidden_layer.output,
            n_in = n_hidden,
            n_out = n_out)

        # pack parameters of the model
        self.parameter_list = self.hidden_layer.parameter_list \
                            + self.logistic_regression_layer.parameter_list

    def cost(self, y, L1_weight = 0.00, L2_squared_weight = 0.00):
        '''
        Return the regularized cost of the model. The regularized cost
        of the model is the negative log likelihood of the output of
        the model plus the regularization terms (L1 and L2_squared),
        and the negative log likelihood of the output of the model is
        computed in the logistic regression layer.

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example
                  the correct label

        :type L1_weight: float
        :param L1_weight: L1-norm's weight when added to the cost

        :type L2_squared_weight: float
        :param L2_squared_weight: L2-norm's weight when added to the cost
        '''

        # L1 normaliztion term
        L1 = abs(self.hidden_layer.W).sum() \
           + abs(self.logistic_regression_layer.W).sum()

        # L2 normalization term
        L2_squared = (self.hidden_layer.W ** 2).sum() \
                   + (self.logistic_regression_layer.W ** 2).sum()

        # negative log likelihood term
        negative_log_likelihood = self.logistic_regression_layer.cost(y)

        # construct the cost
        cost = negative_log_likelihood \
             + L1_weight * L1 \
             + L2_squared_weight * L2_squared

        return cost

    def error(self, y):
        '''
        Return the error of the model, which is computed in the logistic
        regression layer.

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example
                  the correct label
        '''
        return self.logistic_regression_layer.error(y)
