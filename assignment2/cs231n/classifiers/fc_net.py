from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.params['W1'] = np.random.normal(loc=0.0, scale=weight_scale, size=(input_dim, hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.normal(loc=0.0, scale=weight_scale, size=(hidden_dim, num_classes))
        self.params['b2'] = np.zeros(num_classes)


        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        """
        Reminder : 

        1. The architecture of the network :
            input - FC - ReLU - FC - softmax

        2. loss function = softmax

        3. input - FC - ReLU - FC - softmax
                                  <--here the scores

        4. FC = (x -> x*w + b)
           ReLU = (x -> max(0,x))
        """

        # unpack
        W1, b1  = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        # output layer-to-layer
        out_fc, cache_fc_1 = affine_forward(X, W1, b1)
        out_relu, cache_relu = relu_forward(out_fc.reshape(np.prod(out_fc.shape)))
        scores, cache_fc_2 = affine_forward(out_relu.reshape(out_fc.shape), W2, b2)

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        """
        Reminder :
        A/ score structure
        1. scores.shape = (number_of_images, number_of_labels)
        2. scores[i] is the score vector of image i
        3. scores[i][j] is the scores for label j of image i

        B/ score computation
        1. stability trick : scores -= np.max(scores, axis=1) before exponential
        2. loss = data_loss + reg_loss
        3. reg_loss = self.reg * (np.sum(W1*W1) + np.sum(W2*W2))

        """
        N = X.shape[0]

        scores -= np.max(scores, axis=1).reshape(scores.shape[0],1)
        exp_scores = np.exp(scores)
        sum_exp_vec = np.sum(exp_scores, axis=1).reshape(exp_scores.shape[0],1)
        softmax_mat = exp_scores/sum_exp_vec

        #print('fc_net.py')
        #print(y.dtype)
        #y = y.astype(int)
        L_images = np.log(softmax_mat[np.arange(N), y.astype(int)])

        data_loss = np.sum(L_images)/N
        reg_loss = 0.5*self.reg * (np.sum(W1*W1) + np.sum(W2*W2))
        loss = reg_loss - data_loss

        pass


        softmax_mat[np.arange(N), y] -= 1
        dReLU, dW2, db2 = affine_backward(softmax_mat, cache_fc_2)
        dfc1 = relu_backward(dReLU, cache_relu)
        _, dW1, db1 = affine_backward(dfc1.reshape(N, len(b1)), cache_fc_1)

        dW1 /= N
        dW2 /= N
        db1 /= N
        db2 /= N

        dW1 += self.reg*W1
        dW2 += self.reg*W2 

        grads = {'W1':dW1, 'b1':db1, 'W2':dW2, 'b2':db2}

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        string_w = 'W'
        w_keys = [string_w + str(i) for i in range(1, self.num_layers + 1)]
        string_b = 'b'
        b_keys = [string_b + str(i) for i in range(1, self.num_layers + 1)]
        print(w_keys)
        print(b_keys)

        # print(hidden_dims)
        # add input_dim at the begin of list hidden_dim
        temp_1 = hidden_dims[::-1] + [input_dim]
        temp_1 = temp_1[::-1]
        # add num_classes at the end of list hidden_dim
        temp_2 = hidden_dims + [num_classes]

        temp = list(zip(temp_1, temp_2))
        print(temp)

        list_w = list(zip(w_keys, temp))
        list_b = list(zip(b_keys, temp_2))

        # print(list_w)
        # print(list_b)

        for key, size in list_w:
            self.params[key] = np.random.normal(loc=0.0, scale=weight_scale, size=size)

        for key, val in list_b:
            self.params[key] = np.zeros(val)

        """    
        for k, v in self.params.items():
            print(k,v)
        """

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        w_keys = ['W' + str(i) for i in range(1, self.num_layers + 1)]
        w_vals = [self.params[key] for key in w_keys]
        b_keys = ['b' + str(i) for i in range(1, self.num_layers + 1)]
        b_vals = [self.params[key] for key in b_keys]

        # print(w_vals)
        # print(b_vals)
        out = X
        outs = []
        caches = []
        for i in range(self.num_layers):
            out, cache = affine_relu_forward(out, w_vals[i], b_vals[i])
            outs.append(out)
            caches.append(cache)

        scores = outs[-1]

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        scores -= np.max(scores, axis=1).reshape(scores.shape[0],1)
        exp_scores = np.exp(scores)
        sum_exp_vec = np.sum(exp_scores, axis=1).reshape(exp_scores.shape[0],1)
        softmax_mat = exp_scores/sum_exp_vec

        #print('fc_net.py')
        #print(y.dtype)
        #y = y.astype(int)
        N = X.shape[0]
        L_images = np.log(softmax_mat[np.arange(N), y.astype(int)])

        data_loss = np.sum(L_images)/N

        w_2 = np.array([w*w for w in w_vals])
        temp = [np.sum(w) for w in w_2]
        reg_loss = 0.5*self.reg * np.sum(temp)
        loss = reg_loss - data_loss

        # d_loss/d_scores
        softmax_mat[np.arange(N), y.astype(int)] -= 1

        rev_outs = outs[::-1]
        rev_caches = caches[::-1]

        dws = []
        dbs = []

        dout = softmax_mat
        for i in range(self.num_layers):
            dout, dw, db = affine_relu_backward(dout, rev_caches[i])
            dws.append(dw)
            dbs.append(db)

        dws.reverse()
        dbs.reverse()

        dws = np.array(dws)/N
        dbs = np.array(dbs)/N

        dws = dws + self.reg*np.array(w_vals)

        grads = dict(zip(w_keys, dws))
        grads.update(dict(zip(b_keys, dbs)))

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
