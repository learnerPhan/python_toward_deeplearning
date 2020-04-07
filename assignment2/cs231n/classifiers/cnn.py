from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params. Store weights and biases for the convolutional   #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #                           
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # parameters W1, b1 for convolutional layer
        C, H, W = input_dim
        self.params['W1'] = np.random.normal(loc=0.0, scale=weight_scale, size=(num_filters, C*filter_size*filter_size)).reshape(num_filters, C, filter_size, filter_size)
        self.params['b1'] = np.zeros((num_filters,))

        # calculate the output size of convolution layer
        stride = 1
        pad = (filter_size-1) // 2

        Hpad = H + 2*pad
        Wpad = W + 2*pad

        oconv_H = 1 + (Hpad - filter_size)//stride
        oconv_W = 1 + (Wpad - filter_size)//stride

        # calculate the output size of max-pool layer
        pool_H, pool_W = 2, 2
        pool_stride = 2

        op_H = 1 + (oconv_H - pool_H)//pool_stride
        op_W = 1 + (oconv_W - pool_W)//pool_stride

        # calculate the input size of affine layer
        i_affine_size = num_filters * op_H * op_W

        # parameters W2, b2 for hidden affine layer
        self.params['W2'] = np.random.normal(loc=0.0, scale=weight_scale, size=(i_affine_size, hidden_dim))
        self.params['b2'] = np.zeros(hidden_dim)

        # parameters W3, b3 for last affine layter
        self.params['W3'] = np.random.normal(loc=0.0, scale=weight_scale, size=(hidden_dim, num_classes))
        self.params['b3'] = np.zeros(num_classes)

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        """
        Reminder : Architecture 
                    conv - relu - 2x2 max pool - affine - relu - affine - softmax
                    |---frist combined layer---||---second----||--third--|
        """
        out, c_r_p_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        # cache = (conv_cache, relu_cache, pool_cache)
        out, a_r_cache = affine_relu_forward(out, W2, b2)
        # cache = (fc_cache, relu_cache)
        scores, a_r_cache2 = affine_relu_forward(out, W3, b3)
        # cache = (fc_cache, relu_cache)



        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        """
        Reminder:
        1. loss = data_loss + reg_loss
        2. reg_loss = self.reg * (np.sum(W1*W1) + np.sum(W2*W2))
        """
        data_loss, dout = softmax_loss(scores, y)
        reg_loss = 0.5*self.reg * (np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3))
        loss = reg_loss + data_loss






        # grads['W1' : dW1, 'b1' : db1, 'W2' : dW2, 'b2' : db2, 'W3' : dW3, 'b3' : db3]

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
