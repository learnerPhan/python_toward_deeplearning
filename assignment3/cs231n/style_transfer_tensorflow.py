import tensorflow as tf
import numpy as np

def tv_loss(img, tv_weight):
    """
    Compute total variation loss.

    Inputs:
    - img: Tensor of shape (1, H, W, 3) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.

    Returns:
    - loss: Tensor holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    # Your implementation should be vectorized and not require any loops!
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    H, W = tf.shape(img)[1:3]
    # Be careful with slice in python
    # H ---> range of indices [0, H-1]
    # Here we want the first to be   from 1 to H-1,
    #              the second to be  from 0 to H-2
    # And because in python writting [a:b] means from a to b-1
    # So we end up with the first  is [1:H]
    # and               the second is [0:H-1]
    
    loss = tf.reduce_sum((img[:,1:H,:,:] - img[:,0:H-1,:,:])**2)
    loss += tf.reduce_sum((img[:,:,1:W,:] - img[:,:,0:W-1,:])**2)
    loss *= tv_weight
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss

def style_loss(feats, style_layers, style_targets, style_weights):
    """
    Computes the style loss at a set of layers.

    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a Tensor giving the Gram matrix of the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].

    Returns:
    - style_loss: A Tensor containing the scalar style loss.
    """
    # Hint: you can do this with one for loop over the style layers, and should
    # not be short code (~5 lines). You will need to use your gram_matrix function.
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    style_loss = 0
    i = 0;
    for idx in style_layers:
        style_cur = gram_matrix(feats[idx], normalize=True) 
        dif = style_cur - style_targets[i]
        dif_sqr = dif**2
        loss_layer = tf.reduce_sum(dif_sqr) * style_weights[i]
        style_loss += loss_layer
        i = i + 1

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return style_loss

def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.

    Inputs:
    - features: Tensor of shape (1, H, W, C) giving features for
      a single image.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)

    Returns:
    - gram: Tensor of shape (C, C) giving the (optionally normalized)
      Gram matrices for the input image.
    """
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #get the dims
    # flatten
    H, W, C = tf.shape(features)[1:]
    flat_feature = tf.reshape(features, (H*W, C))

    gram = tf.matmul(tf.transpose(flat_feature), flat_feature)
    if (normalize==True):
        norm = tf.cast(H*W*C, tf.float32)
        gram = gram/norm

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return gram

def content_loss(content_weight, content_current, content_original):
    """
    Compute the content loss for style transfer.

    Inputs:
    - content_weight: scalar constant we multiply the content_loss by.
    - content_current: features of the current image, Tensor with shape [1, height, width, channels]
    - content_original: features of the content image, Tensor with shape [1, height, width, channels]

    Returns:
    - scalar content loss
    """
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    cc = tf.shape(content_current)[-1]
    ct = tf.shape(content_original)[-1]

    cur_new = tf.reshape(content_current, (-1,cc))
    targ_new = tf.reshape(content_original, (-1,ct))

    # print(tf.shape(cur_new))
    # print(tf.shape(targ_new))

    # content_l = np.sum((cur_new - targ_new)**2)
    # tf way to do such task (np.sum)
    content_l = tf.reduce_sum((cur_new - targ_new)**2, [0,1])
    content_l *= content_weight
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return content_l

# We provide this helper code which takes an image, a model (cnn), and returns a list of
# feature maps, one per layer.
def extract_features(x, cnn):
    """
    Use the CNN to extract features from the input image x.

    Inputs:
    - x: A Tensor of shape (N, H, W, C) holding a minibatch of images that
      will be fed to the CNN.
    - cnn: A Tensorflow model that we will use to extract features.

    Returns:
    - features: A list of feature for the input images x extracted using the cnn model.
      features[i] is a Tensor of shape (N, H_i, W_i, C_i); recall that features
      from different layers of the network may have different numbers of channels (C_i) and
      spatial dimensions (H_i, W_i).
    """
    features = []
    prev_feat = x
    for i, layer in enumerate(cnn.net.layers[:-2]):
        next_feat = layer(prev_feat)
        features.append(next_feat)
        prev_feat = next_feat
    return features

def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
