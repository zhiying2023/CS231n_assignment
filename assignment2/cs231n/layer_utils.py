from .layers import *
from .fast_layers import *


def affine_relu_forward(x, w, b):
    """Convenience layer that performs an affine transform followed by a ReLU.

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """Backward pass for the affine-relu convenience layer.
    Inputs:
    - dout,cache

    Returns a tuple of:
    - dx,dw,db
    """

    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
    """
    Input:
    - x: Data of shape (N, D)
    - w, b: Weights for the affine layer
    - gamma: Scale parameter
    - beta: Shift paremeter
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: A tuple of values needed in the backward pass(cache1,cache2,cache3)
        - cache1=(x1,w,b)
        - cache2=(x2,mean,var,gamma,beta)
        - cache3=(x3)
    """
    fc_out, fc_cache = affine_forward(x, w, b)
    bn_out, bn_cache = batchnorm_forward(fc_out, gamma, beta, bn_param)
    relu_out, relu_cache = relu_forward(bn_out)
    cache = (fc_cache, bn_cache, relu_cache)
    return relu_out, cache


def affine_bn_relu_backward(dout, cache):
    """
    Input:
    - dout,cache

    Return:
    - dx,dw,db,dgamma,dbeta
    """
    fc_cache, bn_cache, relu_cache = cache
    dx_relu = relu_backward(dout, relu_cache)
    dx_bn, dgamma, dbeta = batchnorm_backward(dx_relu, bn_cache)
    dx_fc, dw, db = affine_backward(dx_bn, fc_cache)
    return dx_fc, dw, db, dgamma, dbeta


def affine_ln_relu_forward(x, w, b, gamma, beta, bn_param):
    """
    Input:
    - x: Data of shape (N, D)
    - w, b: Weights for the affine layer
    - gamma: Scale parameter
    - beta: Shift paremeter
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: A tuple of values needed in the backward pass(cache1,cache2,cache3)
        - cache1=(x1,w,b)
        - cache2=(x2,mean,var,gamma,beta)
        - cache3=(x3)
    """
    fc_out, fc_cache = affine_forward(x, w, b)
    ln_out, ln_cache = layernorm_forward(fc_out, gamma, beta, bn_param)
    relu_out, relu_cache = relu_forward(ln_out)
    cache = (fc_cache, ln_cache, relu_cache)
    return relu_out, cache


def affine_ln_relu_backward(dout, cache):
    """
    Input:
    - dout,cache

    Return:
    - dx,dw,db,dgamma,dbeta
    """
    fc_cache, ln_cache, relu_cache = cache
    dx_relu = relu_backward(dout, relu_cache)
    dx_ln, dgamma, dbeta = layernorm_backward(dx_relu, ln_cache)
    dx_fc, dw, db = affine_backward(dx_ln, fc_cache)
    return dx_fc, dw, db, dgamma, dbeta


def affine_relu_drop_forward(x, w, b, dropout_param):
    """
    Input:
    - x: Data of shape (N, D)
    - w, b: Weights for the affine layer
    - bn_param: Dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: A tuple of values needed in the backward pass(cache1,cache2,cache3)
        - cache1=(x1,w,b)
        - cache2=(x2)
        - cache3=(p,mode,seed,mask)
    """
    out_fc, cache_fc = affine_forward(x, w, b)
    out_relu, cache_relu = relu_forward(out_fc)
    out_drop, cache_drop = dropout_forward(out_relu, dropout_param)
    cache = (cache_fc, cache_relu, cache_drop)
    return out_drop, cache


def affine_relu_drop_backward(dout, cache):
    """
    Input:
    - dout,cache

    Return:
    - dx,dw,db
    """
    cache_fc, cache_relu, cache_drop = cache
    dx_drop = dropout_backward(dout, cache_drop)
    dx_relu = relu_backward(dx_drop, cache_relu)
    dx_fc, dw, db = affine_backward(dx_relu, cache_fc)
    return dx_fc, dw, db


# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


def conv_relu_forward(x, w, b, conv_param):
    """A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """Backward pass for the conv-relu convenience layer.
    - Output: dx,dw,db
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    """Convenience layer that performs a convolution, a batch normalization, and a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer
    - gamma, beta: Arrays of shape (D2,) and (D2,) giving scale and shift
      parameters for batch normalization.
    - bn_param: Dictionary of parameters for batch normalization.

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache


def conv_bn_relu_backward(dout, cache):
    """Backward pass for the conv-bn-relu convenience layer."""
    conv_cache, bn_cache, relu_cache = cache
    dan = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db, dgamma, dbeta


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """Backward pass for the conv-relu-pool convenience layer."""
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db
