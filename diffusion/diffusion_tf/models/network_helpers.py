# Adapted from https://github.com/icon-lab/SLATER/tree/main
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import dnnlib.tflib as tflib
from dnnlib.tflib.ops.upfirdn_2d import upsample_2d, downsample_2d, upsample_conv_2d, conv_downsample_2d
from dnnlib.tflib.ops.fused_bias_act import fused_bias_act
from dnnlib import EasyDict
from dnnlib import misc
import math

#************************************************************************************************************
# return the shape of a tensor as a list
def get_shape(x):
    shape, dyn_shape = x.shape.as_list().copy(), tf.shape(x)
    for index, dim in enumerate(shape):
        if dim is None:
            shape[index] = dyn_shape[index]
    return shape

#************************************************************************************************************
# return dimensions of elements in a tensor
def element_dim(x):
    return np.prod(get_shape(x)[1:])

#************************************************************************************************************
# convert tensors to 2d
def to_2d(x, mode):
    shape = get_shape(x)
    if len(shape) == 2:
        return x
    if mode == "last":
        return tf.reshape(x, [-1, shape[-1]])
    else:
        return tf.reshape(x, [shape[0], element_dim(x)])

#************************************************************************************************************
# get/create a weight tensor for a convolution or fully-connected layer
def get_weight(shape, gain = 1, use_wscale = True, lrmul = 1, weight_var = "weight"):
    fan_in = np.prod(shape[:-1])
    he_std = gain / np.sqrt(fan_in)

    # Equalized learning rate
    if use_wscale:
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
    else:
        init_std = he_std / lrmul
        runtime_coef = lrmul

    init = tf.initializers.random_normal(0, init_std)
    return tf.get_variable(weight_var, shape = shape, initializer = init) * runtime_coef

#************************************************************************************************************
# fully-connected layer
def dense_layer(x, dim, gain = 1, use_wscale = True, lrmul = 1, weight_var = "weight", name = None):
    if name is not None:
        weight_var = "{}_{}".format(weight_var, name)

    if len(get_shape(x)) > 2:
        x = to_2d(x, "first")

    w = get_weight([get_shape(x)[1], dim], gain = gain, use_wscale = use_wscale,
        lrmul = lrmul, weight_var = weight_var)

    return tf.matmul(x, w)
#************************************************************************************************************

# apply bias and activation function
def apply_bias_act(x, act = "linear", alpha = None, gain = None, lrmul = 1, bias_var = "bias", name = None):
    if name is not None:
        bias_var = "{}_{}".format(bias_var, name)
    b = tf.get_variable(bias_var, shape = [get_shape(x)[1]], initializer = tf.initializers.zeros()) * lrmul
    return fused_bias_act(x, b = b, act = act, alpha = alpha, gain = gain)

#************************************************************************************************************
# normalization types instance, batch or layer-wise.
def norm(x, norm_type, parametric = True):
    if norm_type == "instance":
        x = tf.contrib.layers.instance_norm(x, data_format = "NCHW", center = parametric, scale = parametric)
    elif norm_type == "batch":
        x = tf.contrib.layers.batch_norm(x, data_format = "NCHW", center = parametric, scale = parametric)
    elif norm_type == "layer":
        x = tf.contrib.layers.layer_norm(inputs = x, begin_norm_axis = -1, begin_params_axis = -1)
    return x

#************************************************************************************************************
# normalize tensor according to the integration type
def attention_normalize(x, num, integration, norm):
    shape = get_shape(x)
    x = tf.reshape(x, [-1, num] + get_shape(x)[1:])
    x = tf.cast(x, tf.float32)

    norm_axis = 1 if norm == "instance" else 2

    if integration in ["add", "both"]:
        x -= tf.reduce_mean(x, axis = norm_axis, keepdims = True)
    if integration in ["mul", "both"]:
        x *= tf.rsqrt(tf.reduce_mean(tf.square(x), axis = norm_axis, keepdims = True) + 1e-8)

    # return x to its original shape
    x = tf.reshape(x, shape)
    return x

#************************************************************************************************************
# minibatch standard deviation layer see StyleGAN for details.
def minibatch_stddev_layer(x, group_size = 4, num_new_features = 1, dims = 2):
    shape = get_shape(x) 
    last_dims = [shape[3]] if dims == 2 else []
    group_size = tf.minimum(group_size, shape[0])
    y = tf.reshape(x, [group_size, -1, num_new_features, shape[1]//num_new_features, shape[2]] + last_dims) 
    y = tf.cast(y, tf.float32)
    y -= tf.reduce_mean(y, axis = 0, keepdims = True) 
    y = tf.reduce_mean(tf.square(y), axis = 0) 
    y = tf.sqrt(y + 1e-8) 
    y = tf.reduce_mean(y, axis = [2, 3] + ([4] if dims == 2 else []), keepdims = True) 
    y = tf.reduce_mean(y, axis = [2]) 
    y = tf.tile(y, [group_size, 1, shape[2]] + last_dims) 
    return tf.concat([x, y], axis = 1) 

#************************************************************************************************************
# create a random dropout mask
def random_dp_binary(shape, dropout):
    if dropout == 0:
        return tf.ones(shape)
    eps = tf.random.uniform(shape)
    keep_mask = (eps >= dropout)
    return keep_mask

#************************************************************************************************************
# perform dropout
def dropout(x, dp, noise_shape = None):
    if dp is None or dp == 0.0:
        return x
    return tf.nn.dropout(x, keep_prob = 1.0 - dp, noise_shape = noise_shape)

#************************************************************************************************************
# set a mask for logits
def logits_mask(x, mask):
    return x + tf.cast(1 - tf.cast(mask, tf.int32), tf.float32) * -10000.0


#************************************************************************************************************
# 2d linear embeddings
def get_linear_embeddings(size, dim, num, rng = 1.0):
    pi = tf.constant(math.pi)
    theta = tf.range(0, pi, pi / num)
    dirs = tf.stack([tf.cos(theta), tf.sin(theta)], axis = -1)
    embs = tf.get_variable(name = "emb", shape = [num, int(dim / num)],
        initializer = tf.random_uniform_initializer())

    c = tf.linspace(-rng, rng, size)
    x = tf.tile(tf.expand_dims(c, axis = 0), [size, 1])
    y = tf.tile(tf.expand_dims(c, axis = 1), [1, size])
    xy = tf.stack([x,y], axis = -1)

    lens = tf.reduce_sum(tf.expand_dims(xy, axis = 2) * dirs, axis = -1, keepdims = True)
    emb = tf.reshape(lens * embs, [size, size, dim])
    return emb

#************************************************************************************************************
# construct sinusoidal embeddings spanning the 2d space
def get_sinusoidal_embeddings(size, dim, num = 2):
    if num == 2:
        c = tf.expand_dims(tf.to_float(tf.linspace(-1.0, 1.0, size)), axis = -1)
        i = tf.to_float(tf.range(int(dim / 4)))

        peSin = tf.sin(c / (tf.pow(10000.0, 4 * i / dim)))
        peCos = tf.cos(c / (tf.pow(10000.0, 4 * i / dim)))

        peSinX = tf.tile(tf.expand_dims(peSin, axis = 0), [size, 1, 1])
        peCosX = tf.tile(tf.expand_dims(peCos, axis = 0), [size, 1, 1])
        peSinY = tf.tile(tf.expand_dims(peSin, axis = 1), [1, size, 1])
        peCosY = tf.tile(tf.expand_dims(peCos, axis = 1), [1, size, 1])

        emb = tf.concat([peSinX, peCosX, peSinY, peCosY], axis = -1)
    else:
        pi = tf.constant(math.pi)
        theta = tf.range(0, pi, pi / num)
        dirs = tf.stack([tf.cos(theta), tf.sin(theta)], axis = -1)

        c = tf.linspace(-1.0, 1.0, size)
        x = tf.tile(tf.expand_dims(c, axis = 0), [size, 1])
        y = tf.tile(tf.expand_dims(c, axis = 1), [1, size])
        xy = tf.stack([x,y], axis = -1)

        lens = tf.reduce_sum(tf.expand_dims(xy, axis = -2) * dirs, axis = -1, keepdims = True)

        i = tf.to_float(tf.range(int(dim / (2 * num))))
        sins = tf.sin(lens / (tf.pow(10000.0, 2 * num * i / dim)))
        coss = tf.cos(lens / (tf.pow(10000.0, 2 * num * i / dim)))
        emb = tf.reshape(tf.concat([sins, coss], axis = -1), [size, size, dim])

    return emb

#************************************************************************************************************
# construct positional embeddings with different types (sinusoidal, linear or trainable)    
def get_positional_embeddings(max_res, dim, pos_type = "sinus", dir_num = 2, init = "uniform", shared = False):
    embs = []
    initializer = tf.random_uniform_initializer() if init == "uniform" else tf.initializers.random_normal()
    for res in range(max_res + 1):
        with tf.variable_scope("pos_emb%d" % res):
            size = 2 ** max_res
            if pos_type == "sinus":
                emb = get_sinusoidal_embeddings(size, dim, num = dir_num)
            elif pos_type == "linear":
                emb = get_linear_embeddings(size, dim, num = dir_num)
            elif pos_type == "trainable2d":
                emb = tf.get_variable(name = "emb", shape = [size, size, dim], initializer = initializer)
            else: # pos_type == "trainable"
                xemb = tf.get_variable(name = "x_emb", shape = [size, int(dim / 2)], initializer = initializer)
                yemb = xemb if shared else tf.get_variable(name = "y_emb", shape = [size, int(dim / 2)],
                    initializer = initializer)
                xemb = tf.tile(tf.expand_dims(xemb, axis = 0), [size, 1, 1])
                yemb = tf.tile(tf.expand_dims(yemb, axis = 1), [1, size, 1])
                emb = tf.concat([xemb, yemb], axis = -1)
            embs.append(emb)
    return embs


def get_positional_embedding(max_res, dim, pos_type = "sinus", dir_num = 2, init = "uniform", shared = False,resolution_array=[]):
    embs = []
    initializer = tf.random_uniform_initializer() if init == "uniform" else tf.initializers.random_normal()
    for idx in range(len(resolution_array)):
        with tf.variable_scope("pos_emb%d" % idx):
            size = resolution_array[idx]
            if pos_type == "sinus":
                emb = get_sinusoidal_embeddings(size, dim, num = dir_num)
        embs.append(emb)
    return embs

#************************************************************************************************************
def get_embeddings(size, dim, init = "uniform", name = None):
    initializer = tf.random_uniform_initializer() if init == "uniform" else tf.initializers.random_normal()
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        emb = tf.get_variable(name = "emb", shape = [size, dim], initializer = initializer)
    return emb

#************************************************************************************************************
def get_relative_embeddings(l, dim, embs):
    diffs = tf.expand_dims(tf.range(l), axis = -1) - tf.range(l)
    diffs -= tf.reduce_min(diffs)
    ret = tf.gather(embs, tf.reshape(diffs, [-1]))
    ret = tf.reshape(ret, [1, l, l, dim])
    return ret

#************************************************************************************************************
# non-linear layer with a resnet connection optionally perform attention    
def nnlayer(x, dim, act, lrmul = 1, y = None, ff = True, pool = False, name = "", **kwargs):
    shape = get_shape(x)
    _x = x
    # Split attention types only for convention
    if y is not None and y != x: # cross-attention
        x = cross_attention_transformer_block(from_tensor = x, to_tensor = y, dim = dim, name = name, **kwargs)[0]
    elif y is not None and y==x: # self-attention
        x = self_attention_transformer_block(from_tensor = x, to_tensor = y, dim = dim, name = name, **kwargs)[0]

    if ff: # feed-forward
        if pool:
            x = to_2d(x, "last")

        with tf.variable_scope("Dense%s_0" % name):
            x = apply_bias_act(dense_layer(x, dim, lrmul = lrmul), act = act, lrmul = lrmul)
        with tf.variable_scope("Dense%s_1" % name):
            x = apply_bias_act(dense_layer(x, dim, lrmul = lrmul), lrmul = lrmul)

        if pool:
            x = tf.reshape(x, shape)

    #    x = tf.nn.leaky_relu(x + _x) # resnet connection
    return x

#************************************************************************************************************
# multi-layer perceptron with a nonlinearity 'act'.
# optionally use resnet connections and self-attention.
def mlp(x, resnet, layers_num, dim, act, lrmul, pooling = "mean", transformer = False, norm_type = None, **kwargs):
    shape = get_shape(x)

    if len(get_shape(x)) > 2:
        if pooling == "cnct":
            with tf.variable_scope("Dense_pool"):
                x = apply_bias_act(dense_layer(x, dim), act = act)
        elif pooling == "batch":
            x = to_2d(x, "last")
        else:
            pool_shape = (get_shape(x)[-2], get_shape(x)[-1])
            x = tf.nn.avg_pool(x, pool_shape, pool_shape, padding = "SAME", data_format = "NCHW")
            x = to_2d(x, "first")

    if resnet:
        half_layers_num = int(layers_num / 2)
        for layer_idx in range(half_layers_num):
            y = x if transformer else None
            x = nnlayer(x, dim, act, lrmul, y = y, name = layer_idx, **kwargs)
            x = norm(x, norm_type)

        with tf.variable_scope("Dense%d" % layer_idx):
            x = apply_bias_act(dense_layer(x, dim, lrmul = lrmul), act = act, lrmul = lrmul)

    else:
        for layer_idx in range(layers_num):
            with tf.variable_scope("Dense%d" % layer_idx):
                x = apply_bias_act(dense_layer(x, dim, lrmul = lrmul), act = act, lrmul = lrmul)
                x = norm(x, norm_type)

    x = tf.reshape(x, [-1] + shape[1:-1] + [dim])
    return x


#************************************************************************************************************
# convolution layer with optional upsampling or downsampling
def conv2d_layer(x, dim, kernel, up = False, factor=2, down = False, resample_kernel = None, gain = 1, use_wscale = True, lrmul = 1, weight_var = "weight",layer_idx='0'):
    assert not (up and down)
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, get_shape(x)[1], dim], gain = gain, use_wscale = use_wscale, lrmul = lrmul, weight_var = weight_var +'_' + str(layer_idx))
    if up:
        x = upsample_conv_2d(x, w, data_format = "NCHW", k = resample_kernel, factor=factor)
    elif down:
        x = conv_downsample_2d(x, w, data_format = "NCHW", k = resample_kernel, factor=factor)
    else:
        x = tf.nn.conv2d(x, w, data_format = "NCHW", strides = [1, 1, 1, 1], padding = "SAME")
    return x

#************************************************************************************************************
# modulated convolution layer (see StyleGAN for details)
def modulated_convolution_layer(x, y, dim, kernel,layer_idx,up = False, down = False,resample_kernel = None, modulate = True, demodulate = True, fused_modconv = True,  
        noconv = False, gain = 1, use_wscale = True, lrmul = 1, weight_var = "weight", mod_weight_var = "mod_weight", mod_bias_var = "mod_bias"):

    weight_var = weight_var + "_" + str(layer_idx)
    mod_weight_var = mod_weight_var + "_" + str(layer_idx)
    mod_bias_var = mod_bias_var + "_" + str(layer_idx)
    assert not (up and down)
    assert kernel >= 1 and kernel % 2 == 1

    w = get_weight([kernel, kernel, get_shape(x)[1], dim], gain = gain, use_wscale = use_wscale,
        lrmul = lrmul, weight_var = weight_var)
    ww = w[np.newaxis]

    s = dense_layer(y, dim = get_shape(x)[1], weight_var = mod_weight_var) 
    s = apply_bias_act(s, bias_var = mod_bias_var) + 1 
    
    if modulate:
        ww *= s[:, np.newaxis, np.newaxis, :, np.newaxis] 
        if demodulate:
            d = tf.rsqrt(tf.reduce_sum(tf.square(ww), axis = [1, 2, 3]) + 1e-8) 
            ww *= d[:, np.newaxis, np.newaxis, np.newaxis, :] 
    else:
        ww += tf.zeros_like(s[:, np.newaxis, np.newaxis, :, np.newaxis])

    if fused_modconv:
        x = tf.reshape(x, [1, -1, get_shape(x)[-2], get_shape(x)[-1]])  
        w = tf.reshape(tf.transpose(ww, [1, 2, 3, 0, 4]), get_shape(ww)[1:4] + [-1])
    else:
        if modulate:
            x *= s[:, :, np.newaxis, np.newaxis] 

    if noconv:
        if up:
            x = upsample_2d(x, k = resample_kernel)
        elif down:
            x = downsample_2d(x, k = resample_kernel)
    else:
        if up:
            x = upsample_conv_2d(x, w, data_format = "NCHW", k = resample_kernel)
        elif down:
            x = conv_downsample_2d(x, w, data_format = "NCHW", k = resample_kernel)
        else:
            x = tf.nn.conv2d(x, w, data_format = "NCHW", strides = [1,1,1,1], padding = "SAME")

    if fused_modconv:
        x = tf.reshape(x, [-1, dim] + get_shape(x)[-2:]) 
    elif modulate and demodulate:
        x *= d[:, :, np.newaxis, np.newaxis] 

    return x

#************************************************************************************************************
# validate transformer input shape (reshape to 2d)
def process_input(t, t_pos, t_len, name):
    shape = get_shape(t)

    if len(shape) > 3:
        misc.error("Transformer {}_tensor has {} shape. should be up to 3 dims.".format(name, shape))
    elif len(shape) == 3:
        batch_size, t_len, _ = shape
    else:
        if t_len is None:
            misc.error("If {}_tensor has two dimensions, must specify {}_len.".format(name, name))
        batch_size = tf.cast(shape[0] / t_len, tf.int32)

    # reshape tensors to 2d
    t = to_2d(t, "last")
    if t_pos is not None:
        t_pos = tf.tile(to_2d(t_pos, "last"), [batch_size, 1])

    return t, t_pos, shape, t_len, batch_size

#************************************************************************************************************
# transpose tensor to scores
def transpose_for_scores(x, batch_size, num_heads, elem_num, head_size):
    x = tf.reshape(x, [batch_size, elem_num, num_heads, head_size])
    x = tf.transpose(x, [0, 2, 1, 3]) 
    return x

#************************************************************************************************************
# calculate attention probabilities using tf.nn.softmax
def compute_probs(scores, dp):
    probs = tf.nn.softmax(scores)
    shape = get_shape(probs)
    shape[-2] = 1
    probs = dropout(probs, dp / 2)
    probs = dropout(probs, dp / 2, shape)
    return probs

#************************************************************************************************************
# scale and bias the given tensor in transformer
def integrate(tensor, tensor_len, control, integration, norm, layer_idx="0"):
    dim = get_shape(tensor)[-1]

    # normalization
    if norm is not None:
        tensor = attention_normalize(tensor, tensor_len, integration, norm)

    # compute gain/bias
    control_dim = {"add": dim, "mul": dim, "both": 2 * dim}[integration]
    bias = gain = control = apply_bias_act(dense_layer(control, control_dim, name = "out"+ str(layer_idx)), name = "out"+ str(layer_idx))
    if integration == "both":
        gain, bias = tf.split(control, 2, axis = -1)

    # modulation
    if integration != "add":
        tensor *= (gain + 1)
    if integration != "mul":
        tensor += bias

    return tensor

#************************************************************************************************************
# computing centroid assignments (see k-means algorithm for details, Lloyd et al. 1982)
def compute_assignments(att_probs):
    centroid_assignments = (att_probs / (tf.reduce_sum(att_probs, axis = -2, keepdims = True) + 1e-8))
    centroid_assignments = tf.transpose(centroid_assignments, [0, 1, 3, 2]) # [B, N, T, F]
    return centroid_assignments

#************************************************************************************************************
# using centroids for attention calculations (see k-means algorithm for details, Lloyd et al. 1982)
def compute_centroids(_queries, queries, to_from, to_len, from_len, batch_size, num_heads, 
        size_head, parametric):
    
    dim = 2 * size_head
    from_elements = tf.concat([_queries, queries - _queries], axis = -1)
    from_elements = transpose_for_scores(from_elements, batch_size, num_heads, from_len, dim) 

    if to_from is not None:

        if get_shape(to_from)[-2] < to_len:
            s = int(math.sqrt(get_shape(to_from)[-2]))
            to_from = upsample_2d(tf.reshape(to_from, [batch_size * num_heads, s, s, from_len]), factor = 2, data_format = "NHWC")
            to_from = tf.reshape(to_from, [batch_size, num_heads, to_len, from_len])

        if get_shape(to_from)[-1] < from_len:
            s = int(math.sqrt(get_shape(to_from)[-1]))
            to_from = upsample_2d(tf.reshape(to_from, [batch_size * num_heads, to_len, s, s]), factor = 2, data_format = "NCHW")
            to_from = tf.reshape(to_from, [batch_size, num_heads, to_len, from_len])

        if get_shape(to_from)[-2] > to_len:
            s = int(math.sqrt(get_shape(to_from)[-2]))
            to_from = downsample_2d(tf.reshape(to_from, [batch_size * num_heads, s, s, from_len]), factor = 2, data_format = "NHWC")
            to_from = tf.reshape(to_from, [batch_size, num_heads, to_len, from_len])

        if get_shape(to_from)[-1] > from_len:
            s = int(math.sqrt(get_shape(to_from)[-1]))
            to_from = downsample_2d(tf.reshape(to_from, [batch_size * num_heads, to_len, s, s]), factor = 2, data_format = "NCHW")
            to_from = tf.reshape(to_from, [batch_size, num_heads, to_len, from_len])

        to_centroids = tf.matmul(to_from, from_elements)

    if to_from is None or parametric:
        if parametric:
            to_centroids = tf.tile(tf.get_variable("toasgn_init", shape = [1, num_heads, to_len, dim],
                initializer = tf.initializers.random_normal()), [batch_size, 1, 1, 1])
        else:
            to_centroids = apply_bias_act(dense_layer(queries, dim * num_heads, name = "key2"), name = "key2")
            to_centroids = transpose_for_scores(to_centroids, batch_size, num_heads, dim, dim)

    return from_elements, to_centroids

#************************************************************************************************************
# construct cross-attention-transformer used between latents and images
def cross_attention_transformer_block(
        dim,                                  # dimension of the layer
        from_tensor,        to_tensor,        
        from_len = None,    to_len = None,    
        from_pos = None,    to_pos = None,    # the positional encodings for the cross attention tensors
        num_heads = 1,                        # number of attention heads (default value is 1 for slater)
        att_dp = 0.12,                        # dropout rate of attention
        att_mask = None,                      # Attention mask to block from/to elements [batch_size, from_len, to_len]
        integration = "mul",                  # integration type (default value is 'mul' for slater)
        norm = "layer",                       # normalization type
        kmeans = False,                       # see k-means algorithm (Lloyd et al 1982).
        kmeans_iters = 1,                     # number of k-means iterations per layer
        att_vars = {},                        # variables used in k-means algorithm carried through layers
                                              # suffix
        name = "",
        layer_idx=0): 

    assert from_tensor != to_tensor # be sure for cross-attention
    from_tensor, from_pos, from_shape, from_len, batch_size = process_input(from_tensor, from_pos, from_len, "from")

    to_tensor,   to_pos,   to_shape,   to_len,   _          = process_input(to_tensor, to_pos, to_len, "to")

    size_head = int(dim / num_heads)
    to_from = att_vars.get("centroid_assignments")

    with tf.variable_scope("AttLayer_{}".format(name)):
        queries = apply_bias_act(dense_layer(from_tensor, dim, name = "query" ), name = "query") 
        keys    = apply_bias_act(dense_layer(to_tensor, dim, name = "key"), name = "key")      
        values  = apply_bias_act(dense_layer(to_tensor, dim, name = "value"), name = "value")  

        _queries = queries

        if from_pos is not None:
            queries += apply_bias_act(dense_layer(from_pos, dim, name = "from_pos" + str(layer_idx)), name = "from_pos"+ str(layer_idx))

        if to_pos is not None:
            keys += apply_bias_act(dense_layer(to_pos, dim, name = "to_pos" + str(layer_idx)), name = "to_pos"+ str(layer_idx))

        if kmeans:
            from_elements, to_centroids = compute_centroids(_queries, queries, to_from,
                to_len, from_len, batch_size, num_heads, size_head, parametric = True)

        values = transpose_for_scores(values, batch_size, num_heads, to_len, size_head)     
        queries = transpose_for_scores(queries, batch_size, num_heads, from_len, size_head) 
        keys = transpose_for_scores(keys, batch_size, num_heads, to_len, size_head)         
        att_scores = tf.matmul(queries, keys, transpose_b = True)                           
        att_probs = None

        for i in range(kmeans_iters):
            with tf.variable_scope("iter_{}".format(i)):
                if kmeans:
                    if i > 0:
                        to_from = compute_assignments(att_probs)
                        to_centroids = tf.matmul(to_from, from_elements)

                    w = tf.get_variable(name = "st_weights" + str(layer_idx), shape = [num_heads, 1, get_shape(from_elements)[-1]],
                        initializer = tf.ones_initializer())
                    att_scores = tf.matmul(from_elements * w, to_centroids, transpose_b = True)

                att_scores = tf.multiply(att_scores, 1.0 / math.sqrt(float(size_head)))
                if att_mask is not None:
                    att_scores = logits_mask(att_scores, tf.expand_dims(att_mask, axis = 1))
                att_probs = compute_probs(att_scores, att_dp)

        if kmeans:
            to_from = compute_assignments(att_probs)

        control = tf.matmul(att_probs, values) 
        control = tf.transpose(control, [0, 2, 1, 3]) 
        control = tf.reshape(control, [batch_size * from_len, dim]) 
        from_tensor = integrate(from_tensor, from_len, control, integration, norm, layer_idx=layer_idx)

    if len(from_shape) > 2:
        from_tensor = tf.reshape(from_tensor, from_shape)

    return from_tensor, att_probs, {"centroid_assignments": to_from}

#************************************************************************************************************
# construct a self-attention-transformer block
def self_attention_transformer_block(
        dim,                                  # dimension of the layer
        from_tensor,        to_tensor,        
        from_len = None,    to_len = None,    
        from_pos = None,    to_pos = None,    # the positional encodings for the cross attention tensors
        num_heads = 1,                        # number of attention heads (default value is 1 for slater)
        att_dp = 0.12,                        # dropout rate of attention
        att_mask = None,                      # Attention mask to block from/to elements [batch_size, from_len, to_len]
        integration = "mul",                  # integration type (default value is 'mul' for slater)
        norm = "layer",                       # normalization type
        kmeans = False,                       # see k-means algorithm (Lloyd et al 1982).
        kmeans_iters = 1,                     # number of k-means iterations per layer
        att_vars = {},                        # variables used in k-means algorithm carried through layers
                                              # suffix
        name = "",
        layer_idx=0): 
    
    assert from_tensor == to_tensor # be sure for self-attention
    
    from_tensor, from_pos, from_shape, from_len, batch_size = process_input(from_tensor, from_pos, from_len, "from")
    to_tensor,   to_pos,   to_shape,   to_len,   _          = process_input(to_tensor, to_pos, to_len, "to")

    size_head = int(dim / num_heads)
    to_from = att_vars.get("centroid_assignments")

    with tf.variable_scope("AttLayer_{}".format(name)):
        queries = apply_bias_act(dense_layer(from_tensor, dim, name = "query" + str(layer_idx)), name = "query") 
        keys    = apply_bias_act(dense_layer(to_tensor, dim, name = "key" + str(layer_idx)), name = "key")       
        values  = apply_bias_act(dense_layer(to_tensor, dim, name = "value" + str(layer_idx)), name = "value")   
        _queries = queries

        if from_pos is not None:
            queries += apply_bias_act(dense_layer(from_pos, dim, name = "from_pos" + str(layer_idx)), name = "from_pos")
        if to_pos is not None:
            keys += apply_bias_act(dense_layer(to_pos, dim, name = "to_pos" + str(layer_idx)), name = "to_pos")

        if kmeans:
            from_elements, to_centroids = compute_centroids(_queries, queries, to_from,
                to_len, from_len, batch_size, num_heads, size_head, parametric = True)

        values = transpose_for_scores(values, batch_size, num_heads, to_len, size_head)    
        queries = transpose_for_scores(queries, batch_size, num_heads, from_len, size_head) 
        keys = transpose_for_scores(keys, batch_size, num_heads, to_len, size_head)       
        att_scores = tf.matmul(queries, keys, transpose_b = True)                       
        att_probs = None

        for i in range(kmeans_iters):
            with tf.variable_scope("iter_{}".format(i)):
                if kmeans:
                    if i > 0:
                        to_from = compute_assignments(att_probs)
                        to_centroids = tf.matmul(to_from, from_elements)

                    w = tf.get_variable(name = "st_weights" + str(layer_idx), shape = [num_heads, 1, get_shape(from_elements)[-1]],
                        initializer = tf.ones_initializer())
                    att_scores = tf.matmul(from_elements * w, to_centroids, transpose_b = True)

            att_scores = tf.multiply(att_scores, 1.0 / math.sqrt(float(size_head)))
            if att_mask is not None:
                att_scores = logits_mask(att_scores, tf.expand_dims(att_mask, axis = 1))
            att_probs = compute_probs(att_scores, att_dp)


        if kmeans:
            to_from = compute_assignments(att_probs)

        control = tf.matmul(att_probs, values) 
        control = tf.transpose(control, [0, 2, 1, 3])
        control = tf.reshape(control, [batch_size * from_len, dim]) 
        from_tensor = integrate(from_tensor, from_len, control, integration, norm)

    if len(from_shape) > 2:
        from_tensor = tf.reshape(from_tensor, from_shape)

    return from_tensor, att_probs, {"centroid_assignments": to_from}


def get_timestep_embedding(timesteps, embedding_dim: int):
  """
  From Fairseq.
  Build sinusoidal embeddings.
  This matches the implementation in tensor2tensor, but differs slightly
  from the description in Section 3.5 of "Attention Is All You Need".
  """
  assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32

  half_dim = embedding_dim // 2
  emb = math.log(10000) / (half_dim - 1)
  emb = tf.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
  emb = tf.cast(timesteps, dtype=tf.float32)[:, None] * emb[None, :]
  emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=1)
  if embedding_dim % 2 == 1:  # zero pad
    emb = tf.pad(emb, [[0, 0], [0, 1]])

  return emb