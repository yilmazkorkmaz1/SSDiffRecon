import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# import sys
# sys.path.append("..") # Adds higher directory to python modules path.
from .network_helpers import *
import dnnlib.tflib as tflib


class SSDiffRecon_Model():

    def __init__(self):
        tflib.init_tf()
        self.Diff_Network = None
        self.Mapper = None
        self.load_networks()
        self.latent_pos = get_embeddings(16, 32, name = "ltnt_emb")
        

    def load_networks(self):
        kwargs = dict()
        kwargs['components_num'] = 16
        kwargs['latent_size'] = 32
        kwargs["dlatent_size"] = 32
        self.Mapper = tflib.Network("Mapper", dlatent_broadcast = 12, func_name=globals()["Mapper"],**kwargs)
        self.Diff_Network = tflib.Network("DiffModel_network", func_name = globals()["DiffModel"])

    def get_trainable_variables(self):
        vars = []
        mapper = 1
        diff = 1
        if diff == 1:
            for var in self.Diff_Network.trainables.values():
                vars.append(var)
        if mapper == 1:
            for var in self.Mapper.trainables.values():
                vars.append(var)
        return vars

    def model(self, us_im, noisy_sample, label, time, mask):
        dlatent = self.Mapper.get_output_for(time,label,self.latent_pos,None)
        denoised_im, _ = self.Diff_Network.get_output_for(noisy_sample, us_im, dlatent, mask)
        return denoised_im
    

def Mapper(
    time,                                   # time index
    labels_in,                              # labels 
    latent_pos,                             # positional embeddings for latents (used in self-attention-transformer)
    component_mask,                         # drop out mask 
    components_num          = 16,           # number of local latent components z_1,...,z_k
    latent_size             = 512,          # latent dimensionality per component.
    dlatent_size            = 512,          # disentangled latent dimensionality
    label_size              = 5,            # label dimensionality, 0 if no labels 
    dlatent_broadcast       = None,         # tile latent vectors to num_layer to control style in all layers 
    normalize_latents       = True,         # normalize latent vectors (z)
    mapping_layersnum       = 8,            # number of mapping layers
    mapping_dim             = None,         # number of activations in the mapping layers
    mapping_lrmul           = 0.01,         # learning rate multiplier for the mapping layers
    mapping_nonlinearity    = "lrelu",      # activation function
    num_heads               = 1,            # number of attention heads
    attention_dropout       = 0.12,         # attention dropout rate
    **_kwargs):

    act = mapping_nonlinearity
    k = components_num
    latents_num = k + 1 # total number of latents = num(local_latents) + global_latent

    net_dim = mapping_dim
    layersnum = mapping_layersnum
    lrmul = mapping_lrmul
    ltnt2ltnt = True
    resnet = True

    # input tensors
    time.set_shape([None])
    labels_in.set_shape([None, label_size])
   
    latent_pos.set_shape([k, dlatent_size])
    component_mask.set_shape([None, 1, k])

    batch_size = get_shape(time)[0]
    x = None

    with tf.variable_scope("TimeConcat"):
        temb = get_timestep_embedding(time, latent_size)
        w = tf.get_variable("weight", shape = [latent_size, latent_size], initializer = tf.initializers.random_normal())
        x = tf.tile(tf.expand_dims(tf.matmul(temb, w), axis = 1), (1, latents_num, 1))
           

    if net_dim is None:
        net_dim = dlatent_size
    else:
        x = to_2d(x, "last")
        x = apply_bias_act(dense_layer(x, net_dim, name = "map_start"), name = "map_start")
        x = tf.reshape(x, [batch_size, latents_num, net_dim])
        if latent_pos is not None:
            latent_pos = apply_bias_act(dense_layer(latent_pos, net_dim, name = "map_pos"), name = "map_pos")

    if label_size:
        with tf.variable_scope("LabelConcat"):
            w = tf.get_variable("weight", shape = [label_size, latent_size], initializer = tf.initializers.random_normal())
            l = tf.tile(tf.expand_dims(tf.matmul(labels_in, w), axis = 1), (1, latents_num, 1))
            x = tf.concat([x, l], axis = 2)

    # splitting latent vectors to global and local
    x, g = tf.split(x, [k, 1], axis = 1)
    g = tf.squeeze(g, axis = 1)

    # normalize latent vectors
    if normalize_latents:
        with tf.variable_scope("Normalize"):
            x *= tf.rsqrt(tf.reduce_mean(tf.square(x), axis = -1, keepdims = True) + 1e-8)

    mlp_kwargs = {}
    if ltnt2ltnt:
        mlp_kwargs.update({         "transformer": ltnt2ltnt,
           "num_heads": 1,          "att_dp": attention_dropout,
           "from_pos": latent_pos,  "to_pos": latent_pos,
           "from_len": k,           "to_len": k,
        })

    # mapper layers
    if k == 0:
        x = tf.zeros([batch_size, 0, net_dim])
    else:
        x = mlp(x, resnet, layersnum, net_dim, act, lrmul, pooling = "batch",
            att_mask = component_mask, **mlp_kwargs)

    with tf.variable_scope("global"):
        # mapping global latent separately
        g = mlp(g, resnet, layersnum, net_dim, act, lrmul)
    # concatenate back global and local latent vectors
    x = tf.concat([x, tf.expand_dims(g, axis = 1)], axis = 1)

    # tile latent vectors to all resolution layers to control style and local features in each layer.
    if dlatent_broadcast is not None:
        with tf.variable_scope("Broadcast"):
            x = tf.tile(x[:, :, np.newaxis], [1, 1, dlatent_broadcast, 1])

    x = tf.identity(x, name = "dlatents_out")
    return x # [batch size, num_layers, number of latent vectors, latent dimension]


def DiffModel(    
    noisy_sample,                        # input sample iterated through noise to fully_sampled
    us_im,                              # undersampled image
    dlatents_in,                        # intermediate latent vectors (W) : k local + 1 global
    mask,                               # undersampling mask
    dlatent_size        = 32,           # latent dimension
    pos_dim             = None,         # positional embeddings dimension
    num_channels        = 2,            # number of channels (rgb default)
    resolution          = 256,          # resolution (overwritten by used dataset)
    nonlinearity        = "lrelu",      # activation function
    local_noise         = True,         # add stochastic noise to activations
    randomize_noise     = False,        # change noise variables every time
    components_num      = 16,           # number of local latent vectors
    num_heads           = 1,            # number of attention heads
    attention_dropout   = 0.12,         # attention dropout rate
    integration         = "mul",        # feature integration type: additive, multiplicative or both
    norm                = 'layer',      # feature normalization type (optional): instance, batch or layer
    use_pos             = True,         # use positional encoding for latents
    **kwargs,
    ):                         

    # settings
    k = components_num
    act = nonlinearity
    latents_num = k + 1 # k local + 1 global latent
    resolution_log2 = int(np.log2(resolution))
    num_layers = 12
    noisy_sample.set_shape([None, num_channels, resolution, resolution])
    us_im.set_shape([None, num_channels, resolution, resolution])

    if pos_dim is None:
        pos_dim = dlatent_size
    assert resolution == 2**resolution_log2 and resolution >= 4

    def get_global(dlatents):
        return dlatents[:, -1]
    # inputs
    dlatents_in.set_shape([None, latents_num, num_layers, dlatent_size])
    mask.set_shape([None, 1, resolution, resolution])
    batch_size = get_shape(dlatents_in)[0]
    component_mask = random_dp_binary([batch_size, k], 0.0)
    component_mask = tf.expand_dims(component_mask, axis = 1)
    latent_pos = get_embeddings(k, dlatent_size, name = "ltnt_emb")

    if not use_pos:
        latent_pos = None

    num_resnet_blocks = 6
    resolution_array = []
    for i in range(num_resnet_blocks*2):
        resolution_array.append(256)

    noise_layers = []
    for layer_idx in range(len(resolution_array)):
        batch_multiplier = 1
        noise_shape = [batch_multiplier, 1, resolution_array[layer_idx], resolution_array[layer_idx]]
        # local noise variables
        noise_layers.append(tf.get_variable("noise%d" % layer_idx, shape = noise_shape,
            initializer = tf.initializers.random_normal(), trainable = True))

    def add_noise(x, layer_idx):
        if randomize_noise:
            shape = get_shape(x) 
            shape[1] = 1
            noise = tf.random_normal(shape)
        else:
            noise = noise_layers[layer_idx]
        strength = tf.get_variable("noise_strength" + str(layer_idx), shape = [], initializer = tf.initializers.zeros())
        x += strength * noise
        return x

    def synthesizer_layer(x, dlatents, layer_idx, dim, kernel, att_vars, up = False,down=False, transformer=True):
        att_map = None
        dlatent_global = get_global(dlatents_in)[:,  layer_idx]
        
        new_dlatents = None
        if dlatents is None:
            dlatents = dlatents_in[:, :-1, layer_idx]


        # perform modulated_convolution
        x = modulated_convolution_layer(x, dlatent_global, dim, kernel, fused_modconv = True, modulate = True, layer_idx=layer_idx)
        

        shape = get_shape(x)
        if transformer:
            x = tf.transpose(tf.reshape(x, [shape[0], shape[1], shape[2] * shape[3]]), [0, 2, 1])

            # arguments used in attention-blocks see run_network.py for explanation of each argument
            kwargs = {
                "num_heads": num_heads,
                "integration": integration,     
                "norm": norm,                   
                "att_mask": component_mask,     
                "att_dp": attention_dropout,    
                "from_pos": get_sinusoidal_embeddings(shape[2], pos_dim),    
                "to_pos": latent_pos,           
                "att_vars": att_vars,     
                "layer_idx": layer_idx                                                      
            }
            # cross Attention Transformer Layer information flow from local latent vectors to images
            x, att_map, att_vars = cross_attention_transformer_block(from_tensor = x, to_tensor = dlatents, dim = shape[1],
                name = "l2n" + str(layer_idx), **kwargs)

            x = tf.reshape(tf.transpose(x, [0, 2, 1]), shape)

        if local_noise:
            x = add_noise(x, layer_idx)
        
        x = apply_bias_act(x, act = act, name=str(layer_idx))

        return x, new_dlatents, att_map, att_vars

    def fft2c(im):
        return tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(im, axes=[-1,-2])), axes=[-1,-2]) 
    
    def ifft2c(d):
        return tf.signal.fftshift(tf.signal.ifft2d(tf.signal.ifftshift(d, axes=[-1,-2])), axes=[-1,-2])

    def data_consistency_layer(x, target, mask):
        x_real = x[:,0,:,:]
        x_imag = x[:,1,:,:]
        x = tf.complex(x_real, x_imag)
        target_real = target[:,0,:,:]
        target_imag = target[:,1,:,:]
        target = tf.complex(target_real, target_imag)
        kspace_recon = fft2c(x)
        kspace_target = fft2c(target)
        masked = tf.greater(mask[:,0,:,:],0)
        new_kspace = tf.where(masked, kspace_target, kspace_recon)
        new_im = ifft2c(new_kspace)
        sep_im_real = tf.real(new_im)
        sep_im_imag = tf.imag(new_im)
        return tf.stack([sep_im_real, sep_im_imag], axis=1)

    def torgb(t, y, dlatents): 
        with tf.variable_scope("ToRGB"):
            t = modulated_convolution_layer(t, dlatents[:, -1], dim = num_channels,
                kernel = 7, demodulate = False, fused_modconv = True, modulate = True, layer_idx="toRGB")
            t = apply_bias_act(t)
            t = data_consistency_layer(t, us_im, mask)
            
            if y is not None:
                t += y

            return t
        
    def block(x,  dlatents,  att_vars, idx=0): 
        # convolution + cross attention transformer
        att_maps = []

        x = conv2d_layer(x, dim = 128, kernel = 7)

        for i in range(num_resnet_blocks):
            t = x
            with tf.variable_scope("64x64-first-conv-block-" + str(i)):
                x, dlatents, att_map1, att_vars = synthesizer_layer(x, dlatents,layer_idx = idx,
                    dim = 128, kernel = 3, att_vars = att_vars, transformer=True)
            idx +=1 
            with tf.variable_scope("64x64-second-cov-block-" + str(i)):
                x, dlatents, att_map2, att_vars = synthesizer_layer(x, dlatents,layer_idx = idx,
                    dim = 128, kernel = 3, att_vars = att_vars, transformer=True)
            idx +=1
            x = (x + t)
            t = x  
            with tf.variable_scope("data-cons-layer-first" + str(i)):
                x = apply_bias_act(conv2d_layer(x, dim = num_channels, kernel = 1))
            with tf.variable_scope("data-cons-layer" + str(i)):
               x = data_consistency_layer(x, us_im, mask)
            with tf.variable_scope("data-cons-layer-second" + str(i)):
                x = apply_bias_act(conv2d_layer(x, dim = 128, kernel = 1), act=act)
            x = (x + t)  
            att_maps.append(att_map1)
            att_maps.append(att_map2)

        return x, dlatents, att_maps, att_vars, idx

    imgs_out, dlatents, att_maps = None, None, []
    att_vars = {"centroid_assignments": None}

    x = noisy_sample
    idx = 0
    resolution = 8
    x, dlatents, _att_maps, att_vars, idx = block(x, dlatents,  att_vars = att_vars, idx=idx)
    att_maps += _att_maps
    imgs_out = torgb(x, imgs_out,  get_global(dlatents_in))



    def list2tensor(att_list):
        att_list = [att_map for att_map in att_list if att_map is not None]
        if len(att_list) == 0:
            return None
        resolution = 2**resolution_log2
        maps_out = []
        for att_map in att_list:
            s = int(math.sqrt(get_shape(att_map)[2]))
            att_map = tf.transpose(tf.reshape(att_map, [-1, s, s, k]), [0, 3, 1, 2]) 
            if s < resolution:
                att_map = upsample_2d(att_map, factor = int(resolution / s))
            att_map = tf.reshape(att_map, [-1, num_heads, k, resolution, resolution]) 
            maps_out.append(att_map)

        maps_out = tf.transpose(tf.stack(maps_out, axis = 1), [0, 3, 1, 2, 4, 5]) 
        return maps_out

    maps_out = list2tensor(att_maps)

    if maps_out == None:
        maps_out = tf.zeros([1, 16, num_layers, 1, 256, 256])
    
    imgs_out = tf.identity(imgs_out, name = "diff_out") # [batch_size, number of channels, height, width]
    maps_out = tf.identity(maps_out, name = "diff_map_out") # [batch_size, number of local latent vectors, number of layers, heads_num, height, width]

    return imgs_out, maps_out
    
