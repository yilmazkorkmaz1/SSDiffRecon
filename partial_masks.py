import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# centered fft
def fft2c(im):
    return tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(im, axes=[-1,-2])), axes=[-1,-2]) 
# centered ifft
def ifft2c(d):
    return tf.signal.fftshift(tf.signal.ifft2d(tf.signal.ifftshift(d, axes=[-1,-2])), axes=[-1,-2])


# creates a new (more) undersampled image for ixi
def us_im_creator_ixi(new_mask, us_im):
    us_im = tf.complex(us_im[:,0,:,:], us_im[:,1,:,:])
    us_im_kspace = fft2c(us_im)
    us_im_kspace = us_im_kspace * tf.cast(new_mask, dtype=tf.complex64)
    new_us_im = ifft2c(us_im_kspace)
    return tf.concat([tf.real(new_us_im), tf.imag(new_us_im)], axis=1)

# creates a couple of new us and loss masks at the given rate of alpha
def partial_mask_creator(mask, alpha):

  mask_mean = tf.reduce_mean(mask)

  org_shape = tf.shape(mask)

  mask = tf.reshape(mask, shape=[-1])

  indexes_non_zero = mask_mean * tf.cast(tf.shape(mask), tf.float32)
  sample_num = tf.cast(indexes_non_zero * alpha, dtype=tf.int32)
  idxs = tf.where(mask>0)
  ridxs = tf.cast(tf.gather(tf.random.shuffle(idxs), tf.range(0,tf.squeeze(sample_num),1)), dtype=tf.int32)
  zero_array = tf.zeros(tf.shape(mask))
  zero_array = tf.tensor_scatter_nd_update(zero_array, ridxs, tf.ones(sample_num))

  new_mask = tf.reshape(zero_array, org_shape)
  mask = tf.reshape(mask, org_shape)
  loss_mask = mask - new_mask

  return new_mask, loss_mask

# creates a new (more) undersampled image for fastmri (requires crop/padding)
def us_im_creator_fastmri(new_mask, us_im, coil_map):
    coil_map = tf.complex(coil_map[:, 0, :, :, :] , coil_map[:, 1, :, :, :])

    pad_x = tf.cast((tf.shape(us_im)[2] - tf.shape(coil_map)[2]) / 2, dtype=tf.int32)
    pad_y = tf.cast((tf.shape(us_im)[3] - tf.shape(coil_map)[3]) / 2, dtype=tf.int32)

    us_im = tf.complex(us_im[:,0],us_im[:,1])

    indices_x, indices_y = tf.meshgrid(tf.range(pad_x, tf.shape(us_im)[1]-pad_x), tf.range(pad_y, tf.shape(us_im)[2]-pad_y),  indexing='ij')

    batch_size = tf.shape(new_mask)[0]
    indices_x = tf.tile(tf.expand_dims(indices_x, axis=0), [batch_size, 1, 1])
    indices_y = tf.tile(tf.expand_dims(indices_y, axis=0), [batch_size, 1, 1])
    indices = tf.stack([indices_x, indices_y], axis=-1)
    
    us_im = tf.gather_nd(indices=indices, params=us_im, batch_dims=1) 

    new_mask = tf.tile(new_mask, [1,5,1,1])
    us_im = tf.tile(tf.expand_dims(us_im, axis=1), [1,5,1,1])

    us_im_kspace = fft2c(us_im * coil_map)
    us_im_kspace = us_im_kspace * tf.cast(new_mask, dtype=tf.complex64)
    new_us_im = ifft2c(us_im_kspace)
    new_us_im = tf.reduce_sum(new_us_im * tf.math.conj(coil_map),axis=1)
    paddings = tf.convert_to_tensor([[0,0], [pad_x, pad_x], [pad_y, pad_y]])
    sep_im_real = tf.pad(tf.real(new_us_im), paddings, "CONSTANT")
    sep_im_imag = tf.pad(tf.imag(new_us_im), paddings, "CONSTANT")
    return tf.stack([sep_im_real, sep_im_imag], axis=1)
   