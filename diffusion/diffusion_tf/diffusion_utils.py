# Adapted from https://github.com/hojonathanho/diffusion
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def meanflat(x):
  return tf.reduce_mean(x, axis=list(range(1, len(x.shape))))

def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
  betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
  warmup_time = int(num_diffusion_timesteps * warmup_frac)
  betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
  return betas


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
  if beta_schedule == 'quad':
    betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
  elif beta_schedule == 'linear':
    betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
  elif beta_schedule == 'warmup10':
    betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
  elif beta_schedule == 'warmup50':
    betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
  elif beta_schedule == 'const':
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
  elif beta_schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
    betas = 1. / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
  else:
    raise NotImplementedError(beta_schedule)
  assert betas.shape == (num_diffusion_timesteps,)
  return betas


class GaussianDiffusion2:
  """
  Contains utilities for the diffusion model.

  Arguments:
  - what the network predicts (x_{t-1}, x_0, or epsilon)
  - which loss function (kl or unweighted MSE)
  - what is the variance of p(x_{t-1}|x_t) (learned, fixed to beta, or fixed to weighted beta)
  - what type of decoder, and how to weight its loss? is its variance learned too?
  """

  def __init__(self, *, betas, model_mean_type, model_var_type):
    self.model_mean_type = model_mean_type  # xprev, xstart, eps
    self.model_var_type = model_var_type  # learned, fixedsmall, fixedlarge

    assert isinstance(betas, np.ndarray)
    self.betas = betas = betas.astype(np.float64)  # computations here in float64 for accuracy
    assert (betas > 0).all() and (betas <= 1).all()
    timesteps, = betas.shape
    self.num_timesteps = int(timesteps)

    alphas = 1. - betas
    self.alphas_cumprod = np.cumprod(alphas, axis=0)
    self.alphas_cumprod_prev = np.append(1., self.alphas_cumprod[:-1])
    assert self.alphas_cumprod_prev.shape == (timesteps,)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
    self.sqrt_one_minus_alphas_cumprod = np.sqrt(1. - self.alphas_cumprod)
    self.log_one_minus_alphas_cumprod = np.log(1. - self.alphas_cumprod)
    self.sqrt_recip_alphas_cumprod = np.sqrt(1. / self.alphas_cumprod)
    self.sqrt_recipm1_alphas_cumprod = np.sqrt(1. / self.alphas_cumprod - 1)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    self.posterior_variance = betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
    self.posterior_log_variance_clipped = np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
    self.posterior_mean_coef1 = betas * np.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1. - self.alphas_cumprod)

  @staticmethod
  def _extract(a, t, x_shape):
    """
    Extract some coefficients at specified timesteps,
    then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    bs, = t.shape

    out = tf.gather(tf.convert_to_tensor(a, dtype=tf.float32), t)
 
    return tf.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))

  def q_mean_variance(self, x_start, t):
    mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
    variance = self._extract(1. - self.alphas_cumprod, t, x_start.shape)
    log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
    return mean, variance, log_variance

  def q_sample(self, x_start, t, noise=None):
    """
    Diffuse the data (t == 0 means diffused for 1 step)
    """
    if noise is None:
      noise = tf.random_normal(shape=x_start.shape)
  #  assert noise.shape == x_start.shape
    return (
        self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
        self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
    )

  def q_posterior_mean_variance(self, x_start, x_t, t):
    """
    Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
    """

    posterior_mean = (
        self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
        self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
    )
    posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
    posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
    assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] ==
            x_start.shape[0])
    return posterior_mean, posterior_variance, posterior_log_variance_clipped

  def p_mean_variance(self, denoise_fn, *, x, t,us_im=None, clip_denoised: bool, return_pred_xstart: bool):
    B, H, W, C = x.shape
    assert t.shape == [B]
    pred = tf.math.equal(t[0],tf.constant(0, dtype=tf.int32))
    if us_im != None:
      new_us_im = tf.cond(pred, lambda:us_im, lambda:self.q_sample(x_start=us_im, t=t,noise=tf.random_normal(shape=us_im.shape, mean=0, stddev=0.01)))
      model_output = denoise_fn(x=x, t=t, us_im=new_us_im)
    else:
      model_output = denoise_fn(x=x, t=t)

    # Learned or fixed variance?
    if self.model_var_type == 'learned':
      assert model_output.shape == [B, H, W, C * 2]
      model_output, model_log_variance = tf.split(model_output, 2, axis=-1)
      model_variance = tf.exp(model_log_variance)
    elif self.model_var_type in ['fixedsmall', 'fixedlarge']:
      # below: only log_variance is used in the KL computations
      model_variance, model_log_variance = {
        # for fixedlarge, we set the initial (log-)variance like so to get a better decoder log likelihood
        'fixedlarge': (self.betas, np.log(np.append(self.posterior_variance[1], self.betas[1:]))),
        'fixedsmall': (self.posterior_variance, self.posterior_log_variance_clipped),
      }[self.model_var_type]
      model_variance = self._extract(model_variance, t, x.shape) * tf.ones(x.shape.as_list())
      model_log_variance = self._extract(model_log_variance, t, x.shape) * tf.ones(x.shape.as_list())
    else:
      raise NotImplementedError(self.model_var_type)

    # Mean parameterization
    _maybe_clip = lambda x_: (tf.clip_by_value(x_, -1., 1.) if clip_denoised else x_)
    if self.model_mean_type == 'xprev':  # the model predicts x_{t-1}
      pred_xstart = _maybe_clip(self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output))
      model_mean = model_output
    elif self.model_mean_type == 'xstart':  # the model predicts x_0
      pred_xstart = _maybe_clip(model_output)
      model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
    elif self.model_mean_type == 'eps':  # the model predicts epsilon
      pred_xstart = _maybe_clip(self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output))
      model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
    else:
      raise NotImplementedError(self.model_mean_type)

    #assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
    if return_pred_xstart:
      return model_mean, model_variance, model_log_variance, pred_xstart
    else:
      return model_mean, model_variance, model_log_variance

  def _predict_xstart_from_eps(self, x_t, t, eps):
    assert x_t.shape == eps.shape
    return (
        self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
        self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
    )

  def _predict_xstart_from_xprev(self, x_t, t, xprev):
    assert x_t.shape == xprev.shape
    return (  # (xprev - coef2*x_t) / coef1
        self._extract(1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
        self._extract(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape) * x_t
    )

  # === Sampling ===

  def p_sample(self, denoise_fn, *, x, t, noise_fn, us_im=None, clip_denoised=True, return_pred_xstart: bool):
    """
    Sample from the model
    """
    model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
      denoise_fn, x=x, t=t, us_im=us_im, clip_denoised=clip_denoised, return_pred_xstart=True)
    noise = noise_fn(shape=x.shape, dtype=x.dtype)
   
    # no noise when t == 0
    nonzero_mask = tf.reshape(1 - tf.cast(tf.equal(t, 0), tf.float32), [x.shape[0]] + [1] * (len(x.shape) - 1))
    sample = model_mean + nonzero_mask * tf.exp(0.5 * model_log_variance) * noise
 
    return (sample, pred_xstart) if return_pred_xstart else sample

  def p_sample_loop(self, denoise_fn, *, shape, us_im, noise_fn=tf.random_normal):
    """
    Generate samples
    """

    i_0 = tf.constant(self.num_timesteps - 1, dtype=tf.int32)

    img_0 = us_im

    _, img_final = tf.while_loop(
      cond=lambda i_, _: tf.greater_equal(i_, 0),
      body=lambda i_, img_: [
        i_ - 1,
        self.p_sample(
          denoise_fn=denoise_fn, x=img_, t=tf.fill([shape[0]], i_), us_im=us_im, noise_fn=noise_fn, return_pred_xstart=False)
      ],
      loop_vars=[i_0, img_0],
      shape_invariants=[i_0.shape, img_0.shape],
      back_prop=False
    )
  #  assert img_final.shape == shape
    return img_final

  def p_sample_loop_progressive(self, denoise_fn, *, shape, us_im, noise_fn=tf.random_normal, include_xstartpred_freq=50):
    """
    Generate samples and keep track of prediction of x0
    """

    i_0 = tf.constant(self.num_timesteps - 1, dtype=tf.int32)
  
    img_0 = us_im

    num_recorded_xstartpred = self.num_timesteps // include_xstartpred_freq
    xstartpreds_0 = tf.zeros([shape[0], num_recorded_xstartpred, *shape[1:]], dtype=tf.float32)  # [B, N, H, W, C]

    def _loop_body(i_, img_, xstartpreds_):
      # Sample p(x_{t-1} | x_t) as usual
      sample, pred_xstart = self.p_sample(
        denoise_fn=denoise_fn, x=img_, t=tf.fill([shape[0]], i_), us_im=us_im, noise_fn=noise_fn, return_pred_xstart=True)

      # Keep track of prediction of x0
      insert_mask = tf.equal(tf.floordiv(i_, include_xstartpred_freq),
                             tf.range(num_recorded_xstartpred, dtype=tf.int32))
      insert_mask = tf.reshape(tf.cast(insert_mask, dtype=tf.float32),
                               [1, num_recorded_xstartpred, *([1] * len(shape[1:]))])  # [1, N, 1, 1, 1]
      new_xstartpreds = insert_mask * pred_xstart[:, None, ...] + (1. - insert_mask) * xstartpreds_
      return [i_ - 1, sample, new_xstartpreds]

    _, img_final, xstartpreds_final = tf.while_loop(
      cond=lambda i_, img_, xstartpreds_: tf.greater_equal(i_, 0),
      body=_loop_body,
      loop_vars=[i_0, img_0, xstartpreds_0],
      shape_invariants=[i_0.shape, img_0.shape, xstartpreds_0.shape],
      back_prop=False
    )

    return img_final, xstartpreds_final  # xstart predictions should agree with img_final at step 0

  def p_sample_loop_trajectory(self, denoise_fn, *, shape, us_im, noise_fn=tf.random_normal, repeat_noise_steps=-1):
      """
      Generate samples, returning intermediate images
      Useful for visualizing how denoised images evolve over time
      Args:
        repeat_noise_steps (int): Number of denoising timesteps in which the same noise
          is used across the batch. If >= 0, the initial noise is the same for all batch elemements.
      """
      i_0 = tf.constant(self.num_timesteps - 1, dtype=tf.int32)

      img_0 = us_im
      times = tf.Variable([i_0])
      imgs = tf.Variable([img_0])
      # Steps with different noise for each batch element
      times, imgs = tf.while_loop(
        cond=lambda times_, _: tf.greater_equal(times_[-1], 0),
        body=lambda times_, imgs_: [
          tf.concat([times_, [times_[-1] - 1]], 0),
          tf.concat([imgs_, [self.p_sample(denoise_fn=denoise_fn,
                                          us_im = us_im,
                                          x=imgs_[-1],
                                          t=tf.fill([shape[0]], times_[-1]),
                                          noise_fn=noise_fn,return_pred_xstart=False)]], 0)
        ],
        loop_vars=[times, imgs],
        shape_invariants=[tf.TensorShape([None, *i_0.shape]),
                          tf.TensorShape([None, *img_0.shape])],
        back_prop=False
      )
 
      return times, imgs
  

  def training_losses_ixi_ssdu(self, denoise_fn, x_start, t, us_im, loss_mask, noise=None):

    # Add noise to data
    assert t.shape == [x_start.shape[0]]
    if noise is None:
      noise = tf.random_normal(shape=x_start.shape, dtype=x_start.dtype)
    assert noise.shape == x_start.shape and noise.dtype == x_start.dtype
    x_t = self.q_sample(x_start=x_start, t=t, noise=noise)

    # Calculate the loss
    model_output = denoise_fn(x=x_t, t=t)
    losses = self.partial_kspace_loss_ixi(us_im=us_im, loss_mask=loss_mask, generated = model_output)
    assert losses.shape == t.shape
    return losses


  def training_losses_fastmri_ssdu(self, denoise_fn, x_start, t, us_im, coil_map, label, loss_mask, noise=None):
    """
    Training loss calculation
    """

    if noise is None:
      noise = tf.random_normal(shape=tf.shape(x_start), dtype=x_start.dtype)
    x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
    assert self.model_var_type != 'learned'

    model_output = denoise_fn(x=x_t, t=t)
    partial_kspace_loss = self.partial_kspace_loss_fastmri(us_im, coil_map, loss_mask, model_output)
    losses = partial_kspace_loss

    assert losses.shape == t.shape
    return losses

  def fft2c(self, im):
      return tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(im, axes=[-1,-2])), axes=[-1,-2]) 

  def ifft2c(self, d):
      return tf.signal.fftshift(tf.signal.ifft2d(tf.signal.ifftshift(d, axes=[-1,-2])), axes=[-1,-2])


  
  def partial_kspace_loss_ixi(self, us_im, loss_mask, generated):
        
      us_im_real = us_im[:,0,:,:]
      us_im_imag = us_im[:,1,:,:]
      us_im = tf.complex(us_im_real, us_im_imag)

      gen_real = generated[:,0,:,:]
      gen_imag = generated[:,1,:,:]
      generated = tf.complex(gen_real, gen_imag)      

      us_im_kspace = self.fft2c(us_im)
      generated_im_kspace = self.fft2c(generated)
      loss_mask = tf.cast(loss_mask, dtype=tf.complex64)
      loss = tf.abs(us_im_kspace * loss_mask - generated_im_kspace * loss_mask)

      return meanflat(loss)
  

  def partial_kspace_loss_fastmri(self, us_im, coil_map, loss_mask, generated):
      coil_map = tf.complex(coil_map[:, 0] , coil_map[:, 1])

      pad_x = tf.cast((tf.shape(us_im)[2] - tf.shape(coil_map)[2]) / 2, dtype=tf.int32)
      pad_y = tf.cast((tf.shape(us_im)[3] - tf.shape(coil_map)[3]) / 2, dtype=tf.int32)
      us_im = tf.complex(us_im[:,0],us_im[:,1])
      generated = tf.complex(generated[:,0],generated[:,1])
  
      indices_x, indices_y = tf.meshgrid(tf.range(pad_x, tf.shape(us_im)[1]-pad_x), tf.range(pad_y, tf.shape(us_im)[2]-pad_y),  indexing='ij')

      batch_size = tf.shape(us_im)[0]
      indices_x = tf.tile(tf.expand_dims(indices_x, axis=0), [batch_size, 1, 1])
      indices_y = tf.tile(tf.expand_dims(indices_y, axis=0), [batch_size, 1, 1])
      indices = tf.stack([indices_x, indices_y], axis=-1)

      us_im = tf.gather_nd(indices=indices, params=us_im, batch_dims=1) 
      generated = tf.gather_nd(indices=indices, params=generated, batch_dims=1) 

      us_im = tf.tile(tf.expand_dims(us_im, axis=1), [1,5,1,1])
      generated = tf.tile(tf.expand_dims(generated, axis=1), [1,5,1,1])

      us_im_kspace = self.fft2c(us_im * coil_map)
      generated_im_kspace = self.fft2c(generated * coil_map)


      loss_mask = tf.cast(loss_mask, dtype=tf.complex64)
      loss = tf.abs(us_im_kspace * loss_mask - generated_im_kspace * loss_mask)

      return meanflat(loss)
