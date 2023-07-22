# Adapted from https://github.com/hojonathanho/diffusion to work on a single GPU instead of TPU
import os
import time
import numpy as np
import glob
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from natsort import natsorted # pip install natsort


from .tpu_summaries import TpuSummaries
from .. import utils


# ========== Training ==========

normalize_data = lambda x_: (x_-0.5)*2
unnormalize_data = lambda x_: (x_ + 1.) / 2
unnormalize_data_255 = lambda x: (x+1.) * 255/2


class Model:
  # All images (inputs and outputs) should be normalized to [-1, 1]
  def train_fn(self, x, y) -> dict:
    raise NotImplementedError

  def samples_fn(self, dummy_x, y) -> dict:
    raise NotImplementedError

  def bpd_fn(self, x, y) -> dict:
    return None


def make_ema(global_step, ema_decay, trainable_variables):
  ema = tf.train.ExponentialMovingAverage(decay=tf.where(tf.less(global_step, 1), 1e-10, ema_decay))
  ema_op = ema.apply(trainable_variables)
  return ema, ema_op


def run_training(
    *, model_constructor, train_input_fn, total_bs,
    optimizer, lr, warmup, grad_clip, ema_decay=0.9999,
    log_dir, exp_name, max_steps=int(1e10)
):
  tf.logging.set_verbosity(tf.logging.INFO)

  # Create checkpoint directory
  model_dir = os.path.join(log_dir + '/' +  exp_name)
  print('model dir:', model_dir)
  if tf.io.gfile.exists(model_dir):
    print('model dir already exists: {}'.format(model_dir))
    if input('continue training? [y/n] ') != 'y':
      print('aborting')
      return


  # model_fn for TPUEstimator
  def model_fn(features, params, mode,labels):        
    num_cores = 1
    local_bs = total_bs // num_cores
    print('Global batch size: {}, local batch size: {}'.format(total_bs, local_bs))
    assert mode == tf.estimator.ModeKeys.TRAIN, 'only TRAIN mode supported'
    assert features['us_im'].shape[0] == local_bs
   
    features['us_im'].set_shape([local_bs,2,512,512])
    features['mask'].set_shape([local_bs,1,None,None])
    features['coil_map'].set_shape([local_bs,2,5,None,None])



    del params

    ##########

    # create model
    model = model_constructor()
    assert isinstance(model, Model)
    # training loss
    train_info_dict = model.train_fn(tf.cast(features['us_im'], tf.float32), 
    labels['label'], tf.cast(features['mask'], tf.float32), tf.cast(features["coil_map"], tf.float32))
    loss = train_info_dict['loss']
    assert loss.shape == []

    # train op
    trainable_variables = model.get_trainables()
    print('num params: {:,}'.format(sum(int(np.prod(p.shape.as_list())) for p in trainable_variables)))
    global_step = tf.train.get_or_create_global_step()
    warmed_up_lr = utils.get_warmed_up_lr(max_lr=lr, warmup=warmup, global_step=global_step)
    train_op, gnorm = utils.make_optimizer(
      loss=loss,
      trainable_variables=trainable_variables,
      global_step=global_step,
      lr=warmed_up_lr,
      optimizer=optimizer,
      grad_clip=grad_clip
    )


    # ema
    ema, ema_op = make_ema(global_step=global_step, ema_decay=ema_decay, trainable_variables=trainable_variables)
    with tf.control_dependencies([train_op]):
      train_op = tf.group(ema_op)


    # summary
    tpu_summary = TpuSummaries(model_dir, save_summary_steps=100)
    tpu_summary.scalar('train/loss', loss)
    tpu_summary.scalar('train/gnorm', gnorm)
    tpu_summary.scalar('train/pnorm', utils.rms(trainable_variables))
    tpu_summary.scalar('train/lr', warmed_up_lr)
    return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, train_op=train_op)

  # Set up Estimator and train

  estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    config=tf.estimator.RunConfig(
      model_dir=model_dir,
      session_config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True),
      save_checkpoints_secs=1600, 
      keep_checkpoint_max=10000
    ),
    warm_start_from=None
  )
  estimator.train(input_fn=train_input_fn, max_steps=max_steps)


# ========== Evaluation / sampling ==========
class EvalWorker:
  def __init__(self, model_constructor, total_bs, dataset):

    self.total_bs = total_bs
    self.num_cores = 1
    self.local_bs = total_bs // self.num_cores
    print('num cores: {}'.format(self.num_cores))
    print('total batch size: {}'.format(self.total_bs))
    print('local batch size: {}'.format(self.local_bs))
    self.dataset = dataset.eval_input_fn()
    self.it = tf.data.Iterator.from_structure(self.dataset.output_types, self.dataset.output_shapes)
    self.init_dataset = self.it.make_initializer(self.dataset)

    img_batch_shape = [self.local_bs,2,512,512]

    # Model
    self.model = model_constructor()
    assert isinstance(self.model, Model)

    # Eval/samples graphs
    self.samples_outputs = self._make_sampling_graph(
      img_shape=img_batch_shape[1:])

    # Model with EMA parameters
    self.global_step = tf.train.get_or_create_global_step()
    ema, _ = make_ema(global_step=self.global_step, ema_decay=1e-10, trainable_variables=self.model.get_trainables())

    # # EMA versions of the above
    with utils.ema_scope(ema):
      self.ema_samples_outputs= self._make_sampling_graph(
        img_shape=img_batch_shape[1:])

  def _make_sampling_graph(self, img_shape):

    def _make_inputs(total_bs, local_bs):
      # Dummy inputs to feed to samplers
      input_x = tf.random.normal([local_bs, *img_shape])
      input_mask_and_us_and_coil_map, input_y = self.it.get_next()
      input_mask_and_us_and_coil_map['us_im'].set_shape([local_bs,2,512,512])
      input_mask_and_us_and_coil_map['mask'].set_shape([local_bs,1,None,None])
      input_mask_and_us_and_coil_map['coil_map'].set_shape([local_bs, 2, 5, None, None])

      return input_x, input_mask_and_us_and_coil_map, input_y

    # Samples
    x,im_us_and_mask_and_coil_map,y = _make_inputs(self.total_bs, self.local_bs)
    samples_outputs = self.model.samples_fn(dummy_noise=x,y=y['label'], us_im=im_us_and_mask_and_coil_map['us_im'], mask=im_us_and_mask_and_coil_map['mask'], coil_map=im_us_and_mask_and_coil_map['coil_map'])
    samples_outputs_abs = {'samples':tf.abs(tf.complex(samples_outputs['samples'][0,0,:,:],samples_outputs['samples'][0,1,:,:]))}
    return samples_outputs_abs


  def _run_sampling(self, sess, ema: bool):
    out = {}
    print('sampling...')
    tstart = time.time()
    
    samples = sess.run(self.ema_samples_outputs if ema else self.samples_outputs)
    print('sampling done in {} sec'.format(time.time() - tstart))
    for k, v in samples.items():
      out[k] = v
    return out



  def _write_eval_and_samples(self, sess, log: utils.SummaryWriter, curr_step, prefix, ema: bool, idx):
    # Samples
    direc = self._run_sampling(sess, ema=ema).items()
    for k, v in direc:
      np.save(self.logdir + "/test/im_" + prefix + str(idx) + k + ".npy", v)
    log.flush()


  def run(self, logdir, seed=0, eval_checkpoint=""):
    """Runs the eval/sampling worker loop.
    Args:
      logdir: directory to read checkpoints from
      eval_checkpoint: checkpoint to load weights from
    """
    self.logdir = logdir
    path = self.logdir + "/test"
    isExist = os.path.exists(path)
    if not isExist:

   # Create a new directory because it does not exist
      os.makedirs(path)

    path = self.logdir + "/val"
    isExist = os.path.exists(path)
    if not isExist:

   # Create a new directory because it does not exist
      os.makedirs(path)
      
      
    tf.logging.set_verbosity(tf.logging.INFO)


    assert tf.io.gfile.isdir(logdir), 'expected {} to be a directory'.format(logdir)

    # Set up eval SummaryWriter
    eval_logdir = os.path.join(logdir, 'eval')
    print('Writing eval data to: {}'.format(eval_logdir))
    eval_log = utils.SummaryWriter(eval_logdir, write_graph=False)
    for ckpt in glob.glob(logdir + "/*" + eval_checkpoint + ".index"):
      print(ckpt)
    # Make the session
    config = tf.ConfigProto()
    config.allow_soft_placement = True
   
    print('making session...')
    with tf.Session(config=config) as sess:
      sess.run(self.init_dataset)
      new_variables = []
      sess.run(tf.global_variables_initializer())
      vars_to_load = [i[0] for i in tf.train.list_variables(ckpt[:-6])]
      for variable in tf.global_variables():
        if variable.op.name in vars_to_load:
          print("listed")
          print(variable.op.name)
          new_variables.append(variable)
          time.sleep(0.05)
        else:
          print("not listed")
          print(variable.op.name)
          time.sleep(0.1)
      

      saver = tf.train.Saver(new_variables)
        

      # if eval checkpoint is not specified takes the last
      for ckpt in natsorted(glob.glob(logdir + "/*" + eval_checkpoint + ".index"), reverse=True):
        
        print("checkpoint loaded: ", ckpt)
        sess.run(self.init_dataset)
        saver.restore(sess, ckpt[:-6])

        global_step_val = sess.run(self.global_step)
        print('initializing global variables')
        print('restored global step: {}'.format(global_step_val))

        print('seeding')
        utils.seed_all(seed)
        for idx in range(400*6):
          self._write_eval_and_samples(sess, log=eval_log, curr_step=global_step_val, prefix='eval', ema=False, idx=idx)
        exit()
