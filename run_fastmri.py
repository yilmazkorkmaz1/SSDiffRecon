import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import functools
import argparse
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from diffusion.diffusion_tf.diffusion_utils import get_beta_schedule, GaussianDiffusion2
from diffusion.diffusion_tf.models.model_multicoil import SSDiffRecon_Model_Multicoil
from diffusion.diffusion_tf.gpu_utils import gpu_tpu_utils_fastmri as gpu_utils
from diffusion.diffusion_tf.gpu_utils import datasets
import partial_masks


class Model(gpu_utils.Model):
  
  def __init__(self, *, model_mean_type, model_var_type, betas: np.ndarray):

    self.diffusion = GaussianDiffusion2(
      betas=betas, model_mean_type=model_mean_type, model_var_type=model_var_type)
    self.model_class = SSDiffRecon_Model_Multicoil()

  def get_trainables(self):
      return self.model_class.get_trainable_variables()


  def _denoise(self, x, us_im, t, y, mask, coil_map):
    B = x.shape[0]
    assert x.dtype == tf.float32
    assert t.shape == [B] and t.dtype in [tf.int32, tf.int64]

    out = self.model_class.model(us_im=us_im, noisy_sample=x, label=y, time=t, mask=mask, coil_map=coil_map)
    return out

  def train_fn(self, us_im, y, mask, coil_map, alpha=0.95):
    B = us_im.shape[0]
    t = tf.random_uniform([B], 0, self.diffusion.num_timesteps, dtype=tf.int32)

    new_mask, loss_mask = partial_masks.partial_mask_creator(mask, alpha)
    new_us_im = partial_masks.us_im_creator_fastmri(new_mask=new_mask, us_im=us_im, coil_map=coil_map)
    x_start = new_us_im

    losses = self.diffusion.training_losses_fastmri_ssdu(
      denoise_fn=functools.partial(self._denoise,  us_im=new_us_im, y=y, mask=new_mask, coil_map=coil_map), 
      x_start=x_start, 
      t=t,
      loss_mask=loss_mask,
      us_im=us_im,
      coil_map=coil_map,
      label=y)

    assert losses.shape == t.shape == [B]
    return {'loss': tf.reduce_mean(losses)}

  def samples_fn(self, dummy_noise, y, us_im, mask, coil_map):
    sample = self.diffusion.p_sample_loop(
          denoise_fn=functools.partial(self._denoise, y=y, us_im=us_im, mask=mask, coil_map=coil_map),
          shape=dummy_noise.shape.as_list(),
          us_im=us_im,
          noise_fn=tf.random_normal)
    return {'samples': sample}


def evaluation(args):
  ds = datasets.get_dataset(args.dataset, batch_size=args.batch_size, phase='test')
  worker = gpu_utils.EvalWorker(
    model_constructor=lambda: Model(
     model_mean_type='xstart',
     model_var_type='fixedlarge',
     betas=get_beta_schedule(
        args.beta_schedule, beta_start=args.beta_start, beta_end=args.beta_end, num_diffusion_timesteps=args.num_diffusion_timesteps
      ),
    ),
    total_bs=args.batch_size, dataset=ds)
  worker.run(logdir=args.results_dir)


def train(args):
  ds = datasets.get_dataset(args.dataset, batch_size=args.batch_size, phase='train')
  gpu_utils.run_training(
    exp_name=args.exp_name,
    model_constructor=lambda: Model(
     model_mean_type='xstart',
     model_var_type='fixedlarge',
      betas=get_beta_schedule(
        args.beta_schedule, beta_start=args.beta_start, beta_end=args.beta_end, num_diffusion_timesteps=args.num_diffusion_timesteps
      ),
    ),
    optimizer=args.optimizer, total_bs=args.batch_size, lr=args.lr, warmup=args.warmup, grad_clip=args.grad_clip,
    train_input_fn=ds.train_input_fn, log_dir=args.results_dir
  )


def get_args_parser():
      parser = argparse.ArgumentParser('SSDiffRecon train and evaluate for fastMRI', add_help=False)
      parser.add_argument('--train', action='store_true', default=False)
      parser.add_argument('--eval', action='store_true', default=False)
      parser.add_argument('--results_dir', type=str, default="../results/")
      parser.add_argument('--exp_name', type=str, default="")
      parser.add_argument('--gpu', type=str, default='0')
      parser.add_argument('--dataset', type=str, default='fastMRI')
      parser.add_argument('--batch_size', type=int, default=1)
      parser.add_argument('--optimizer', type=str, default='adam')
      parser.add_argument('--grad_clip', type=float, default=1.)
      parser.add_argument('--lr', type=float, default=2e-3)
      parser.add_argument('--warmup', type=int, default=5000)
      parser.add_argument('--num_diffusion_timesteps', type=int, default=1000)
      parser.add_argument('--beta_start', type=float, default=0.0001)
      parser.add_argument('--beta_end', type=int, default=0.02)
      parser.add_argument('--beta_schedule', type=str, default='linear')
      parser.add_argument('--eval_checkpoint',type=str,default='')
      
      return parser


if __name__ == '__main__':
       
    args = get_args_parser()
    args = args.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    if args.train:
       train(args)
    elif args.eval:
       args.num_diffusion_timesteps=5   
       evaluation(args)
    else:
       print("specify mode")
