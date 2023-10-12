<p align="center">
  <a href="" rel="noopener">
</p>

<h3 align="center">Self-Supervised MRI Reconstruction with Unrolled Diffusion Models</h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---

<p align="center"> Official Tensorflow implementation of SSDiffRecon (MICCAI2023)
    <br> 
</p>


## About <a name = "about"></a>

Magnetic Resonance Imaging (MRI) produces excellent soft tissue contrast, albeit it is an inherently slow imaging modality. Promising deep learning methods have recently been proposed to reconstruct accelerated MRI scans. However, existing methods still suffer from various limitations regarding image fidelity, contextual sensitivity, and reliance on fully-sampled acquisitions for model training. To comprehensively address these limitations, we propose a novel self-supervised deep reconstruction model, named Self-Supervised Diffusion Reconstruction (SSDiffRecon). SSDiffRecon expresses a conditional diffusion process as an unrolled architecture that interleaves cross-attention transformers for reverse diffusion steps with data-consistency blocks for physics-driven processing. Unlike recent diffusion methods for MRI reconstruction, a self-supervision strategy is adopted to train SSDiffRecon using only undersampled k-space data. Comprehensive experiments on public brain MR datasets demonstrates the superiority of SSDiffRecon against state-of-the-art supervised, and self-supervised baselines in terms of reconstruction speed and quality. 

## Prerequisites <a name = "Prerequisites"></a>
Required packages can easily be installed via conda:
```
conda env create -f environment.yml
```
Then:
```
conda activate ssdiffrecon_env
```
Tensorflow 1.14+ should also work fine since we do not use TF2 specific functionalities. 

## Datasets

1) IXI dataset: https://brain-development.org/ixi-dataset/
2) fastMRI Brain dataset: https://fastmri.med.nyu.edu/

For IXI dataset image dimensions are 256x256. For fastMRI dataset image dimensions vary with contrasts. (T1: 256x320, T2: 288x384, FLAIR: 256x320).

Tensorflow requires datasets in the tfrecords format. To create tfrecords file containing new datasets you can use dataset_tool.py.

Tfrecords files created for fastMRI and IXI datasets can be downloaded from the link:

https://drive.google.com/drive/folders/1h1kt8b4JgPOG-tNtRxAeEfMLUMIEzw9r?usp=drive_link

Coil-sensitivity-maps are estimated using ESPIRIT (http://people.eecs.berkeley.edu/~mlustig/Software.html). 

## Run Commands
After setting up the environment and downloading dataset files, you can simply run the following commands.

To train the single-coil model (IXI) with default parameters:
```
python run_ixi.py --train --exp_name ixi_trial1 --gpu 1
```
To train the multi-coil model (fastMRI) with learning rate 1e-5, run the following:
```
python run_fastmri.py --train --exp_name fastmri_trial1 --gpu 1 --lr 1e-5
```
To evaluate the single-coil model (IXI) with default parameters using the checkpoint at 1000th step of training, run the following:
```
python run_ixi.py --eval --results_dir ./results/ixi_trial1 --eval_checkpoint 1000
```
To evaluate the multi-coil model (fastMRI) with a beta_start parameter 0.005, run the following:
```
python run_fastmri.py --eval --results_dir ./results/fastmri_trial1 --beta_start 0.005
```
## Trained Models

Trained model weights for both datasets can be downloaded from this link: https://drive.google.com/drive/folders/1ApxzBWqyD7Km0vAm-pILCyN6nvlsfjSg?usp=drive_link

## Citation 

You are encouraged to modify/distribute this code. However, please acknowledge this code and cite the paper appropriately.
```
@article{korkmaz2023self,
  title={Self-Supervised MRI Reconstruction with Unrolled Diffusion Models},
  author={Korkmaz, Yilmaz and Cukur, Tolga and Patel, Vishal M.},
  journal={arXiv preprint arXiv:2306.16654},
  year={2023}
}
```

## Acknowledgements

This github page utilizes libraries from https://github.com/hojonathanho/diffusion/tree/master and https://github.com/icon-lab/SLATER/tree/main.

