

## Ultra-fast in vivo deep-tissue 3D imaging with selective-illumination NIR-II light-field microscopy and aberration-corrected implicit neural representation reconstruction (AIR)


AIR-net is a self-supervised deep learning-based light field reconstruction algorithm with adaptive aberration correction, 
specifically designed for in vivo imaging applications in light field microscopy.


## System Requirements
- Windows 10. Linux should be able to run the Python-based codes.
- Graphics: Nvidia GPU with >16 GB graphic memory (RTX 3090 recommended or multi-GPUs for parallel computing)
- Memory: >=32 GB RAM 
- Hard Drive: ~50GB free space (SSD recommended)

## Dependencies
```
  - python=3.10.8
  - cudatoolkit=11.8
  - pytorch=2.0.1
  - torchvision=0.15.2
  - tensorboard==2.10.1
  - scipy==1.12.0
  - imageio
  - tifffile
  - six==1.16
  - tqdm==4.64.1
  - mat73==0.59
```

***Note: We provide a packages' list (".code/create_environment.yml"), containing the required dependencies can help users to fast install the running environment.
To use this, run the following commend inside a [conda](https://docs.conda.io/en/latest/) console. This step will cost ~10 minutes (depend on the network quality)***
  ```
  conda env create -f create_environment.yml
  ```

## Overview of repository
```
├── code:
    The source codes of AIR-net 
    └── configs:
            The configuration files of different sample.
    └── core: 
            Network model and image-processing codes
    └── data 
            Contains real-captured and simulated light field images along with corresponding ideal PSFs (Point Spread Functions) as network inputs.
    └── logs 
            Contains pre-trained network weights.
    └── twophoton_data 
            Contains three-dimensional data of mouse cerebrovascular system captured by two-photon microscopy, used for simulation experiments.
    '.py' files:
    · glbSettings.py  (Device settings)
    · main.py (main function)
    · test.py (network inference with trained weights)
    · train_calibrate.py (train AIR-net)
    . sim_data_gen.py (Generate PSFs with random aberrations and the corresponding projected light field image, as shown in Fig. 10 and Fig. 11 of the supplementary files.)
    
├── FLFM_simulation
    The codes for FLFM PSF simulation and light-field deconvolution with/without DAO
```

## Get Started 

### Training  data preparation:
If you want to run this code on your own light field microscopy data, first perform background subtraction preprocessing. 
Here, we used ImageJ plugins "Subtract Background" and "DeconvolutionLab2" for this step. Then, generate the corresponding 
ideal PSF "HS.mat" by running the ./FLFM_matlab/PSF_simulation.m file in MATLAB, based on your optic system parameters.


### Commands for Training
***Note***: Users can specify the GPU device for network training or inference in *'glbSettings.py'*.
#### Example 1: Experimental data training
This command was used to represent one scene  with AIR-net. The network parameters and corresponding data directory have been included in the config file ('*exp_train.cfg*'). 
Users can modify the parameters following the code annotations in "main.py". Typing the following command in console to start training.
  ```
  python main.py --flagfile=configs/exp_train.cfg
  ```
  #### Example 2: Simulated data training
 To validate the accuracy of our AIR-net algorithm, we run ./code/sim_data_gen.py to generate PSFs with random aberrations(PSF GT) and corresponding aberrated light field image. 
 The aberrated light field image, along with the ideal PSF(HS.mat), were then used as inputs to train the AIR-net.
  ```
  python main.py --flagfile=configs/sim_train.cfg
  ```

### Commands for Test
The path to pre-trained weights is pre-configured in the configuration file via the "basedir" and "expname" parameters.
#### Example 1: Experimental data testing
* Commands:
  ```
  python main.py --flagfile=configs/exp_test.cfg
  ```
#### Example 1: Simulated data testing
* Commands:
  ```
  python main.py --flagfile=configs/sim_test.cfg
  ```

## Citation
If you use this code and relevant data, please cite the corresponding paper where original methods appeared:
Fenghe Zhong, Xue Li, Mian He, et al. Ultra-fast in vivo deep-tissue 3D imaging with selective-illumination NIR-II light-field microscopy and aberration-corrected implicit neural representation reconstruction (AIR). 
