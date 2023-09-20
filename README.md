# SCR-Torch

Python implementation of the NeurIPS'22 paper ["Selective compression learning of latent representations for variable-rate image compression”](https://arxiv.org/abs/2211.04104)

Official code repository [Link](https://github.com/JooyoungLeeETRI/SCR)
## Introduction

---

This repo is pytorch implementation of SCR. Code is very simple and easy to understand fastly.
Some of these codes are based on [CompressAI](https://github.com/InterDigitalInc/CompressAI)

Currently the test code has been verified to produce results consistent with the officially released code. However, the training code has not been verified yet. Therefore, to run the training code, you should refer to the configuration of the official code.

## Required packages

---

- Python = 3.11.5
- Pytorch = 2.0.1
- CompressAI = 1.2.4
  - g++ installation required 
  
### Installation

---

```bash
$ conda install -c nvidia cudatoolkit=11.8
$ conda install -c conda-forge cudnn
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 
$ pip install tensorboard
$ pip install compressai
$ pip install pyyaml
$ pip install tqdm
```


## Data preparation

---

**Important Note:** The current code is specifically designed to work with the TempDataset. We recommend either overwriting the CSV file located at ./TempDataset/metadata or creating a new **Dataset directory for use with this code.

- **Case 1: Overwrite the CSV file in ./TempDataset/metadata**
    - Refer to the dataset instructions in `./TempDataset/README.md`
    - Modify `temp_train.csv` and `temp_test.csv` to match your target dataset
    - Update the —data_path parameter in the .sh file
- **Case 2: Create a New Dataset Directory**
    - Refer to the dataset instructions in `./TempDataset/README.md`
    - Create a new directory (e.g., `./**Dataset`) and adjust the Dataset functions in `./**Dataset/**Dataset.py` accordingly
    - Write `**_train.csv` and `**_test.csv` to suit your target dataset
    - Update function references in `Train_Codec.py` (line 14, 95, 105) and `Test_Codec.py` (line 7, 52)
    - Update the —data_path parameter in the .sh file

## Training

---

- Ensure that you check the .sh files and set the `$ export CUDA_VISIBLE_DEVICES=”**”` according to your hardware setup.
- Make sure that `—model_name` corresponds to the configuration file located at `./config/model/{model_name}.yaml`.
- Model files (.pth) will be saved in the directory `{—save_path}/Train_record/{model_name}_{exp_name}/`.
- Review the configuration settings in `./config/train/{train_config}.yaml` to ensure they match your training requirements.
- Choose one of the following methods to initiate training:

```bash
$ sh Experiment_Temp.sh. # For single GPU setup
$ sh Distributed_Experiment.sh. # For multi-GPU setup (DDP)
```

## Test

---

- Before testing, please review the .sh file and set the `$ export CUDA_VISIBLE_DEVICES=”**”` environment variable according to your hardware configuration.
- Ensure that the `—model_name` parameter corresponds to the configuration file located at `./config/model/{model_name}.yaml`.
- Model files (.pth) located in the directory `{—save_path}/{model_name}_{exp_name}/Param_{epochs}.pth` will be used for testing.
- The `—epochs` parameter can accept either an integer or a list of integers (e.g., 1, 2, 3).
- If `—epochs` is left unspecified (null), the default model file `{—save_path}/Train_record/{model_name}_{exp_name}/Param.pth` will be used for testing.
- The `—quality_level` parameter can be either an integer or a floating-point number ranging from 1.0 to 8.0.

```bash
$ sh Test_PTModels
```

## P**retrained models**

---

**Important Note:** After downloading the Param.pth file, move it to the directory `{—save_path}/Train_record/{model_name}_{exp_name}/` before use. 
Additionally, for testing, set the `--epochs` parameter to None or null in the `Test_PTModels.sh` script.

- Hyperprior: [Download Link] will be update very soon
- SCR_wo_SC: [Download Link] will be update very soon
- SCR_Full: [[Download Link]](https://drive.google.com/file/d/1Tsz1NKK8jvdOMioBY40mqiwS6mUsmY28/view?usp=sharing)

## **Citation**

---

If you use this project, please cite the relevant original publications for the models and datasets, and cite this project as:

```
@inproceedings{
lee2022selective,
title={Selective compression learning of latent representations for variable-rate image compression},
author={Jooyoung Lee and Seyoon Jeong and Munchurl Kim},
booktitle={Thirty-Sixth Conference on Neural Information Processing Systems},
year={2022},
url={https://openreview.net/forum?id=xI5660uFUr}
}
```