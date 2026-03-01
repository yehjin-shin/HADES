<div align=center>
<h1>Graph Signal Processing Meets Mamba2: Adaptive Filter Bank via Delta Modulation</h1>

![GitHub Repo stars](https://img.shields.io/github/stars/yehjin-shin/HADES)
[![OpenReview](https://img.shields.io/badge/OpenReview-Forum-blue)](https://openreview.net/forum?id=w0XhHcXfKv)
 [![arXiv](https://img.shields.io/badge/arXiv-1234.12345-b31b1b.svg)](https://arxiv.org/abs/1234.12345)

<div>
    <a href="https://yehjin-shin.github.io/" target="_blank">Yehjin Shin*</a>,
      <a href="https://scholar.google.com/citations?user=4GpvarsAAAAJ&hl=en" target="_blank">Seojin Kim*</a>,
      <a href="https://sites.google.com/view/noseong" target="_blank">Noseong Park</a>,
    <div>
     Korea Advanced Institute of Science and Technology (KAIST)
    </div>
</div>
</div>

---

This is the official PyTorch implementation of the ICLR 2026 paper "Graph Signal Processing Meets Mamba2: Adaptive Filter Bank via Delta Modulation". We introduce **H**ierarchical **AD**aptive filter bank for **E**fficient **S**SMs (*HADES*), a GSP-inspired framework that reinterprets Mamba2 as an adaptive filter bank on a line graph. 

# Setup
## Create Environment
To set up our environment, please run:
```
conda env create -f environment.yaml
conda activate hades
```

Install Mamba:
```
pip install mamba-ssm==2.2.4
pip install causal-conv1d
```

## Additional Requirements - lm_harness_eval
```
pip install lm_eval==0.4.2
```

# Evaluation
To evaluate our model, you should first download the checkpoint from this URL.

https://drive.google.com/drive/folders/1gkKPHcEmepQEvRqAKKwPtwIrV0dYpWDL?usp=share_link


To run language modeling and commonsense reasoning benchmarks:
```
sh scripts/lm_eval.sh 
```
You can modify arguments in .sh file. The primary values are the best hyperparameter values.

Arguments:
* MODEL_NAME=HADES # Model name to test
* TRAIN_NAME=path/to/checkpoint # checkpoint directory.You should put correct dir to test result.
* NUM_FILTERS=16 # Number of selected filters to use (H)
* SHARED_FILTERS=8 # Number of shared filters to use (S)
* GAMMA=0.25 # strength of spectral bias
* GPU=0
* BATCH_SIZE=64

To run passkey retrieval task:
```
sh scripts/passkey_eval.sh 
```
You can modify arguments in .sh file. The primary values are the best hyperparameter values.

Arguments:
* MODEL_NAME=HADES # Model name to test
* TRAIN_NAME=path/to/checkpoint/pytorch_model.bin # checkpoint directory. You should put correct dir to test result.
* NUM_FILTERS=16 # Number of selected filters to use (H)
* SHARED_FILTERS=8 # Number of shared filters to use (S)
* GAMMA=0.25 # strength of spectral bias
* GPU=0

