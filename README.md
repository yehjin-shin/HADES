# Graph Signal Processing Meets Mamba2: Adaptive Filter Bank via Delta Modulation

We introduce **H**ierarchical **AD**aptive filter bank for **E**fficient **S**SMs (*HADES*), a GSP-inspired framework that reinterprets Mamba2 as an adaptive filter bank on a line graph. [Paper Link](https://openreview.net/forum?id=cH0OxrmfdL&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICML.cc%2F2025%2FWorkshop%2FES-FoMo-III%2FAuthors%23your-submissions))

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

