
MODEL_NAME=HADES
TRAIN_NAME=Path/to/checkpoint/pytorch_model.bin
NUM_FILTERS=16
SHARED_FILTERS=8
GAMMA=0.25
GPU=3

CUDA_VISIBLE_DEVICES=$GPU python -m eval.pass_key_batch \
                                --base_model $MODEL_NAME \
                                --pretrained_path $TRAIN_NAME \
                                --num_filters $NUM_FILTERS \
                                --shared_filters $SHARED_FILTERS \
                                --gamma $GAMMA \
