
MODEL_NAME=HADES
TRAIN_NAME=/Path/to/checkpoint
NUM_FILTERS=16
SHARED_FILTERS=8
GAMMA=0.25
GPU=3
BATCH_SIZE=64

CUDA_VISIBLE_DEVICES=$GPU python -m eval.lm_harness_eval --model $MODEL_NAME \
                                --model_args pretrained=$TRAIN_NAME,num_filters=$NUM_FILTERS,shared_filters=$SHARED_FILTERS,gamma=$GAMMA \
                                --tasks lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,boolq,wikitext \
                                --batch_size $BATCH_SIZE