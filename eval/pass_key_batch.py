import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
import sys
import math
import torch
import argparse
import random
import numpy as np
from numpy import random
from tqdm import tqdm
import transformers
import pandas as pd
import logging
from transformers import AutoTokenizer

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from HADES import MODEL_DICT, CONFIG_DICT


def set_logger(log_path, log_name='lm-eval', mode='a'):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s')

    file_handler = logging.FileHandler(log_path, mode=mode)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

def load_trained_model(model_name, pretrained_path, num_filters=16, shared_filters=16, gamma=0, device='cuda'):
    config = CONFIG_DICT[model_name]()
    config.num_filters = num_filters
    config.shared_filters = shared_filters
    config.gamma = gamma
    model = MODEL_DICT[model_name](config)
    model.load_state_dict(torch.load(pretrained_path))
    model.to(device=device, dtype=torch.float32)
    return model

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--base_model', type=str, default="HADES")
    parser.add_argument('--pretrained_path', type=str, default="./output/hades_best_checkpoint/epoch_0/pytorch_model.bin")
    parser.add_argument('--result_path', type=str, default="./output")
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--max_tokens', type=int, default=17000, help='maximum token length for evaluation')
    parser.add_argument('--interval', type=int, default=1000, help='interval for evaluation')
    parser.add_argument('--num_tests', type=int, default=10, help='number of repeat testing for each length')
    parser.add_argument('--num_filters', type=int, default=16)
    parser.add_argument('--shared_filters', type=int, default=8)
    parser.add_argument('--gamma', type=float, default=0)
    args = parser.parse_args()
    return args

def generate_prompt_landmark(n_garbage, seed, n_garbage_prefix, tokenizer, answer_prompt_len=2, answer_token_len=2):
    """Generates a text file and inserts an passkey at a random position."""
    rnd_state = random.get_state()
    random.seed(seed)
    n_garbage_suffix = n_garbage - n_garbage_prefix

    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    final_question = "What is the pass key? The pass key is"
    garbage_inf = " ".join([garbage] * 5000)
    assert len(garbage_inf) >= n_garbage
    garbage_prefix = garbage_inf[:n_garbage_prefix]
    garbage_suffix = garbage_inf[:n_garbage_suffix]
    if answer_prompt_len == -1:
        pass_key = random.randint(1, 50000)
        sample_line = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
        sample_lines = [
            task_description,
            garbage_prefix,
            sample_line,
            garbage_suffix,
            final_question,
        ]
        sample_lines = "\n".join(sample_lines)
        sample_lines_len = len(tokenizer(sample_lines).input_ids)
        return sample_lines_len
    
    pass_key_token_len = 0
    pass_key_prompt_len = 0
    while pass_key_prompt_len != answer_prompt_len or pass_key_token_len != answer_token_len:
        pass_key = random.randint(1, 50000)
        pass_key_token_len = tokenizer(str(pass_key), return_tensors="pt").input_ids.shape[-1]
    
        information_line = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
        
        lines = [
            task_description,
            garbage_prefix,
            information_line,
            garbage_suffix,
            final_question,
        ]
        lines = "\n".join(lines)
        lines_ids = tokenizer(lines, return_tensors='pt').input_ids
        pass_key_prompt_len = lines_ids.shape[-1]
    
    random.set_state(rnd_state)
    return lines, str(pass_key)


def passkey_retrieval_test(model, tokenizer, logger, device, n_garbage_prefix, n_garbage=60000, num_test=10, seed=42):
    prompt_len = generate_prompt_landmark(n_garbage, 0, n_garbage_prefix, tokenizer, -1, 2)

    prompts, answers = [], []
    random.seed(seed)
    for i in range(num_test):
        seed = i + random.randint(1, 50000)
        prompt, answer = generate_prompt_landmark(n_garbage, seed, n_garbage_prefix, tokenizer, prompt_len, 2)
        prompts.append(prompt)
        answers.append(answer)

    input_ids = tokenizer(prompts, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    len_token = input_ids.shape[-1] 
    answer_ids = tokenizer(answers, return_tensors="pt").input_ids
        
    max_new_tokens=answer_ids.shape[-1]
    if 'hades' in model.__class__.__name__.lower():
        generation_output = model.generate(
                input_ids=input_ids,
                max_length=len(input_ids[0])+max_new_tokens,
                cg=True,
                return_dict_in_generate=True,
                output_scores=False,
                enable_timing=False,
                temperature=0,
                top_k=1,
            )
    else:
        generation_output = model.generate(
            input_ids=input_ids, 
            max_new_tokens=max_new_tokens, 
            return_dict_in_generate=True, 
            temperature=0,
            num_beams=1, 
            use_cache=True,#False,
            pad_token_id=tokenizer.eos_token_id,  
            # attention_mask=attn_mask, 
            ) 
    
    model_answer = generation_output.sequences[:, -max_new_tokens:].cpu()
    model_answer = [v.strip() for v in tokenizer.batch_decode(model_answer)]
    correct_answer = [v.strip() for v in tokenizer.batch_decode(answer_ids)]
    passed_tests = 0
    total_tokens = 0
    for i in range(num_test):
        is_correct = (model_answer[i] == correct_answer[i])
        logger.info(f"{correct_answer[i]} | {model_answer[i]} | {is_correct}")
        passed_tests += is_correct
        total_tokens += len_token

    return passed_tests, total_tokens


def main(args, logger):
    device = "cuda:0"
    torch.cuda.set_device(device)

    logger.info(f"base model {args.base_model}")
    model_name = args.base_model.lower()
    if model_name in MODEL_DICT:
        model = load_trained_model(model_name, args.pretrained_path, num_filters=args.num_filters, shared_filters=args.shared_filters, gamma=args.gamma, device='cuda')
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", trust_remote_code=True)
    else:
        logger.info("Unable to load model")
        exit()

    logger.info(model)
    total_test_points = args.max_tokens // args.interval
    all_accuries = []
    for i in range(total_test_points):
        # This is a rough ratio to control the number of texts and tokens
        n_garbage = int(3.75 * (i + 1) * args.interval // 1024 * 1024)
        # 10 diffierent n_garbage_prefix for each n_garbage that uniformly distributed
        avg_tokens = None
        for n_garbage_prefix in np.linspace(0, n_garbage, num=20, endpoint=False, dtype=int): # range(0, n_garbage, n_garbage // 20):
            passed_tests, total_tokens = passkey_retrieval_test(model, tokenizer, logger, device, n_garbage_prefix, n_garbage=n_garbage, num_test=args.num_tests, seed=n_garbage_prefix)
            avg_tokens = total_tokens//args.num_tests if avg_tokens is None else avg_tokens
            accuracy = float(passed_tests)/args.num_tests
            depth = n_garbage_prefix/n_garbage
            logger.info("accuracy on the token length %d, depth %f, is %f"%(avg_tokens,depth, accuracy))
            result = {"Context Length": avg_tokens, "Document Depth": round(depth*100, -1),"Score": passed_tests}
            all_accuries.append(result)
    df = pd.DataFrame(all_accuries)

    df['Document Depth'] = [i for i in range(0, 100, 10) for _ in range(2)] * 17
    df.to_csv(f'{args.result_path}/passkey_result_before_pivot.csv')

    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])
    
    pivot_table = pd.pivot_table(df, values='Score', index=['Document Depth', 'Context Length'], aggfunc='mean').reset_index() # This will aggregate
    pivot_table = pivot_table.pivot(index="Document Depth", columns="Context Length", values="Score")

    # Create the heatmap with better aesthetics
    plt.figure(figsize=(20, 11))  # Can adjust these dimensions as needed
    sns.heatmap(
        pivot_table,
        # annot=True,
        fmt="g",
        cmap=cmap,
        cbar_kws={'label': 'Score'},
        vmax=10,
        vmin=0,
    )
    # More aesthetics
    plt.xlabel('Token Limit', fontsize=70)  # X-axis label
    plt.ylabel('Depth Percent', fontsize=70)  # Y-axis label
    plt.xticks(ticks=[0, 4, 8, 12, 16], labels=['1K','4K', '8K','12K','16K'], fontsize=60) 
    plt.yticks(ticks=[0,5,9], labels=[0, 50, 100], fontsize=60)  

    cbar = plt.gca().collections[-1].colorbar
    cbar.ax.tick_params(labelsize=60)  
    cbar.set_label('Score', fontsize=70)  
    plt.tight_layout() 
    # save
    plt.savefig(os.path.join(args.result_path, 'passkey_heatmap_paper.pdf'))
    logger.info(f'save passkey result (heatmap) at {args.result_path}/passkey_heatmap_paper.pdf')
    
    
if __name__ == "__main__":
    args = parse_config()

    # make result_path
    if args.base_model.lower() in MODEL_DICT.keys(): # HADES
        args.result_path = os.path.dirname(args.pretrained_path)
    elif args.baseline:
        args.result_path = os.path.dirname(args.pretrained_path)
    else: # official model
        args.result_path = os.path.join('./evaluation', args.base_model.lower())
        os.makedirs(args.result_path, exist_ok=True)

    log_path = os.path.join(args.result_path, 'passkey.log')
    logger = set_logger(log_path)
    logger.info(' '.join(sys.argv))
    
    main(args, logger)