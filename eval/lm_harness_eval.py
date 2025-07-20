# usage: lm_harness_eval.py [-h] [--model MODEL] [--tasks task1,task2] [--model_args MODEL_ARGS] [--num_fewshot N] 
                            # [--batch_size auto|auto:N|N] [--max_batch_size N] [--device DEVICE] [--output_path DIR|DIR/file.json] [--limit N|0<N<1] [--use_cache DIR]
#                           [--cache_requests {true,refresh,delete}] [--check_integrity] [--write_out] [--log_samples] 
                            # [--system_instruction SYSTEM_INSTRUCTION] [--apply_chat_template [APPLY_CHAT_TEMPLATE]] [--fewshot_as_multiturn] [--show_config]
#                           [--include_path DIR] [--gen_kwargs GEN_KWARGS] [--verbosity CRITICAL|ERROR|WARNING|INFO|DEBUG] 
                            # [--wandb_args WANDB_ARGS] [--hf_hub_log_args HF_HUB_LOG_ARGS] [--predict_only] [--seed SEED] [--trust_remote_code]

import sys
import torch
import subprocess
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file
from lm_eval.api.model import LM
from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model
from lm_eval.__main__ import cli_evaluate
from accelerate import Accelerator, DistributedType
import accelerate
import logging

from typing import List, Literal, Optional, Tuple, Union
import os
import json
import tqdm
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_scheduler,
)
from HADES import MODEL_DICT, CONFIG_DICT

os.makedirs('output/_logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('output/_logs/log.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = accelerate.logging.get_logger('lm-eval')


class BaseEvalWrapper(HFLM):

    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(
        self,
        pretrained="",
        max_length=2000,
        batch_size=64,
        tokenizer='EleutherAI/gpt-neox-20b',
        device=f"cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float32,
        truncation = False,
        logits_cache = True,
        revision = "main",
        backend = 'causal',
        peft = None,
        delta = None,
        prefix_token_id = None,
        num_filters=None,
        shared_filters=None,
        load_balance_coef=None,
        diversity_coef=None,
        gamma=None,
    ):  # training is everything 32
        LM.__init__(self)

        # Parameters
        self._batch_size = int(batch_size) if batch_size is not None else 64
        self._max_length = max_length
        self._device = torch.device(device)
        self._dtype = dtype
        # Tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.add_bos_token = False
        self.vocab_size = self.tokenizer.vocab_size
        self.truncation = truncation
        self.logits_cache = logits_cache
        self.vocab_size = self.tokenizer.vocab_size
        self.revision = revision
        self.peft = peft
        self.delta = delta
        self.backend = backend
        self.pretrained = pretrained
        self.custom_prefix_token_id = prefix_token_id
        ## custom filters
        self.num_filters=num_filters
        self.shared_filters=shared_filters
        self.load_balance_coef=load_balance_coef
        self.diversity_coef=diversity_coef
        self.gamma=gamma

    def load_model_from_index_json(self, json_path):
        with open(json_path, "r") as f:
            index_data = json.load(f)

        state_dict = {}
        for key, file_name in tqdm.tqdm(index_data["weight_map"].items()):
            file_path = os.path.join(self.pretrained, file_name)
            partial_state_dict = torch.load(file_path)
            state_dict.update(partial_state_dict)

        return state_dict

    def make_pytorch_model(self):
        script_path = os.path.join(self.pretrained, "zero_to_fp32.py")
        try:
            result = subprocess.run(
                ["python", script_path, self.pretrained, self.pretrained, "--max_shard_size=10GB"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                )
        except:
            print(f"You should run {script_path} to get pytorch_model.bin")

    @property
    def batch_size(self):
        return self._batch_size

    def _model_generate(self, **kwargs):
        raise NotImplementedError()
     
     
@register_model("HADES")
class HADESEvalWrapper(BaseEvalWrapper):
    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert os.path.exists(self.pretrained)
        if not os.path.exists(os.path.join(self.pretrained, "pytorch_model.bin")):
            self.make_pytorch_model()

        config = CONFIG_DICT["hades"]()
        config.num_filters = self.num_filters
        config.shared_filters = self.shared_filters
        config.load_balance_coef= self.load_balance_coef
        config.diversity_coef= self.diversity_coef
        config.gamma = self.gamma
        model = MODEL_DICT["hades"](config)
        accelerator = Accelerator(mixed_precision="bf16")
        model = accelerator.prepare(model)
        accelerator.load_state(self.pretrained)
        self._model = model.to(torch.bfloat16)

   
if __name__ == "__main__":
    cli_evaluate()