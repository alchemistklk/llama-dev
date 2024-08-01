import json
from pathlib import Path
from typing import Optional
from typing import Union

from transformers import AutoTokenizer, T5Tokenizer

DEFAULT_HF_MODEL_NAME_DIRS = {
    "llama": "meta-llama/Llama-2-7b-hf",
}

DEFAULT_PROMPT_TEMPLATES = {
    'internlm':
        "<|User|>:{input_text}<eoh>\n<|Bot|>:",
    'qwen':
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n",
}


def get_engine_version(engine_dir: str) -> Union[None, str]:
    engine_dir = Path(engine_dir)
    config_file = engine_dir / "config.json"
    with open(config_file, "r") as f:
        config = json.load(f)

    if 'version' not in config:
        return None
    return config['version']


def read_model_name(engine_dir: str):
    engine_version = get_engine_version(engine_dir)
    with open(Path(engine_dir) / "config.json", "r") as f:
        config = json.load(f)
    if engine_version is None:
        return config["builder_config"]["name"]

    return config["pretrained_config"]["architectures"]


def throttle_generator(generator, stream_interval):
    for i, out in enumerate(generator):
        if not i % stream_interval:
            yield out

    if i % stream_interval:
        yield out


def load_tokenizer(tokenizer_dir: Optional[str] = None, vocab_file: Optional[str] = None,
                   model_name: Optional[str] = "gpt", tokenizer_type: Optional[str] = None):
    if vocab_file is None:
        use_fast = True
        if tokenizer_type is not None and tokenizer_type == "llama":
            use_fast = False
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, legacy=False, padding_side='left',
                                                  trancation_side='left', trust_remote_code=True,
                                                  tokenizer_type=tokenizer_type, use_fast=use_fast)
    else:
        # For gpt-next, directly load from tokenizer.model
        tokenizer = T5Tokenizer(vocab_file=vocab_file, padding_side='left', truncation_side='left')

    if model_name == "qwen":
        with open(Path(tokenizer_dir)/"generation_conifg.json", "r") as f:
            gene_config = json.load(f)
        chat_format = gene_config["chat_format"]
        if chat_format == "raw":
            pad_id = gene_config["pad_token_id"]
            end_id = gene_config["eos_token_id"]
        elif chat_format == "chatml":
            pad_id = tokenizer.im_end_id
            end_id = tokenizer.im_end_id
        else:
            raise Exception(f"Unknown chat_format: {chat_format}")
    elif model_name == 'glm_10b':
        pad_id = tokenizer.pad_token_id
        end_id = tokenizer.eop_token_id
    else:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id
        end_id = tokenizer.eos_token_id
    return tokenizer, pad_id, end_id