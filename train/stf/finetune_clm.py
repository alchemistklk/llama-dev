import logging
import math
import os
import sys
import random
from dataclasses import dataclass, field
from itertools import chain
import deepspeed
from typing import Optional, List, Union

import datasets
import evaluate
import torch
from datasets import load_dataset
from peft import (  # noqa: E402
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
import transformers
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    BitsAndBytesConfig,
    is_torch_tpu_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

import pdb


logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/tokenizer/config we are going to finetune
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            + ", ".join(MODEL_TYPES)
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )

    torch_type: Optional[str] = field(
        default=None,
        metadata={
            "help": {
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the dtype will be automatically derived from model's weights"
            },
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (
            self.config_name is not None or self.model_name_or_path is not None
        ):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTraningArguments:
    """
    Arguments pertraining to what data we are going to input our model for training and eval
    """

    training_on_inputs: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation set"},
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_files: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "The input training data files (a dataset or a dataset script)."
        },
    )
    validation_files: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data files (a dataset or a dataset script)."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(
        default=False,
        metadata={
            "help": "Whether to load the dataset as a stream or not. Default is False."
        },
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True,
        metadata={"help": "Whether to keep linebreaks in the text when tokenizing."},
    )

    def __post_init__(self):
        if self.streaming:
            require_version(
                "datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`"
            )
        if self.dataset_name is None and self.train_files is None:
            raise ValueError("Need either a dataset name or a training data file.")
        else:
            if self.train_files is not None:
                extension = self.train_files[0].split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "txt",
                ], "`train_file` should be a csv, a json or a txt file."
            if self.validation_files is not None:
                extension = self.validation_files[0].split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "txt",
                ], "`validation_file` should be a csv, a json or a txt file."


def main():
    parser = HfArgumentParser((ModelArguments, DataTraningArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file, let's parse it to get our arguments
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracing the example usage to help us better allocate resources to maintain them.
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    set_seed(training_args.seed)
    """
    Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    """

    if True:
        data_files = {}
        dataset_args = {}
        if training_args.train_file is not None:
            data_files["train"] = training_args.train_file
        if training_args.validation_file is not None:
            data_files["validation"] = training_args.validation_file
        extension = data_files["train"][0].split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks

        raw_datasets = load_dataset( 
            extension,
            data_files=data_files,
            cache_dir=os.path.join(training_args.output_dir, "dataset_cache"),
            use_auth_token=True if data_args.use_auth_token else None,
            **dataset_args,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=os.path.join(training_args.output_dir, "dataset_cache"),
                use_auth_token=True if data_args.use_auth_token else None,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=os.path.join(training_args.output_dir, "dataset_cache"),
                use_auth_token=True if data_args.use_auth_token else None,
                **dataset_args,
            )

    # Setup configuration
    config_kwargs = {
        "cache_dir": training_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if data_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, **config_kwargs
        )
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info("Overriding config with: %s", model_args.config_overrides)
            config.update_from_string(model_args.config_overrides)
            logger.info("New config: %s", config)
    # Setup tokenizer

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if data_args.use_auth_token else None,
        "padding_side": "left",
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, **tokenizer_kwargs
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, **tokenizer_kwargs
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    tokenizer.pad_token = tokenizer.eos_token
    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        print(torch_dtype)
        torch_dtype = torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if data_args.use_auth_token else None,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            use_flash_attention2=True,
            device_map={"": int(os.environ.get("LOCAL_RANK", 0))},
        )
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum({p.dict_pair(): p.numel() for p in model.parameters()}.values())
        logger.info(
            f"Training new model from scratch with {n_params / 2**20:.2f} parameters."
        )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)

    train_on_input = True
    if len(column_names) == 1:
        text_column_name = "text" if "text" in column_names else column_names[0]
    elif len(column_names) == 2:
        input_column_name = "input" if "input" in column_names else column_names[0]
        target_column_name = "target" if "target" in column_names else column_names[1]
        train_on_input = False
    else:
        raise ValueError("输入的列数不对")
    print("train_on_input", train_on_input)

    tok_logger = transformers.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(
                [item for item in examples[text_column_name]],
                truncation=True,
                max_length=data_args.block_size,
                padding=False,
                return_tensors=None,
            )
            output["lables"] = output["input_ids"].clone()
            return output

    def tokenize(prompt):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=data_args.block_size,
            padding=False,
            return_tensors=None,
        )
        result["labels"] = result["input_ids"].clone()
        return result

    def generate_and_tokenize_prompt(data_point):
        input_text = data_point[input_column_name]
        target_text = data_point[target_column_name]
        full_prompt = input_text + target_text
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_input:
            user_prompt = input_text
            tokenized_user_prompt = tokenize(user_prompt)
            user_prompt_length = len(tokenized_user_prompt["input_ids"])
            # [-100, ..., -100, tokenized(input + target)["labels"][user_prompt_length], ...]
            tokenized_full_prompt["labels"] = [-100] * user_prompt_length
            +tokenized_full_prompt["labels"][user_prompt_length:]
        return tokenized_full_prompt

    with training_args.main_process_first(desc="tokenizing datasets"):
        if not data_args.streaming:
            tokenized_datasets = raw_datasets.map(
                (tokenize_function if train_on_input else generate_and_tokenize_prompt),
                batched=True if train_on_input else False,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                (tokenize_function if train_on_input else generate_and_tokenize_prompt),
                batched=True if train_on_input else False,
                remove_columns=column_names,
                desc="Running tokenizer on dataset (streaming)",
            )
    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 2048:
            block_size = 2048

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(data_args.max_train_samples, len(train_dataset))
            train_dataset = train_dataset.select(range(max_train_samples))
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
        train_dataset = train_dataset.shuffle(seed=training_args.seed)

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(data_args.max_eval_samples, len(eval_dataset))
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                logits = logits[0]
            return logits.argmax(dim=-1)

        metrics = evaluate.load("accuracy.py")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)

            # .reshape(-1)
            # print(labels.shape)
            # true_predictions = [
            #     [p for (p, l) in zip(pred, gold_label) if l != -100]
            #     for pred, gold_label in zip(preds, labels)
            # ]
            # true_labels = [
            #     [l for (p, l) in zip(pred, gold_label) if l != -100]
            #     for pred, gold_label in zip(preds, labels)
            # ]
            # preds = np.array(true_predictions).reshape(-1)
            # labels = np.array(true_labels).reshape(-1)
            return metrics.compute(predictions=preds, references=labels)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        compute_metrics=(compute_metrics if training_args.do_eval else None),
        preprocess_logits_for_metrics=(
            preprocess_logits_for_metrics if training_args.do_eval else None
        ),
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        print(training_args.local_rank, "start training")

        if torch.__version__ >= "2" and sys.platform != "win32":
            torch.compile(model)
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )

        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def _mp_fn(index):
    main()


if __name__ == "main":
    main()
