import os
import sys
import math
import torch
import random
import logging
import datasets
import evaluate
from datasets import load_dataset
from typing import Optional
from dataclasses import dataclass, field
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    HfArgumentParser,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoConfig,
    TrainerState,
    TrainerControl,
    AutoTokenizer,
    TrainerCallback,
    BitsAndBytesConfig,
    is_torch_tpu_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger

from peft import (  # noqa: E402
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers.utils.versions import require_version
from transformers.utils import check_min_version, send_example_telemetry
import transformers.utils.logging

# Declare the path to load model and save model
out_dir = "llama3_lora_finetune"
logger = logging.getLogger(__name__)


# Load parameters of model
MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/tokenizer/config class to finetune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list."
            + ",".join(MODEL_TYPES)
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing settings when the model is trained from scratch. Example: "
            "num_attention_heads=12,hidden_size=768"
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
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded"
        },
    )

    # LoRA parameters
    lora_r: Optional[int] = field(default=16)
    lora_alpha: Optional[float] = field(default=32)

    target_modules: Optional[str] = field(
        default="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj",
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
        },
    )
    use_fast_tokenizer: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to use the fast tokenizer (backed by the tokenizers library) or the original one."
        },
    )
    load_in_bits: Optional[int] = field(default=8)
    model_revision: Optional[str] = field(
        default="main", metadata={"help": "The specific model version to use"}
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

    torch_dtype: Optional[torch.dtype] = field(
        default=None,
        metadata={
            "help": "The dtype of the model. Default to torch.float32",
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (
            self.config_name is not None or self.model_name_or_path is not None
        ):
            raise ValueError(
                "You cannot define both config_overrides and a model configuration/model path."
            )
        if type(self.target_modules) == str:
            self.target_modules = self.target_modules.split(",")


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_on_inputs: bool = field(
        default=False,
        metadata={"help": "Overwrite the cache training and evaluation sets"},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use"}
    )
    train_files: Optional[list[str]] = field(
        default=None, metadata={"help": "The input training data file (a text file)"}
    )
    validation_files: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)"
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this"
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this"
        },
    )
    streaming: bool = field(
        default=False, metadata={"help": "Whether to use streaming or not"}
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={"help": "Optional input sentence length after tokenization"},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    validation_split_percentages: Optional[int] = field(
        default=5,
        metadata={"help": "Percentage of the training data used for validation"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing"},
    )
    keep_linebreaks: bool = field(
        default=True,
        metadata={"help": "Whether to keep the linebreaks in the TXT files"},
    )

    def __post_init__(self):
        if self.streaming:
            require_version(
                "datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`"
            )

        if (
            self.dataset_name is None
            and self.train_files is None
            and self.validation_files is None
        ):
            raise ValueError("Need either a dataset name or a training file.")
        if self.train_files is not None:
            extension = self.train_files[0].split(".")[-1]
            assert extension in [
                "txt",
                "json",
                "csv",
            ], "`train_files` should be a list of files with extensions .txt, .json or csv"
        if self.validation_files is not None:
            extension = self.train_files[0].split(".")[-1]
            assert extension in [
                "txt",
                "json",
                "csv",
            ], "`validation_files` should be a list of files with extensions .txt, .json or csv"


class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.is_world_process_zero:
            print("+++++++++++++++++save call back++++++++++++++++")
            checkpoint_folder = os.path.join(
                args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )
            kwargs["model"].save_pretrained(checkpoint_folder)

            pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                os.remove(pytorch_model_path)
            return control


def main():
    # see all possible arguments in src/transformers/training_args.py
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith("json"):
        # If we pass only one argument and it's the path to a json file, let's parse it to get the arguments
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracing the example usage help us better allocate resources
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
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
        last_checkpoint = transformers.utils.logging.get_last_checkpoint(
            training_args.output_dir
        )
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # set seed before initializing the model
    set_seed(training_args.seed)

    # Process dataset
    if True:
        data_files = {}
        dataset_args = {}
        if data_args.train_files is not None:
            data_files["train"] = data_args.train_files
        if data_args.validation_files is not None:
            data_files["validation"] = data_args.validation_files
        extension = (
            data_args.train_files[0].split(".")[-1]
            if data_args.train_files is not None
            else data_args.validation_files[0].split(".")[-1]
        )

        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=os.path.join(training_args.output_dir, "cache"),
            use_auth_token=True if model_args.use_auth_token else None,
            **dataset_args,
        )
        # If validation data is there, validation_split_percentage will be used to divide the dataset
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentages}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentages}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                **dataset_args,
            )

        # Load pretrained model and tokenizer
        #
        # Distributed training:
        # The .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab
        config_kwargs = {
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "use_auth_token": True if model_args.use_auth_token else None,
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
                logger.info(f"Overriding config with: {model_args.config_overrides}")
                config.update_from_string(model_args.config_overrides)
                logger.info(f"New config: {config}")

        tokenizer_kwargs = {
            "cache_dir": model_args.cache_dir,
            "use_fast": model_args.use_fast_tokenizer,
            "revision": model_args.model_revision,
            "use_auth_token": True if model_args.use_auth_token else None,
            "padding_size": "left",
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

        # LoRA Config
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            # target_modules=["query_key_value"],
            # target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            target_modules=model_args.target_modules,
            fan_in_fan_out=False,
            lora_dropout=0.05,
            inference_mode=False,
            bias="none",
            task_type="CAUSAL_LM",
        )
        print(lora_config)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        torch_dtype = torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            torch_dtype=torch_dtype,
            load_in_8bit=True if model_args.load_in_bits == 8 else False,
            trust_remote_code=True,
            use_flash_attention_2=True,
            quantization_config=(bnb_config if model_args.load_in_bits == 4 else None),
            device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
        )
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Number of parameters: {n_params/2**20:.2f}M")

    # Resize the embeddings only when necessary to avoid index errors
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        tokenizer.resize_token_embeddings(len(tokenizer))
    if model_args.load_in_bits == 8:
        model = prepare_model_for_int8_training(model)
    elif model_args.load_in_bits == 4:
        model = prepare_model_for_kbit_training(model)

    # Process dataset
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)

    train_on_inputs = True
    if len(column_names) == 1:
        text_column_name = "text" if "text" in column_names else column_names[0]
    elif len(column_names) == 2:
        input_column_name = "input" if "input" in column_names else column_names[0]
        target_column_name = "target" if "target" in column_names else column_names[0]
        train_on_inputs = False
    else:
        raise ValueError("输入文件列数不正确")
    print("train_on_inputs", train_on_inputs)

    tak_logger = transformers.utils.logging.get_logger(
        "transformers.data.data_collator"
    )

    def tokenizer_function(examples):
        with CaptureLogger(tak_logger) as cl:
            output = tokenizer(
                examples[text_column_name],
                truncation=True,
                max_length=data_args.block_size,
                padding=False,
                return_tensors=None,
            )
            output["labels"] = output["input_ids"].copy()
        return output

    def tokenize(prompt):
        result = tokenizer(
            prompt,
            truncation=True,
            padding=False,
            max_length=data_args.block_size,
            return_tensors=None,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        input_text = data_point[input_column_name]
        target_text = data_point[target_column_name]
        full_prompt = input_text + target_text
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = input_text
            tokenizer_user_prompt = tokenize(user_prompt)
            user_prompt_length = len(tokenizer_user_prompt["input_ids"])
            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_length + tokenized_full_prompt["labels"][
                user_prompt_length:
            ]
        return tokenized_full_prompt

    with training_args.main_process_first(desc="dataset map tokenization"):
        if not data_args.streaming:
            tokenized_datasets = raw_datasets.map(
                tokenizer_function if train_on_inputs else generate_and_tokenize_prompt,
                batched=True if train_on_inputs else False,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                tokenizer_function if train_on_inputs else generate_and_tokenize_prompt,
                batched=True if train_on_inputs else False,
                remove_columns=column_names,
            )

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 2048:
            block_size = 2048
    else:
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}")
        train_dataset = train_dataset.shuffle(seed=training_args.seed)

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                logits = logits[0]

            return logits.argmax(dim=-1)

        metrics = evaluate.load("accuracy.py")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # align elements of labels and preds
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metrics.compute(predictions=preds, references=labels)

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Initialize Trainer
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        compute_metrics=(
            compute_metrics
            if training_args.do_eval and not is_torch_tpu_available()
            else None
        ),
        preprocess_logits_for_metrics=(
            preprocess_logits_for_metrics
            if training_args.do_eval and not is_torch_tpu_available()
            else None
        ),
        callbacks=([SavePeftModelCallback] if isinstance(model, PeftModel) else None)
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            resume_from_checkpoint = training_args.resume_from_checkpoint
            checkpoint_name = os.path.join(resume_from_checkpoint, "pytorch_model.bin")
            if not os.path.exists(checkpoint_name):
                checkpoint_name = os.path.join(
                    resume_from_checkpoint, "adapter_model.bin"
                )
                resume_from_checkpoint = False

            if os.path.exists(checkpoint_name):
                print("resume from {checkpoint_name}")
                adapters_weights = torch.load(checkpoint_name)
                set_peft_model_state_dict(model, adapters_weights)
            else:
                print(f"Checkpoint {checkpoint_name} not found")
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(len(train_dataset), max_train_samples)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(len(eval_dataset), max_eval_samples)

        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")

        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()
    # main()
    # if training_args.do_train:
    #     if training_args.local_rank not in [-1, 0]:
    #         torch.distributed.barrier()
    #     if training_args.local_rank == 0:
    #         torch.distributed.barrier()
    #         main()
    # elif training_args.do_eval:
    #     main()
    # else:
    #     raise ValueError("At least one of `do_train` or `do_eval` must be True.")
