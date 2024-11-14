import transformers
from transformers import (
    AutoConfig,
    BertTokenizer,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    DataCollatorWithPadding,
    default_data_collator,
    Trainer,
    EvalPrediction,
)
from adapters import BnConfig,AutoAdapterModel,AdapterTrainer
import adapters
import numpy as np
import evaluate
from sklearn.metrics import f1_score
from dataHelper import get_dataset
from functools import partial
from typing import Callable,Optional,List
from dataclasses import dataclass, field
import rich
import logging
import wandb
import os
import sys
import random


"""
initialize seed
"""
random.seed(2024)

def compute_metrics(metric:Callable,lables:np.ndarray,p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    # 得到输出的类
    preds = np.argmax(preds, axis=1)
    #计算accuracy
    result = metric.compute(predictions=preds, references=p.label_ids) 
    result["micro_f1"] = f1_score(
        y_true=preds, y_pred=p.label_ids, average="micro", sample_weight=None,labels=lables #计算micro_f1
    )
    result["macro_f1"] = f1_score(
        y_true=preds, y_pred=p.label_ids, average="macro", sample_weight=None,labels=lables,zero_division=1.0,  #对于全0的类，直接忽略
    )
    return result

logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    dataset_name: str = field(
        default=None,metadata={"help": "The names of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    def __post_init__(self):
        if self.dataset_name is not None:
            self.dataset_name = self.dataset_name.split(",")
    
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_adapter:bool = field(
        default=False,
        metadata={"help": "Whether to use adapter or not."},
    )
    mh_adapter:bool = field(
        default=True,
        metadata={"help": "Whether add adapter after multihead attention layers."},
    )
    output_adapter:bool = field(
        default=True,
        metadata={"help": "Whether add adapter before linear layer."},
    )
    reduction_factor:int = field(
        default=12,#768/12=64
        metadata={"help": "defines the ratio between a model’s layer hidden dimension and the bottleneck dimension."},
    )
    non_linearity:str = field(
        default="gelu",
        metadata={"help": "use what activation function"},

    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )

def main():
    # 解析输入参数
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    parser = model_args, data_args, training_args = parser.parse_args_into_dataclasses()   #命令行解析
    
    ################
    # Logging
    ################
    
    # Setup logging
    # Configure the logging with specific format and date format
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Check if logging is enabled in the training arguments
    if training_args.should_log:
        # Set the logging level to INFO as the default log level in training_args is passive
        transformers.utils.logging.set_verbosity_info()

    # Retrieve and set the log level based on the training arguments
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)  # Set the log level for the logger
    transformers.utils.logging.set_verbosity(log_level)  # Set the verbosity for the transformers logging utility
    transformers.utils.logging.enable_default_handler()  # Enable the default handler to handle the logging
    transformers.utils.logging.enable_explicit_format()  # Enable explicit format for the logging

    # Log a summary of the process, device, and training settings on each process
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    # Log the training and evaluation parameters
    logger.info(f"Training/evaluation parameters {training_args}")
    
    ################
    # Config, Model, Tokenizer (and Dataset)
    ################
    
    # create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    sep_token  = tokenizer.sep_token if tokenizer.sep_token else '<sep>'
    
    # load dataset and get label numbers
    dataset = get_dataset(data_args.dataset_name,sep_token=sep_token)
    num_labels = len(dataset["train"].unique("label"))  

    # prepare config for model
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    # create model 
    model = None
    if model_args.use_adapter:  #if use adapter, then set up models by AutoAdapterModel
        # If using adapters, check if the specified model is "roberta-base
        if "roberta-base" in model_args.model_name_or_path:
            adapters_config = BnConfig(
                mh_adapter=model_args.mh_adapter,
                output_adapter=model_args.output_adapter,
                non_linearity=model_args.non_linearity,
                reduction_factor=model_args.reduction_factor
            )
            
            model = AutoAdapterModel.from_pretrained(
                "roberta-base",
                config=config,
            )
            model.add_classification_head("rotten_tomatoes", num_labels=num_labels)
            model.add_adapter("rotten_tomatoes", config=adapters_config)
            model.train_adapter("rotten_tomatoes")
            model.set_active_adapters("rotten_tomatoes")
        else:
            raise ValueError(f"sorry,the model:{model_args.model_name_or_path} are not on the list")
    else:
        # else we use AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )

    
    # Prints the number of trainable parameters in the model.
    trainable_params = 0
    all_param = 0
    model_shapes = []
    for name, param in model.named_parameters():
        model_shapes.append([param.requires_grad,name,param.shape])
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    import json
    with open("model_structure.json",mode="w",encoding="UTF-8") as f:
        json.dump(model_shapes, f, indent=4)
    logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    ################
    # Processing Dataset and data_collator
    ################
    
    # get the proper max_seq_length for tokenizing
    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    
    def preprocess_function(examples):
        result = tokenizer(examples["text"], padding='max_length', max_length=max_seq_length, truncation=True,return_tensors="pt")
        result["label"] = examples["label"]
        return result
    
    # preprocessing the datasets
    dataset = dataset.map(preprocess_function,
        batched=True,
        desc="Running tokenizer on dataset",
    )

    # the data collator which will be used to process the datasets
    data_collator = default_data_collator #DataCollatorWithPadding(tokenizer=tokenizer,max_length=max_seq_length)
    
    # split the dataset into two parts
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    
    # Get the metric function
    metric = evaluate.load("accuracy")
    # Calculat the total label numbers
    
    # Set the final metrics with accuracy, f1——micro, f1-macro
    my_metrics = partial(compute_metrics,metric,np.arange(num_labels))

    ################
    # Training
    ################
    if model_args.use_adapter: 
        # if is a model with adapter, use AdapterTrainer
        trainer = AdapterTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            compute_metrics=my_metrics,
            tokenizer = tokenizer,
            data_collator=data_collator,
        )
        
        logger.info(
            "start trainning the adapters"
        )
        
    else:
        # else use Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            compute_metrics=my_metrics,
            processing_class=tokenizer,
            data_collator=data_collator,
        )
        logger.info(
                "start trainning the full layer finetune"
            )
    
    if training_args.do_train:
        # training
        train_result = trainer.train()

        # compute train results
        metrics = train_result.metrics
        
        # save train results
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        if training_args.do_eval:
            # compute evaluation results
            metrics = trainer.evaluate()
            # save evaluation results
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
    elif training_args.do_eval:
        # compute evaluation results
        metrics = trainer.evaluate()
        # save evaluation results
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

if __name__ == "__main__":

    main()