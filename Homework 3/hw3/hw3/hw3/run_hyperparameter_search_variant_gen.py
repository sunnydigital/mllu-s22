"""Run a hyperparameter search on a RoBERTa model fine-tuned on BoolQ.

Example usage:
    python run_hyperparameter_search.py BoolQ/
"""
import argparse
import boolq
import data_utils
import finetuning_utils
import json
import pandas as pd
import transformers
import ray.tune as tune

from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizerFast

from ray.tune.suggest.basic_variant import BasicVariantGenerator

parser = argparse.ArgumentParser(
    description="Run a hyperparameter search for finetuning a RoBERTa model on the BoolQ dataset."
)

parser.add_argument(
    "data_dir",
    type=str,
    help="Directory containing the BoolQ dataset. Can be downloaded from https://dl.fbaipublicfiles.com/glue/superglue/data/v2/BoolQ.zip.",
)

args = parser.parse_args()

# Since the labels for the test set have not been released, we will use half of the
# validation dataset as our test dataset for the purposes of this assignment.
train_df = pd.read_json(f"{args.data_dir}/train.jsonl", lines=True, orient="records")
val_df, test_df = train_test_split(
    pd.read_json(f"{args.data_dir}/val.jsonl", lines=True, orient="records"),
    test_size=0.5,
)

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
train_data = boolq.BoolQDataset(train_df, tokenizer)
val_data = boolq.BoolQDataset(val_df, tokenizer)
test_data = boolq.BoolQDataset(test_df, tokenizer)

## TODO: Initialize a transformers.TrainingArguments object here for use in
## training and tuning the model. Consult the assignment handout for some
## sample hyperparameter values.

training_args = transformers.TrainingArguments(output_dir='/scratch/sons01/search_checkpoint',
                                               per_gpu_train_batch_size=8,
                                               per_gpu_eval_batch_size=64,
                                               num_train_epochs=3,
                                               logging_steps=500,
                                               logging_first_step=True,
                                               save_steps=1000,
                                               weight_decay=0.01)

## TODO: Initialize a transformers.Trainer object and run a Bayesian
## hyperparameter search for at least 5 trials (but not too many) on the 
## learning rate. Hint: use the model_init() and
## compute_metrics() methods from finetuning_utils.py as arguments to
## Trainer(). Use the hp_space parameter in hyperparameter_search() to specify
## your hyperparameter search space. (Note that this parameter takes a function
## as its value.)
## Also print out the run ID, objective value,
## and hyperparameters of your best run.

trainer = transformers.Trainer(model_init=finetuning_utils.model_init,
                               args=training_args,
                               train_dataset=train_data,
                               eval_dataset=val_data,
                               tokenizer=tokenizer,
                               compute_metrics=finetuning_utils.compute_metrics)

best_result = trainer.hyperparameter_search(search_alg=BasicVariantGenerator(),
                                               hp_space=lambda _: {'learning_rate': tune.uniform(1e-5, 5e-5)},
                                               n_trials=5,
                                               direction='minimize',
                                               backend='ray',
                                               compute_objective=lambda temp: temp['eval_loss'],
                                               mode='min',
                                               log_to_file=True)

print(best_result.run_id, best_result.objective, best_result.hyperparameters)
