import sklearn
import torch
from transformers import RobertaForSequenceClassification

def compute_metrics(eval_pred):
    """Computes accuracy, f1, precision, and recall from a 
    transformers.trainer_utils.EvalPrediction object.
    """
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)

    ## TODO: Return a dictionary containing the accuracy, f1, precision, and recall scores.
    ## You may use sklearn's precision_recall_fscore_support and accuracy_score methods.
    accuracy = sklearn.metrics.accuracy_score(labels, preds)
    others = sklearn.metrics.precision_recall_fscore_support(labels, preds)

    return {'accuracy': accuracy,
            'precision': others[0], 
            'recall': others[1], 
            'f1': others[2]}

def model_init():
    """Returns an initialized model for use in a Hugging Face Trainer."""
    ## TODO: Return a pretrained RoBERTa model for sequence classification.
    ## See https://huggingface.co/transformers/model_doc/roberta.html#robertaforsequenceclassification.
    return RobertaForSequenceClassification.from_pretrained('roberta-base')