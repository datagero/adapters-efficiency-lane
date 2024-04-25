import numpy as np
from transformers import EvalPrediction
from sklearn.metrics import f1_score

def compute_accuracy(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": (preds == p.label_ids).mean()}

def macro_f1(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    return {"macro_f1": f1_score(p.label_ids, preds, average='macro')}

def micro_f1(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    return {"micro_f1": f1_score(p.label_ids, preds, average='micro')}