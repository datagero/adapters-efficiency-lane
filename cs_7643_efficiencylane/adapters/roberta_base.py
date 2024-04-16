from data_loaders.citation_intent_data_loader import CitationIntentDataLoader
from transformers import Trainer, TrainingArguments

from adapters import RobertaAdapterModel, AdapterConfig
from transformers import RobertaTokenizer, RobertaConfig, EvalPrediction, RobertaForSequenceClassification
import torch
import numpy as np
from sklearn.metrics import f1_score

# ======================================================
# Set-up and Load Data
# ======================================================
model_name = 'roberta-base'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = CitationIntentDataLoader(model_name="roberta-base",
                                  dataset_name="citation_intent",
                                  path=f"data/citation_intent/",
                                  checkpoint_path="data/citation_intent/processed_dataset.pt")

dataset = loader.load_dataset(overwrite=False)

# ======================================================
# Model Config & Training
# ======================================================

# Set up training for the Model and Adapter
config = RobertaConfig.from_pretrained(
    model_name,
    num_labels=loader.num_labels,
    problem_type="single_label_classification",
    hidden_dropout_prob=0.1,  # Dropout probability as specified
    # classifier_dropout=0.1    # Additional dropout in the classification head
)

model_variant = model_name
# model_variant = './mlm_model'
# model_variant = "allenai/dsp_roberta_base_tapt_citation_intent_1688" 

model = RobertaForSequenceClassification.from_pretrained(model_variant, config=config)#.to(device)

# Test, using off-the-shelf transformer model
# from transformers import AutoModel, AutoTokenizer 
# model = AutoModel.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# inputs = tokenizer("Hello world!", return_tensors="pt")
# outputs = model(**inputs)

# model.eval()
model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

def compute_accuracy(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": (preds == p.label_ids).mean()}

def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    return {"macro_f1": f1_score(p.label_ids, preds, average='macro')}

# training_args = TrainingArguments(
#     output_dir='./results',
#     num_train_epochs=3,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=16,
#     warmup_steps=500,
#     weight_decay=0.01,
#     logging_dir='./logs',
#     logging_steps=10,
#     evaluation_strategy="epoch",
#     save_strategy="epoch"
# )

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,  # Adjust as needed
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    warmup_steps=500,
    # weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=500,  # Evaluate the model every 500 steps
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_macro_f1"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    compute_metrics=compute_metrics
)

# Start training
trainer.train()
evaluation_results = trainer.evaluate()
print(evaluation_results)
print("Test Macro-F1 Score:", evaluation_results['eval_macro_f1'])

# # Initialize Trainer
# trainer = Trainer(
#     model=model,
#     args=TrainingArguments(
#         output_dir="./results",
#         per_device_eval_batch_size=16,
#         no_cuda=False if torch.cuda.is_available() else True,
#     ),
#     compute_metrics=compute_accuracy,
# )

# # Evaluate the model
# results = trainer.evaluate(dataset['test'])
# print(f"Accuracy: {results['eval_accuracy']}")